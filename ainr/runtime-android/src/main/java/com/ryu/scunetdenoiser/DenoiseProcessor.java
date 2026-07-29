package com.ryu.scunetdenoiser;

import android.os.SystemClock;
import android.util.Log;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

final class DenoiseProcessor {
    enum Mode {
        GPU,
        NPU,
        DUAL
    }

    enum OverlapMode {
        FAST,
        HIGH
    }

    interface ProgressListener {
        void onProgress(int complete, int total, long elapsedMillis);
    }

    static final int TILE = 192;
    static final int PADDING = 8;
    static final int CORE = TILE - 2 * PADDING;
    private static final int PLANE = TILE * TILE;
    private static final int TENSOR_ELEMENTS = 3 * PLANE;
    private static final String TAG = "SCUNetProcessor";

    byte[] process(
        byte[] source,
        int width,
        int height,
        Mode mode,
        OverlapMode overlapMode,
        AcceleratorEngine gpu,
        AcceleratorEngine npu,
        AtomicBoolean canceled,
        ProgressListener listener
    ) throws Exception {
        validateImage(source, width, height);
        if (overlapMode == OverlapMode.HIGH) {
            return processHighOverlap(source, width, height, mode, gpu, npu, canceled, listener);
        }
        int columns = divideRoundUp(width, CORE);
        int rows = divideRoundUp(height, CORE);
        int total = columns * rows;
        byte[] result = new byte[source.length];
        AtomicInteger complete = new AtomicInteger();
        long started = SystemClock.elapsedRealtime();

        if (mode == Mode.GPU || mode == Mode.NPU || rows < 2) {
            AcceleratorEngine engine = mode == Mode.NPU ? require(npu, "NPU") : require(gpu, "GPU");
            processRows(source, result, width, height, columns, 0, rows, engine,
                canceled, complete, total, started, listener);
            return result;
        }

        require(gpu, "GPU");
        require(npu, "NPU");
        TileScheduler scheduler = new TileScheduler(total);
        OrderedComposer composer = new OrderedComposer(result, width, height, columns, total);
        ExecutorService workers = Executors.newFixedThreadPool(2, runnable -> {
            Thread thread = new Thread(runnable);
            thread.setPriority(Thread.NORM_PRIORITY);
            return thread;
        });
        try {
            Future<WorkerResult> npuFuture = workers.submit(() -> processDynamicTiles(
                source, width, height, columns, npu, true, scheduler, composer,
                canceled, complete, total, started, listener));
            Future<WorkerResult> gpuFuture = workers.submit(() -> processDynamicTiles(
                source, width, height, columns, gpu, false, scheduler, composer,
                canceled, complete, total, started, listener));
            WorkerResult npuResult = await(npuFuture);
            WorkerResult gpuResult = await(gpuFuture);
            checkCanceled(canceled);
            composer.finish();
            logDualProfile(npuResult, gpuResult, total);
            return result;
        } finally {
            workers.shutdownNow();
        }
    }

    static int tileCount(int width, int height) {
        return divideRoundUp(width, CORE) * divideRoundUp(height, CORE);
    }

    static int tileCount(int width, int height, OverlapMode overlapMode) {
        if (overlapMode == OverlapMode.HIGH) {
            return (divideRoundUp(width, TILE / 2) + 1)
                * (divideRoundUp(height, TILE / 2) + 1);
        }
        return tileCount(width, height);
    }

    private byte[] processHighOverlap(
        byte[] source,
        int width,
        int height,
        Mode mode,
        AcceleratorEngine gpu,
        AcceleratorEngine npu,
        AtomicBoolean canceled,
        ProgressListener listener
    ) throws Exception {
        final int stride = TILE / 2;
        int columns = divideRoundUp(width, stride) + 1;
        int rows = divideRoundUp(height, stride) + 1;
        int total = columns * rows;
        byte[] result = new byte[source.length];
        AtomicInteger complete = new AtomicInteger();
        long started = SystemClock.elapsedRealtime();
        WeightedOrderedComposer composer = new WeightedOrderedComposer(
            result, width, height, columns, total, stride);

        // Mixing delegate outputs tile-by-tile can create a new 96 px pattern.
        // High overlap prioritizes consistent output and uses one backend.
        if (mode == Mode.GPU || mode == Mode.NPU || mode == Mode.DUAL) {
            AcceleratorEngine engine = mode == Mode.NPU ? require(npu, "NPU") : require(gpu, "GPU");
            processHighTiles(source, width, height, columns, engine, null, composer,
                canceled, complete, total, started, listener);
            composer.finish();
            return result;
        }

        throw new IllegalStateException("Unsupported accelerator mode");
    }

    private WorkerResult processHighTiles(
        byte[] source, int width, int height, int columns,
        AcceleratorEngine engine, TileScheduler scheduler,
        WeightedOrderedComposer composer, AtomicBoolean canceled,
        AtomicInteger complete, int total, long started, ProgressListener listener
    ) throws Exception {
        long wallStarted = SystemClock.elapsedRealtimeNanos();
        long inferenceNanos = 0;
        int processed = 0;
        int maximumIndex = -1;
        float[] input = new float[engine.inputElements()];
        NoiseStrengthEstimator.Workspace noiseWorkspace =
            conditionedWorkspace(input);
        while (true) {
            checkCanceled(canceled);
            int index = scheduler == null ? processed : scheduler.claim(true);
            if (index < 0 || index >= total) break;
            int row = index / columns;
            int column = index % columns;
            int stride = TILE / 2;
            prepareTile(
                source,
                width,
                height,
                input,
                column * stride - stride,
                row * stride - stride,
                noiseWorkspace);
            long inferenceStarted = SystemClock.elapsedRealtimeNanos();
            float[] output = engine.infer(input);
            inferenceNanos += SystemClock.elapsedRealtimeNanos() - inferenceStarted;
            if (output.length != TENSOR_ELEMENTS) {
                throw new IllegalStateException(engine.kind() + " returned " + output.length + " values");
            }
            composer.submit(index, output);
            processed++;
            maximumIndex = Math.max(maximumIndex, index);
            int done = complete.incrementAndGet();
            if (done == 1 || done == total || done % 4 == 0) {
                listener.onProgress(done, total, SystemClock.elapsedRealtime() - started);
            }
        }
        return new WorkerResult(processed, maximumIndex,
            SystemClock.elapsedRealtimeNanos() - wallStarted, inferenceNanos);
    }

    private void processRows(
        byte[] source,
        byte[] result,
        int width,
        int height,
        int columns,
        int firstRow,
        int rowLimit,
        AcceleratorEngine engine,
        AtomicBoolean canceled,
        AtomicInteger complete,
        int total,
        long started,
        ProgressListener listener
    ) throws Exception {
        float[] input = new float[engine.inputElements()];
        NoiseStrengthEstimator.Workspace noiseWorkspace =
            conditionedWorkspace(input);
        List<float[]> previousRow = null;
        for (int row = firstRow; row < rowLimit; row++) {
            checkCanceled(canceled);
            List<float[]> currentRow = new ArrayList<>(columns);
            int coreY = row * CORE;
            for (int column = 0; column < columns; column++) {
                checkCanceled(canceled);
                int coreX = column * CORE;
                prepareTile(
                    source,
                    width,
                    height,
                    input,
                    coreX - PADDING,
                    coreY - PADDING,
                    noiseWorkspace);
                float[] output = engine.infer(input);
                if (output.length != TENSOR_ELEMENTS) {
                    throw new IllegalStateException(
                        engine.kind() + " returned " + output.length + " values");
                }
                composeTile(
                    output,
                    row,
                    column,
                    result,
                    width,
                    height,
                    currentRow,
                    previousRow);

                int done = complete.incrementAndGet();
                if (done == 1 || done == total || done % 4 == 0) {
                    listener.onProgress(done, total, SystemClock.elapsedRealtime() - started);
                }
            }
            previousRow = currentRow;
        }
    }

    private WorkerResult processDynamicTiles(
        byte[] source,
        int width,
        int height,
        int columns,
        AcceleratorEngine engine,
        boolean fromStart,
        TileScheduler scheduler,
        OrderedComposer composer,
        AtomicBoolean canceled,
        AtomicInteger complete,
        int total,
        long started,
        ProgressListener listener
    ) throws Exception {
        long wallStarted = SystemClock.elapsedRealtimeNanos();
        long inferenceNanos = 0;
        int processed = 0;
        int maximumIndex = -1;
        float[] input = new float[engine.inputElements()];
        NoiseStrengthEstimator.Workspace noiseWorkspace =
            conditionedWorkspace(input);

        while (true) {
            checkCanceled(canceled);
            int index = scheduler.claim(fromStart);
            if (index < 0) break;
            int row = index / columns;
            int column = index % columns;
            prepareTile(
                source,
                width,
                height,
                input,
                column * CORE - PADDING,
                row * CORE - PADDING,
                noiseWorkspace);
            long inferenceStarted = SystemClock.elapsedRealtimeNanos();
            float[] output = engine.infer(input);
            long tileInference = SystemClock.elapsedRealtimeNanos() - inferenceStarted;
            inferenceNanos += tileInference;
            if (output.length != TENSOR_ELEMENTS) {
                throw new IllegalStateException(
                    engine.kind() + " returned " + output.length + " values");
            }
            composer.submit(index, output);
            processed++;
            maximumIndex = Math.max(maximumIndex, index);
            if (processed <= 3 || processed % 50 == 0) {
                Log.i(TAG, String.format(
                    Locale.US,
                    "DUAL_TILE backend=%s index=%d worker_tile=%d inference_ms=%.3f",
                    engine.kind(),
                    index,
                    processed,
                    tileInference / 1_000_000.0));
            }

            int done = complete.incrementAndGet();
            if (done == 1 || done == total || done % 4 == 0) {
                listener.onProgress(done, total, SystemClock.elapsedRealtime() - started);
            }
        }

        return new WorkerResult(
            processed,
            maximumIndex,
            SystemClock.elapsedRealtimeNanos() - wallStarted,
            inferenceNanos);
    }

    private static <T> T await(Future<T> future) throws Exception {
        try {
            return future.get();
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            throw new CancellationException("Processing interrupted");
        } catch (ExecutionException error) {
            Throwable cause = error.getCause();
            if (cause instanceof Exception) throw (Exception) cause;
            if (cause instanceof Error) throw (Error) cause;
            throw new IllegalStateException(cause);
        }
    }

    private static void prepareTile(
        byte[] image,
        int width,
        int height,
        float[] tile,
        int startX,
        int startY,
        NoiseStrengthEstimator.Workspace noiseWorkspace
    ) {
        for (int y = 0; y < TILE; y++) {
            int sourceY = reflect(startY + y, height);
            for (int x = 0; x < TILE; x++) {
                int sourceX = reflect(startX + x, width);
                int source = (sourceY * width + sourceX) * 3;
                int destination = y * TILE + x;
                tile[destination] = (image[source] & 0xff) / 255.0f;
                tile[PLANE + destination] = (image[source + 1] & 0xff) / 255.0f;
                tile[2 * PLANE + destination] = (image[source + 2] & 0xff) / 255.0f;
            }
        }
        if (noiseWorkspace != null) {
            NoiseStrengthEstimator.estimateAndFill(tile, noiseWorkspace);
        }
    }

    private static NoiseStrengthEstimator.Workspace conditionedWorkspace(float[] input) {
        if (input.length == TENSOR_ELEMENTS) return null;
        if (input.length == 5 * PLANE || input.length == 7 * PLANE) {
            return new NoiseStrengthEstimator.Workspace();
        }
        throw new IllegalArgumentException("Unsupported model input size " + input.length);
    }

    private static void copyCore(
        float[] tile,
        byte[] image,
        int width,
        int height,
        int destinationX,
        int destinationY
    ) {
        int copyWidth = Math.min(CORE, width - destinationX);
        int copyHeight = Math.min(CORE, height - destinationY);
        for (int y = 0; y < copyHeight; y++) {
            int tileRow = (PADDING + y) * TILE + PADDING;
            int imageRow = ((destinationY + y) * width + destinationX) * 3;
            for (int x = 0; x < copyWidth; x++) {
                int source = tileRow + x;
                int destination = imageRow + x * 3;
                image[destination] = toByte(tile[source]);
                image[destination + 1] = toByte(tile[PLANE + source]);
                image[destination + 2] = toByte(tile[2 * PLANE + source]);
            }
        }
    }

    private static void blendHorizontal(
        float[] left,
        float[] right,
        byte[] image,
        int width,
        int height,
        int seamX,
        int startY
    ) {
        int span = PADDING * 2;
        int yEnd = Math.min(height, startY + CORE);
        for (int y = Math.max(0, startY); y < yEnd; y++) {
            int tileY = PADDING + y - startY;
            for (int x = seamX - PADDING; x < seamX + PADDING && x < width; x++) {
                if (x < 0) continue;
                int offset = x - (seamX - PADDING);
                float weight = (offset + 0.5f) / span;
                putBlendedPixel(image, width, x, y, left, right,
                    tileY * TILE + CORE + offset,
                    tileY * TILE + offset,
                    weight);
            }
        }
    }

    private static void blendVertical(
        float[] top,
        float[] bottom,
        byte[] image,
        int width,
        int height,
        int startX,
        int seamY
    ) {
        int span = PADDING * 2;
        int xEnd = Math.min(width, startX + CORE);
        for (int y = seamY - PADDING; y < seamY + PADDING && y < height; y++) {
            if (y < 0) continue;
            int offset = y - (seamY - PADDING);
            float weight = (offset + 0.5f) / span;
            for (int x = Math.max(0, startX); x < xEnd; x++) {
                int tileX = PADDING + x - startX;
                putBlendedPixel(image, width, x, y, top, bottom,
                    (CORE + offset) * TILE + tileX,
                    offset * TILE + tileX,
                    weight);
            }
        }
    }

    private static void blendCorner(
        float[] topLeft,
        float[] top,
        float[] left,
        float[] current,
        byte[] image,
        int width,
        int height,
        int seamX,
        int seamY
    ) {
        int span = PADDING * 2;
        for (int y = seamY - PADDING; y < seamY + PADDING && y < height; y++) {
            if (y < 0) continue;
            int offsetY = y - (seamY - PADDING);
            float weightY = (offsetY + 0.5f) / span;
            for (int x = seamX - PADDING; x < seamX + PADDING && x < width; x++) {
                if (x < 0) continue;
                int offsetX = x - (seamX - PADDING);
                float weightX = (offsetX + 0.5f) / span;
                int topLeftIndex = (CORE + offsetY) * TILE + CORE + offsetX;
                int topIndex = (CORE + offsetY) * TILE + offsetX;
                int leftIndex = offsetY * TILE + CORE + offsetX;
                int currentIndex = offsetY * TILE + offsetX;
                int destination = (y * width + x) * 3;
                for (int channel = 0; channel < 3; channel++) {
                    int channelOffset = channel * PLANE;
                    float topValue = lerp(
                        topLeft[channelOffset + topLeftIndex],
                        top[channelOffset + topIndex],
                        weightX);
                    float bottomValue = lerp(
                        left[channelOffset + leftIndex],
                        current[channelOffset + currentIndex],
                        weightX);
                    image[destination + channel] = toByte(
                        lerp(topValue, bottomValue, weightY));
                }
            }
        }
    }

    private static void putBlendedPixel(
        byte[] image,
        int width,
        int x,
        int y,
        float[] first,
        float[] second,
        int firstIndex,
        int secondIndex,
        float weight
    ) {
        int destination = (y * width + x) * 3;
        for (int channel = 0; channel < 3; channel++) {
            int channelOffset = channel * PLANE;
            image[destination + channel] = toByte(lerp(
                first[channelOffset + firstIndex],
                second[channelOffset + secondIndex],
                weight));
        }
    }

    private static int reflect(int value, int limit) {
        if (limit <= 1) return 0;
        while (value < 0 || value >= limit) {
            value = value < 0 ? -value : 2 * limit - 2 - value;
        }
        return value;
    }

    private static byte toByte(float value) {
        return (byte) Math.max(0, Math.min(255, Math.round(value * 255.0f)));
    }

    private static float lerp(float first, float second, float weight) {
        return first + (second - first) * weight;
    }

    private static int divideRoundUp(int value, int divisor) {
        return (value + divisor - 1) / divisor;
    }

    private static void logDualProfile(
        WorkerResult npu,
        WorkerResult gpu,
        int total
    ) {
        double concurrentNanos = Math.max(npu.wallNanos, gpu.wallNanos);
        double overlap = concurrentNanos > 0
            ? (npu.inferenceNanos + gpu.inferenceNanos) / concurrentNanos
            : 0;
        Log.i(TAG, String.format(
            Locale.US,
            "DUAL_PROFILE boundary=%d/%d npu_tiles=%d npu_wall_ms=%.3f "
                + "npu_inference_ms=%.3f npu_mean_ms=%.3f gpu_tiles=%d "
                + "gpu_wall_ms=%.3f gpu_inference_ms=%.3f gpu_mean_ms=%.3f "
                + "concurrent_wall_ms=%.3f overlap=%.3f",
            npu.maximumIndex + 1,
            total,
            npu.tiles,
            npu.wallNanos / 1_000_000.0,
            npu.inferenceNanos / 1_000_000.0,
            npu.tiles == 0 ? 0 : npu.inferenceNanos / 1_000_000.0 / npu.tiles,
            gpu.tiles,
            gpu.wallNanos / 1_000_000.0,
            gpu.inferenceNanos / 1_000_000.0,
            gpu.tiles == 0 ? 0 : gpu.inferenceNanos / 1_000_000.0 / gpu.tiles,
            concurrentNanos / 1_000_000.0,
            overlap));
    }

    private static AcceleratorEngine require(AcceleratorEngine engine, String name) {
        if (engine == null) throw new IllegalStateException(name + " engine is unavailable");
        return engine;
    }

    private static void validateImage(byte[] image, int width, int height) {
        if (width <= 0 || height <= 0) throw new IllegalArgumentException("Invalid image size");
        long expected = (long) width * height * 3;
        if (expected > Integer.MAX_VALUE || image.length != expected) {
            throw new IllegalArgumentException(
                "Expected " + expected + " RGB bytes, got " + image.length);
        }
    }

    private static void checkCanceled(AtomicBoolean canceled) {
        if (canceled.get() || Thread.currentThread().isInterrupted()) {
            throw new CancellationException("Denoising canceled");
        }
    }

    private static final class TileScheduler {
        private int nextFirst;
        private int nextLast;

        TileScheduler(int tileCount) {
            nextLast = tileCount - 1;
        }

        synchronized int claim(boolean fromStart) {
            if (nextFirst > nextLast) return -1;
            if (fromStart) return nextFirst++;
            return nextLast--;
        }
    }

    private static final class OrderedComposer {
        private final byte[] result;
        private final int width;
        private final int height;
        private final int columns;
        private final float[][] pending;
        private int nextIndex;
        private List<float[]> previousRow;
        private List<float[]> currentRow;

        OrderedComposer(byte[] result, int width, int height, int columns, int total) {
            this.result = result;
            this.width = width;
            this.height = height;
            this.columns = columns;
            pending = new float[total][];
        }

        synchronized void submit(int index, float[] output) {
            if (index < nextIndex || index >= pending.length || pending[index] != null) {
                throw new IllegalStateException("Duplicate or invalid completed tile " + index);
            }
            pending[index] = output;
            while (nextIndex < pending.length && pending[nextIndex] != null) {
                int row = nextIndex / columns;
                int column = nextIndex % columns;
                if (column == 0) currentRow = new ArrayList<>(columns);
                float[] tile = pending[nextIndex];
                pending[nextIndex] = null;
                composeTile(
                    tile,
                    row,
                    column,
                    result,
                    width,
                    height,
                    currentRow,
                    previousRow);
                if (column == columns - 1) previousRow = currentRow;
                nextIndex++;
            }
        }

        synchronized void finish() {
            if (nextIndex != pending.length) {
                throw new IllegalStateException(
                    "Missing completed tile " + nextIndex + " of " + pending.length);
            }
        }
    }

    /** Streaming whole-tile overlap-add; only two tile rows are retained. */
    private static final class WeightedOrderedComposer {
        private final byte[] result;
        private final int width;
        private final int height;
        private final int columns;
        private final int stride;
        private final float[][] pending;
        private final float[] weights = new float[TILE];
        private final float[] band;
        private int nextIndex;
        private int bandOrigin;

        WeightedOrderedComposer(
            byte[] result, int width, int height, int columns, int total, int stride
        ) {
            this.result = result;
            this.width = width;
            this.height = height;
            this.columns = columns;
            this.stride = stride;
            pending = new float[total][];
            band = new float[width * TILE * 3];
            bandOrigin = -stride;
            for (int i = 0; i < TILE; i++) {
                double angle = Math.PI * (i + 0.5) / TILE;
                weights[i] = (float) (Math.sin(angle) * Math.sin(angle));
            }
        }

        synchronized void submit(int index, float[] output) {
            if (index < nextIndex || index >= pending.length || pending[index] != null) {
                throw new IllegalStateException("Duplicate or invalid completed tile " + index);
            }
            pending[index] = output;
            while (nextIndex < pending.length && pending[nextIndex] != null) {
                int row = nextIndex / columns;
                int column = nextIndex % columns;
                float[] tile = pending[nextIndex];
                pending[nextIndex] = null;
                add(tile, column * stride - stride, row * stride - stride);
                nextIndex++;
                if (column == columns - 1) flush(stride);
            }
        }

        private void add(float[] tile, int startX, int startY) {
            int x0 = Math.max(0, startX);
            int x1 = Math.min(width, startX + TILE);
            int y0 = Math.max(0, startY);
            int y1 = Math.min(height, startY + TILE);
            for (int y = y0; y < y1; y++) {
                int tileY = y - startY;
                int bandY = y - bandOrigin;
                float wy = weights[tileY];
                for (int x = x0; x < x1; x++) {
                    int tileIndex = tileY * TILE + x - startX;
                    int bandIndex = (bandY * width + x) * 3;
                    float weight = wy * weights[x - startX];
                    band[bandIndex] += tile[tileIndex] * weight;
                    band[bandIndex + 1] += tile[PLANE + tileIndex] * weight;
                    band[bandIndex + 2] += tile[2 * PLANE + tileIndex] * weight;
                }
            }
        }

        private void flush(int rows) {
            int y0 = Math.max(0, bandOrigin);
            int y1 = Math.min(height, bandOrigin + rows);
            for (int y = y0; y < y1; y++) {
                int bandOffset = (y - bandOrigin) * width * 3;
                int resultOffset = y * width * 3;
                for (int i = 0; i < width * 3; i++) {
                    result[resultOffset + i] = toByte(band[bandOffset + i]);
                }
            }
            int retained = (TILE - rows) * width * 3;
            System.arraycopy(band, rows * width * 3, band, 0, retained);
            java.util.Arrays.fill(band, retained, band.length, 0f);
            bandOrigin += rows;
        }

        synchronized void finish() {
            if (nextIndex != pending.length) {
                throw new IllegalStateException("Missing completed tile " + nextIndex);
            }
            flush(TILE);
        }
    }

    private static void composeTile(
        float[] output,
        int row,
        int column,
        byte[] result,
        int width,
        int height,
        List<float[]> currentRow,
        List<float[]> previousRow
    ) {
        int coreX = column * CORE;
        int coreY = row * CORE;
        copyCore(output, result, width, height, coreX, coreY);
        float[] left = column > 0 ? currentRow.get(column - 1) : null;
        float[] top = previousRow == null ? null : previousRow.get(column);
        float[] topLeft = previousRow == null || column == 0
            ? null : previousRow.get(column - 1);
        if (left != null) {
            blendHorizontal(left, output, result, width, height, coreX, coreY);
        }
        if (top != null) {
            blendVertical(top, output, result, width, height, coreX, coreY);
        }
        if (topLeft != null) {
            blendCorner(topLeft, top, left, output, result, width, height, coreX, coreY);
        }
        currentRow.add(output);
    }

    private static final class WorkerResult {
        final int tiles;
        final int maximumIndex;
        final long wallNanos;
        final long inferenceNanos;

        WorkerResult(
            int tiles,
            int maximumIndex,
            long wallNanos,
            long inferenceNanos
        ) {
            this.tiles = tiles;
            this.maximumIndex = maximumIndex;
            this.wallNanos = wallNanos;
            this.inferenceNanos = inferenceNanos;
        }
    }
}
