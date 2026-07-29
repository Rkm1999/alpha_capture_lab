package com.ryu.scunetdenoiser;

import android.content.Context;
import android.graphics.Bitmap;

import java.io.File;
import java.util.concurrent.CancellationException;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Shared full-resolution AINR entry point used by Alpha Capture Lab and the
 * standalone validation app.
 */
public final class AinrProcessor implements AutoCloseable {
    public enum Model {
        DISTILLED("Distilled"),
        SCUNET("SCUNet");

        public final String label;

        Model(String label) {
            this.label = label;
        }
    }

    public enum Phase {
        INSTALLING_MODEL,
        PREPARING_ACCELERATOR,
        PROCESSING
    }

    public interface ProgressListener {
        void onProgress(Phase phase, int completed, int total, String detail);
    }

    private final Context context;
    private final ModelStore modelStore = new ModelStore();
    private final DenoiseProcessor processor = new DenoiseProcessor();
    private Model activeModel;
    private DenoiseProcessor.Mode activeMode;
    private AcceleratorEngine gpu;
    private AcceleratorEngine npu;
    private boolean npuFailed;

    public AinrProcessor(Context context) {
        this.context = context.getApplicationContext();
    }

    public synchronized String prepare(
        Model model,
        AtomicBoolean canceled,
        ProgressListener listener
    ) throws Exception {
        if (activeModel == model && activeMode != null) {
            return backendLabel();
        }
        closeEngines();
        checkCanceled(canceled);
        ModelStore.Variant variant = variant(model);
        listener.onProgress(Phase.INSTALLING_MODEL, 0, 0, model.label);
        File modelFile = modelStore.ensureInstalled(
            context,
            variant,
            canceled,
            (copied, total) -> listener.onProgress(
                Phase.INSTALLING_MODEL,
                safeInt(copied),
                safeInt(total),
                model.label));
        checkCanceled(canceled);

        boolean canUseNpu = !npuFailed
            && NpuSupport.detect(context) != NpuSupport.Vendor.UNSUPPORTED;
        if (canUseNpu) {
            listener.onProgress(Phase.PREPARING_ACCELERATOR, 0, 0, "NPU");
            try {
                npu = AcceleratorEngine.createNpu(
                    context,
                    modelFile,
                    modelStore.cacheKey(variant),
                    modelStore.tensorSpec(variant));
                npu.warmUp();
            } catch (Throwable error) {
                closeNpu();
                npuFailed = true;
                if (error instanceof Error && !(error instanceof LinkageError)) {
                    throw (Error) error;
                }
            }
        }

        if (npu == null) {
            listener.onProgress(Phase.PREPARING_ACCELERATOR, 0, 0, "GPU");
            gpu = AcceleratorEngine.createGpu(
                context,
                modelFile,
                modelStore.cacheKey(variant),
                modelStore.tensorSpec(variant));
            gpu.warmUp();
        }

        activeModel = model;
        activeMode = npu != null
            ? (gpu != null ? DenoiseProcessor.Mode.DUAL : DenoiseProcessor.Mode.NPU)
            : DenoiseProcessor.Mode.GPU;
        return backendLabel();
    }

    public synchronized Bitmap process(
        Bitmap source,
        Model model,
        AtomicBoolean canceled,
        ProgressListener listener
    ) throws Exception {
        prepare(model, canceled, listener);
        checkCanceled(canceled);
        int width = source.getWidth();
        int height = source.getHeight();
        byte[] input = bitmapToRgb(source, canceled);
        byte[] output;
        try {
            output = run(input, width, height, canceled, listener);
        } catch (CancellationException error) {
            throw error;
        } catch (Throwable acceleratedFailure) {
            if (activeMode == DenoiseProcessor.Mode.GPU) {
                if (acceleratedFailure instanceof Exception) {
                    throw (Exception) acceleratedFailure;
                }
                throw (Error) acceleratedFailure;
            }
            closeEngines();
            npuFailed = true;
            activeModel = null;
            prepareGpu(model, canceled, listener);
            output = run(input, width, height, canceled, listener);
        }
        return rgbToBitmap(output, width, height, canceled);
    }

    public synchronized String backendLabel() {
        if (activeMode == DenoiseProcessor.Mode.DUAL) return "GPU + NPU";
        if (activeMode == DenoiseProcessor.Mode.NPU) return "NPU";
        if (activeMode == DenoiseProcessor.Mode.GPU) return "GPU";
        return "Not prepared";
    }

    private byte[] run(
        byte[] input,
        int width,
        int height,
        AtomicBoolean canceled,
        ProgressListener listener
    ) throws Exception {
        int total = DenoiseProcessor.tileCount(
            width, height, DenoiseProcessor.OverlapMode.FAST);
        listener.onProgress(Phase.PROCESSING, 0, total, backendLabel());
        return processor.process(
            input,
            width,
            height,
            activeMode,
            DenoiseProcessor.OverlapMode.FAST,
            gpu,
            npu,
            canceled,
            (complete, count, elapsed) -> listener.onProgress(
                Phase.PROCESSING,
                complete,
                count,
                backendLabel()));
    }

    private void prepareGpu(
        Model model,
        AtomicBoolean canceled,
        ProgressListener listener
    ) throws Exception {
        ModelStore.Variant variant = variant(model);
        File modelFile = modelStore.ensureInstalled(
            context,
            variant,
            canceled,
            (copied, total) -> listener.onProgress(
                Phase.INSTALLING_MODEL,
                safeInt(copied),
                safeInt(total),
                model.label));
        listener.onProgress(Phase.PREPARING_ACCELERATOR, 0, 0, "GPU fallback");
        gpu = AcceleratorEngine.createGpu(
            context,
            modelFile,
            modelStore.cacheKey(variant),
            modelStore.tensorSpec(variant));
        gpu.warmUp();
        activeModel = model;
        activeMode = DenoiseProcessor.Mode.GPU;
    }

    private static ModelStore.Variant variant(Model model) {
        return model == Model.DISTILLED
            ? ModelStore.Variant.HIGH_PERFORMANCE_GPU
            : ModelStore.Variant.HIGH_QUALITY;
    }

    private static byte[] bitmapToRgb(Bitmap bitmap, AtomicBoolean canceled) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        byte[] result = new byte[Math.multiplyExact(Math.multiplyExact(width, height), 3)];
        int[] row = new int[width];
        for (int y = 0; y < height; y++) {
            checkCanceled(canceled);
            bitmap.getPixels(row, 0, width, 0, y, width, 1);
            int destination = y * width * 3;
            for (int color : row) {
                result[destination++] = (byte) ((color >>> 16) & 0xff);
                result[destination++] = (byte) ((color >>> 8) & 0xff);
                result[destination++] = (byte) (color & 0xff);
            }
        }
        return result;
    }

    private static Bitmap rgbToBitmap(
        byte[] rgb,
        int width,
        int height,
        AtomicBoolean canceled
    ) {
        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        int[] row = new int[width];
        for (int y = 0; y < height; y++) {
            checkCanceled(canceled);
            int source = y * width * 3;
            for (int x = 0; x < width; x++) {
                row[x] = 0xff000000
                    | ((rgb[source++] & 0xff) << 16)
                    | ((rgb[source++] & 0xff) << 8)
                    | (rgb[source++] & 0xff);
            }
            bitmap.setPixels(row, 0, width, 0, y, width, 1);
        }
        return bitmap;
    }

    private static int safeInt(long value) {
        return (int) Math.min(Integer.MAX_VALUE, Math.max(0, value));
    }

    private static void checkCanceled(AtomicBoolean canceled) {
        if (canceled.get() || Thread.currentThread().isInterrupted()) {
            throw new CancellationException("AINR canceled");
        }
    }

    private void closeEngines() {
        if (gpu != null) {
            gpu.close();
            gpu = null;
        }
        closeNpu();
        activeModel = null;
        activeMode = null;
    }

    private void closeNpu() {
        if (npu != null) {
            npu.close();
            npu = null;
        }
    }

    @Override
    public synchronized void close() {
        closeEngines();
    }
}
