package com.ryu.scunetdenoiser;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Intent;
import android.content.ClipData;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.Switch;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

public final class MainActivity extends Activity {
    private static final String TAG = "SCUNetDenoiser";
    private static final int OPEN_IMAGE_REQUEST = 1001;
    private static final int PREVIEW_MAXIMUM = 1600;

    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private final ExecutorService tasks = Executors.newSingleThreadExecutor(runnable -> {
        Thread thread = new Thread(runnable, "scunet-app-worker");
        thread.setPriority(Thread.NORM_PRIORITY);
        return thread;
    });
    private final ModelStore modelStore = new ModelStore();
    private final ImageStore imageStore = new ImageStore();
    private final DenoiseProcessor processor = new DenoiseProcessor();
    private NpuSupport.Vendor npuVendor = NpuSupport.Vendor.UNSUPPORTED;

    private TextView sessionStatus;
    private TextView emptyPreview;
    private TextView imageDetails;
    private TextView progressStatus;
    private TextView progressPercent;
    private ImageView imagePreview;
    private ProgressBar progressBar;
    private RadioGroup backendGroup;
    private RadioGroup qualityGroup;
    private Switch highOverlapSwitch;
    private RadioButton npuButton;
    private RadioButton dualButton;
    private Button chooseButton;
    private Button runButton;
    private Button cancelButton;
    private AlertDialog cancelConfirmation;
    private Button saveButton;

    private final List<SelectedImage> selectedImages = new ArrayList<>();
    private final List<CompletedImage> completedImages = new ArrayList<>();
    private String selectedName = "image";
    private Bitmap displayedPreview;
    private File completedFile;
    private int completedWidth;
    private int completedHeight;
    private AcceleratorEngine gpuEngine;
    private AcceleratorEngine npuEngine;
    private ModelStore.Quality engineQuality;
    private AtomicBoolean currentCancellation = new AtomicBoolean();
    private boolean busy;
    private boolean destroyed;
    private volatile boolean compilingNpu;
    private long npuCompileStarted;

    private final Runnable compilationTicker = new Runnable() {
        @Override
        public void run() {
            if (!compilingNpu || destroyed) return;
            long elapsed = SystemClock.elapsedRealtime() - npuCompileStarted;
            progressStatus.setText(R.string.compiling_npu);
            progressPercent.setText(formatDuration(elapsed));
            mainHandler.postDelayed(this, 1000);
        }
    };

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        npuVendor = NpuSupport.detect(this);
        Log.i(TAG, "NPU_SUPPORT vendor=" + npuVendor
            + " device=" + NpuSupport.deviceSummary()
            + " android_api=" + Build.VERSION.SDK_INT);
        setContentView(R.layout.activity_main);
        bindViews();
        bindActions();
        handleIncomingIntent(getIntent());
    }

    @Override
    protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        setIntent(intent);
        handleIncomingIntent(intent);
    }

    private void bindViews() {
        sessionStatus = findViewById(R.id.session_status);
        emptyPreview = findViewById(R.id.empty_preview);
        imageDetails = findViewById(R.id.image_details);
        progressStatus = findViewById(R.id.progress_status);
        progressPercent = findViewById(R.id.progress_percent);
        imagePreview = findViewById(R.id.image_preview);
        progressBar = findViewById(R.id.progress_bar);
        backendGroup = findViewById(R.id.backend_group);
        qualityGroup = findViewById(R.id.quality_group);
        qualityGroup.check(getPreferences(MODE_PRIVATE).getBoolean(
            "high_performance_model", false)
            ? R.id.quality_fast
            : R.id.quality_high);
        highOverlapSwitch = findViewById(R.id.high_overlap_switch);
        highOverlapSwitch.setChecked(getPreferences(MODE_PRIVATE)
            .getBoolean("high_overlap", false));
        npuButton = findViewById(R.id.backend_npu);
        dualButton = findViewById(R.id.backend_dual);
        chooseButton = findViewById(R.id.choose_button);
        runButton = findViewById(R.id.run_button);
        cancelButton = findViewById(R.id.cancel_button);
        saveButton = findViewById(R.id.save_button);
        applyNpuAvailability();
        updateSessionLabel();
    }

    private void bindActions() {
        chooseButton.setOnClickListener(view -> openImagePicker());
        runButton.setOnClickListener(view -> startDenoising());
        cancelButton.setOnClickListener(view -> showCancelConfirmation());
        saveButton.setOnClickListener(view -> saveCompletedImage());
        backendGroup.setOnCheckedChangeListener((group, checkedId) -> updateSessionLabel());
        qualityGroup.setOnCheckedChangeListener((group, checkedId) -> {
            boolean highPerformance = checkedId == R.id.quality_fast;
            getPreferences(MODE_PRIVATE).edit()
                .putBoolean("high_performance_model", highPerformance)
                .apply();
            applyNpuAvailability();
            updateSessionLabel();
        });
        highOverlapSwitch.setOnCheckedChangeListener((button, checked) -> {
            getPreferences(MODE_PRIVATE).edit().putBoolean("high_overlap", checked).apply();
            if (checked && selectedMode() == DenoiseProcessor.Mode.DUAL) {
                backendGroup.check(R.id.backend_gpu);
            }
            applyNpuAvailability();
        });
    }

    private void openImagePicker() {
        Intent intent = new Intent(Intent.ACTION_PICK);
        intent.setDataAndType(
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
            "image/*");
        intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true);
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        startActivityForResult(intent, OPEN_IMAGE_REQUEST);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode != OPEN_IMAGE_REQUEST || resultCode != RESULT_OK || data == null) return;
        ArrayList<Uri> uris = new ArrayList<>();
        ClipData clip = data.getClipData();
        if (clip != null) {
            for (int index = 0; index < clip.getItemCount(); index++) {
                Uri uri = clip.getItemAt(index).getUri();
                if (uri != null && !uris.contains(uri)) uris.add(uri);
            }
        } else if (data.getData() != null) {
            uris.add(data.getData());
        }
        if (uris.isEmpty()) return;
        for (Uri uri : uris) persistReadPermission(data, uri);
        selectImages(uris);
    }

    private void handleIncomingIntent(Intent intent) {
        if (intent == null) return;
        Uri uri = null;
        if (Intent.ACTION_VIEW.equals(intent.getAction())) {
            uri = intent.getData();
        } else if (Intent.ACTION_SEND.equals(intent.getAction())
            && intent.getType() != null
            && intent.getType().startsWith("image/")) {
            if (Build.VERSION.SDK_INT >= 33) {
                uri = intent.getParcelableExtra(Intent.EXTRA_STREAM, Uri.class);
            } else {
                @SuppressWarnings("deprecation")
                Uri legacy = intent.getParcelableExtra(Intent.EXTRA_STREAM);
                uri = legacy;
            }
        }
        if (uri != null) selectImages(List.of(uri));
    }

    private void persistReadPermission(Intent data, Uri uri) {
        if ((data.getFlags() & Intent.FLAG_GRANT_READ_URI_PERMISSION) == 0) return;
        try {
            getContentResolver().takePersistableUriPermission(
                uri, Intent.FLAG_GRANT_READ_URI_PERMISSION);
        } catch (SecurityException ignored) {
            // Some providers grant access only for the current process.
        }
    }

    private void selectImages(List<Uri> uris) {
        if (busy) return;
        selectedImages.clear();
        for (int index = 0; index < uris.size(); index++) {
            selectedImages.add(new SelectedImage(
                uris.get(index),
                sourceName(uris.get(index), "image_" + (index + 1))));
        }
        Uri uri = selectedImages.get(0).uri;
        clearCompletedImage();
        setBusy(true);
        currentCancellation = new AtomicBoolean();
        setIndeterminate("Loading image preview");
        tasks.execute(() -> {
            try {
                ImageStore.Preview preview = imageStore.decodePreview(
                    getContentResolver(), uri, PREVIEW_MAXIMUM);
                post(() -> {
                    if (selectedImages.isEmpty()
                        || !uri.equals(selectedImages.get(0).uri)) {
                        preview.bitmap.recycle();
                        return;
                    }
                    selectedName = preview.displayName;
                    selectedImages.get(0).name = preview.displayName;
                    showPreview(preview.bitmap);
                    imageDetails.setText(getString(
                        R.string.image_details_format,
                        selectionLabel(),
                        preview.width,
                        preview.height));
                    emptyPreview.setVisibility(View.GONE);
                    setProgressIdle("Ready to denoise");
                });
            } catch (Throwable error) {
                postError("Could not open image", error);
            } finally {
                post(() -> setBusy(false));
            }
        });
    }

    private void startDenoising() {
        if (busy || selectedImages.isEmpty()) return;
        List<SelectedImage> batch = new ArrayList<>(selectedImages);
        DenoiseProcessor.Mode mode = selectedMode();
        ModelStore.Quality quality = selectedQuality();
        DenoiseProcessor.OverlapMode overlapMode = highOverlapSwitch.isChecked()
            ? DenoiseProcessor.OverlapMode.HIGH
            : DenoiseProcessor.OverlapMode.FAST;
        currentCancellation = new AtomicBoolean();
        AtomicBoolean cancellation = currentCancellation;
        clearCompletedImage();
        setBusy(true);
        setIndeterminate("Preparing model");
        long operationStarted = SystemClock.elapsedRealtime();

        tasks.execute(() -> {
            try {
                boolean needsGpu = mode == DenoiseProcessor.Mode.GPU
                    || mode == DenoiseProcessor.Mode.DUAL;
                boolean needsNpu = mode == DenoiseProcessor.Mode.NPU
                    || mode == DenoiseProcessor.Mode.DUAL;
                ModelStore.Variant gpuVariant = quality == ModelStore.Quality.HIGH_PERFORMANCE
                    ? ModelStore.Variant.HIGH_PERFORMANCE_GPU
                    : ModelStore.Variant.HIGH_QUALITY;
                ModelStore.Variant npuVariant = quality == ModelStore.Quality.HIGH_PERFORMANCE
                    ? ModelStore.Variant.HIGH_PERFORMANCE_NPU
                    : ModelStore.Variant.HIGH_QUALITY;
                File gpuModel = needsGpu
                    ? modelStore.ensureInstalled(
                        this, gpuVariant, cancellation,
                        (copied, total) -> postModelCopyProgress(copied, total))
                    : null;
                File npuModel = needsNpu
                    ? modelStore.ensureInstalled(
                        this, npuVariant, cancellation,
                        (copied, total) -> postModelCopyProgress(copied, total))
                    : null;
                checkCanceled(cancellation);
                ensureEngines(
                    mode, quality, gpuVariant, gpuModel, npuVariant, npuModel,
                    cancellation);
                checkCanceled(cancellation);

                File outputDirectory = new File(getCacheDir(), "completed");
                ArrayList<CompletedImage> outputs = new ArrayList<>();
                Bitmap preview = null;
                int lastWidth = 0;
                int lastHeight = 0;
                for (int batchIndex = 0; batchIndex < batch.size(); batchIndex++) {
                    checkCanceled(cancellation);
                    SelectedImage selected = batch.get(batchIndex);
                    int imageNumber = batchIndex + 1;
                    post(() -> setIndeterminate(batchStatus(
                        imageNumber, batch.size(), "Loading full-resolution image")));
                    ImageStore.RgbImage input = imageStore.decodeFull(
                        getContentResolver(), selected.uri, cancellation);
                    int width = input.width;
                    int height = input.height;
                    int tiles = DenoiseProcessor.tileCount(width, height, overlapMode);
                    post(() -> setDeterminate(
                        batchStatus(imageNumber, batch.size(), "0 / " + tiles + " tiles"),
                        batchProgress(imageNumber, batch.size(), 0),
                        imageNumber + " / " + batch.size()));
                    byte[] denoised = processor.process(
                        input.pixels, width, height, mode, overlapMode,
                        gpuEngine, npuEngine, cancellation,
                        (complete, total, elapsed) -> postBatchTileProgress(
                            imageNumber, batch.size(), complete, total));
                    input = null;
                    checkCanceled(cancellation);
                    post(() -> setIndeterminate(batchStatus(
                        imageNumber, batch.size(), "Encoding JPEG")));
                    File output = new File(outputDirectory, "scunet_completed_" + batchIndex + ".jpg");
                    imageStore.encodeJpeg(denoised, width, height, output, 96, cancellation);
                    imageStore.copyExif(
                        getContentResolver(), selected.uri, output, width, height);
                    denoised = null;
                    outputs.add(new CompletedImage(output, selected.name, width, height));
                    if (preview != null) preview.recycle();
                    preview = imageStore.decodeFilePreview(output, PREVIEW_MAXIMUM);
                    lastWidth = width;
                    lastHeight = height;
                }
                long elapsed = SystemClock.elapsedRealtime() - operationStarted;
                Log.i(TAG, String.format(
                    Locale.US,
                    "DENOISE_COMPLETE mode=%s quality=%s width=%d height=%d images=%d elapsed_ms=%d bytes=%d",
                    mode,
                    quality,
                    lastWidth,
                    lastHeight,
                    outputs.size(),
                    elapsed,
                    outputs.get(outputs.size() - 1).file.length()));

                Bitmap finalPreview = preview;
                int finalWidth = lastWidth;
                int finalHeight = lastHeight;
                post(() -> {
                    completedImages.clear();
                    completedImages.addAll(outputs);
                    CompletedImage last = outputs.get(outputs.size() - 1);
                    completedFile = last.file;
                    completedWidth = finalWidth;
                    completedHeight = finalHeight;
                    showPreview(finalPreview);
                    emptyPreview.setVisibility(View.GONE);
                    progressBar.setIndeterminate(false);
                    progressBar.setProgress(1000);
                    progressStatus.setText(outputs.size() == 1
                        ? getString(R.string.denoising_complete)
                        : outputs.size() + " images completed");
                    progressPercent.setText(formatDuration(elapsed));
                    sessionStatus.setText(getString(
                        R.string.accelerator_ready, modeLabel(mode)));
                });
            } catch (CancellationException error) {
                post(() -> setProgressIdle("Canceled"));
            } catch (OutOfMemoryError error) {
                postError("Not enough memory. Close other apps and retry", error);
            } catch (Throwable error) {
                postError("Denoising failed", error);
            } finally {
                stopNpuTicker();
                post(() -> setBusy(false));
            }
        });
    }

    private void ensureEngines(
        DenoiseProcessor.Mode mode,
        ModelStore.Quality quality,
        ModelStore.Variant gpuVariant,
        File gpuModel,
        ModelStore.Variant npuVariant,
        File npuModel,
        AtomicBoolean cancellation
    ) throws Exception {
        if (engineQuality != null && engineQuality != quality) {
            closeEngines();
        }
        boolean needsNpu = mode == DenoiseProcessor.Mode.NPU || mode == DenoiseProcessor.Mode.DUAL;
        boolean needsGpu = mode == DenoiseProcessor.Mode.GPU || mode == DenoiseProcessor.Mode.DUAL;

        // NPU is intentionally compiled first so the Qualcomm JIT peak does not include a GPU session.
        if (needsNpu && npuEngine == null) {
            if (gpuEngine != null) {
                gpuEngine.close();
                gpuEngine = null;
            }
            startNpuTicker();
            long createStarted = SystemClock.elapsedRealtime();
            AcceleratorEngine created = AcceleratorEngine.createNpu(
                this,
                npuModel,
                modelStore.cacheKey(npuVariant),
                modelStore.tensorSpec(npuVariant));
            stopNpuTicker();
            try {
                created.warmUp();
            } catch (Throwable error) {
                created.close();
                throw error;
            }
            npuEngine = created;
            engineQuality = quality;
            Log.i(TAG, "NPU_JIT_READY quality=" + quality + " vendor=" + npuVendor + " elapsed_ms="
                + (SystemClock.elapsedRealtime() - createStarted));
            checkCanceled(cancellation);
        }
        if (needsGpu && gpuEngine == null) {
            post(() -> setIndeterminate("Preparing GPU"));
            long createStarted = SystemClock.elapsedRealtime();
            AcceleratorEngine created = AcceleratorEngine.createGpu(
                this,
                gpuModel,
                modelStore.cacheKey(gpuVariant),
                modelStore.tensorSpec(gpuVariant));
            try {
                created.warmUp();
            } catch (Throwable error) {
                created.close();
                throw error;
            }
            gpuEngine = created;
            engineQuality = quality;
            Log.i(TAG, "GPU_READY quality=" + quality + " elapsed_ms="
                + (SystemClock.elapsedRealtime() - createStarted));
            checkCanceled(cancellation);
        }
        post(this::updateSessionLabel);
    }

    private void saveCompletedImage() {
        List<CompletedImage> outputs = new ArrayList<>(completedImages);
        if (busy || outputs.isEmpty()) return;
        setBusy(true);
        setIndeterminate("Saving image");
        tasks.execute(() -> {
            try {
                for (int index = 0; index < outputs.size(); index++) {
                    CompletedImage output = outputs.get(index);
                    int number = index + 1;
                    post(() -> setIndeterminate(batchStatus(
                        number, outputs.size(), "Saving to gallery")));
                    imageStore.saveToGallery(getContentResolver(), output.file,
                        ImageStore.outputName(output.name), output.width, output.height);
                }
                post(() -> {
                    progressBar.setIndeterminate(false);
                    progressBar.setProgress(1000);
                    progressStatus.setText(R.string.saved_location);
                    progressPercent.setText(getString(R.string.percent_format, 100));
                    Toast.makeText(
                        this,
                        outputs.size() == 1
                            ? getString(R.string.saved_image,
                                ImageStore.outputName(outputs.get(0).name))
                            : "Saved " + outputs.size() + " images",
                        Toast.LENGTH_LONG).show();
                });
            } catch (Throwable error) {
                postError("Could not save image", error);
            } finally {
                post(() -> setBusy(false));
            }
        });
    }

    private void requestCancellation() {
        currentCancellation.set(true);
        cancelButton.setEnabled(false);
        progressStatus.setText(compilingNpu
            ? R.string.cancel_compiler
            : R.string.cancel_tile);
        progressPercent.setText("");
    }

    private void showCancelConfirmation() {
        if (!busy || currentCancellation.get()) return;
        if (cancelConfirmation != null && cancelConfirmation.isShowing()) return;
        AlertDialog dialog = new AlertDialog.Builder(this)
            .setTitle(R.string.cancel_denoising_title)
            .setMessage(R.string.cancel_denoising_message)
            .setNegativeButton(R.string.continue_denoising, null)
            .setPositiveButton(R.string.confirm_cancel, (ignored, which) -> {
                if (busy && !currentCancellation.get()) requestCancellation();
            })
            .create();
        cancelConfirmation = dialog;
        dialog.setOnDismissListener(ignored -> {
            if (cancelConfirmation == dialog) cancelConfirmation = null;
        });
        dialog.show();
    }

    private void startNpuTicker() {
        compilingNpu = true;
        npuCompileStarted = SystemClock.elapsedRealtime();
        post(() -> {
            sessionStatus.setText(R.string.npu_setup);
            progressBar.setIndeterminate(true);
            mainHandler.removeCallbacks(compilationTicker);
            mainHandler.post(compilationTicker);
        });
    }

    private void stopNpuTicker() {
        compilingNpu = false;
        post(() -> mainHandler.removeCallbacks(compilationTicker));
    }

    private void postModelCopyProgress(long copied, long total) {
        int value = total <= 0 ? 0 : (int) Math.min(1000, copied * 1000 / total);
        post(() -> setDeterminate(
            "Installing denoise model",
            value,
            getString(R.string.percent_format, value / 10)));
    }

    private void postTileProgress(int complete, int total, long elapsedMillis) {
        int value = (int) Math.min(1000, complete * 1000L / total);
        post(() -> setDeterminate(
            getResources().getQuantityString(
                R.plurals.denoising_tiles, total, complete, total),
            value,
            getString(R.string.percent_format, value / 10)));
    }

    private void setDeterminate(String status, int progress, String percent) {
        progressBar.setIndeterminate(false);
        progressBar.setProgress(progress);
        progressStatus.setText(status);
        progressPercent.setText(percent);
    }

    private void setIndeterminate(String status) {
        progressBar.setIndeterminate(true);
        progressStatus.setText(status);
        progressPercent.setText("");
    }

    private void setProgressIdle(String status) {
        progressBar.setIndeterminate(false);
        progressBar.setProgress(0);
        progressStatus.setText(status);
        progressPercent.setText("");
    }

    private void setBusy(boolean value) {
        busy = value;
        chooseButton.setEnabled(!value);
        runButton.setEnabled(!value && !selectedImages.isEmpty());
        saveButton.setEnabled(!value && !completedImages.isEmpty());
        backendGroup.setEnabled(!value);
        qualityGroup.setEnabled(!value);
        highOverlapSwitch.setEnabled(!value);
        for (int index = 0; index < backendGroup.getChildCount(); index++) {
            backendGroup.getChildAt(index).setEnabled(!value);
        }
        for (int index = 0; index < qualityGroup.getChildCount(); index++) {
            qualityGroup.getChildAt(index).setEnabled(!value);
        }
        boolean npuAvailable = npuVendor != NpuSupport.Vendor.UNSUPPORTED;
        npuButton.setEnabled(!value && npuAvailable);
        dualButton.setEnabled(!value && npuAvailable && !highOverlapSwitch.isChecked());
        cancelButton.setVisibility(value ? View.VISIBLE : View.GONE);
        cancelButton.setEnabled(value);
        LinearLayout.LayoutParams saveLayout =
            (LinearLayout.LayoutParams) saveButton.getLayoutParams();
        saveLayout.setMarginStart(value ? dp(8) : 0);
        saveButton.setLayoutParams(saveLayout);
        if (value) {
            getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        } else {
            if (cancelConfirmation != null) cancelConfirmation.dismiss();
            getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        }
    }

    private void updateSessionLabel() {
        DenoiseProcessor.Mode mode = selectedMode();
        if (mode == DenoiseProcessor.Mode.GPU) {
            sessionStatus.setText(gpuEngine == null ? R.string.gpu : R.string.gpu_ready);
        } else if (mode == DenoiseProcessor.Mode.NPU) {
            sessionStatus.setText(npuEngine == null ? R.string.npu_jit : R.string.npu_ready);
        } else {
            sessionStatus.setText(
                gpuEngine != null && npuEngine != null
                    ? R.string.dual_ready
                    : R.string.dual_jit);
        }
    }

    private void applyNpuAvailability() {
        boolean hardwareAvailable = npuVendor != NpuSupport.Vendor.UNSUPPORTED;
        boolean available = hardwareAvailable;
        npuButton.setEnabled(available);
        dualButton.setEnabled(available && !highOverlapSwitch.isChecked());
        if (available) {
            String description = NpuSupport.displayName(npuVendor);
            npuButton.setContentDescription("NPU · " + description);
            dualButton.setContentDescription("GPU + NPU · " + description);
        } else {
            npuButton.setText(R.string.npu_unavailable);
            dualButton.setText(R.string.gpu_npu_unavailable);
            String description = "NPU unavailable on " + NpuSupport.deviceSummary();
            npuButton.setContentDescription(description);
            dualButton.setContentDescription(description);
            backendGroup.check(R.id.backend_gpu);
        }
    }

    private DenoiseProcessor.Mode selectedMode() {
        int checked = backendGroup.getCheckedRadioButtonId();
        if (checked == R.id.backend_npu) return DenoiseProcessor.Mode.NPU;
        if (checked == R.id.backend_dual) return DenoiseProcessor.Mode.DUAL;
        return DenoiseProcessor.Mode.GPU;
    }

    private ModelStore.Quality selectedQuality() {
        return qualityGroup.getCheckedRadioButtonId() == R.id.quality_fast
            ? ModelStore.Quality.HIGH_PERFORMANCE
            : ModelStore.Quality.HIGH_QUALITY;
    }

    private static String modeLabel(DenoiseProcessor.Mode mode) {
        if (mode == DenoiseProcessor.Mode.NPU) return "NPU";
        if (mode == DenoiseProcessor.Mode.DUAL) return "GPU + NPU";
        return "GPU";
    }

    private void showPreview(Bitmap bitmap) {
        Bitmap previous = displayedPreview;
        displayedPreview = bitmap;
        imagePreview.setImageBitmap(bitmap);
        if (previous != null && previous != bitmap && !previous.isRecycled()) previous.recycle();
    }

    private void clearCompletedImage() {
        completedImages.clear();
        completedFile = null;
        completedWidth = 0;
        completedHeight = 0;
        saveButton.setEnabled(false);
    }

    private String selectionLabel() {
        if (selectedImages.size() <= 1) return selectedName;
        return selectedName + " + " + (selectedImages.size() - 1) + " more";
    }

    private String sourceName(Uri uri, String fallback) {
        String value = ImageStore.displayName(getContentResolver(), uri);
        return value == null || value.isBlank() ? fallback : value;
    }

    private static String batchStatus(int current, int total, String detail) {
        return total <= 1 ? detail : "Image " + current + " of " + total + " · " + detail;
    }

    private static int batchProgress(int current, int total, int imageProgress) {
        return (int) Math.min(1000,
            ((current - 1L) * 1000L + imageProgress) / Math.max(1, total));
    }

    private void postBatchTileProgress(int image, int imageTotal, int complete, int total) {
        int imageProgress = (int) Math.min(1000, complete * 1000L / total);
        post(() -> setDeterminate(
            batchStatus(image, imageTotal, complete + " / " + total + " tiles"),
            batchProgress(image, imageTotal, imageProgress),
            image + " / " + imageTotal));
    }

    private static final class SelectedImage {
        final Uri uri;
        String name;

        SelectedImage(Uri uri, String name) {
            this.uri = uri;
            this.name = name;
        }
    }

    private static final class CompletedImage {
        final File file;
        final String name;
        final int width;
        final int height;

        CompletedImage(File file, String name, int width, int height) {
            this.file = file;
            this.name = name;
            this.width = width;
            this.height = height;
        }
    }

    private void postError(String message, Throwable error) {
        Log.e(TAG, message, error);
        String detail = rootMessage(error);
        post(() -> {
            progressBar.setIndeterminate(false);
            progressBar.setProgress(0);
            progressStatus.setText(message);
            progressPercent.setText("");
            Toast.makeText(
                this,
                detail.isBlank() ? message : message + ": " + detail,
                Toast.LENGTH_LONG).show();
        });
    }

    private void post(Runnable runnable) {
        mainHandler.post(() -> {
            if (!destroyed) runnable.run();
        });
    }

    private static String rootMessage(Throwable error) {
        Throwable current = error;
        while (current.getCause() != null && current.getCause() != current) {
            current = current.getCause();
        }
        String value = current.getMessage();
        if (value == null || value.isBlank()) value = current.getClass().getSimpleName();
        return value;
    }

    private static String formatDuration(long millis) {
        long seconds = Math.max(0, millis / 1000);
        return String.format(Locale.US, "%d:%02d", seconds / 60, seconds % 60);
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    private static void checkCanceled(AtomicBoolean cancellation) {
        if (cancellation.get() || Thread.currentThread().isInterrupted()) {
            throw new CancellationException("Operation canceled");
        }
    }

    @Override
    protected void onDestroy() {
        destroyed = true;
        compilingNpu = false;
        mainHandler.removeCallbacksAndMessages(null);
        currentCancellation.set(true);
        if (cancelConfirmation != null) cancelConfirmation.dismiss();
        if (displayedPreview != null && !displayedPreview.isRecycled()) {
            displayedPreview.recycle();
            displayedPreview = null;
        }
        try {
            tasks.execute(() -> {
                closeEngines();
            });
        } catch (Throwable ignored) {
            closeEngines();
        }
        tasks.shutdown();
        super.onDestroy();
    }

    private void closeEngines() {
        if (gpuEngine != null) {
            gpuEngine.close();
            gpuEngine = null;
        }
        if (npuEngine != null) {
            npuEngine.close();
            npuEngine = null;
        }
        engineQuality = null;
    }
}
