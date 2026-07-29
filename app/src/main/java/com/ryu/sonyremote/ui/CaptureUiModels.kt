package com.ryu.sonyremote.ui

import android.graphics.Bitmap
import android.net.Uri
import com.ryu.sonyremote.data.CaptureWorkspace
import com.ryu.sonyremote.processing.LutPreset
import com.ryu.sonyremote.processing.AinrDenoiseModel
import com.ryu.sonyremote.processing.EditorGeometry
import com.ryu.sonyremote.processing.AinrProgress
import com.ryu.sonyremote.processing.CubeLut
import com.ryu.sonyremote.model.CameraSetting
import com.ryu.sonyremote.model.CameraSettingId
import java.net.URI
import java.util.UUID

enum class CaptureMode(val label: String) {
    Photo("Photo"),
    LiveNd("Live ND"),
    LiveComposite("Composite"),
    Panorama("Panorama"),
}

enum class StackCaptureStrategy(val label: String) {
    SingleSequential("Single"),
    ContinuousBurst("Burst"),
}

enum class AutoDenoiseMode(val label: String) {
    Off("Off"),
    Always("Always"),
    IsoThreshold("ISO threshold"),
}

internal fun shouldAutoDenoise(mode: AutoDenoiseMode, iso: Int?, threshold: Int): Boolean = when (mode) {
    AutoDenoiseMode.Off -> false
    AutoDenoiseMode.Always -> true
    AutoDenoiseMode.IsoThreshold -> iso != null && iso >= threshold
}

enum class CaptureAssetKind(val label: String) {
    Photo("Photo"),
    SourceFrame("Frame"),
    LiveNd("Live ND"),
    LiveComposite("Composite"),
    Panorama("Panorama"),
    Lut("LUT"),
}

enum class OriginalImportState {
    NotAvailable,
    Available,
    Queued,
    Downloading,
    Imported,
    Failed,
}

data class LutCaptureState(
    val preset: LutPreset = LutPreset.Neutral,
    val presetIntensities: Map<LutPreset, Float> = LutPreset.entries.associateWith { 1f },
    val visiblePresets: List<LutPreset> = DEFAULT_VISIBLE_LUTS,
    val bakeIntoPhotos: Boolean = false,
    val importedLuts: List<ImportedLut> = emptyList(),
    val selectedImportedLabel: String? = null,
    val importedIntensities: Map<String, Float> = emptyMap(),
) {
    init {
        require(presetIntensities.values.all { it in 0f..1f })
        require(LutPreset.Neutral in visiblePresets)
        require(visiblePresets.distinct().size == visiblePresets.size)
    }

    val importedLut: ImportedLut? get() = importedLuts.firstOrNull { it.label == selectedImportedLabel }
    val importedSelected: Boolean get() = importedLut != null
    val importedIntensity: Float get() = selectedImportedLabel?.let { importedIntensities[it] } ?: 1f
    val intensity: Float get() = if (importedSelected) importedIntensity else presetIntensities[preset] ?: 1f
    val isOriginal: Boolean get() = !importedSelected && (preset == LutPreset.Neutral || intensity == 0f)

    fun select(preset: LutPreset): LutCaptureState = copy(preset = preset, selectedImportedLabel = null)

    fun selectImported(label: String): LutCaptureState =
        if (importedLuts.none { it.label == label }) this else copy(selectedImportedLabel = label)

    fun import(lut: ImportedLut): LutCaptureState = copy(
        importedLuts = importedLuts.filterNot { it.label == lut.label } + lut,
        selectedImportedLabel = lut.label,
        importedIntensities = importedIntensities + (lut.label to 1f),
    )

    fun removeImported(label: String): LutCaptureState = copy(
        importedLuts = importedLuts.filterNot { it.label == label },
        selectedImportedLabel = selectedImportedLabel.takeUnless { it == label },
        importedIntensities = importedIntensities - label,
    )

    fun withIntensity(preset: LutPreset, intensity: Float): LutCaptureState = copy(
        presetIntensities = presetIntensities + (preset to intensity.coerceIn(0f, 1f)),
    )

    fun withImportedIntensity(intensity: Float): LutCaptureState =
        selectedImportedLabel?.let {
            copy(importedIntensities = importedIntensities + (it to intensity.coerceIn(0f, 1f)))
        } ?: this

    fun addPreset(preset: LutPreset): LutCaptureState =
        if (preset in visiblePresets) this else copy(visiblePresets = visiblePresets + preset)

    fun removePreset(preset: LutPreset): LutCaptureState {
        if (preset == LutPreset.Neutral || preset !in visiblePresets) return this
        val remaining = visiblePresets - preset
        return copy(
            visiblePresets = remaining,
            preset = if (this.preset == preset) LutPreset.Neutral else this.preset,
        )
    }

    companion object {
        val DEFAULT_VISIBLE_LUTS = listOf(
            LutPreset.Neutral,
            LutPreset.Cinema,
            LutPreset.Warm,
            LutPreset.Cool,
            LutPreset.Mono,
        )
    }
}

data class ImportedLut(val label: String, val cube: CubeLut, val thumbnail: Bitmap? = null)

data class LutPreviewItem(
    val preset: LutPreset,
    val thumbnail: Bitmap? = null,
    val isLoading: Boolean = false,
    val failed: Boolean = false,
)

data class ImportedCapture(
    val id: String = UUID.randomUUID().toString(),
    val previewUri: Uri? = null,
    val originalUri: Uri? = null,
    val thumbnailRemoteUrl: URI? = null,
    val postviewRemoteUrl: URI? = null,
    val postviewIsOriginal: Boolean = false,
    val cameraContentId: String? = null,
    val isLiveViewPlaceholder: Boolean = false,
    val originalImportState: OriginalImportState = determineOriginalImportState(
        originalUri = originalUri,
        postviewRemoteUrl = postviewRemoteUrl,
        postviewIsOriginal = postviewIsOriginal,
        cameraContentId = cameraContentId,
    ),
    val width: Int? = null,
    val height: Int? = null,
    val downloadBytes: Long = 0,
    val downloadTotalBytes: Long? = null,
    val downloadAttempt: Int = 1,
    val downloadFailed: Boolean = false,
) {
    val downloadFraction: Float?
        get() = downloadTotalBytes?.takeIf { it > 0 }?.let { (downloadBytes.toFloat() / it).coerceIn(0f, 1f) }
    val displayUri: Uri? get() = originalUri ?: previewUri
    val qualityLabel: String
        get() = when {
            originalUri != null -> "Original"
            previewUri != null && originalImportState == OriginalImportState.NotAvailable -> "Preview only"
            previewUri != null -> "Preview"
            downloadFailed -> "Download failed"
            isLiveViewPlaceholder -> if (downloadAttempt > 1) "Retry $downloadAttempt" else "Downloading"
            else -> "Pending"
        }
}

data class OriginalImportRequest(
    val capture: ImportedCapture,
    val shouldEnqueue: Boolean,
)

internal fun requestOriginalImport(capture: ImportedCapture): OriginalImportRequest =
    when (capture.originalImportState) {
        OriginalImportState.Available,
        OriginalImportState.Failed,
        -> OriginalImportRequest(
            capture.copy(originalImportState = OriginalImportState.Queued),
            shouldEnqueue = true,
        )
        OriginalImportState.NotAvailable,
        OriginalImportState.Queued,
        OriginalImportState.Downloading,
        OriginalImportState.Imported,
        -> OriginalImportRequest(capture, shouldEnqueue = false)
    }

internal fun markOriginalImportDownloading(capture: ImportedCapture): ImportedCapture =
    if (capture.originalImportState == OriginalImportState.Queued) {
        capture.copy(originalImportState = OriginalImportState.Downloading)
    } else {
        capture
    }

internal fun markOriginalImportFailed(capture: ImportedCapture): ImportedCapture =
    capture.copy(originalImportState = OriginalImportState.Failed)

internal fun cancelOriginalImport(capture: ImportedCapture): ImportedCapture =
    if (capture.originalImportState in setOf(
            OriginalImportState.Queued,
            OriginalImportState.Downloading,
        )
    ) {
        capture.copy(
            originalImportState = determineOriginalImportState(
                originalUri = capture.originalUri,
                postviewRemoteUrl = capture.postviewRemoteUrl,
                postviewIsOriginal = capture.postviewIsOriginal,
                cameraContentId = capture.cameraContentId,
            ),
        )
    } else {
        capture
    }

internal fun determineOriginalImportState(
    originalUri: Uri?,
    postviewRemoteUrl: URI?,
    postviewIsOriginal: Boolean,
    cameraContentId: String?,
): OriginalImportState =
    if (originalUri != null) {
        OriginalImportState.Imported
    } else if ((postviewRemoteUrl != null && postviewIsOriginal) || cameraContentId != null) {
        OriginalImportState.Available
    } else {
        OriginalImportState.NotAvailable
    }

sealed interface CaptureAssetSource {
    data class MediaStore(val uri: Uri) : CaptureAssetSource
    data class Workspace(val item: CaptureWorkspace.Item) : CaptureAssetSource
    data object LiveViewPlaceholder : CaptureAssetSource
}

data class FilmstripItem(
    val id: String = UUID.randomUUID().toString(),
    val kind: CaptureAssetKind,
    val title: String,
    val source: CaptureAssetSource,
    val thumbnail: Bitmap,
    val sourceCount: Int = 1,
    val importedCapture: ImportedCapture? = null,
    val relatedSourceIds: List<String> = emptyList(),
    val appliedLutName: String? = null,
    val appliedLutStrength: Float? = null,
    val appliedDenoiseModel: AinrDenoiseModel? = null,
    val appliedDenoiseStrength: Float? = null,
    val appliedGeometry: EditorGeometry? = null,
)

data class CaptureSessionUiState(
    val mode: CaptureMode? = null,
    val isActive: Boolean = false,
    val isFinishing: Boolean = false,
    val frameCount: Int = 0,
    val targetFrames: Int? = null,
    val preview: Bitmap? = null,
    val processingProgress: Float? = null,
) {
    val canFinishPanorama: Boolean
        get() = mode == CaptureMode.Panorama && isActive && frameCount >= 2 && !isFinishing
}

data class RoutedPreviews<T>(val main: T?, val pip: T?)

internal fun <T> routeCapturePreviews(
    liveView: T?,
    processedPreview: T?,
    mode: CaptureMode?,
): RoutedPreviews<T> = if (processedPreview != null && mode in setOf(
        CaptureMode.LiveNd,
        CaptureMode.LiveComposite,
        CaptureMode.Panorama,
    )
) {
    RoutedPreviews(
        main = processedPreview,
        pip = liveView.takeIf { mode == CaptureMode.Panorama },
    )
} else {
    RoutedPreviews(main = liveView, pip = null)
}

internal fun exposureModeShortLabel(raw: String): String = when (raw.trim().lowercase()) {
    "program auto", "program", "p" -> "P"
    "aperture priority", "aperture", "a" -> "A"
    "shutter priority", "shutter", "s" -> "S"
    "manual exposure", "manual", "m" -> "M"
    else -> raw
}

internal fun exposurePrioritySetting(
    capabilities: com.ryu.sonyremote.model.CameraCapabilities,
    direction: Int,
): Pair<CameraSetting, String>? {
    if (direction == 0) return null
    val mode = capabilities.settings[CameraSettingId.ExposureMode]?.currentLabel.orEmpty()
    val id = when (exposureModeShortLabel(mode)) {
        "A" -> CameraSettingId.FNumber
        "S" -> CameraSettingId.ShutterSpeed
        "M" -> CameraSettingId.FNumber
        else -> return null
    }
    val setting = capabilities.settings[id]?.takeIf { it.isWritable && it.options.size > 1 } ?: return null
    val current = setting.options.indexOfFirst { it.wireValue == setting.currentWireValue }
        .takeIf { it >= 0 } ?: return null
    val next = (current + if (direction < 0) -1 else 1).coerceIn(0, setting.options.lastIndex)
    return if (next == current) null else setting to setting.options[next].wireValue
}

internal fun zoomTargetReached(current: Int, target: Int, direction: String, tolerance: Int = 1): Boolean =
    if (direction == "in") current >= target - tolerance else current <= target + tolerance

internal fun shutterDurationSeconds(value: String?): Double? {
    val normalized = value?.trim()?.removeSuffix("s") ?: return null
    val parts = normalized.split('/')
    return when (parts.size) {
        1 -> parts[0].toDoubleOrNull()
        2 -> {
            val numerator = parts[0].toDoubleOrNull() ?: return null
            val denominator = parts[1].toDoubleOrNull()?.takeIf { it != 0.0 } ?: return null
            numerator / denominator
        }
        else -> null
    }
}

internal fun liveNdExposureLabel(shutter: String?, frames: Int): String? =
    shutterDurationSeconds(shutter)?.let { seconds ->
        val total = seconds * (frames + 1)
        when {
            total < 1.0 -> "${(total * 1_000).toInt()}ms"
            total < 10.0 -> String.format(java.util.Locale.US, "%.1fs", total)
            else -> "${total.toInt()}s"
        }
    }

internal fun prioritizedPhotoSettings(
    settings: Collection<CameraSetting>,
): Pair<List<CameraSetting>, List<CameraSetting>> {
    val quickOrder = listOf(
        CameraSettingId.ExposureMode,
        CameraSettingId.FNumber,
        CameraSettingId.ShutterSpeed,
        CameraSettingId.IsoSpeedRate,
        CameraSettingId.ExposureCompensation,
    )
    val byId = settings.associateBy(CameraSetting::id)
    val quick = quickOrder.mapNotNull(byId::get)
    return quick to settings.filterNot { it.id in quickOrder }
}

internal fun photoSettingsAfterShutterControls(
    capabilities: com.ryu.sonyremote.model.CameraCapabilities,
): List<CameraSetting> {
    val exposureMode = capabilities.settings[CameraSettingId.ExposureMode]
    val priorityId = when (exposureMode?.currentLabel?.let(::exposureModeShortLabel)) {
        "A", "M" -> CameraSettingId.FNumber
        "S" -> CameraSettingId.ShutterSpeed
        else -> null
    }
    val hidden = setOfNotNull(CameraSettingId.ShootMode, CameraSettingId.ExposureMode, priorityId)
    val (quick, secondary) = prioritizedPhotoSettings(capabilities.settings.values)
    return (quick + secondary).filterNot { it.id in hidden }
}

data class LutEditorUiState(
    val item: FilmstripItem,
    val source: EditorImageSource = EditorImageSource.Processed,
    val hasRetainedOriginal: Boolean = false,
    val preset: LutPreset = LutPreset.Neutral,
    val intensity: Float = 1f,
    val exposure: Float = 0f,
    val contrast: Float = 0f,
    val saturation: Float = 0f,
    val denoiseEnabled: Boolean = false,
    val denoiseStrength: Float = 0.6f,
    val denoiseModel: AinrDenoiseModel = AinrDenoiseModel.Distilled,
    val geometry: EditorGeometry = EditorGeometry(),
    val lutThumbnails: Map<LutPreset, Bitmap> = emptyMap(),
    val importedLuts: List<ImportedLut> = emptyList(),
    val selectedImportedLabel: String? = null,
    val importedLutThumbnails: Map<String, Bitmap> = emptyMap(),
    val isLutThumbnailsLoading: Boolean = false,
    val originalPreview: Bitmap? = null,
    val preview: Bitmap? = null,
    val isProcessing: Boolean = false,
    val ainrProgress: AinrProgress? = null,
    val canUndo: Boolean = false,
    val canRedo: Boolean = false,
) {
    val effectiveDenoiseStrength: Float get() = denoiseStrength.takeIf { denoiseEnabled } ?: 0f
    val hasEdits: Boolean
        get() = selectedImportedLabel != null || preset != LutPreset.Neutral ||
            exposure != 0f || contrast != 0f || saturation != 0f ||
            denoiseEnabled || geometry.hasChanges
}

enum class EditorImageSource(val label: String) {
    Processed("Processed"),
    Original("Original"),
}

enum class EditorTool(val label: String) {
    Adjust("Adjust"),
    Lut("LUT"),
    Denoise("Denoise"),
    Crop("Crop"),
    Rotate("Rotate"),
    Perspective("Perspective"),
}
