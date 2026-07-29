package com.ryu.sonyremote.processing

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.Canvas
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint
import androidx.exifinterface.media.ExifInterface
import com.ryu.scunetdenoiser.AinrProcessor
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.util.concurrent.atomic.AtomicBoolean

enum class AinrDenoiseModel(
    val label: String,
    val runtimeModel: AinrProcessor.Model,
) {
    Distilled("Distilled", AinrProcessor.Model.DISTILLED),
    Scunet("SCUNet", AinrProcessor.Model.SCUNET),
}

data class AinrProgress(
    val phase: AinrProcessor.Phase,
    val completed: Int,
    val total: Int,
    val detail: String,
) {
    val fraction: Float?
        get() = if (total > 0) completed.toFloat() / total else null
}

class JpegImageProcessor(
    private val ainrProcessor: AinrProcessor? = null,
) {
    private val geometryProcessor = EditorGeometryProcessor()

    fun photographicSensitivity(jpeg: ByteArray): Int? = runCatching {
        ExifInterface(ByteArrayInputStream(jpeg))
            .getAttribute(ExifInterface.TAG_PHOTOGRAPHIC_SENSITIVITY)
            ?.substringBefore(',')
            ?.trim()
            ?.toIntOrNull()
    }.getOrNull()

    fun prepareEditorPreview(
        jpeg: ByteArray,
        geometry: EditorGeometry = EditorGeometry(),
        denoiseStrength: Float = 0f,
        denoiseModel: AinrDenoiseModel = AinrDenoiseModel.Distilled,
        ainrProgress: ((AinrProgress) -> Unit)? = null,
        ainrCanceled: AtomicBoolean = AtomicBoolean(false),
    ): Bitmap = applyDenoise(
        geometryProcessor.apply(
            decode(jpeg, maxEdge = EDITOR_PREVIEW_EDGE, mutable = true),
            geometry,
        ),
        denoiseStrength,
        denoiseModel,
        ainrProgress,
        ainrCanceled,
)

data class LutThumbnailBatch(
    val presets: Map<LutPreset, Bitmap>,
    val imported: Map<String, Bitmap>,
)

    fun renderEditorPreview(
        base: Bitmap,
        preset: LutPreset,
        intensity: Float,
        exposure: Float,
        contrast: Float,
        saturation: Float,
    ): Bitmap = applyBasicEdits(
        applyLut(base.copy(Bitmap.Config.ARGB_8888, true), preset, intensity),
        exposure,
        contrast,
        saturation,
    )

    fun renderEditorPreview(
        base: Bitmap,
        lut: CubeLut,
        intensity: Float,
        exposure: Float,
        contrast: Float,
        saturation: Float,
    ): Bitmap = applyBasicEdits(
        applyLut(base.copy(Bitmap.Config.ARGB_8888, true), lut, intensity),
        exposure,
        contrast,
        saturation,
    )

    fun editPreview(
        jpeg: ByteArray,
        preset: LutPreset,
        intensity: Float,
        exposure: Float,
        contrast: Float,
        saturation: Float,
        denoiseStrength: Float = 0f,
        denoiseModel: AinrDenoiseModel = AinrDenoiseModel.Distilled,
    ): Bitmap = applyBasicEdits(
        applyLut(applyDenoise(decode(jpeg, maxEdge = previewEdge(denoiseStrength), mutable = true), denoiseStrength, denoiseModel), preset, intensity),
        exposure,
        contrast,
        saturation,
    )

    fun applyEditsToJpeg(
        jpeg: ByteArray,
        preset: LutPreset,
        intensity: Float,
        exposure: Float,
        contrast: Float,
        saturation: Float,
        denoiseStrength: Float = 0f,
        denoiseModel: AinrDenoiseModel = AinrDenoiseModel.Distilled,
        geometry: EditorGeometry = EditorGeometry(),
        ainrProgress: ((AinrProgress) -> Unit)? = null,
        ainrCanceled: AtomicBoolean = AtomicBoolean(false),
    ): ByteArray {
        val decoded = decode(jpeg, maxEdge = exportEdge(denoiseStrength), mutable = true)
        val transformed = geometryProcessor.apply(decoded, geometry)
        val bitmap = applyBasicEdits(
            applyLut(
                applyDenoise(
                    transformed,
                    denoiseStrength,
                    denoiseModel,
                    ainrProgress,
                    ainrCanceled,
                ),
                preset,
                intensity,
            ),
            exposure,
            contrast,
            saturation,
        )
        return try { encode(bitmap, JPEG_QUALITY) } finally { bitmap.recycle() }
    }

    fun editPreview(
        jpeg: ByteArray, lut: CubeLut, intensity: Float,
        exposure: Float, contrast: Float, saturation: Float,
        denoiseStrength: Float = 0f,
        denoiseModel: AinrDenoiseModel = AinrDenoiseModel.Distilled,
    ): Bitmap = applyBasicEdits(
        applyLut(applyDenoise(decode(jpeg, maxEdge = previewEdge(denoiseStrength), mutable = true), denoiseStrength, denoiseModel), lut, intensity),
        exposure, contrast, saturation,
    )

    fun applyEditsToJpeg(
        jpeg: ByteArray, lut: CubeLut, intensity: Float,
        exposure: Float, contrast: Float, saturation: Float,
        denoiseStrength: Float = 0f,
        denoiseModel: AinrDenoiseModel = AinrDenoiseModel.Distilled,
        geometry: EditorGeometry = EditorGeometry(),
        ainrProgress: ((AinrProgress) -> Unit)? = null,
        ainrCanceled: AtomicBoolean = AtomicBoolean(false),
    ): ByteArray {
        val decoded = decode(jpeg, maxEdge = exportEdge(denoiseStrength), mutable = true)
        val transformed = geometryProcessor.apply(decoded, geometry)
        val bitmap = applyBasicEdits(
            applyLut(
                applyDenoise(
                    transformed,
                    denoiseStrength,
                    denoiseModel,
                    ainrProgress,
                    ainrCanceled,
                ),
                lut,
                intensity,
            ),
            exposure, contrast, saturation,
        )
        return try { encode(bitmap, JPEG_QUALITY) } finally { bitmap.recycle() }
    }

    fun detectAutoGeometry(
        bitmap: Bitmap,
        includePerspective: Boolean,
    ): AutoGeometrySuggestion = geometryProcessor.detect(bitmap, includePerspective)
    fun thumbnail(jpeg: ByteArray, maxEdge: Int = THUMBNAIL_EDGE): Bitmap =
        decode(jpeg, maxEdge = maxEdge, mutable = false)

    fun lutPreview(jpeg: ByteArray, preset: LutPreset, intensity: Float = 1f): Bitmap =
        applyLut(decode(jpeg, maxEdge = PREVIEW_EDGE, mutable = true), preset, intensity)

    fun liveViewLutPreview(jpeg: ByteArray, preset: LutPreset, intensity: Float): Bitmap =
        applyLut(decode(jpeg, maxEdge = LIVE_PREVIEW_EDGE, mutable = true), preset, intensity)

    fun liveViewLutPreview(jpeg: ByteArray, lut: CubeLut, intensity: Float): Bitmap =
        applyLut(decode(jpeg, maxEdge = LIVE_PREVIEW_EDGE, mutable = true), lut, intensity)

    fun lutThumbnail(jpeg: ByteArray, preset: LutPreset, intensity: Float): Bitmap =
        applyLut(decode(jpeg, maxEdge = LUT_THUMBNAIL_EDGE, mutable = true), preset, intensity)

    fun lutThumbnail(jpeg: ByteArray, lut: CubeLut, intensity: Float): Bitmap =
        applyLut(decode(jpeg, maxEdge = LUT_THUMBNAIL_EDGE, mutable = true), lut, intensity)

    fun lutThumbnailBatch(
        jpeg: ByteArray,
        presets: List<LutPreset>,
        imported: Map<String, CubeLut>,
        intensity: Float,
    ): LutThumbnailBatch {
        val source = decode(jpeg, maxEdge = LUT_THUMBNAIL_EDGE, mutable = true)
        return try {
            LutThumbnailBatch(
                presets = presets.associateWith { preset ->
                    applyLut(source.copy(Bitmap.Config.ARGB_8888, true), preset, intensity)
                },
                imported = imported.mapValues { (_, lut) ->
                    applyLut(source.copy(Bitmap.Config.ARGB_8888, true), lut, intensity)
                },
            )
        } finally {
            source.recycle()
        }
    }

    fun applyLutToJpeg(jpeg: ByteArray, preset: LutPreset, intensity: Float = 1f): ByteArray {
        if (preset == LutPreset.Neutral || intensity <= 0f) return jpeg.copyOf()
        val bitmap = decode(jpeg, maxEdge = null, mutable = true)
        return try {
            encode(applyLut(bitmap, preset, intensity), JPEG_QUALITY)
        } finally {
            if (!bitmap.isRecycled) bitmap.recycle()
        }
    }

    fun applyLutToJpeg(jpeg: ByteArray, lut: CubeLut, intensity: Float = 1f): ByteArray {
        if (intensity <= 0f) return jpeg.copyOf()
        val bitmap = decode(jpeg, maxEdge = null, mutable = true)
        return try {
            encode(applyLut(bitmap, lut, intensity), JPEG_QUALITY)
        } finally {
            if (!bitmap.isRecycled) bitmap.recycle()
        }
    }

    fun decode(jpeg: ByteArray, maxEdge: Int?, mutable: Boolean): Bitmap {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size, bounds)
        require(bounds.outWidth > 0 && bounds.outHeight > 0) { "Could not decode JPEG dimensions" }
        val sampleSize = maxEdge?.let { calculateSampleSize(bounds.outWidth, bounds.outHeight, it) } ?: 1
        val options = BitmapFactory.Options().apply {
            inSampleSize = sampleSize
            inPreferredConfig = Bitmap.Config.ARGB_8888
            inMutable = mutable
        }
        var decoded = requireNotNull(BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size, options)) {
            "Could not decode JPEG"
        }
        if (maxEdge != null && maxOf(decoded.width, decoded.height) > maxEdge) {
            val scale = maxEdge.toFloat() / maxOf(decoded.width, decoded.height)
            val scaled = Bitmap.createScaledBitmap(
                decoded,
                (decoded.width * scale).toInt().coerceAtLeast(1),
                (decoded.height * scale).toInt().coerceAtLeast(1),
                true,
            )
            if (scaled !== decoded) decoded.recycle()
            decoded = scaled
        }
        decoded = applyExifOrientation(decoded, jpeg)
        if (mutable && !decoded.isMutable) {
            val copy = decoded.copy(Bitmap.Config.ARGB_8888, true)
            decoded.recycle()
            decoded = copy
        }
        return decoded
    }

    fun encode(bitmap: Bitmap, quality: Int = JPEG_QUALITY): ByteArray {
        require(quality in 1..100)
        return ByteArrayOutputStream().use { output ->
            check(bitmap.compress(Bitmap.CompressFormat.JPEG, quality, output)) {
                "Could not encode processed JPEG"
            }
            output.toByteArray()
        }
    }

    private fun applyLut(bitmap: Bitmap, preset: LutPreset, intensity: Float): Bitmap {
        if (preset == LutPreset.Neutral || intensity <= 0f) return bitmap
        return applyLut(bitmap, CubeLut.forPreset(preset), intensity)
    }

    private fun applyLut(bitmap: Bitmap, lut: CubeLut, intensity: Float): Bitmap {
        if (intensity <= 0f) return bitmap
        val rows = minOf(STRIPE_ROWS, bitmap.height)
        val pixels = IntArray(bitmap.width * rows)
        var y = 0
        while (y < bitmap.height) {
            val rowCount = minOf(rows, bitmap.height - y)
            val count = bitmap.width * rowCount
            bitmap.getPixels(pixels, 0, bitmap.width, 0, y, bitmap.width, rowCount)
            for (index in 0 until count) pixels[index] = lut.mapArgb(pixels[index], intensity)
            bitmap.setPixels(pixels, 0, bitmap.width, 0, y, bitmap.width, rowCount)
            y += rowCount
        }
        return bitmap
    }

    private fun applyExifOrientation(bitmap: Bitmap, jpeg: ByteArray): Bitmap {
        val exif = runCatching { ExifInterface(ByteArrayInputStream(jpeg)) }.getOrNull() ?: return bitmap
        if (exif.rotationDegrees == 0 && !exif.isFlipped) return bitmap
        val matrix = Matrix().apply {
            if (exif.isFlipped) postScale(-1f, 1f)
            if (exif.rotationDegrees != 0) postRotate(exif.rotationDegrees.toFloat())
        }
        val oriented = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        if (oriented !== bitmap) bitmap.recycle()
        return oriented
    }

    private fun applyDenoise(
        bitmap: Bitmap,
        denoiseStrength: Float,
        denoiseModel: AinrDenoiseModel,
        ainrProgress: ((AinrProgress) -> Unit)? = null,
        ainrCanceled: AtomicBoolean = AtomicBoolean(false),
    ): Bitmap {
        if (denoiseStrength <= 0f) return bitmap
        val processor = checkNotNull(ainrProcessor) { "AINR processing is unavailable" }
        val denoised = processor.process(
            bitmap,
            denoiseModel.runtimeModel,
            ainrCanceled,
        ) { phase, completed, total, detail ->
            ainrProgress?.invoke(AinrProgress(phase, completed, total, detail))
        }
        return blendDenoise(bitmap, denoised, denoiseStrength)
    }

    private fun blendDenoise(original: Bitmap, denoised: Bitmap, strength: Float): Bitmap {
        val amount = strength.coerceIn(0f, 1f)
        if (amount >= 1f) {
            if (original !== denoised) original.recycle()
            return denoised
        }
        if (amount <= 0f) {
            if (denoised !== original) denoised.recycle()
            return original
        }
        val output = Bitmap.createBitmap(original.width, original.height, Bitmap.Config.ARGB_8888)
        Canvas(output).apply {
            drawBitmap(original, 0f, 0f, null)
            drawBitmap(denoised, 0f, 0f, Paint(Paint.ANTI_ALIAS_FLAG).apply {
                alpha = (amount * 255f).toInt()
            })
        }
        original.recycle()
        denoised.recycle()
        return output
    }

    fun blendEditorDenoise(original: Bitmap, denoised: Bitmap, strength: Float): Bitmap {
        require(original.width == denoised.width && original.height == denoised.height)
        val output = Bitmap.createBitmap(original.width, original.height, Bitmap.Config.ARGB_8888)
        Canvas(output).apply {
            drawBitmap(original, 0f, 0f, null)
            drawBitmap(denoised, 0f, 0f, Paint(Paint.ANTI_ALIAS_FLAG).apply {
                alpha = (strength.coerceIn(0f, 1f) * 255f).toInt()
            })
        }
        return output
    }

    private fun previewEdge(denoiseStrength: Float) =
        if (denoiseStrength > 0f) ML_PREVIEW_EDGE else PREVIEW_EDGE

    private fun exportEdge(denoiseStrength: Float): Int? =
        ML_EXPORT_EDGE.takeIf { denoiseStrength > 0f }

    private fun applyBasicEdits(
        bitmap: Bitmap,
        exposure: Float,
        contrast: Float,
        saturation: Float,
    ): Bitmap {
        if (exposure == 0f && contrast == 0f && saturation == 0f) return bitmap
        val output = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
        val matrix = ColorMatrix().apply {
            setSaturation((1f + saturation).coerceIn(0f, 2f))
            val contrastScale = (1f + contrast).coerceIn(0.25f, 2f)
            val exposureOffset = exposure.coerceIn(-1f, 1f) * 96f
            val translate = 128f * (1f - contrastScale) + exposureOffset
            postConcat(ColorMatrix(floatArrayOf(
                contrastScale, 0f, 0f, 0f, translate,
                0f, contrastScale, 0f, 0f, translate,
                0f, 0f, contrastScale, 0f, translate,
                0f, 0f, 0f, 1f, 0f,
            )))
        }
        Canvas(output).drawBitmap(bitmap, 0f, 0f, Paint(Paint.ANTI_ALIAS_FLAG).apply {
            colorFilter = ColorMatrixColorFilter(matrix)
        })
        bitmap.recycle()
        return output
    }

    private fun calculateSampleSize(width: Int, height: Int, maxEdge: Int): Int {
        require(maxEdge > 0)
        var sample = 1
        while (maxOf(width / (sample * 2), height / (sample * 2)) >= maxEdge) sample *= 2
        return sample
    }

    private companion object {
        const val THUMBNAIL_EDGE = 240
        const val PREVIEW_EDGE = 1600
        const val ML_PREVIEW_EDGE = 960
        const val EDITOR_PREVIEW_EDGE = 960
        const val ML_EXPORT_EDGE = Int.MAX_VALUE
        const val LIVE_PREVIEW_EDGE = 960
        const val LUT_THUMBNAIL_EDGE = 200
        const val STRIPE_ROWS = 32
        const val JPEG_QUALITY = 95
    }
}
