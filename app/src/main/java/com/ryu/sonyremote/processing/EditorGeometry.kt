package com.ryu.sonyremote.processing

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.RectF
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.hypot

data class NormalizedPoint(
    val x: Float,
    val y: Float,
) {
    fun clamped(): NormalizedPoint = NormalizedPoint(
        x.coerceIn(0f, 1f),
        y.coerceIn(0f, 1f),
    )
}

data class NormalizedCrop(
    val left: Float = 0f,
    val top: Float = 0f,
    val right: Float = 1f,
    val bottom: Float = 1f,
) {
    fun normalized(minimumSize: Float = 0.05f): NormalizedCrop {
        val l = left.coerceIn(0f, 1f - minimumSize)
        val t = top.coerceIn(0f, 1f - minimumSize)
        return NormalizedCrop(
            left = l,
            top = t,
            right = right.coerceIn(l + minimumSize, 1f),
            bottom = bottom.coerceIn(t + minimumSize, 1f),
        )
    }

    val isFullFrame: Boolean
        get() = left == 0f && top == 0f && right == 1f && bottom == 1f
}

data class EditorGeometry(
    val crop: NormalizedCrop = NormalizedCrop(),
    val quarterTurns: Int = 0,
    val straightenDegrees: Float = 0f,
    val perspective: List<NormalizedPoint> = FULL_FRAME_CORNERS,
) {
    val normalizedQuarterTurns: Int get() = ((quarterTurns % 4) + 4) % 4
    val hasPerspective: Boolean get() = perspective != FULL_FRAME_CORNERS
    val hasChanges: Boolean
        get() = !crop.isFullFrame || normalizedQuarterTurns != 0 ||
            straightenDegrees != 0f || hasPerspective

    fun normalized(): EditorGeometry = copy(
        crop = crop.normalized(),
        quarterTurns = normalizedQuarterTurns,
        straightenDegrees = straightenDegrees.coerceIn(-45f, 45f),
        perspective = normalizePerspective(perspective),
    )

    companion object {
        val FULL_FRAME_CORNERS = listOf(
            NormalizedPoint(0f, 0f),
            NormalizedPoint(1f, 0f),
            NormalizedPoint(1f, 1f),
            NormalizedPoint(0f, 1f),
        )
    }
}

data class AutoGeometrySuggestion(
    val rotationDegrees: Float? = null,
    val perspective: List<NormalizedPoint>? = null,
)

internal fun normalizePerspective(points: List<NormalizedPoint>): List<NormalizedPoint> {
    if (points.size != 4) return EditorGeometry.FULL_FRAME_CORNERS
    val clamped = points.map(NormalizedPoint::clamped)
    val topWidth = clamped[1].x - clamped[0].x
    val bottomWidth = clamped[2].x - clamped[3].x
    val leftHeight = clamped[3].y - clamped[0].y
    val rightHeight = clamped[2].y - clamped[1].y
    return if (
        topWidth >= 0.05f && bottomWidth >= 0.05f &&
        leftHeight >= 0.05f && rightHeight >= 0.05f
    ) clamped else EditorGeometry.FULL_FRAME_CORNERS
}

internal fun cropForAspect(
    current: NormalizedCrop,
    aspectRatio: Float,
    imageAspectRatio: Float,
): NormalizedCrop {
    if (aspectRatio <= 0f || imageAspectRatio <= 0f) return current.normalized()
    val centerX = (current.left + current.right) / 2f
    val centerY = (current.top + current.bottom) / 2f
    var width = current.right - current.left
    var height = current.bottom - current.top
    val normalizedTarget = aspectRatio / imageAspectRatio
    if (width / height > normalizedTarget) width = height * normalizedTarget
    else height = width / normalizedTarget
    return NormalizedCrop(
        centerX - width / 2f,
        centerY - height / 2f,
        centerX + width / 2f,
        centerY + height / 2f,
    ).normalized()
}

internal fun cropPresetForAspect(
    aspectRatio: Float,
    imageAspectRatio: Float,
): NormalizedCrop = cropForAspect(
    current = NormalizedCrop(),
    aspectRatio = aspectRatio,
    imageAspectRatio = imageAspectRatio,
)

class EditorGeometryProcessor {
    fun apply(source: Bitmap, requested: EditorGeometry): Bitmap {
        val geometry = requested.normalized()
        var current = source
        val rotation = geometry.normalizedQuarterTurns * 90f + geometry.straightenDegrees
        if (rotation != 0f) {
            current = replace(current, rotate(current, rotation))
            if (geometry.straightenDegrees != 0f) {
                current = replace(current, cropLargestOpaqueRectangle(current))
            }
        }
        if (geometry.hasPerspective) {
            current = replace(current, warpPerspective(current, geometry.perspective))
            current = replace(current, cropLargestOpaqueRectangle(current))
        }
        if (!geometry.crop.isFullFrame) {
            current = replace(current, crop(current, geometry.crop))
        }
        return current
    }

    fun detect(source: Bitmap, includePerspective: Boolean): AutoGeometrySuggestion {
        if (!OpenCVLoader.initLocal()) return AutoGeometrySuggestion()
        val rgba = Mat()
        val gray = Mat()
        val edges = Mat()
        return try {
            Utils.bitmapToMat(source, rgba)
            Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY)
            Imgproc.GaussianBlur(gray, gray, Size(5.0, 5.0), 0.0)
            Imgproc.Canny(gray, edges, 60.0, 160.0)
            AutoGeometrySuggestion(
                rotationDegrees = detectRotation(edges),
                perspective = if (includePerspective) detectQuadrilateral(edges) else null,
            )
        } finally {
            rgba.release()
            gray.release()
            edges.release()
        }
    }

    private fun rotate(source: Bitmap, degrees: Float): Bitmap {
        val matrix = Matrix().apply { postRotate(degrees) }
        val bounds = RectF(0f, 0f, source.width.toFloat(), source.height.toFloat())
        matrix.mapRect(bounds)
        matrix.postTranslate(-bounds.left, -bounds.top)
        val output = Bitmap.createBitmap(
            kotlin.math.ceil(bounds.width().toDouble()).toInt().coerceAtLeast(1),
            kotlin.math.ceil(bounds.height().toDouble()).toInt().coerceAtLeast(1),
            Bitmap.Config.ARGB_8888,
        )
        output.setHasAlpha(true)
        Canvas(output).apply {
            drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)
            drawBitmap(
                source,
                matrix,
                Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG),
            )
        }
        return output
    }

    private fun warpPerspective(
        source: Bitmap,
        corners: List<NormalizedPoint>,
    ): Bitmap {
        val width = source.width.toDouble()
        val height = source.height.toDouble()
        val points = corners.map { Point(it.x * width, it.y * height) }
        val destinationWidth = maxOf(
            distance(points[0], points[1]),
            distance(points[3], points[2]),
        ).toInt().coerceAtLeast(1)
        val destinationHeight = maxOf(
            distance(points[0], points[3]),
            distance(points[1], points[2]),
        ).toInt().coerceAtLeast(1)
        val input = Mat()
        val output = Mat()
        val sourcePoints = MatOfPoint2f(*points.toTypedArray())
        val destinationPoints = MatOfPoint2f(
            Point(0.0, 0.0),
            Point(destinationWidth - 1.0, 0.0),
            Point(destinationWidth - 1.0, destinationHeight - 1.0),
            Point(0.0, destinationHeight - 1.0),
        )
        return try {
            Utils.bitmapToMat(source, input)
            val transform = Imgproc.getPerspectiveTransform(sourcePoints, destinationPoints)
            try {
                Imgproc.warpPerspective(
                    input,
                    output,
                    transform,
                    Size(destinationWidth.toDouble(), destinationHeight.toDouble()),
                    Imgproc.INTER_CUBIC,
                )
                Bitmap.createBitmap(
                    destinationWidth,
                    destinationHeight,
                    Bitmap.Config.ARGB_8888,
                ).also { Utils.matToBitmap(output, it) }
            } finally {
                transform.release()
            }
        } finally {
            input.release()
            output.release()
            sourcePoints.release()
            destinationPoints.release()
        }
    }

    private fun crop(source: Bitmap, crop: NormalizedCrop): Bitmap {
        val left = (crop.left * source.width).toInt().coerceIn(0, source.width - 1)
        val top = (crop.top * source.height).toInt().coerceIn(0, source.height - 1)
        val right = (crop.right * source.width).toInt().coerceIn(left + 1, source.width)
        val bottom = (crop.bottom * source.height).toInt().coerceIn(top + 1, source.height)
        return Bitmap.createBitmap(source, left, top, right - left, bottom - top)
    }

    private fun cropLargestOpaqueRectangle(source: Bitmap): Bitmap {
        if (!source.hasAlpha()) return source.copy(Bitmap.Config.ARGB_8888, true)
        val width = source.width
        val height = source.height
        val pixels = IntArray(width)
        val histogram = IntArray(width)
        var bestLeft = 0
        var bestTop = 0
        var bestRight = width
        var bestBottom = height
        var bestArea = 0
        for (y in 0 until height) {
            source.getPixels(pixels, 0, width, 0, y, width, 1)
            for (x in 0 until width) {
                histogram[x] = if ((pixels[x] ushr 24) >= 250) histogram[x] + 1 else 0
            }
            val stack = IntArray(width + 1)
            var size = 0
            for (x in 0..width) {
                val currentHeight = if (x == width) 0 else histogram[x]
                while (size > 0 && histogram[stack[size - 1]] > currentHeight) {
                    val index = stack[--size]
                    val rectangleHeight = histogram[index]
                    val left = if (size == 0) 0 else stack[size - 1] + 1
                    val area = rectangleHeight * (x - left)
                    if (area > bestArea) {
                        bestArea = area
                        bestLeft = left
                        bestRight = x
                        bestBottom = y + 1
                        bestTop = bestBottom - rectangleHeight
                    }
                }
                if (x < width) stack[size++] = x
            }
        }
        if (bestArea == 0) return source.copy(Bitmap.Config.ARGB_8888, true)
        return Bitmap.createBitmap(
            source,
            bestLeft,
            bestTop,
            bestRight - bestLeft,
            bestBottom - bestTop,
        )
    }

    private fun detectRotation(edges: Mat): Float? {
        val lines = Mat()
        return try {
            Imgproc.HoughLinesP(
                edges,
                lines,
                1.0,
                Math.PI / 180.0,
                70,
                minOf(edges.cols(), edges.rows()) * 0.18,
                18.0,
            )
            val deviations = buildList {
                for (index in 0 until lines.rows()) {
                    val line = lines.get(index, 0) ?: continue
                    var angle = Math.toDegrees(
                        atan2(line[3] - line[1], line[2] - line[0]),
                    ).toFloat()
                    while (angle <= -90f) angle += 180f
                    while (angle > 90f) angle -= 180f
                    val deviation = when {
                        abs(angle) <= 25f -> angle
                        abs(abs(angle) - 90f) <= 25f ->
                            if (angle > 0f) angle - 90f else angle + 90f
                        else -> continue
                    }
                    add(deviation)
                }
            }.sorted()
            if (deviations.size < 3) null
            else (-deviations[deviations.size / 2]).coerceIn(-15f, 15f)
        } finally {
            lines.release()
        }
    }

    private fun detectQuadrilateral(edges: Mat): List<NormalizedPoint>? {
        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        val contourInput = edges.clone()
        return try {
            Imgproc.findContours(
                contourInput,
                contours,
                hierarchy,
                Imgproc.RETR_LIST,
                Imgproc.CHAIN_APPROX_SIMPLE,
            )
            val imageArea = edges.cols().toDouble() * edges.rows()
            contours.asSequence()
                .mapNotNull { contour ->
                    val curve = MatOfPoint2f(*contour.toArray())
                    val approximation = MatOfPoint2f()
                    try {
                        val perimeter = Imgproc.arcLength(curve, true)
                        Imgproc.approxPolyDP(curve, approximation, perimeter * 0.025, true)
                        val points = approximation.toArray()
                        val polygon = MatOfPoint(*points)
                        try {
                            if (
                                points.size == 4 && Imgproc.isContourConvex(polygon) &&
                                abs(Imgproc.contourArea(approximation)) >= imageArea * 0.18
                            ) points.toList() else null
                        } finally {
                            polygon.release()
                        }
                    } finally {
                        curve.release()
                        approximation.release()
                    }
                }
                .maxByOrNull { polygonArea(it) }
                ?.let(::orderCorners)
                ?.map {
                    NormalizedPoint(
                        (it.x / edges.cols()).toFloat(),
                        (it.y / edges.rows()).toFloat(),
                    ).clamped()
                }
        } finally {
            contours.forEach(Mat::release)
            hierarchy.release()
            contourInput.release()
        }
    }

    private fun orderCorners(points: List<Point>): List<Point> {
        val topLeft = points.minBy { it.x + it.y }
        val bottomRight = points.maxBy { it.x + it.y }
        val topRight = points.maxBy { it.x - it.y }
        val bottomLeft = points.minBy { it.x - it.y }
        return listOf(topLeft, topRight, bottomRight, bottomLeft)
    }

    private fun polygonArea(points: List<Point>): Double =
        points.indices.sumOf { index ->
            val next = points[(index + 1) % points.size]
            points[index].x * next.y - next.x * points[index].y
        }.let(::abs) / 2.0

    private fun distance(first: Point, second: Point): Double =
        hypot(first.x - second.x, first.y - second.y)

    private fun replace(old: Bitmap, replacement: Bitmap): Bitmap {
        if (replacement !== old) old.recycle()
        return replacement
    }
}
