package com.ryu.sonyremote.ui

import kotlin.math.min

internal data class GalleryOffset(val x: Float, val y: Float)

internal fun clampGalleryScale(scale: Float): Float = scale.coerceIn(1f, 5f)

internal fun updateComparisonFraction(
    current: Float,
    deltaPixels: Float,
    viewportWidth: Int,
): Float {
    if (viewportWidth <= 0) return current.coerceIn(0f, 1f)
    return (current + deltaPixels / viewportWidth).coerceIn(0f, 1f)
}

internal fun clampGalleryOffset(
    x: Float,
    y: Float,
    scale: Float,
    viewportWidth: Int,
    viewportHeight: Int,
    imageWidth: Int,
    imageHeight: Int,
): GalleryOffset {
    if (
        scale <= 1f || viewportWidth <= 0 || viewportHeight <= 0 ||
        imageWidth <= 0 || imageHeight <= 0
    ) {
        return GalleryOffset(0f, 0f)
    }
    val fit = min(
        viewportWidth.toFloat() / imageWidth,
        viewportHeight.toFloat() / imageHeight,
    )
    val renderedWidth = imageWidth * fit * scale
    val renderedHeight = imageHeight * fit * scale
    val maxX = ((renderedWidth - viewportWidth) / 2f).coerceAtLeast(0f)
    val maxY = ((renderedHeight - viewportHeight) / 2f).coerceAtLeast(0f)
    return GalleryOffset(x.coerceIn(-maxX, maxX), y.coerceIn(-maxY, maxY))
}

internal fun shouldDismissGalleryDetail(
    offsetY: Float,
    viewportHeight: Int,
    velocityY: Float,
    minimumVelocity: Float,
): Boolean {
    if (offsetY <= 0f || viewportHeight <= 0) return false
    return offsetY >= viewportHeight * 0.25f || velocityY >= minimumVelocity
}
