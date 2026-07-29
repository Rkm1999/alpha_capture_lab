package com.ryu.sonyremote.processing

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class EditorGeometryTest {
    @Test
    fun cropNormalizationKeepsMinimumAreaInsideFrame() {
        val crop = NormalizedCrop(0.9f, -0.2f, 0.1f, 1.4f).normalized()

        assertTrue(crop.left >= 0f)
        assertTrue(crop.top >= 0f)
        assertTrue(crop.right <= 1f)
        assertTrue(crop.bottom <= 1f)
        assertTrue(crop.right - crop.left >= 0.05f)
        assertTrue(crop.bottom - crop.top >= 0.05f)
    }

    @Test
    fun aspectCropAccountsForImageAspectRatio() {
        val square = cropForAspect(
            current = NormalizedCrop(),
            aspectRatio = 1f,
            imageAspectRatio = 2f,
        )

        assertEquals(0.25f, square.left, 0.0001f)
        assertEquals(0.75f, square.right, 0.0001f)
        assertEquals(0f, square.top, 0.0001f)
        assertEquals(1f, square.bottom, 0.0001f)
    }

    @Test
    fun aspectPresetDoesNotCompoundPreviousCrop() {
        val square = cropPresetForAspect(1f, imageAspectRatio = 1.5f)
        val fourByThreeAfterSquare = cropPresetForAspect(
            aspectRatio = 4f / 3f,
            imageAspectRatio = 1.5f,
        )
        val fourByThreeDirect = cropPresetForAspect(
            aspectRatio = 4f / 3f,
            imageAspectRatio = 1.5f,
        )

        assertTrue(square != fourByThreeAfterSquare)
        assertEquals(fourByThreeDirect, fourByThreeAfterSquare)
    }

    @Test
    fun invalidPerspectiveFallsBackToFullFrame() {
        val invalid = listOf(
            NormalizedPoint(0.8f, 0f),
            NormalizedPoint(0.2f, 0f),
            NormalizedPoint(1f, 1f),
            NormalizedPoint(0f, 1f),
        )

        assertEquals(EditorGeometry.FULL_FRAME_CORNERS, normalizePerspective(invalid))
        assertFalse(EditorGeometry(perspective = invalid).normalized().hasPerspective)
    }

    @Test
    fun quarterTurnsAreNormalized() {
        assertEquals(3, EditorGeometry(quarterTurns = -1).normalizedQuarterTurns)
        assertEquals(1, EditorGeometry(quarterTurns = 5).normalizedQuarterTurns)
    }
}
