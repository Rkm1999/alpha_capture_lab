package com.ryu.sonyremote.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class GalleryGestureMathTest {
    @Test
    fun `zoom scale stays within viewer limits`() {
        assertEquals(1f, clampGalleryScale(0.25f))
        assertEquals(2.5f, clampGalleryScale(2.5f))
        assertEquals(5f, clampGalleryScale(8f))
    }

    @Test
    fun `comparison divider follows drag and stays within image`() {
        assertEquals(0.75f, updateComparisonFraction(0.5f, 250f, 1_000))
        assertEquals(0f, updateComparisonFraction(0.1f, -500f, 1_000))
        assertEquals(1f, updateComparisonFraction(0.9f, 500f, 1_000))
        assertEquals(0.4f, updateComparisonFraction(0.4f, 200f, 0))
    }

    @Test
    fun `unzoomed image cannot be panned`() {
        val offset = clampGalleryOffset(
            x = 500f,
            y = -500f,
            scale = 1f,
            viewportWidth = 1_080,
            viewportHeight = 1_800,
            imageWidth = 6_000,
            imageHeight = 4_000,
        )

        assertEquals(GalleryOffset(0f, 0f), offset)
    }

    @Test
    fun `zoomed image pan is clamped to rendered image bounds`() {
        val offset = clampGalleryOffset(
            x = 5_000f,
            y = -5_000f,
            scale = 3f,
            viewportWidth = 1_080,
            viewportHeight = 1_800,
            imageWidth = 6_000,
            imageHeight = 4_000,
        )

        assertEquals(1_080f, offset.x, 0.01f)
        assertEquals(-180f, offset.y, 0.01f)
    }

    @Test
    fun `dismiss requires distance or downward velocity`() {
        assertFalse(shouldDismissGalleryDetail(200f, 1_800, 400f, 1_200f))
        assertTrue(shouldDismissGalleryDetail(450f, 1_800, 0f, 1_200f))
        assertTrue(shouldDismissGalleryDetail(40f, 1_800, 1_300f, 1_200f))
        assertFalse(shouldDismissGalleryDetail(-500f, 1_800, 2_000f, 1_200f))
    }
}
