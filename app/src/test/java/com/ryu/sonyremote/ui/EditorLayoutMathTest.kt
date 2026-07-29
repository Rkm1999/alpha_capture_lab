package com.ryu.sonyremote.ui

import org.junit.Assert.assertEquals
import org.junit.Test

class EditorLayoutMathTest {
    @Test
    fun fittedImageIsCenteredWithoutCropping() {
        val rect = fittedImageRect(
            viewportWidth = 1_000f,
            viewportHeight = 1_000f,
            imageWidth = 2_000,
            imageHeight = 1_000,
        )

        assertEquals(0f, rect.left, 0.001f)
        assertEquals(250f, rect.top, 0.001f)
        assertEquals(1_000f, rect.right, 0.001f)
        assertEquals(750f, rect.bottom, 0.001f)
    }
}
