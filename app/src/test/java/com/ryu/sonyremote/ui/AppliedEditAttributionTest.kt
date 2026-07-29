package com.ryu.sonyremote.ui

import com.ryu.sonyremote.processing.AinrDenoiseModel
import com.ryu.sonyremote.processing.LutPreset
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class AppliedEditAttributionTest {
    @Test
    fun selectedLutAndDenoiseAreCapturedTogether() {
        val lut = LutCaptureState(
            preset = LutPreset.Cinema,
            presetIntensities = LutPreset.entries.associateWith {
                if (it == LutPreset.Cinema) 0.4f else 1f
            },
        )

        val edits = appliedEditsFor(lut, AinrDenoiseModel.Distilled, 0.75f)

        assertEquals("Cinema", edits.lutName)
        assertEquals(0.4f, edits.lutStrength)
        assertEquals("Distilled", edits.denoiseModel)
        assertEquals(0.75f, edits.denoiseStrength)
    }

    @Test
    fun originalSelectionHasNoLutAttribution() {
        val edits = appliedEditsFor(
            LutCaptureState(),
            denoiseModel = null,
            denoiseStrength = 0f,
        )

        assertNull(edits.lutName)
        assertNull(edits.lutStrength)
        assertNull(edits.denoiseModel)
        assertNull(edits.denoiseStrength)
    }
}
