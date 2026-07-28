package com.ryu.scunetdenoiser;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

public final class ImageStoreTest {
    @Test
    public void outputNamePreservesOriginalBaseName() {
        assertEquals("DSC01234_denoised.jpg", ImageStore.outputName("DSC01234.JPG"));
        assertEquals(
            "vacation.edit.v2_denoised.jpg",
            ImageStore.outputName("vacation.edit.v2.jpeg"));
    }

    @Test
    public void outputNameSanitizesUnsupportedCharacters() {
        assertEquals(
            "camera_image_denoised.jpg",
            ImageStore.outputName("camera/image.jpg"));
    }

    @Test
    public void outputNameFallsBackForMissingName() {
        assertEquals("image_denoised.jpg", ImageStore.outputName(null));
        assertEquals("image_denoised.jpg", ImageStore.outputName(" "));
    }

    @Test
    public void fileNameFromPickerDataUsesOriginalBaseName() {
        assertEquals(
            "DSC01234.JPG",
            ImageStore.fileNameFromPath("/storage/emulated/0/DCIM/Camera/DSC01234.JPG"));
        assertEquals(
            "vacation image.jpg",
            ImageStore.fileNameFromPath("file:///storage/emulated/0/Pictures/vacation%20image.jpg"));
    }
}
