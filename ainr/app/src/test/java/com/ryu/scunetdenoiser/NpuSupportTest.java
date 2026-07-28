package com.ryu.scunetdenoiser;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

import org.junit.Test;

public final class NpuSupportTest {
    @Test
    public void mapsOfficialQualcommMobileSocsToHtpArchitectures() {
        assertEquals("V69", NpuSupport.qualcommHtpArchitecture("SM8450"));
        assertEquals("V69", NpuSupport.qualcommHtpArchitecture("SM8475"));
        assertEquals("V73", NpuSupport.qualcommHtpArchitecture("SM8550"));
        assertEquals("V75", NpuSupport.qualcommHtpArchitecture("SM8650"));
        assertEquals("V79", NpuSupport.qualcommHtpArchitecture("SM8750"));
        assertEquals("V81", NpuSupport.qualcommHtpArchitecture("SM8850"));
    }

    @Test
    public void normalizesSocModelCase() {
        assertEquals("V75", NpuSupport.qualcommHtpArchitecture("sm8650"));
    }

    @Test
    public void rejectsUnpackagedQualcommRuntime() {
        assertNull(NpuSupport.qualcommHtpArchitecture("SM8350"));
        assertNull(NpuSupport.qualcommHtpArchitecture(null));
    }

    @Test
    public void acceptsPackagedQualcommTargetsOnAndroid12() {
        assertEquals(
            NpuSupport.Vendor.QUALCOMM,
            NpuSupport.detect(31, "QTI", "SM8450", false, false));
        assertEquals(
            NpuSupport.Vendor.QUALCOMM,
            NpuSupport.detect(31, "Qualcomm", "SM8850", true, false));
    }

    @Test
    public void checkerDoesNotEnableUnpackagedQualcommArchitecture() {
        assertEquals(
            NpuSupport.Vendor.UNSUPPORTED,
            NpuSupport.detect(36, "QTI", "SM8350", true, false));
    }

    @Test
    public void mediatekRequiresAndroid15AndPackagedTarget() {
        assertEquals(
            NpuSupport.Vendor.UNSUPPORTED,
            NpuSupport.detect(34, "Mediatek", "MT6989", false, true));
        assertEquals(
            NpuSupport.Vendor.MEDIATEK,
            NpuSupport.detect(35, "Mediatek", "MT6989(ENG)", false, true));
        assertEquals(
            NpuSupport.Vendor.UNSUPPORTED,
            NpuSupport.detect(35, "Mediatek", "MT6993", false, true));
    }

    @Test
    public void samsungRequiresAndroid16AiLiteCoreTargets() {
        assertEquals(
            NpuSupport.Vendor.UNSUPPORTED,
            NpuSupport.detect(35, "Samsung", "E9965", false, false));
        assertEquals(
            NpuSupport.Vendor.SAMSUNG,
            NpuSupport.detect(36, "Samsung S.LSI", "E9965", false, false));
        assertEquals(
            NpuSupport.Vendor.UNSUPPORTED,
            NpuSupport.detect(36, "Samsung", "E2400", false, false));
    }

    @Test
    public void selectsMatchingQualcommRuntimeFiles() {
        String[] libraries = NpuSupport.requiredPackagedLibraries(
            NpuSupport.Vendor.QUALCOMM, "SM8550");
        assertArrayEquals(
            new String[] {
                "libLiteRtCompilerPlugin_Qualcomm.so",
                "libLiteRtDispatch_Qualcomm.so",
                "libQnnSystem.so",
                "libQnnHtp.so",
                "libQnnHtpPrepare.so",
                "libQnnHtpV73Skel.so",
                "libQnnHtpV73Stub.so",
                "libQnnIr.so",
                "libQnnSaver.so"
            },
            libraries);
    }
}
