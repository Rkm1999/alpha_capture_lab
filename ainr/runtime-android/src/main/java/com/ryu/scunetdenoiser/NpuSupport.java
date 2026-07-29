package com.ryu.scunetdenoiser;

import android.content.Context;
import android.os.Build;

import com.google.ai.edge.litert.NpuCompatibilityChecker;

import java.io.File;
import java.util.Locale;

final class NpuSupport {
    enum Vendor {
        QUALCOMM,
        MEDIATEK,
        SAMSUNG,
        UNSUPPORTED
    }

    private NpuSupport() {}

    static Vendor detect() {
        if (Build.VERSION.SDK_INT < 31) return Vendor.UNSUPPORTED;
        boolean qualcommChecker =
            NpuCompatibilityChecker.Companion.getQualcomm().isDeviceSupported();
        boolean mediatekChecker =
            NpuCompatibilityChecker.Companion.getMediatek().isDeviceSupported();
        return detect(
            Build.VERSION.SDK_INT,
            Build.SOC_MANUFACTURER,
            Build.SOC_MODEL,
            qualcommChecker,
            mediatekChecker);
    }

    static Vendor detect(Context context) {
        Vendor vendor = detect();
        return hasPackagedRuntime(context, vendor) ? vendor : Vendor.UNSUPPORTED;
    }

    static Vendor detect(
        int sdk,
        String socManufacturer,
        String socModel,
        boolean qualcommChecker,
        boolean mediatekChecker
    ) {
        if (sdk < 31) return Vendor.UNSUPPORTED;

        String manufacturer = clean(socManufacturer).toLowerCase(Locale.US);
        String model = clean(socModel).toUpperCase(Locale.US);
        boolean qualcommIdentity = manufacturer.equals("qualcomm")
            || manufacturer.equals("qti")
            || qualcommChecker;
        if (qualcommIdentity && qualcommHtpArchitecture(model) != null) {
            return Vendor.QUALCOMM;
        }

        // This APK relies on the device's NeuroPilot adapter. Android 15 is
        // therefore required; supporting older releases requires bundling the
        // matching MediaTek adapter library with the app.
        boolean mediatekIdentity = manufacturer.equals("mediatek") || mediatekChecker;
        if (sdk >= 35 && mediatekIdentity && isPackagedMediaTekTarget(model)) {
            return Vendor.MEDIATEK;
        }

        // LiteRT 2.1.6's Samsung path requires the Android 16 AI LiteCore and
        // ENN system libraries. The Maven compatibility checker in this
        // release does not expose a Samsung checker.
        boolean samsungIdentity = manufacturer.equals("samsung")
            || manufacturer.equals("s.lsi")
            || manufacturer.equals("samsung s.lsi");
        if (sdk >= 36 && samsungIdentity && isPackagedSamsungTarget(model)) {
            return Vendor.SAMSUNG;
        }
        return Vendor.UNSUPPORTED;
    }

    static String displayName(Vendor vendor) {
        if (vendor == Vendor.QUALCOMM) {
            String architecture = qualcommHtpArchitecture();
            return architecture == null
                ? "Qualcomm HTP"
                : "Qualcomm HTP " + architecture;
        }
        if (vendor == Vendor.MEDIATEK) return "MediaTek NeuroPilot";
        if (vendor == Vendor.SAMSUNG) return "Samsung AI LiteCore";
        return "Unsupported NPU";
    }

    static String qualcommHtpArchitecture() {
        if (Build.VERSION.SDK_INT < 31) return null;
        return qualcommHtpArchitecture(clean(Build.SOC_MODEL).toUpperCase(Locale.US));
    }

    static String qualcommHtpArchitecture(String model) {
        if (model == null) return null;
        switch (model.toUpperCase(Locale.US)) {
            case "SM8450":
            case "SM8475":
                return "V69";
            case "SM8550":
                return "V73";
            case "SM8650":
                return "V75";
            case "SM8750":
                return "V79";
            case "SM8850":
                return "V81";
            default:
                return null;
        }
    }

    static String deviceSummary() {
        if (Build.VERSION.SDK_INT < 31) return "Android " + Build.VERSION.SDK_INT;
        String manufacturer = clean(Build.SOC_MANUFACTURER);
        String model = clean(Build.SOC_MODEL);
        String value = (manufacturer + " " + model).trim();
        return value.isEmpty() ? "unknown SoC" : value;
    }

    private static String clean(String value) {
        if (value == null) return "";
        return value.replace("(ENG)", "").trim();
    }

    private static boolean isPackagedMediaTekTarget(String model) {
        return model.equals("MT6878")
            || model.equals("MT6897")
            || model.equals("MT6983")
            || model.equals("MT6985")
            || model.equals("MT6989")
            || model.equals("MT6991");
    }

    private static boolean isPackagedSamsungTarget(String model) {
        return model.equals("E9955") || model.equals("E9965");
    }

    static IllegalStateException initializationFailure(Vendor vendor, Throwable cause) {
        String detail;
        if (vendor == Vendor.QUALCOMM) {
            detail = "Qualcomm NPU initialization failed. The device firmware must expose "
                + "the CDSP RPC driver required by the packaged QAIRT 2.47 runtime";
        } else if (vendor == Vendor.MEDIATEK) {
            detail = "MediaTek NPU initialization failed. The Android 15 firmware must "
                + "expose a compatible NeuroPilot Neuron adapter";
        } else if (vendor == Vendor.SAMSUNG) {
            detail = "Samsung NPU initialization failed. The Android 16 firmware must "
                + "expose compatible AI LiteCore compiler and ENN runtime libraries";
        } else {
            detail = "NPU initialization failed";
        }
        return new IllegalStateException(detail + " on " + deviceSummary(), cause);
    }

    static boolean hasPackagedRuntime(Context context, Vendor vendor) {
        if (vendor == Vendor.UNSUPPORTED) return false;
        File directory = new File(context.getApplicationInfo().nativeLibraryDir);
        String socModel = Build.VERSION.SDK_INT >= 31 ? Build.SOC_MODEL : "";
        for (String name : requiredPackagedLibraries(vendor, socModel)) {
            if (!new File(directory, name).isFile()) return false;
        }
        return true;
    }

    static String[] requiredPackagedLibraries(Vendor vendor, String socModel) {
        String suffix;
        if (vendor == Vendor.QUALCOMM) {
            suffix = "Qualcomm";
        } else if (vendor == Vendor.MEDIATEK) {
            suffix = "MediaTek";
        } else if (vendor == Vendor.SAMSUNG) {
            suffix = "Samsung";
        } else {
            return new String[0];
        }
        String compiler = "libLiteRtCompilerPlugin_" + suffix + ".so";
        String dispatch = "libLiteRtDispatch_" + suffix + ".so";
        if (vendor != Vendor.QUALCOMM) return new String[] {compiler, dispatch};

        String architecture = qualcommHtpArchitecture(clean(socModel).toUpperCase(Locale.US));
        if (architecture == null) return new String[] {"unsupported-qualcomm-runtime"};
        return new String[] {
            compiler,
            dispatch,
            "libQnnSystem.so",
            "libQnnHtp.so",
            "libQnnHtpPrepare.so",
            "libQnnHtp" + architecture + "Skel.so",
            "libQnnHtp" + architecture + "Stub.so",
            "libQnnIr.so",
            "libQnnSaver.so"
        };
    }
}
