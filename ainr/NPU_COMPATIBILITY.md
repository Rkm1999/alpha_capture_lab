# Android NPU compatibility audit

This audit applies to the Android app's LiteRT 2.1.6 on-device compilation
path. Chipset detection, packaged libraries, and successful execution are
separate requirements. A matching SoC name alone does not prove that NPU
initialization or full model delegation will succeed.

## Current support matrix

| Vendor | App enablement | Packaged by the APK | Verification |
| --- | --- | --- | --- |
| Qualcomm | Android 12+, SM8450, SM8475, SM8550, SM8650, SM8750, or SM8850 | LiteRT compiler/dispatch, QAIRT 2.47 host libraries, and matching V69/V73/V75/V79/V81 HTP stub and DSP image | SM8750 verified; other generations provisional |
| MediaTek | Android 15+, MT6878, MT6897, MT6983, MT6985, MT6989, or MT6991 | LiteRT compiler/dispatch only | Provisional |
| Samsung | Android 16+, E9955 or E9965 | LiteRT compiler/dispatch only | Provisional |

Unsupported or incomplete targets use GPU mode. At startup the app now checks
that the compiler, dispatch, and chipset-specific packaged files are present
before enabling NPU controls.

## Audit findings

1. The previous MediaTek branch accepted Android 12 through 14 but did not
   package `libneuron_adapter.so`. LiteRT's MediaTek compiler dynamically loads
   a firmware-provided Neuron/NeuroPilot adapter, so that claim was not valid
   for those builds. MediaTek is now limited to Android 15+ targets in the
   matching LiteRT JIT sample matrix.
2. The previous Samsung branch accepted Android 12+, while LiteRT's Samsung
   setup requires Android 16. Its compiler loads AI LiteCore graph compiler
   libraries and its dispatch bridge loads the firmware ENN runtime. Samsung
   is now limited to Android 16 and remains provisional.
3. MediaTek detection combined LiteRT 2.1.6's checker with a separate list
   containing MT6990 and MT6993. Those targets are not both represented in the
   matching JIT sample/runtime matrix. Detection now uses only the targets for
   which this APK has a defined deployment path.
4. Qualcomm packaging matches LiteRT 2.1.6's QAIRT 2.47 recipe and includes a
   runtime pair for each enabled HTP architecture. The only physically
   validated target remains SM8750. Other supported Snapdragon generations
   still depend on compatible OEM CDSP/RPC firmware and require device tests.
5. Initialization errors now identify the missing vendor layer: Qualcomm
   CDSP RPC, MediaTek NeuroPilot adapter, or Samsung AI LiteCore/ENN. This
   makes a field report distinguishable from an unsupported model operation
   or a generic LiteRT failure.
6. Static graph inspection found that the width-24 model uses only `ADD`,
   `CONCATENATION`, `CONV_2D`, `MUL`, `PAD`, `RESIZE_NEAREST_NEIGHBOR`,
   `SLICE`, and `TRANSPOSE`. LiteRT 2.1.6 has MediaTek and Samsung builders
   for all of them. The full SCUNet graph's attention, normalization, and
   decoder operators are also represented in both plugin implementations.
   This removes an obvious operator-list mismatch, but only a device run can
   prove the vendor compiler accepts each tensor shape and option.

## Device acceptance test

For every new chipset and Android build:

1. Confirm the app reports the expected `NPU_SUPPORT` vendor and SoC in logcat.
2. Delete app data, select NPU, and complete the first on-device compilation.
3. Confirm `NPU_JIT_READY` appears and no dynamic-loader, CDSP, Neuron, AI
   LiteCore, or ENN error precedes it.
4. Run at least two warm 192 x 192 tiles and a full-resolution image.
5. Compare output numerically with GPU and record median/p95 tile latency.
6. Confirm the run is actually delegated using vendor profiler/driver logs;
   successful `CompiledModel.create` alone is not sufficient evidence.

Do not mark a chipset verified until this test has been completed on that
physical device and firmware build.
