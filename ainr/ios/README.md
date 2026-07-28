# SCUNet Denoiser for iOS

Native SwiftUI port of the standalone full-resolution SCUNet denoiser.

## Features

- Import one or more JPEG, HEIC, PNG, and other iOS-supported images from
  Photos, Files, Open In, or the share sheet
- Process batch selections sequentially with one warm Core ML session, show
  image and tile progress, and save or share all completed results together
- Load a bundled 24 MP Sony ISO 51200 JPEG for repeatable device testing
- Normalize source orientation before processing
- Process the full image with overlapping 192 x 192 tiles and an 8-pixel
  feathered context on every edge
- Select GPU, Neural Engine, GPU + Neural Engine, or CPU execution explicitly
- Select **High Performance** for the FP16 LiteDenoise student or **High
  Quality** for the original FP16 SCUNet. High Quality remains the default.
- Show model preparation, tile progress, elapsed time, and the selected backend
- Optionally use 50% whole-tile weighted overlap to suppress checkerboard patterns
- Cancel between tiles, compare original and result, then save or share the JPEG

The app uses native Core ML ML Programs with FP16 weights and FP32 input and
output tensors. High Performance uses the selected width-24 residual-adapter
LiteDenoise graph; High Quality uses full SCUNet. Core
ML compiles each model on the iPhone once and the app retains separate compiled
models in Application Support for later launches. Neural Engine is the default
because the optimized fixed-shape graph is the fastest sustained path on the
tested iPhone. Previously selected backend and quality settings are remembered.

The checked-in LiteDenoise package is reproducible from the retained checkpoint
with:

```bash
/home/ryu/.cache/ainr-coreml-venv/bin/python \
  ../training/distillation/paper_192/scripts/export_ios_coreml.py \
  --checkpoint ../training/distillation/archive/final_w24/target-best.pt \
  --precomputed-noise-gate \
  --output Sources/SCUNetDenoiser/Resources/Models/litedenoise_192_w24.mlpackage
```

The compute-unit choices constrain Core ML as follows:

| App option | Core ML compute units |
| --- | --- |
| GPU | CPU and GPU |
| Neural Engine | CPU and Neural Engine |
| GPU + Neural Engine | Two concurrent sessions: CPU and GPU plus CPU and Neural Engine |
| CPU | CPU only |

`GPU + Neural Engine` is a real dual-session mode rather than Core ML's `.all`
placement policy. The Neural Engine claims tiles from the start of the image
while the GPU claims them from the end. A shared tile scheduler keeps both
workers busy, restores raster order before composition, and applies the same
16-pixel feathering at accelerator transitions as at every other tile edge.

Core ML can still place unsupported operations on the CPU within the selected
constraint. The app reports the requested configuration and does not silently
rename a failed delegate or CPU fallback as GPU or Neural Engine execution.

## Compatibility

- iOS 17 or later
- 64-bit iPhone or iPad
- GPU mode requires a Core ML-compatible Apple GPU
- Neural Engine performance and graph coverage depend on the Apple SoC and iOS
- The app checks the selected model's Core ML compute plan on the current
  device. Neural Engine choices are disabled when the device has no ANE or the
  selected model has no ANE placement, and a saved ANE selection falls back to
  GPU automatically.
- GPU + Neural Engine requires both accelerator paths and uses more memory and
  power than Neural Engine alone
- CPU mode is the compatibility fallback and is not practical for 24 MP images

The app includes the Apache-2.0 SCUNet license in its `Legal` resource folder.
No ONNX Runtime or LiteRT runtime is linked into the current iOS build.

## Historical Device Run

These measurements describe the earlier LiteDenoise performance export and
remain useful as an iPhone 15 Pro Max backend baseline. The selected W24 model
must be benchmarked separately before publishing current-model figures.

Tested on an iPhone 15 Pro Max running iOS 18.7.2 with the bundled 4000 x 6000
ISO 51200 JPEG:

| Configuration | Result |
| --- | --- |
| Production Neural Engine, first run after install | Completed all 805 tiles and encoded a valid 4000 x 6000 JPEG in 51.888 seconds |
| First-run model preparation | 10.025 seconds |
| First-run Neural Engine prediction | 41.139 seconds total; 51.105 ms per tile |
| Production Neural Engine, cached relaunch | 41.548 seconds end to end; model load was 0.085 seconds |
| Cached Neural Engine prediction | 40.726 seconds total; 50.591 ms per tile |
| GPU + Neural Engine, balanced hot-device run | 57.631 seconds end to end: 11.616 seconds loading both sessions and 45.273 seconds for concurrent inference |
| Balanced dual-session work split | Neural Engine processed 619 tiles while GPU processed 186; worker wall times were 45.218 and 45.273 seconds |
| Cool-device dual-session trial | 45.895 seconds end to end with a 35.331-second concurrent inference stage using the earlier fixed split |
| Dual-session output check | 61.04 dB PSNR versus the Neural Engine-only output; mean absolute difference was 0.039 of one 8-bit level |
| Non-model work | About 0.7 seconds for decode, tile conversion, tensor copies, seam composition, and JPEG encoding |
| Neural Engine memory | About 361 MiB peak anonymous memory; 308 MiB physical footprint after completion |
| GPU + Neural Engine memory | About 740 MiB peak anonymous memory; 213 MiB physical footprint after completion |
| Optimized GPU, short run | 113.8 ms mean prediction time over 20 tiles |
| Earlier GPU full run | 220.8 seconds with the old export, including thermal slowdown to about 0.33 seconds per tile |

The production Neural Engine path stayed near 50-51 ms per tile through the
entire 805-tile run, without the severe sustained slowdown seen on the GPU.
Core ML's optional `fastPrediction` specialization reached 45.798 ms per tile
and a 37.696-second cached run, but added roughly 16 seconds to the first model
load. The app intentionally uses the lighter reshape-only optimization because
it gives the lower first-image latency.

The dual-session mode achieved 1.95-1.98x inference overlap in measured runs,
so GPU and Neural Engine prediction really occurred concurrently. Its dynamic
tile scheduler adapts the split to current accelerator speed; in the hot run,
both workers finished within 0.055 seconds of one another. GPU throughput drops
substantially as the phone heats, however, and loading a second model session
also adds startup cost. Neural Engine therefore remains the production default,
while GPU + Neural Engine is available for maximum throughput on a cool device
or when several images can reuse the already-loaded sessions.

The final dynamic-scheduler output was checked at its L-shaped accelerator
handoff. Local edge change remained within 2.1% of the Neural Engine-only
reference at all three transition segments, with no visible accelerator seam.

The performance fault was in the original model export, not JPEG decoding or
Swift stitching. That graph expanded every GELU into primitive arithmetic and
represented shifted windows with gather operations. Core ML loaded a long
sequence of small ANE programs and prediction took about 0.86-1.03 seconds per
tile. The current export uses Core ML's native tanh-approximate GELU plus fixed
slice/concat window shifts. Core ML's on-device compute plan assigns all 1,316
non-constant operations to the Neural Engine, and prediction now takes about
51 ms per tile. Tensor preparation and input/output copies account for less
than 0.5 ms per tile.

The earlier ONNX Runtime Core ML execution-provider build was terminated by iOS
after reaching roughly 3.5 GB. A constrained LiteRT Core ML delegate avoided
that memory limit but left most of the graph on CPU, producing a 96% CPU resource
report. The optimized direct Core ML program replaces both paths and completed
the full-resolution regression test without a crash.

## Build on Linux

This project uses [xtool](https://github.com/xtool-org/xtool) and a local Darwin
Swift SDK:

```bash
cd /home/ryu/projects/sony-camera-remote/ainr/ios
xtool dev build --configuration release --ipa
```

Install the signed IPA on a connected, trusted iPhone:

```bash
xtool install --usb xtool/SCUNetDenoiser.ipa
```

The model packages must remain under
`Sources/SCUNetDenoiser/Resources/Models/` as
`litedenoise_192_w24.mlpackage` and `scunet_192_fp16.mlpackage`.

## Validation

The Linux build verifies Swift compilation, app signing, resource packaging,
and IPA creation. Accelerator placement, memory use, thermal behavior, and
full-resolution output must be validated on an iPhone because Core ML execution
is unavailable in the Linux build environment.
