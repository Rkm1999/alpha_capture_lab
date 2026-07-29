# Third-Party Notices

This project depends on the following open-source libraries. No Sony SDK binary,
sample source, documentation bundle, logo, or other Sony asset is included.

## OpenCV 4.13.0

- Project: https://opencv.org/
- Source: https://github.com/opencv/opencv
- License: Apache License 2.0

The Android AAR can include third-party codec and optimized-computation
components covered by the notices distributed with the corresponding official
OpenCV Android SDK under `sdk/etc/licenses`.

## AndroidX

- Project: https://developer.android.com/jetpack/androidx
- License: Apache License 2.0
- Components include Activity, Compose, Core, ExifInterface, Lifecycle, and
  AndroidX Test.

## Kotlin Coroutines And Serialization

- Project: https://github.com/Kotlin/kotlinx.coroutines and
  https://github.com/Kotlin/kotlinx.serialization
- License: Apache License 2.0

## JUnit 4

- Project: https://github.com/junit-team/junit4
- License: Eclipse Public License 1.0

## RawRefinery

- Source: https://github.com/rymuelle/RawRefinery
- License: MIT
- The iOS app includes the Deep Sharpen ONNX model weights.

## SCUNet And Distilled AINR

- SCUNet source: https://github.com/cszn/SCUNet
- SCUNet license: Apache License 2.0
- Alpha Capture Lab includes the official SCUNet color-real-PSNR model and a
  width-24 mobile student distilled from that frozen teacher.
- Shared runtime and accelerator notices are documented in
  `ainr/THIRD_PARTY.md`. The SCUNet license text is bundled with each platform.

## LiteRT

- Source: https://github.com/google-ai-edge/LiteRT
- License: Apache License 2.0
- Android uses LiteRT 2.1.6 with GPU and supported vendor NPU backends.

## ONNX Runtime

- Source: https://github.com/microsoft/onnxruntime
- License: MIT
- The iOS app uses the official ONNX Runtime 1.20.0 iOS framework with the
  Core ML execution provider.

## libwebp

- Source: https://chromium.googlesource.com/webm/libwebp
- Swift package: https://github.com/SDWebImage/libwebp-Xcode
- License: BSD 3-Clause
- Used by the iOS app for WebP encoding.

## ZIPFoundation

- Source: https://github.com/weichsel/ZIPFoundation
- License: MIT
- Used by the iOS app to import `.cube` LUTs from ZIP archives.
