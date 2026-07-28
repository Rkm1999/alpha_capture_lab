# AI NR Engine Test v0.1 Beta 3

This prerelease is a standalone engineering harness for validating the SCUNet
denoising pipeline before it is integrated into Alpha Capture Lab. It is not a
separate production product.

## Android artifact

`AINR-Android-v0.1.0-beta.3-debug.apk` is an arm64-only, debug-key-signed APK
for direct device testing. It supports GPU execution and supported Qualcomm,
MediaTek, and Samsung NPU backends. Unsupported devices can use GPU mode.

The APK is intentionally large because it includes the generic SCUNet model and
the runtime libraries needed for on-device accelerator compilation. Verify the
download against the attached `SHA256SUMS.txt` before installing it.

## iOS artifact

`AINR-iOS-v0.1.0-beta.3.ipa` is a development-signed IPA for sideloading. Its
provisioning profile is tied to the signing account and registered test device;
other users may need to re-sign it with their own Apple developer identity.

## Scope

- Import or share one or more images and process them as a batch.
- Preserve standard camera EXIF metadata in saved JPEGs.
- Use the epoch-177 LiteDenoise student in Performance mode.
- Run full-resolution SCUNet denoising with 192 x 192 tiles.
- Choose GPU, NPU, or concurrent GPU + NPU execution when supported.
- Use optional 50% weighted tile overlap to suppress checkerboard artifacts.
- Disable combined accelerators in high-overlap mode to keep tile output consistent.
- Monitor setup and tile progress.
- Save the completed JPEG.

Detailed compatibility, validation results, limitations, and build instructions
are in [README.md](README.md).
