# Alpha Capture Lab for iOS

Native SwiftUI port of Alpha Capture Lab, built and sideloaded from Linux with
[xtool](https://github.com/xtool-org/xtool). The camera transport has been
validated against a Sony ILCE-6300 over its direct Wi-Fi network.

## Features

- Sony ScalarWebAPI discovery, live view, app shutter, physical-shutter events,
  continuous capture, exposure settings, drive/burst settings, and remote zoom
- Photo, Live ND, Live Composite, and Panorama shooting workspaces
- Immediate original/reduced downloads with visible progress, sequential queue,
  and five-attempt interruption recovery
- Persistent camera-style gallery with full-screen viewing, private originals,
  and source-frame inspection for computational captures
- Built-in and imported `.cube` LUTs, including multi-LUT Lumix Lab ZIP imports,
  live-view preview, per-LUT strength, baked output, and reversible editing
- Full-screen nondestructive editor with basic adjustments, LUTs, RawRefinery
  Light denoise, and Deep Sharpen
- ONNX Runtime Core ML execution with automatic CPU fallback
- JPEG/WebP output when supported by ImageIO, optional phone GPS EXIF, automatic
  denoise policy, live-view timeout, paired-camera memory, and reachability-based
  auto-connect

The iOS sandbox cannot silently join an arbitrary camera access point. The user
must approve/join the camera Wi-Fi in iOS; auto-connect starts the remote session
when a previously paired camera endpoint becomes reachable.

## Linux build

Install xtool and its Darwin Swift SDK, then install the ignored ONNX Runtime
binary dependency and build:

```bash
cd ios
./scripts/install-onnxruntime.sh
xtool dev build
```

To sign, install, and launch on a USB-connected iPhone:

```bash
xtool dev run --configuration debug --usb
```

Free Apple Account provisioning expires after seven days. Initial signature
verification requires an internet-connected network; join the camera Wi-Fi only
after installation finishes.
