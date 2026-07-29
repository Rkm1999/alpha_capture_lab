# Third-party accelerator components

The MediaTek and Samsung LiteRT compiler and dispatch plugins in
`runtime-android/src/main/jniLibs/arm64-v8a` were built from the official LiteRT v2.1.6
source tag (`1461b6b2def31713f5c71446eab844aae05d02e9`). LiteRT is licensed
under Apache-2.0.

The APK does not redistribute MediaTek NeuroPilot or Samsung AI LiteCore
system runtimes. The plugins dynamically use the runtime and NPU driver
libraries exposed by the device manufacturer on supported phones. For that
reason the app limits MediaTek NPU mode to Android 15 or later and Samsung NPU
mode to Android 16 or later. These paths remain provisional until verified on
physical target devices.

At runtime, the app creates a private plugin directory containing links to
only the compiler and dispatch pair for the detected SoC. LiteRT 2.1.6 cannot
safely select among several dispatch libraries in one directory.

Qualcomm QNN components retain their accompanying vendor terms. The APK bundles
matching QAIRT 2.47.0.260601 runtime pairs for the officially supported SM8450,
SM8475, SM8550, SM8650, SM8750, and SM8850 mobile SoCs. At runtime, only the
detected SoC's HTP stub and DSP image are exposed to LiteRT.

## Reproducibility

The plugins were built with Bazel 7.7.0 and Android NDK 29 from these LiteRT
targets:

```text
//litert/vendors/mediatek/compiler:compiler_plugin_so
//litert/vendors/mediatek/dispatch:dispatch_api_so
//litert/vendors/samsung/compiler:compiler_plugin_so
//litert/vendors/samsung/dispatch:dispatch_api_so
```

Packaged SHA-256 values:

```text
d05723c938f622a76ccf94796892336432788a143cf3faf78ace0b3a8f08f19f  libLiteRtCompilerPlugin_MediaTek.so
40ec74300e5eb00012b0a0f47191ef9ba250fc08a6fe97ef56643a3266ef5846  libLiteRtDispatch_MediaTek.so
434ec1ab0a4ca07ed5557c8bb2d56b83f9119708c01a08f5d386dbb94b990297  libLiteRtCompilerPlugin_Samsung.so
76cc5ad4817439af7135afdcce6346adf0ea86bd649a1aac49704217b5dd3232  libLiteRtDispatch_Samsung.so
87f6054306a2045c5c3efbfd2ae2b7ec9f8b42f605abec369468d73858264452  libQnnHtpV69Skel.so
e9121b1fe3b5b0e6a8eab0b6d6442c6e57e1822da201cba86f91d1a567a91875  libQnnHtpV69Stub.so
b5084469a693b05e372eedf8861e29e92c6e012c1cf543144c894f6141a5348f  libQnnHtpV73Skel.so
baeaa618d456ec59a40007084a411eb172e530701f770ef8b71a629cc0f12d97  libQnnHtpV73Stub.so
607be3d7ec64df053f019438ad6eea59dd3eea5d5f985001709d52d5663479ba  libQnnHtpV75Skel.so
b126a79ebeabf09949656be42326546b03f3216ad671c9c0e4ac935cefa5a631  libQnnHtpV75Stub.so
dd8987c17928e53877e51ce5739c550d3e74393bd5b1b5ccf311326eddb08612  libQnnHtpV79Skel.so
5f7460da65239d97be7b1845c2cebe20a4b8a92805721d9be796b2c4e706f132  libQnnHtpV79Stub.so
0b4fa7419e7265ae33d5e57f214d49cd11abbf7302149bb7ff78d9b602903899  libQnnHtpV81Skel.so
479da62bd52bb7cb5791d5898442457272d5403e42244af690a137b52d67f939  libQnnHtpV81Stub.so
```
