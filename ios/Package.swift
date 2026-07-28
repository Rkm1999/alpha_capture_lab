// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "AlphaCaptureLab",
    platforms: [.iOS(.v17)],
    products: [
        .library(name: "AlphaCaptureLab", targets: ["AlphaCaptureLab"]),
    ],
    dependencies: [
        .package(url: "https://github.com/SDWebImage/libwebp-Xcode.git", exact: "1.5.0"),
    ],
    targets: [
        .binaryTarget(name: "ONNXRuntimeBinary", path: "Vendor/onnxruntime.xcframework"),
        .target(
            name: "RawRefineryRuntime",
            dependencies: ["ONNXRuntimeBinary"],
            path: "Sources/RawRefineryRuntime",
            publicHeadersPath: "include",
            linkerSettings: [
                .linkedLibrary("c++"),
                .linkedLibrary("z"),
                .linkedFramework("CoreML"),
                .linkedFramework("Accelerate"),
            ]
        ),
        .target(
            name: "AlphaCaptureLab",
            dependencies: ["RawRefineryRuntime", .product(name: "libwebp", package: "libwebp-Xcode")],
            path: "Sources/AlphaCaptureLab",
            resources: [.process("Resources")]
        ),
        .testTarget(
            name: "AlphaCaptureLabTests",
            dependencies: ["AlphaCaptureLab"],
            path: "Tests/AlphaCaptureLabTests"
        ),
    ]
)
