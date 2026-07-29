// swift-tools-version: 6.0

import PackageDescription
import Foundation

let vendorPath = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()
    .appendingPathComponent("Vendor")
    .path

let package = Package(
    name: "AlphaCaptureLab",
    platforms: [.iOS(.v17)],
    products: [
        .library(name: "AlphaCaptureLab", targets: ["AlphaCaptureLab"]),
    ],
    dependencies: [
        .package(url: "https://github.com/SDWebImage/libwebp-Xcode.git", exact: "1.5.0"),
        .package(url: "https://github.com/weichsel/ZIPFoundation.git", exact: "0.9.20"),
        .package(path: "../ainr/ios-runtime"),
    ],
    targets: [
        .target(
            name: "AlphaCaptureLab",
            dependencies: [
                .product(name: "libwebp", package: "libwebp-Xcode"),
                .product(name: "AINRRuntime", package: "ios-runtime"),
                "PanoramaOpenCVBridge",
                "ZIPFoundation",
            ],
            path: "Sources/AlphaCaptureLab",
            resources: [.copy("Resources/PrivacyInfo.xcprivacy")]
        ),
        .target(
            name: "PanoramaOpenCVBridge",
            path: "Sources/PanoramaOpenCVBridge",
            publicHeadersPath: "include",
            cxxSettings: [
                .unsafeFlags(["-F", vendorPath]),
            ],
            linkerSettings: [
                .unsafeFlags(["-F", vendorPath]),
                .linkedFramework("opencv2"),
                .linkedFramework("Accelerate"),
                .linkedFramework("AVFoundation"),
                .linkedFramework("CoreGraphics"),
                .linkedFramework("CoreImage"),
                .linkedFramework("CoreMedia"),
                .linkedFramework("QuartzCore"),
                .linkedFramework("UIKit"),
            ]
        ),
        .testTarget(
            name: "AlphaCaptureLabTests",
            dependencies: ["AlphaCaptureLab"],
            path: "Tests/AlphaCaptureLabTests"
        ),
    ]
)
