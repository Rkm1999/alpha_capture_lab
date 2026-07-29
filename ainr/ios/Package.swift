// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "SCUNetDenoiser",
    platforms: [.iOS(.v17)],
    products: [
        .library(name: "SCUNetDenoiser", targets: ["SCUNetDenoiser"]),
    ],
    dependencies: [
        .package(path: "../ios-runtime"),
    ],
    targets: [
        .target(
            name: "SCUNetDenoiser",
            dependencies: [
                .product(name: "AINRRuntime", package: "ios-runtime"),
            ],
            resources: [
                .copy("Resources/Samples"),
            ]
        ),
    ]
)
