// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "AINRRuntime",
    platforms: [.iOS(.v17)],
    products: [
        .library(name: "AINRRuntime", targets: ["AINRRuntime"]),
    ],
    targets: [
        .target(
            name: "AINRRuntime",
            resources: [.copy("Resources")]
        ),
    ]
)
