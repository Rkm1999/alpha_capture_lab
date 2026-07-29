import CoreImage
import ImageIO
import PanoramaOpenCVBridge
import simd
import Vision

enum ComputationalCapture {
    private static let panoramaRegistrationEdge = 2048
    private static let panoramaPreviewEdge = 1600
    private static let panoramaOutputBytesPerPixel: UInt64 = 8
    private static let panoramaMemoryFraction = 0.08

    static func liveND(_ data: [Data]) throws -> CIImage {
        let images = try aligned(data)
        guard var output = images.first else { throw ProcessingError.noFrames }
        for image in images.dropFirst() { output = image.applyingFilter("CIAdditionCompositing", parameters: [kCIInputBackgroundImageKey: output]) }
        let scale = 1 / CGFloat(images.count)
        return output.applyingFilter("CIColorMatrix", parameters: [
            "inputRVector": CIVector(x: scale, y: 0, z: 0, w: 0),
            "inputGVector": CIVector(x: 0, y: scale, z: 0, w: 0),
            "inputBVector": CIVector(x: 0, y: 0, z: scale, w: 0),
            "inputAVector": CIVector(x: 0, y: 0, z: 0, w: scale),
        ]).cropped(to: output.extent)
    }

    static func composite(_ data: [Data]) throws -> CIImage {
        let images = try aligned(data)
        guard var output = images.first else { throw ProcessingError.noFrames }
        for image in images.dropFirst() {
            output = image.applyingFilter("CILightenBlendMode", parameters: [kCIInputBackgroundImageKey: output]).cropped(to: output.extent)
        }
        return output
    }

    static func panoramaPreview(_ data: [Data]) throws -> CIImage {
        let jpeg = try ACLPanoramaStitcher.stitchFrames(data, preview: true)
        guard let result = CIImage(data: jpeg) else {
            throw ProcessingError.invalidImage
        }
        return result
    }

    static func panorama(_ data: [Data]) throws -> CIImage {
        let jpeg = try ACLPanoramaStitcher.stitchFrames(data, preview: false)
        guard let result = CIImage(data: jpeg) else {
            throw ProcessingError.invalidImage
        }
        return result
    }

    private static func panoramaTransforms(_ images: [CIImage]) throws -> [simd_float3x3] {
        guard !images.isEmpty else { throw ProcessingError.noFrames }
        var transforms: [simd_float3x3] = [matrix_identity_float3x3]
        var accumulated = matrix_identity_float3x3
        for index in 1..<images.count {
            let pair = try homography(moving: images[index], reference: images[index - 1])
            accumulated = simd_mul(accumulated, pair)
            transforms.append(accumulated)
        }
        return transforms
    }

    private static func stitch(
        images: [CIImage],
        transforms: [simd_float3x3]
    ) throws -> CIImage {
        guard !images.isEmpty, images.count == transforms.count else {
            throw ProcessingError.noFrames
        }
        let warped = zip(images, transforms).map { project($0, with: $1) }
        let extents = warped.map(\.extent)
        let canvas = extents.reduce(CGRect.null) { $0.union($1) }
        var output = warped[0]
        for index in 1..<warped.count {
            let moved = warped[index]
            let overlap = output.extent.intersection(moved.extent)
            if overlap.isNull || overlap.width < 8 || overlap.height < 8 {
                output = moved.applyingFilter("CISourceOverCompositing", parameters: [kCIInputBackgroundImageKey: output])
            } else {
                let previous = warped[index - 1].extent
                let horizontal = abs(moved.extent.midX - previous.midX) >= abs(moved.extent.midY - previous.midY)
                let forward = horizontal
                    ? moved.extent.midX >= previous.midX
                    : moved.extent.midY >= previous.midY
                let p0: CIVector, p1: CIVector
                if horizontal {
                    p0 = CIVector(x: forward ? overlap.minX : overlap.maxX, y: overlap.midY)
                    p1 = CIVector(x: forward ? overlap.maxX : overlap.minX, y: overlap.midY)
                } else {
                    p0 = CIVector(x: overlap.midX, y: forward ? overlap.minY : overlap.maxY)
                    p1 = CIVector(x: overlap.midX, y: forward ? overlap.maxY : overlap.minY)
                }
                let mask = CIFilter(name: "CILinearGradient", parameters: [
                    "inputPoint0": p0, "inputPoint1": p1,
                    "inputColor0": CIColor.white, "inputColor1": CIColor.black,
                ])?.outputImage?.cropped(to: output.extent.union(moved.extent))
                output = moved.applyingFilter("CIBlendWithMask", parameters: [
                    kCIInputBackgroundImageKey: output,
                    kCIInputMaskImageKey: mask as Any,
                ])
            }
        }
        return output.cropped(to: canvas).transformed(by: .init(translationX: -canvas.minX, y: -canvas.minY))
    }

    private static func panoramaFrames(_ data: [Data]) throws -> [PanoramaFrame] {
        guard !data.isEmpty else { throw ProcessingError.noFrames }
        return try data.map { value in
            guard let source = CIImage(
                data: value,
                options: [.applyOrientationProperty: true]
            )?.transformed(by: .init(translationX: 0, y: 0)),
            let imageSource = CGImageSourceCreateWithData(value as CFData, nil),
            let thumbnail = CGImageSourceCreateThumbnailAtIndex(
                imageSource,
                0,
                [
                    kCGImageSourceCreateThumbnailFromImageAlways: true,
                    kCGImageSourceCreateThumbnailWithTransform: true,
                    kCGImageSourceThumbnailMaxPixelSize: panoramaRegistrationEdge,
                    kCGImageSourceShouldCacheImmediately: true,
                ] as CFDictionary
            ) else {
                throw ProcessingError.invalidImage
            }
            return PanoramaFrame(
                source: source.transformed(
                    by: .init(translationX: -source.extent.minX, y: -source.extent.minY)
                ),
                preview: CIImage(cgImage: thumbnail)
            )
        }
    }

    private static func scaledToPanoramaBudget(_ image: CIImage) -> CIImage {
        let physicalMemory = ProcessInfo.processInfo.physicalMemory
        let pixelBudget = max(
            12_000_000,
            Int(Double(physicalMemory) * panoramaMemoryFraction /
                Double(panoramaOutputBytesPerPixel))
        )
        let pixels = image.extent.width * image.extent.height
        guard pixels > CGFloat(pixelBudget) else { return image }
        let scale = sqrt(CGFloat(pixelBudget) / pixels)
        return image.transformed(by: .init(scaleX: scale, y: scale))
    }

    private static func scaledToFit(_ image: CIImage, maxEdge: Int) -> CIImage {
        let edge = max(image.extent.width, image.extent.height)
        guard edge > CGFloat(maxEdge) else { return image }
        let scale = CGFloat(maxEdge) / edge
        return image.transformed(by: .init(scaleX: scale, y: scale))
    }

    private static func scaleMatrix(x: CGFloat, y: CGFloat) -> simd_float3x3 {
        simd_float3x3(
            SIMD3(Float(x), 0, 0),
            SIMD3(0, Float(y), 0),
            SIMD3(0, 0, 1)
        )
    }

    private static func aligned(_ data: [Data]) throws -> [CIImage] {
        let images = try sourceImages(data)
        guard let reference = images.first else { throw ProcessingError.noFrames }
        return [reference] + (try images.dropFirst().map { image in
            image.transformed(by: try registration(moving: image, reference: reference)).cropped(to: reference.extent)
        })
    }

    private static func sourceImages(_ data: [Data]) throws -> [CIImage] {
        try data.map {
            guard let image = CIImage(data: $0, options: [.applyOrientationProperty: true]) else { throw ProcessingError.invalidImage }
            return image
        }
    }

    private static func registration(moving: CIImage, reference: CIImage) throws -> CGAffineTransform {
        let request = VNTranslationalImageRegistrationRequest(targetedCIImage: reference)
        let handler = VNImageRequestHandler(ciImage: moving)
        try handler.perform([request])
        guard let observation = request.results?.first as? VNImageTranslationAlignmentObservation else { throw ProcessingError.alignmentFailed }
        return observation.alignmentTransform
    }

    private static func homography(moving: CIImage, reference: CIImage) throws -> simd_float3x3 {
        let request = VNHomographicImageRegistrationRequest(targetedCIImage: reference)
        let handler = VNImageRequestHandler(ciImage: moving)
        try handler.perform([request])
        guard let observation = request.results?.first as? VNImageHomographicAlignmentObservation else {
            throw ProcessingError.alignmentFailed
        }
        return observation.warpTransform
    }

    private static func project(_ image: CIImage, with matrix: simd_float3x3) -> CIImage {
        func point(_ x: CGFloat, _ y: CGFloat) -> CIVector {
            let value = simd_mul(matrix, SIMD3<Float>(Float(x), Float(y), 1))
            let divisor = abs(value.z) < 0.00001 ? 1 : value.z
            return CIVector(x: CGFloat(value.x / divisor), y: CGFloat(value.y / divisor))
        }
        let extent = image.extent
        return image.applyingFilter("CIPerspectiveTransform", parameters: [
            "inputTopLeft": point(extent.minX, extent.maxY),
            "inputTopRight": point(extent.maxX, extent.maxY),
            "inputBottomRight": point(extent.maxX, extent.minY),
            "inputBottomLeft": point(extent.minX, extent.minY),
        ])
    }
}

private struct PanoramaFrame {
    let source: CIImage
    let preview: CIImage
}

enum ProcessingError: LocalizedError {
    case noFrames, invalidImage, alignmentFailed
    var errorDescription: String? {
        switch self {
        case .noFrames: "No source frames were captured."
        case .invalidImage: "A source frame could not be decoded."
        case .alignmentFailed: "The frames do not contain enough overlap to align reliably."
        }
    }
}
