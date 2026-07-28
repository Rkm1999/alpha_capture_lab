import CoreImage
import Vision

enum ComputationalCapture {
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

    static func panorama(_ data: [Data]) throws -> CIImage {
        let images = try sourceImages(data)
        guard !images.isEmpty else { throw ProcessingError.noFrames }
        var transforms: [CGAffineTransform] = [.identity]
        var accumulated = CGAffineTransform.identity
        for index in 1..<images.count {
            let pair = try registration(moving: images[index], reference: images[index - 1])
            accumulated = pair.concatenating(accumulated)
            transforms.append(accumulated)
        }
        let extents = zip(images, transforms).map { $0.transformed(by: $1).extent }
        let canvas = extents.reduce(CGRect.null) { $0.union($1) }
        var output = images[0].transformed(by: transforms[0])
        for index in 1..<images.count {
            let image = images[index], transform = transforms[index]
            let moved = image.transformed(by: transform)
            let overlap = output.extent.intersection(moved.extent)
            if overlap.isNull || overlap.width < 8 || overlap.height < 8 {
                output = moved.applyingFilter("CISourceOverCompositing", parameters: [kCIInputBackgroundImageKey: output])
            } else {
                let horizontal = abs(transform.tx - transforms[index - 1].tx) >= abs(transform.ty - transforms[index - 1].ty)
                let forward = horizontal ? transform.tx >= transforms[index - 1].tx : transform.ty >= transforms[index - 1].ty
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
