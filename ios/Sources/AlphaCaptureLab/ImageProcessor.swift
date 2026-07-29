import CoreImage
import CoreImage.CIFilterBuiltins
import ImageIO
import UIKit
import UniformTypeIdentifiers
import CoreLocation
import libwebp

struct EditParameters: Equatable {
    var exposure = 0.0
    var contrast = 1.0
    var saturation = 1.0
    var denoise = 0.0
    var lut = LUTSelection()
    var geometry = EditorGeometry()
}

@MainActor
final class ImageProcessor {
    static let shared = ImageProcessor()
    private let context = CIContext(options: [.cacheIntermediates: true])

    func render(_ image: CIImage, edits: EditParameters, library: LUTLibrary) -> CIImage {
        var output = applyGeometry(image.oriented(.up), geometry: edits.geometry)
        let color = CIFilter.colorControls()
        color.inputImage = output
        color.contrast = Float(edits.contrast)
        color.saturation = Float(edits.saturation)
        output = color.outputImage ?? output
        let exposure = CIFilter.exposureAdjust()
        exposure.inputImage = output
        exposure.ev = Float(edits.exposure)
        output = exposure.outputImage ?? output
        return applyLUT(output, selection: edits.lut, library: library)
    }

    func applyGeometry(_ source: CIImage, geometry: EditorGeometry) -> CIImage {
        var image = source.oriented(.up)
        let extent = image.extent
        if geometry.perspective.count == 4,
           geometry.perspective != EditorGeometry.identityPerspective {
            func vector(_ point: NormalizedPoint) -> CIVector {
                CIVector(
                    x: extent.minX + point.x * extent.width,
                    y: extent.minY + (1 - point.y) * extent.height
                )
            }
            image = image.applyingFilter("CIPerspectiveCorrection", parameters: [
                "inputTopLeft": vector(geometry.perspective[0]),
                "inputTopRight": vector(geometry.perspective[1]),
                "inputBottomRight": vector(geometry.perspective[2]),
                "inputBottomLeft": vector(geometry.perspective[3]),
            ])
        }
        let radians = Double(geometry.normalizedQuarterTurns) * .pi / 2 +
            geometry.straightenDegrees * .pi / 180
        if abs(radians) > 0.0001 {
            let center = CGPoint(x: image.extent.midX, y: image.extent.midY)
            image = image.transformed(by:
                CGAffineTransform(translationX: center.x, y: center.y)
                    .rotated(by: radians)
                    .translatedBy(x: -center.x, y: -center.y)
            )
        }
        let crop = geometry.crop.normalized()
        if !crop.isIdentity {
            let current = image.extent
            let rect = CGRect(
                x: current.minX + crop.left * current.width,
                y: current.minY + (1 - crop.bottom) * current.height,
                width: (crop.right - crop.left) * current.width,
                height: (crop.bottom - crop.top) * current.height
            )
            image = image.cropped(to: rect)
        }
        let finalExtent = image.extent.integral
        return image.transformed(by: .init(
            translationX: -finalExtent.minX,
            y: -finalExtent.minY
        ))
    }

    func applyLUT(_ image: CIImage, selection: LUTSelection, library: LUTLibrary) -> CIImage {
        guard selection.identifier != "Original", selection.strength > 0 else { return image }
        let transformed: CIImage
        if let lut = library.lut(id: selection.identifier) {
            let filter = CIFilter.colorCube()
            filter.inputImage = image
            filter.cubeDimension = Float(lut.dimension)
            filter.cubeData = lut.cubeData
            transformed = filter.outputImage ?? image
        } else {
            transformed = preset(selection.identifier, image: image)
        }
        guard selection.strength < 0.999 else { return transformed }
        let blend = CIFilter.dissolveTransition()
        blend.inputImage = image
        blend.targetImage = transformed
        blend.time = Float(selection.strength)
        return blend.outputImage?.cropped(to: image.extent) ?? transformed
    }

    func jpeg(_ image: CIImage, quality: CGFloat = 0.94) -> Data? {
        guard let cg = context.createCGImage(image, from: image.extent) else { return nil }
        return UIImage(cgImage: cg).jpegData(compressionQuality: quality)
    }

    func encode(
        _ image: CIImage,
        format: OutputFormat,
        location: CLLocation? = nil,
        sourceData: Data? = nil,
        quality: CGFloat = 0.94
    ) -> (Data, String)? {
        guard let cg = context.createCGImage(image, from: image.extent) else { return nil }
        if format == .webp, location == nil, let data = webP(cg, quality: Float(quality * 100)) { return (data, "webp") }
        let requestedType = format == .webp ? UTType.webP.identifier as CFString : UTType.jpeg.identifier as CFString
        let supported = (CGImageDestinationCopyTypeIdentifiers() as? [String])?.contains(requestedType as String) == true
        let type = supported ? requestedType : UTType.jpeg.identifier as CFString
        let output = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(output, type, 1, nil) else { return nil }
        var properties = sourceData
            .flatMap { CGImageSourceCreateWithData($0 as CFData, nil) }
            .flatMap { CGImageSourceCopyPropertiesAtIndex($0, 0, nil) as? [CFString: Any] }
            ?? [:]
        properties[kCGImageDestinationLossyCompressionQuality] = quality
        properties[kCGImagePropertyOrientation] = 1
        if let location {
            let coordinate = location.coordinate
            properties[kCGImagePropertyGPSDictionary] = [
                kCGImagePropertyGPSLatitude: abs(coordinate.latitude),
                kCGImagePropertyGPSLatitudeRef: coordinate.latitude >= 0 ? "N" : "S",
                kCGImagePropertyGPSLongitude: abs(coordinate.longitude),
                kCGImagePropertyGPSLongitudeRef: coordinate.longitude >= 0 ? "E" : "W",
                kCGImagePropertyGPSAltitude: abs(location.altitude),
                kCGImagePropertyGPSAltitudeRef: location.altitude >= 0 ? 0 : 1,
                kCGImagePropertyGPSDateStamp: Self.gpsDate.string(from: location.timestamp),
            ]
        }
        CGImageDestinationAddImage(destination, cg, properties as CFDictionary)
        guard CGImageDestinationFinalize(destination) else { return nil }
        return (output as Data, type == UTType.webP.identifier as CFString ? "webp" : "jpg")
    }

    private func webP(_ image: CGImage, quality: Float) -> Data? {
        let width = image.width, height = image.height, stride = width * 4
        var pixels = [UInt8](repeating: 0, count: stride * height)
        guard let bitmap = CGContext(data: &pixels, width: width, height: height, bitsPerComponent: 8,
                                     bytesPerRow: stride, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                     bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        bitmap.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        var encoded: UnsafeMutablePointer<UInt8>?
        let size = pixels.withUnsafeBufferPointer { WebPEncodeRGBA($0.baseAddress, Int32(width), Int32(height), Int32(stride), quality, &encoded) }
        guard size > 0, let encoded else { return nil }
        defer { WebPFree(encoded) }
        return Data(bytes: encoded, count: size)
    }

    func preview(_ image: CIImage, maxDimension: CGFloat = 1600) -> UIImage? {
        let scale = min(1, maxDimension / max(image.extent.width, image.extent.height))
        let scaled = image.transformed(by: .init(scaleX: scale, y: scale))
        guard let cg = context.createCGImage(scaled, from: scaled.extent) else { return nil }
        return UIImage(cgImage: cg)
    }

    func previewJPEG(_ image: CIImage, maxDimension: CGFloat = 960) -> Data? {
        preview(image, maxDimension: maxDimension)?.jpegData(compressionQuality: 0.96)
    }

    func blend(_ original: CIImage, _ denoised: CIImage, strength: Double) -> CIImage {
        let amount = max(0, min(1, strength))
        guard amount > 0 else { return original }
        guard amount < 0.999 else { return denoised }
        let filter = CIFilter.dissolveTransition()
        filter.inputImage = original
        filter.targetImage = denoised
        filter.time = Float(amount)
        return filter.outputImage?.cropped(to: original.extent) ?? denoised
    }

    private func preset(_ name: String, image: CIImage) -> CIImage {
        switch name {
        case "Mono": return image.applyingFilter("CIPhotoEffectNoir")
        case "Fade": return image.applyingFilter("CIPhotoEffectFade")
        case "Punch": return image.applyingFilter("CIVibrance", parameters: [kCIInputAmountKey: 0.7])
        case "Warm": return image.applyingFilter("CITemperatureAndTint", parameters: ["inputNeutral": CIVector(x: 6500, y: 0), "inputTargetNeutral": CIVector(x: 7600, y: 0)])
        case "Cool": return image.applyingFilter("CITemperatureAndTint", parameters: ["inputNeutral": CIVector(x: 6500, y: 0), "inputTargetNeutral": CIVector(x: 5200, y: 0)])
        case "Cinema": return image.applyingFilter("CIPhotoEffectProcess")
        case "Teal": return image.applyingFilter("CIColorMatrix", parameters: ["inputRVector": CIVector(x: 0.92, y: 0, z: 0.06, w: 0), "inputBVector": CIVector(x: 0.02, y: 0.08, z: 1.05, w: 0)])
        default: return image
        }
    }

    private static let gpsDate: DateFormatter = {
        let formatter = DateFormatter(); formatter.locale = Locale(identifier: "en_US_POSIX"); formatter.timeZone = .gmt
        formatter.dateFormat = "yyyy:MM:dd"
        return formatter
    }()
}
