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
    var highlights = 1.0
    var shadows = 0.0
    var warmth = 0.0
    var sharpen = 0.0
    var denoise = 0.0
    var lut = LUTSelection()
}

@MainActor
final class ImageProcessor {
    static let shared = ImageProcessor()
    private let context = CIContext(options: [.cacheIntermediates: true])

    func render(_ image: CIImage, edits: EditParameters, library: LUTLibrary) -> CIImage {
        var output = image.oriented(.up)
        let color = CIFilter.colorControls()
        color.inputImage = output
        color.contrast = Float(edits.contrast)
        color.saturation = Float(edits.saturation)
        output = color.outputImage ?? output
        let exposure = CIFilter.exposureAdjust()
        exposure.inputImage = output
        exposure.ev = Float(edits.exposure)
        output = exposure.outputImage ?? output
        let tone = CIFilter.highlightShadowAdjust()
        tone.inputImage = output
        tone.highlightAmount = Float(edits.highlights)
        tone.shadowAmount = Float(edits.shadows)
        output = tone.outputImage ?? output
        if edits.warmth != 0 {
            let temperature = CIFilter.temperatureAndTint()
            temperature.inputImage = output
            temperature.neutral = CIVector(x: 6500, y: 0)
            temperature.targetNeutral = CIVector(x: 6500 + edits.warmth * 2500, y: 0)
            output = temperature.outputImage ?? output
        }
        if edits.denoise > 0 {
            let noise = CIFilter.noiseReduction()
            noise.inputImage = output
            noise.noiseLevel = Float(0.02 + edits.denoise * 0.08)
            noise.sharpness = 0.4
            output = noise.outputImage ?? output
        }
        if edits.sharpen > 0 {
            let sharp = CIFilter.sharpenLuminance()
            sharp.inputImage = output
            sharp.sharpness = Float(edits.sharpen * 1.5)
            output = sharp.outputImage ?? output
        }
        return applyLUT(output, selection: edits.lut, library: library)
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

    func encode(_ image: CIImage, format: OutputFormat, location: CLLocation? = nil, quality: CGFloat = 0.94) -> (Data, String)? {
        guard let cg = context.createCGImage(image, from: image.extent) else { return nil }
        if format == .webp, location == nil, let data = webP(cg, quality: Float(quality * 100)) { return (data, "webp") }
        let requestedType = format == .webp ? UTType.webP.identifier as CFString : UTType.jpeg.identifier as CFString
        let supported = (CGImageDestinationCopyTypeIdentifiers() as? [String])?.contains(requestedType as String) == true
        let type = supported ? requestedType : UTType.jpeg.identifier as CFString
        let output = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(output, type, 1, nil) else { return nil }
        var properties: [CFString: Any] = [kCGImageDestinationLossyCompressionQuality: quality]
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
