import CoreImage
import Foundation
import RawRefineryRuntime

actor RawRefineryProcessor {
    static let shared = RawRefineryProcessor()
    private var denoise: OpaquePointer?
    private var sharpen: OpaquePointer?
    private let context = CIContext()

    func process(_ source: CIImage, iso: Int, denoiseStrength: Double, sharpenStrength: Double) throws -> CIImage {
        guard let initial = context.createCGImage(source, from: source.extent) else { throw ProcessingError.invalidImage }
        var pixels = try Self.rgba(initial)
        let width = initial.width, height = initial.height
        if denoiseStrength > 0 {
            let session = try session(named: "rawrefinery_denoise_light", cache: &denoise)
            pixels = try Self.run(pixels, width: width, height: height, session: session, iso: iso,
                                  strength: Float(denoiseStrength), affineMatch: false)
        }
        if sharpenStrength > 0 {
            let session = try session(named: "rawrefinery_deep_sharpen", cache: &sharpen)
            pixels = try Self.run(pixels, width: width, height: height, session: session, iso: iso,
                                  strength: Float(sharpenStrength), affineMatch: true)
        }
        guard let provider = CGDataProvider(data: Data(pixels) as CFData),
              let cg = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                               bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                               bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                               provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent) else {
            throw ProcessingError.invalidImage
        }
        return CIImage(cgImage: cg)
    }

    private func session(named name: String, cache: inout OpaquePointer?) throws -> OpaquePointer {
        if let cache { return cache }
        guard let url = Bundle.module.url(forResource: name, withExtension: "onnx", subdirectory: "Models") else {
            throw RawRefineryError("Missing \(name) model")
        }
        var message = [CChar](repeating: 0, count: 1024)
        let created = url.path.withCString { rr_session_create($0, 1, &message, message.count) }
        guard let created else { throw RawRefineryError(Self.errorMessage(message)) }
        cache = created
        return created
    }

    private static func rgba(_ image: CGImage) throws -> [UInt8] {
        var pixels = [UInt8](repeating: 0, count: image.width * image.height * 4)
        guard let context = CGContext(data: &pixels, width: image.width, height: image.height,
                                      bitsPerComponent: 8, bytesPerRow: image.width * 4,
                                      space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { throw ProcessingError.invalidImage }
        context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        return pixels
    }

    private static func run(_ pixels: [UInt8], width: Int, height: Int, session: OpaquePointer,
                            iso: Int, strength: Float, affineMatch: Bool) throws -> [UInt8] {
        var destination = pixels
        let condition = max(1 / 6400, Float(min(max(iso, 0), 6400)) / 6400)
        let core = 448, padding = 16
        for coreY in stride(from: 0, to: height, by: core) {
            for coreX in stride(from: 0, to: width, by: core) {
                let coreWidth = min(core, width - coreX), coreHeight = min(core, height - coreY)
                let tileWidth = coreWidth + padding * 2, tileHeight = coreHeight + padding * 2
                let plane = tileWidth * tileHeight
                var input = [Float](repeating: 0, count: plane * 3)
                for y in 0..<tileHeight { for x in 0..<tileWidth {
                    let sx = min(max(coreX + x - padding, 0), width - 1)
                    let sy = min(max(coreY + y - padding, 0), height - 1)
                    let pixel = (sy * width + sx) * 4, index = y * tileWidth + x
                    let r = linear(Float(pixels[pixel]) / 255), g = linear(Float(pixels[pixel + 1]) / 255), b = linear(Float(pixels[pixel + 2]) / 255)
                    input[index] = 0.627404*r + 0.329282*g + 0.0433136*b
                    input[plane + index] = 0.069097*r + 0.91954*g + 0.0113612*b
                    input[plane * 2 + index] = 0.0163916*r + 0.0880132*g + 0.895595*b
                }}
                var output = [Float](repeating: 0, count: input.count)
                let inputCount = input.count
                var message = [CChar](repeating: 0, count: 1024)
                let success = input.withUnsafeMutableBufferPointer { inputPointer in
                    output.withUnsafeMutableBufferPointer { outputPointer in
                        rr_session_run(session, inputPointer.baseAddress, inputCount, condition,
                                       tileWidth, tileHeight, outputPointer.baseAddress, &message, message.count)
                    }
                }
                guard success != 0 else { throw RawRefineryError(errorMessage(message)) }
                if affineMatch { match(&output, target: input, plane: plane) }
                for y in 0..<coreHeight { for x in 0..<coreWidth {
                    let tile = (y + padding) * tileWidth + x + padding
                    let r20 = mix(input[tile], output[tile], strength)
                    let g20 = mix(input[plane + tile], output[plane + tile], strength)
                    let b20 = mix(input[plane * 2 + tile], output[plane * 2 + tile], strength)
                    let pixel = ((coreY + y) * width + coreX + x) * 4
                    destination[pixel] = byte(srgb(1.660491*r20 - 0.587641*g20 - 0.072850*b20))
                    destination[pixel + 1] = byte(srgb(-0.124550*r20 + 1.132900*g20 - 0.008349*b20))
                    destination[pixel + 2] = byte(srgb(-0.018151*r20 - 0.100579*g20 + 1.118730*b20))
                    destination[pixel + 3] = 255
                }}
            }
        }
        return destination
    }

    private static func match(_ output: inout [Float], target: [Float], plane: Int) {
        for channel in 0..<3 {
            let start = channel * plane
            let sourceMean = output[start..<(start + plane)].reduce(0, +) / Float(plane)
            let targetMean = target[start..<(start + plane)].reduce(0, +) / Float(plane)
            var variance: Float = 0, covariance: Float = 0
            for index in start..<(start + plane) {
                let delta = output[index] - sourceMean
                variance += delta * delta; covariance += delta * (target[index] - targetMean)
            }
            let scale = covariance / (variance + 1e-8), bias = targetMean - scale * sourceMean
            for index in start..<(start + plane) { output[index] = output[index] * scale + bias }
        }
    }

    private static func linear(_ value: Float) -> Float { value <= 0.04045 ? value / 12.92 : pow((value + 0.055) / 1.055, 2.4) }
    private static func srgb(_ value: Float) -> Float { let v = max(0, value); return min(1, v <= 0.0031308 ? 12.92*v : 1.055*pow(v, 1/2.4)-0.055) }
    private static func byte(_ value: Float) -> UInt8 { UInt8(min(255, max(0, Int(value * 255 + 0.5)))) }
    private static func mix(_ a: Float, _ b: Float, _ strength: Float) -> Float { a + (b - a) * min(1, max(0, strength)) }
    private static func errorMessage(_ bytes: [CChar]) -> String {
        String(decoding: bytes.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }
}

struct RawRefineryError: LocalizedError { let message: String; init(_ message: String) { self.message = message }; var errorDescription: String? { message } }
