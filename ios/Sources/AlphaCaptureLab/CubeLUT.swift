import CoreImage
import Foundation
import RawRefineryRuntime

struct CubeLUT: Identifiable, Hashable {
    let id: String
    let title: String
    let dimension: Int
    let cubeData: Data
    let sourceURL: URL?

    static func parse(_ text: String, title fallback: String, sourceURL: URL? = nil) throws -> CubeLUT {
        var title = fallback
        var dimension = 0
        var values: [Float] = []
        for raw in text.components(separatedBy: .newlines) {
            let line = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !line.isEmpty, !line.hasPrefix("#") else { continue }
            let pieces = line.split(whereSeparator: { $0 == " " || $0 == "\t" })
            if pieces.first == "TITLE" {
                title = line.dropFirst(5).trimmingCharacters(in: CharacterSet(charactersIn: " \t\""))
            } else if pieces.first == "LUT_3D_SIZE", pieces.count > 1 {
                dimension = Int(pieces[1]) ?? 0
            } else if pieces.count >= 3, let r = Float(pieces[0]), let g = Float(pieces[1]), let b = Float(pieces[2]) {
                values.append(contentsOf: [r, g, b, 1])
            }
        }
        guard dimension >= 2, values.count == dimension * dimension * dimension * 4 else { throw LUTError.invalidCube }
        return CubeLUT(id: sourceURL?.lastPathComponent ?? title, title: title, dimension: dimension,
                       cubeData: values.withUnsafeBufferPointer { Data(buffer: $0) }, sourceURL: sourceURL)
    }
}

enum LUTError: LocalizedError {
    case invalidCube
    var errorDescription: String? { "The file is not a valid 3D .cube LUT." }
}

@MainActor
final class LUTLibrary: ObservableObject {
    @Published private(set) var imported: [CubeLUT] = []
    let presets = ["Original", "Cinema", "Warm", "Cool", "Mono", "Fade", "Punch", "Teal"]
    private let directory: URL

    init() {
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        directory = documents.appendingPathComponent("LUTs", isDirectory: true)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        reload()
    }

    var identifiers: [String] { presets + imported.map(\.id) }
    func lut(id: String) -> CubeLUT? { imported.first { $0.id == id } }

    func importFiles(_ urls: [URL]) throws {
        var candidates: [(String, Data)] = []
        for url in urls {
            let access = url.startAccessingSecurityScopedResource()
            defer { if access { url.stopAccessingSecurityScopedResource() } }
            let data = try Data(contentsOf: url)
            if url.pathExtension.lowercased() == "cube" { candidates.append((url.lastPathComponent, data)) }
            if url.pathExtension.lowercased() == "zip" { candidates += try SimpleZIP.cubeFiles(in: data) }
        }
        guard !candidates.isEmpty else { throw LUTError.invalidCube }
        for (name, data) in candidates {
            guard let text = String(data: data, encoding: .utf8) else { continue }
            _ = try CubeLUT.parse(text, title: URL(fileURLWithPath: name).deletingPathExtension().lastPathComponent)
            let base = URL(fileURLWithPath: name).deletingPathExtension().lastPathComponent
            var destination = directory.appendingPathComponent("\(base).cube")
            var suffix = 2
            while FileManager.default.fileExists(atPath: destination.path),
                  (try? Data(contentsOf: destination)) != data {
                destination = directory.appendingPathComponent("\(base) \(suffix).cube")
                suffix += 1
            }
            try data.write(to: destination, options: .atomic)
        }
        reload()
    }

    private func reload() {
        let urls = (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? []
        imported = urls.filter { $0.pathExtension.lowercased() == "cube" }.compactMap { url in
            guard let text = try? String(contentsOf: url, encoding: .utf8) else { return nil }
            return try? CubeLUT.parse(text, title: url.deletingPathExtension().lastPathComponent, sourceURL: url)
        }.sorted { $0.title.localizedStandardCompare($1.title) == .orderedAscending }
    }
}

private enum SimpleZIP {
    static func cubeFiles(in archive: Data) throws -> [(String, Data)] {
        var result: [(String, Data)] = []
        var offset = 0
        while offset + 46 <= archive.count {
            if archive.u32(offset) != 0x02014b50 { offset += 1; continue }
            let method = archive.u16(offset + 10)
            let compressed = Int(archive.u32(offset + 20)), uncompressed = Int(archive.u32(offset + 24))
            let nameLength = Int(archive.u16(offset + 28)), extraLength = Int(archive.u16(offset + 30))
            let commentLength = Int(archive.u16(offset + 32)), localOffset = Int(archive.u32(offset + 42))
            let nameStart = offset + 46
            guard nameStart + nameLength <= archive.count, localOffset + 30 <= archive.count else { break }
            let localNameLength = Int(archive.u16(localOffset + 26)), localExtraLength = Int(archive.u16(localOffset + 28))
            let dataStart = localOffset + 30 + localNameLength + localExtraLength
            guard dataStart + compressed <= archive.count else { break }
            let name = String(data: archive[nameStart..<(nameStart + nameLength)], encoding: .utf8) ?? "lut.cube"
            let payload = Data(archive[dataStart..<(dataStart + compressed)])
            if name.lowercased().hasSuffix(".cube") {
                if method == 0 { result.append((URL(fileURLWithPath: name).lastPathComponent, payload)) }
                else if method == 8, let inflated = payload.inflate(expected: uncompressed) {
                    result.append((URL(fileURLWithPath: name).lastPathComponent, inflated))
                }
            }
            offset = nameStart + nameLength + extraLength + commentLength
        }
        return result
    }
}

private extension Data {
    func u16(_ offset: Int) -> UInt16 { UInt16(self[offset]) | UInt16(self[offset + 1]) << 8 }
    func u32(_ offset: Int) -> UInt32 { UInt32(u16(offset)) | UInt32(u16(offset + 2)) << 16 }
    func inflate(expected: Int) -> Data? {
        var output = Data(count: Swift.max(expected, count * 4, 1024))
        var written = output.count
        let success = output.withUnsafeMutableBytes { destination in
            withUnsafeBytes { source in
                rr_inflate_raw(source.bindMemory(to: UInt8.self).baseAddress!, source.count,
                               destination.bindMemory(to: UInt8.self).baseAddress!, &written)
            }
        }
        guard success != 0, written > 0 else { return nil }
        output.count = written
        return output
    }
}
