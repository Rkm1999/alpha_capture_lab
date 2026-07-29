import CoreImage
import Foundation
import ZIPFoundation

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
            let line = raw.trimmingCharacters(
                in: .whitespacesAndNewlines.union(CharacterSet(charactersIn: "\u{FEFF}"))
            )
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
    case noCubeInArchive
    case noValidCube

    var errorDescription: String? {
        switch self {
        case .invalidCube: "The file is not a valid 3D .cube LUT."
        case .noCubeInArchive: "The archive does not contain a .cube LUT."
        case .noValidCube: "No valid 3D LUT could be imported."
        }
    }
}

struct LUTImportResult {
    let imported: [CubeLUT]
    let newCount: Int
    let duplicateCount: Int
    let rejectedCount: Int

    var summary: String {
        let primary: String
        if newCount == 1 {
            primary = "Imported \(imported.first?.title ?? "1") LUT"
        } else if newCount > 1 {
            primary = "Imported \(newCount) LUTs"
        } else {
            primary = duplicateCount == 1 ? "LUT was already imported" : "\(duplicateCount) LUTs were already imported"
        }
        return rejectedCount > 0 ? "\(primary). Skipped \(rejectedCount) invalid item(s)." : primary
    }
}

@MainActor
final class LUTLibrary: ObservableObject {
    @Published private(set) var imported: [CubeLUT] = []
    let presets = ["Original", "Cinema", "Warm", "Cool", "Mono", "Fade", "Punch", "Teal"]
    private let directory: URL
    private let inboxDirectory: URL

    init() {
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let applicationSupport = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first!
        directory = documents.appendingPathComponent("LUTs", isDirectory: true)
        inboxDirectory = applicationSupport.appendingPathComponent("LUTImportInbox", isDirectory: true)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try? FileManager.default.createDirectory(at: inboxDirectory, withIntermediateDirectories: true)
        reload()
        recoverPendingImports()
    }

    var identifiers: [String] {
        [presets[0]] + imported.map(\.id) + Array(presets.dropFirst())
    }
    func lut(id: String) -> CubeLUT? { imported.first { $0.id == id } }

    @discardableResult
    func importFiles(_ urls: [URL]) throws -> LUTImportResult {
        var staged: [(url: URL, name: String)] = []
        for url in urls {
            let access = url.startAccessingSecurityScopedResource()
            defer { if access { url.stopAccessingSecurityScopedResource() } }
            let data = try Data(contentsOf: url)
            let destination = inboxDirectory.appendingPathComponent(
                "\(UUID().uuidString)__\(url.lastPathComponent)"
            )
            try data.write(to: destination, options: .atomic)
            staged.append((destination, url.lastPathComponent))
        }
        defer { staged.forEach { try? FileManager.default.removeItem(at: $0.url) } }
        return try importStagedFiles(staged)
    }

    private func importStagedFiles(
        _ files: [(url: URL, name: String)]
    ) throws -> LUTImportResult {
        var candidates: [(String, Data)] = []
        for file in files {
            let data = try Data(contentsOf: file.url)
            let fileExtension = URL(fileURLWithPath: file.name).pathExtension.lowercased()
            if fileExtension == "cube" {
                candidates.append((file.name, data))
            } else if fileExtension == "zip" || data.starts(with: [0x50, 0x4b]) {
                candidates += try LUTArchive.cubeFiles(in: data)
            }
        }
        guard !candidates.isEmpty else { throw LUTError.invalidCube }
        var importedIDs: [String] = []
        var newCount = 0
        var duplicateCount = 0
        var rejectedCount = 0
        for (name, data) in candidates {
            guard let text = Self.cubeText(from: data),
                  (try? CubeLUT.parse(
                    text,
                    title: URL(fileURLWithPath: name).deletingPathExtension().lastPathComponent
                  )) != nil else {
                rejectedCount += 1
                continue
            }
            let base = URL(fileURLWithPath: name).deletingPathExtension().lastPathComponent
            var destination = directory.appendingPathComponent("\(base).cube")
            var suffix = 2
            while FileManager.default.fileExists(atPath: destination.path),
                  (try? Data(contentsOf: destination)) != data {
                destination = directory.appendingPathComponent("\(base) \(suffix).cube")
                suffix += 1
            }
            if FileManager.default.fileExists(atPath: destination.path) {
                duplicateCount += 1
            } else {
                try data.write(to: destination, options: .atomic)
                newCount += 1
            }
            importedIDs.append(destination.lastPathComponent)
        }
        guard !importedIDs.isEmpty else { throw LUTError.noValidCube }
        reload()
        return LUTImportResult(
            imported: importedIDs.compactMap(lut(id:)),
            newCount: newCount,
            duplicateCount: duplicateCount,
            rejectedCount: rejectedCount
        )
    }

    private func recoverPendingImports() {
        let pending = (try? FileManager.default.contentsOfDirectory(
            at: inboxDirectory,
            includingPropertiesForKeys: nil
        )) ?? []
        guard !pending.isEmpty else { return }
        let staged = pending.map { url in
            let components = url.lastPathComponent.components(separatedBy: "__")
            return (url: url, name: components.dropFirst().joined(separator: "__"))
        }
        _ = try? importStagedFiles(staged)
        pending.forEach { try? FileManager.default.removeItem(at: $0) }
    }

    private func reload() {
        let urls = (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? []
        imported = urls.filter { $0.pathExtension.lowercased() == "cube" }.compactMap { url in
            guard let data = try? Data(contentsOf: url),
                  let text = Self.cubeText(from: data) else { return nil }
            return try? CubeLUT.parse(text, title: url.deletingPathExtension().lastPathComponent, sourceURL: url)
        }.sorted { $0.title.localizedStandardCompare($1.title) == .orderedAscending }
    }

    private static func cubeText(from data: Data) -> String? {
        String(data: data, encoding: .utf8)
            ?? String(data: data, encoding: .utf16)
            ?? String(data: data, encoding: .utf16LittleEndian)
            ?? String(data: data, encoding: .utf16BigEndian)
    }
}

enum LUTArchive {
    static func cubeFiles(in archive: Data) throws -> [(String, Data)] {
        let archive = try Archive(data: archive, accessMode: .read)
        var result: [(String, Data)] = []
        for entry in archive where entry.type == .file && entry.path.lowercased().hasSuffix(".cube") {
            var data = Data()
            _ = try archive.extract(entry) { data.append($0) }
            result.append((URL(fileURLWithPath: entry.path).lastPathComponent, data))
        }
        guard !result.isEmpty else { throw LUTError.noCubeInArchive }
        return result
    }
}
