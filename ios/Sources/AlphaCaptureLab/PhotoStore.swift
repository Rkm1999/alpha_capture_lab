import Foundation

actor PhotoStore {
    private let directory: URL
    private let privateDirectory: URL
    private let indexURL: URL
    private var index: [SavedPhoto] = []

    init(fileManager: FileManager = .default) {
        let root = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!
            .appendingPathComponent("RemoteCapture", isDirectory: true)
        directory = root.appendingPathComponent("Gallery", isDirectory: true)
        privateDirectory = root.appendingPathComponent("Sources", isDirectory: true)
        indexURL = root.appendingPathComponent("captures.json")
        try? fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        try? fileManager.createDirectory(at: privateDirectory, withIntermediateDirectories: true)
        index = (try? Data(contentsOf: indexURL)).flatMap { try? JSONDecoder().decode([SavedPhoto].self, from: $0) } ?? []
        index = index.filter { fileManager.fileExists(atPath: $0.url.path) }
        let known = Set(index.map { $0.url.standardizedFileURL })
        let legacy = [root, directory].flatMap { location in
            (try? fileManager.contentsOfDirectory(at: location, includingPropertiesForKeys: [.creationDateKey], options: [.skipsHiddenFiles])) ?? []
        }.filter { ["jpg", "jpeg", "webp"].contains($0.pathExtension.lowercased()) && !known.contains($0.standardizedFileURL) }
        index += legacy.map { url in
            let date = (try? url.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? .distantPast
            return SavedPhoto(id: url.lastPathComponent, url: url, capturedAt: date)
        }
    }

    func load() -> [SavedPhoto] { index.sorted { $0.capturedAt > $1.capturedAt } }

    func save(
        _ displayData: Data,
        originalData: Data? = nil,
        kind: CaptureMode = .photo,
        sourceData: [Data] = [],
        lut: LUTSelection? = nil,
        iso: Int? = nil,
        extension fileExtension: String = "jpg"
    ) throws -> SavedPhoto {
        let stem = "RC_\(Self.timestamp.string(from: Date()))_\(UUID().uuidString.prefix(6))"
        let displayURL = directory.appendingPathComponent("\(stem).\(fileExtension)")
        try displayData.write(to: displayURL, options: .atomic)
        var originalURL: URL?
        if let originalData {
            let url = privateDirectory.appendingPathComponent("\(stem)_original.jpg")
            try originalData.write(to: url, options: .atomic)
            originalURL = url
        }
        let sources = try sourceData.enumerated().map { number, data in
            let url = privateDirectory.appendingPathComponent("\(stem)_frame_\(number + 1).jpg")
            try data.write(to: url, options: .atomic)
            return url
        }
        let photo = SavedPhoto(
            id: displayURL.lastPathComponent, url: displayURL, capturedAt: Date(), kind: kind,
            originalURL: originalURL, sourceURLs: sources,
            lutIdentifier: lut?.identifier, lutStrength: lut?.strength, iso: iso
        )
        index.insert(photo, at: 0)
        try persist()
        return photo
    }

    func replace(_ photo: SavedPhoto, data: Data) throws {
        try data.write(to: photo.url, options: .atomic)
    }

    func delete(_ photo: SavedPhoto) throws {
        try? FileManager.default.removeItem(at: photo.url)
        if let originalURL = photo.originalURL { try? FileManager.default.removeItem(at: originalURL) }
        for url in photo.sourceURLs { try? FileManager.default.removeItem(at: url) }
        index.removeAll { $0.id == photo.id }
        try persist()
    }

    private func persist() throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(index).write(to: indexURL, options: .atomic)
    }

    private static let timestamp: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        return formatter
    }()
}
