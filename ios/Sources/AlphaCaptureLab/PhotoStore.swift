import Foundation
import AINRRuntime
import Photos

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
        index = Self.loadIndex(from: indexURL)
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
        denoiseModel: AINRModel? = nil,
        denoiseStrength: Double? = nil,
        iso: Int? = nil,
        geometry: EditorGeometry? = nil,
        derivedFromID: String? = nil,
        originalFilename: String? = nil,
        filenameSuffix: String? = nil,
        extension fileExtension: String = "jpg"
    ) throws -> SavedPhoto {
        let requestedStem = originalFilename
            .map { URL(fileURLWithPath: $0).deletingPathExtension().lastPathComponent }
            .map(Self.sanitizedStem)
        let requested = requestedStem.map { "\($0)\(filenameSuffix ?? "")" }
            ?? "ACL_\(Self.timestamp.string(from: Date()))_\(UUID().uuidString.prefix(6))"
        var stem = requested
        var displayURL = directory.appendingPathComponent("\(stem).\(fileExtension)")
        if FileManager.default.fileExists(atPath: displayURL.path) {
            stem += "_\(UUID().uuidString.prefix(6))"
            displayURL = directory.appendingPathComponent("\(stem).\(fileExtension)")
        }
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
            lutIdentifier: lut?.identifier, lutStrength: lut?.strength,
            denoiseModel: denoiseModel, denoiseStrength: denoiseStrength, iso: iso,
            geometry: geometry, derivedFromID: derivedFromID,
            originalFilename: originalFilename
        )
        index.insert(photo, at: 0)
        try persist()
        return photo
    }

    func replace(
        _ photo: SavedPhoto,
        data: Data,
        denoiseModel: AINRModel?,
        denoiseStrength: Double?
    ) throws {
        try data.write(to: photo.url, options: .atomic)
        guard let index = index.firstIndex(where: { $0.id == photo.id }) else { return }
        self.index[index].denoiseModel = denoiseModel
        self.index[index].denoiseStrength = denoiseStrength
        try persist()
    }

    func updatePhotoLibraryIdentifier(_ identifier: String, for photoID: String) throws {
        guard let item = index.firstIndex(where: { $0.id == photoID }) else { return }
        index[item].photoLibraryIdentifier = identifier
        try persist()
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

    private static func loadIndex(from url: URL) -> [SavedPhoto] {
        guard let data = try? Data(contentsOf: url) else { return [] }
        if let photos = try? JSONDecoder().decode([SavedPhoto].self, from: data) {
            return photos
        }
        guard let objects = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            return []
        }
        return objects.compactMap { object in
            guard let item = try? JSONSerialization.data(withJSONObject: object) else { return nil }
            return try? JSONDecoder().decode(SavedPhoto.self, from: item)
        }
    }

    private static let timestamp: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        return formatter
    }()

    private static func sanitizedStem(_ value: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_"))
        let cleaned = value.unicodeScalars.map { allowed.contains($0) ? Character($0) : "_" }
        let result = String(cleaned).trimmingCharacters(in: CharacterSet(charactersIn: "_"))
        return result.isEmpty ? "Capture" : result
    }
}

actor PhotoLibraryPublisher {
    static let shared = PhotoLibraryPublisher()

    func publish(_ photo: SavedPhoto) async throws -> String {
        let status = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
        guard status == .authorized || status == .limited else {
            throw PhotoLibraryError.permissionDenied
        }
        var placeholderIdentifier: String?
        try await PHPhotoLibrary.shared().performChanges {
            let request = PHAssetCreationRequest.forAsset()
            request.creationDate = photo.capturedAt
            request.addResource(with: .photo, fileURL: photo.url, options: nil)
            placeholderIdentifier = request.placeholderForCreatedAsset?.localIdentifier
        }
        guard let placeholderIdentifier else { throw PhotoLibraryError.publishFailed }
        return placeholderIdentifier
    }
}

enum PhotoLibraryError: LocalizedError {
    case permissionDenied
    case publishFailed

    var errorDescription: String? {
        switch self {
        case .permissionDenied: "Allow Alpha Capture Lab to add photos in Settings."
        case .publishFailed: "Apple Photos did not accept the saved image."
        }
    }
}
