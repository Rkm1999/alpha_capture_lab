import Foundation
import AINRRuntime

enum ConnectionPhase: Equatable {
    case disconnected
    case connecting
    case connected
    case failed(String)

    var title: String {
        switch self {
        case .disconnected: "Not connected"
        case .connecting: "Connecting..."
        case .connected: "Connected"
        case .failed(let message): message
        }
    }
}

enum CaptureMode: String, CaseIterable, Identifiable, Codable {
    case photo = "Photo"
    case liveND = "Live ND"
    case composite = "Composite"
    case panorama = "Panorama"

    var id: String { rawValue }
    var isAvailable: Bool { true }
}

enum CameraSettingID: String, CaseIterable, Codable, Identifiable {
    case drive, burstSpeed, exposureMode, aperture, shutterSpeed, iso, exposureCompensation
    var id: String { rawValue }
    var label: String {
        switch self {
        case .drive: "Drive"
        case .burstSpeed: "Burst speed"
        case .exposureMode: "Exposure mode"
        case .aperture: "Aperture"
        case .shutterSpeed: "Shutter"
        case .iso: "ISO"
        case .exposureCompensation: "Exposure compensation"
        }
    }
}

func normalizedExposureMode(_ value: String?) -> String {
    switch value?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
    case "program auto", "program", "p": "P"
    case "aperture priority", "aperture", "a": "A"
    case "shutter priority", "shutter", "s": "S"
    case "manual exposure", "manual", "m": "M"
    case .some(let value): value.uppercased()
    case nil: "--"
    }
}

func exposurePrioritySettingID(for mode: String?) -> CameraSettingID? {
    switch normalizedExposureMode(mode) {
    case "A", "M": .aperture
    case "S": .shutterSpeed
    default: nil
    }
}

func photoSettingGridOrder(
    settings: [CameraSettingID: CameraSetting],
    exposureMode: String?
) -> [CameraSettingID] {
    let priority = exposurePrioritySettingID(for: exposureMode)
    let continuousDrive = settings[.drive]?.current
        .localizedCaseInsensitiveContains("continuous") == true
    let order: [CameraSettingID] = [
        .aperture,
        .shutterSpeed,
        .iso,
        .exposureCompensation,
        .drive,
        .burstSpeed,
    ]
    return order.filter {
        $0 != priority &&
            settings[$0] != nil &&
            ($0 != .burstSpeed || continuousDrive)
    }
}

func highestBurstSpeed(in options: [String]) -> String? {
    guard var best = options.first else { return nil }
    var bestRank = burstSpeedRank(best)
    for option in options.dropFirst() {
        let rank = burstSpeedRank(option)
        if rank > bestRank {
            best = option
            bestRank = rank
        }
    }
    return best
}

func shutterDurationSeconds(_ rawValue: String) -> Double? {
    let value = rawValue
        .trimmingCharacters(in: .whitespacesAndNewlines)
        .replacingOccurrences(of: "\"", with: "")
        .replacingOccurrences(of: "s", with: "", options: .caseInsensitive)
    guard value.localizedCaseInsensitiveCompare("BULB") != .orderedSame else { return nil }
    let parts = value.split(separator: "/")
    if parts.count == 2,
       let numerator = Double(parts[0]),
       let denominator = Double(parts[1]),
       denominator != 0 {
        return numerator / denominator
    }
    return Double(value)
}

func liveNDBurstDuration(shutterValue: String, requiredFrames: Int) -> Double {
    let exposure = shutterDurationSeconds(shutterValue) ?? 1 / 60
    return max(0, exposure) * Double(max(1, requiredFrames) + 1)
}

private func burstSpeedRank(_ value: String) -> Int {
    let label = value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    if label == "hi" || label.contains("high") || label.contains("fast") { return 3 }
    if label == "mid" || label.contains("medium") || label.contains("normal") { return 2 }
    if label == "lo" || label.contains("low") || label.contains("slow") { return 1 }
    return 0
}

struct CameraSetting: Identifiable, Codable, Equatable {
    let id: CameraSettingID
    var current: String
    var options: [String]
    var writable: Bool = true
}

enum LiveViewQuality: String, CaseIterable, Identifiable, Codable {
    case standard = "M"
    case high = "L"
    var id: String { rawValue }
    var label: String { self == .high ? "High" : "Standard" }
}

enum OutputFormat: String, CaseIterable, Identifiable, Codable {
    case jpeg = "JPEG"
    case webp = "WebP"
    var id: String { rawValue }
}

enum AutoDenoiseMode: String, CaseIterable, Identifiable, Codable {
    case off = "Off"
    case always = "Always"
    case isoThreshold = "ISO threshold"
    var id: String { rawValue }
}

struct LUTSelection: Codable, Equatable {
    var identifier = "Original"
    var strength: Double = 1
}

struct NormalizedPoint: Codable, Equatable, Hashable {
    var x: Double
    var y: Double
}

struct NormalizedCrop: Codable, Equatable, Hashable {
    var left = 0.0
    var top = 0.0
    var right = 1.0
    var bottom = 1.0

    var isIdentity: Bool {
        abs(left) < 0.0001 && abs(top) < 0.0001 &&
            abs(right - 1) < 0.0001 && abs(bottom - 1) < 0.0001
    }

    func normalized(minimumSize: Double = 0.05) -> NormalizedCrop {
        var value = self
        value.left = min(max(value.left, 0), 1 - minimumSize)
        value.top = min(max(value.top, 0), 1 - minimumSize)
        value.right = max(min(value.right, 1), value.left + minimumSize)
        value.bottom = max(min(value.bottom, 1), value.top + minimumSize)
        return value
    }
}

struct EditorGeometry: Codable, Equatable, Hashable {
    var crop = NormalizedCrop()
    var quarterTurns = 0
    var straightenDegrees = 0.0
    var perspective: [NormalizedPoint] = [
        .init(x: 0, y: 0),
        .init(x: 1, y: 0),
        .init(x: 1, y: 1),
        .init(x: 0, y: 1),
    ]

    var hasChanges: Bool {
        !crop.isIdentity || normalizedQuarterTurns != 0 ||
            abs(straightenDegrees) > 0.001 || perspective != Self.identityPerspective
    }

    var normalizedQuarterTurns: Int {
        ((quarterTurns % 4) + 4) % 4
    }

    static let identityPerspective: [NormalizedPoint] = [
        .init(x: 0, y: 0),
        .init(x: 1, y: 0),
        .init(x: 1, y: 1),
        .init(x: 0, y: 1),
    ]
}

struct CameraEventSnapshot {
    var urls: [URL] = []
    var status: String?
    var availableAPIs: Set<String>?
    var settingValues: [CameraSettingID: String] = [:]
    var zoomPosition: Int?
    var zoomSetting: String?
    var zoomBoxCount: Int?
    var zoomBoxIndex: Int?
}

struct PairedCamera: Identifiable, Codable, Equatable {
    var id: String { host }
    let host: String
    var name: String
    var autoConnect: Bool
    var lastConnected: Date
}

enum DownloadQuality: String, CaseIterable, Identifiable, Codable {
    case original = "Original"
    case reduced = "Reduced"

    var id: String { rawValue }
    var sonyValue: String { self == .original ? "Original" : "2M" }
}

struct SavedPhoto: Identifiable, Hashable, Codable {
    let id: String
    let url: URL
    let capturedAt: Date
    var kind: CaptureMode = .photo
    var originalURL: URL?
    var sourceURLs: [URL] = []
    var lutIdentifier: String?
    var lutStrength: Double?
    var denoiseModel: AINRModel?
    var denoiseStrength: Double?
    var iso: Int?
    var geometry: EditorGeometry?
    var derivedFromID: String?
    var photoLibraryIdentifier: String?
    var originalFilename: String?

    private enum CodingKeys: String, CodingKey {
        case id, url, capturedAt, kind, originalURL, sourceURLs
        case lutIdentifier, lutStrength, denoiseModel, denoiseStrength, iso
        case geometry, derivedFromID, photoLibraryIdentifier, originalFilename
    }

    init(
        id: String,
        url: URL,
        capturedAt: Date,
        kind: CaptureMode = .photo,
        originalURL: URL? = nil,
        sourceURLs: [URL] = [],
        lutIdentifier: String? = nil,
        lutStrength: Double? = nil,
        denoiseModel: AINRModel? = nil,
        denoiseStrength: Double? = nil,
        iso: Int? = nil,
        geometry: EditorGeometry? = nil,
        derivedFromID: String? = nil,
        photoLibraryIdentifier: String? = nil,
        originalFilename: String? = nil
    ) {
        self.id = id
        self.url = url
        self.capturedAt = capturedAt
        self.kind = kind
        self.originalURL = originalURL
        self.sourceURLs = sourceURLs
        self.lutIdentifier = lutIdentifier
        self.lutStrength = lutStrength
        self.denoiseModel = denoiseModel
        self.denoiseStrength = denoiseStrength
        self.iso = iso
        self.geometry = geometry
        self.derivedFromID = derivedFromID
        self.photoLibraryIdentifier = photoLibraryIdentifier
        self.originalFilename = originalFilename
    }

    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        id = try values.decode(String.self, forKey: .id)
        url = try values.decode(URL.self, forKey: .url)
        capturedAt = try values.decode(Date.self, forKey: .capturedAt)
        kind = (try? values.decode(CaptureMode.self, forKey: .kind)) ?? .photo
        originalURL = try? values.decodeIfPresent(URL.self, forKey: .originalURL)
        sourceURLs = (try? values.decodeIfPresent([URL].self, forKey: .sourceURLs)) ?? []
        lutIdentifier = try? values.decodeIfPresent(String.self, forKey: .lutIdentifier)
        lutStrength = try? values.decodeIfPresent(Double.self, forKey: .lutStrength)
        denoiseModel = try? values.decodeIfPresent(AINRModel.self, forKey: .denoiseModel)
        denoiseStrength = try? values.decodeIfPresent(Double.self, forKey: .denoiseStrength)
        iso = try? values.decodeIfPresent(Int.self, forKey: .iso)
        geometry = try? values.decodeIfPresent(EditorGeometry.self, forKey: .geometry)
        derivedFromID = try? values.decodeIfPresent(String.self, forKey: .derivedFromID)
        photoLibraryIdentifier = try? values.decodeIfPresent(String.self, forKey: .photoLibraryIdentifier)
        originalFilename = try? values.decodeIfPresent(String.self, forKey: .originalFilename)
    }
}

struct CameraRPCError: LocalizedError {
    let code: Int
    let message: String

    var errorDescription: String? { "Camera error \(code): \(message)" }
}
