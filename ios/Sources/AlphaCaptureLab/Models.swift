import Foundation

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

struct CameraEventSnapshot {
    var urls: [URL] = []
    var status: String?
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

enum DownloadQuality: String, CaseIterable, Identifiable {
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
    var iso: Int?
}

struct CameraRPCError: LocalizedError {
    let code: Int
    let message: String

    var errorDescription: String? { "Camera error \(code): \(message)" }
}
