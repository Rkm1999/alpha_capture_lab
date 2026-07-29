import Foundation
import AINRRuntime

struct ComputationalSessionState: Codable {
    let id: UUID
    let mode: CaptureMode
    let targetCount: Int
    let startedAt: Date
    let quality: DownloadQuality
    let outputFormat: OutputFormat
    let geotagging: Bool
    let lut: LUTSelection
    let autoDenoise: AutoDenoiseMode
    let denoiseThreshold: Int
    let denoiseModel: AINRModel
    let iso: Int?
    var remoteURLs: [String]
    var frameNames: [String]
}

struct RestoredComputationalSession {
    let state: ComputationalSessionState
    let frames: [Data]
}

actor ComputationalSessionStore {
    private let directory: URL
    private let framesDirectory: URL
    private let manifestURL: URL
    private let fileManager: FileManager

    init(root: URL? = nil, fileManager: FileManager = .default) {
        self.fileManager = fileManager
        let base = root ?? fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("AlphaCaptureLab", isDirectory: true)
        directory = base.appendingPathComponent("PendingComputationalCapture", isDirectory: true)
        framesDirectory = directory.appendingPathComponent("Frames", isDirectory: true)
        manifestURL = directory.appendingPathComponent("session.json")
    }

    func begin(_ state: ComputationalSessionState) throws {
        try clearFiles()
        try fileManager.createDirectory(at: framesDirectory, withIntermediateDirectories: true)
        try persist(state)
    }

    func append(_ data: Data, remoteURL: URL) throws -> RestoredComputationalSession {
        guard var state = try loadState() else {
            throw ComputationalSessionError.missingSession
        }
        if !state.remoteURLs.contains(remoteURL.absoluteString) {
            let name = String(format: "frame_%04d.jpg", state.frameNames.count + 1)
            try data.write(to: framesDirectory.appendingPathComponent(name), options: .atomic)
            state.remoteURLs.append(remoteURL.absoluteString)
            state.frameNames.append(name)
            try persist(state)
        }
        return try restored(state)
    }

    func load() throws -> RestoredComputationalSession? {
        guard let state = try loadState() else { return nil }
        return try restored(state)
    }

    func clear() throws {
        try clearFiles()
    }

    private func loadState() throws -> ComputationalSessionState? {
        guard fileManager.fileExists(atPath: manifestURL.path) else { return nil }
        return try JSONDecoder().decode(
            ComputationalSessionState.self,
            from: Data(contentsOf: manifestURL)
        )
    }

    private func restored(_ state: ComputationalSessionState) throws -> RestoredComputationalSession {
        let frames = try state.frameNames.map {
            try Data(contentsOf: framesDirectory.appendingPathComponent($0))
        }
        return RestoredComputationalSession(state: state, frames: frames)
    }

    private func persist(_ state: ComputationalSessionState) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(state).write(to: manifestURL, options: .atomic)
    }

    private func clearFiles() throws {
        if fileManager.fileExists(atPath: directory.path) {
            try fileManager.removeItem(at: directory)
        }
    }
}

enum ComputationalSessionError: LocalizedError {
    case missingSession

    var errorDescription: String? {
        "The pending computational capture could not be restored."
    }
}
