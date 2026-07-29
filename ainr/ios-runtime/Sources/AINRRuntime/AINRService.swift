import Foundation

public enum AINRModel: String, CaseIterable, Identifiable, Codable, Hashable, Sendable {
    case distilled = "Distilled"
    case scunet = "SCUNet"

    public var id: Self { self }
}

public struct AINRProcessingProgress: Sendable {
    public enum Phase: String, Sendable {
        case loading
        case preparing
        case processing
        case encoding
    }

    public let phase: Phase
    public let completed: Int
    public let total: Int
    public let detail: String

    public var fraction: Double? {
        guard total > 0 else { return nil }
        return Double(completed) / Double(total)
    }
}

public actor AINRService {
    public static let shared = AINRService()

    private let processor = SCUNetProcessor.shared
    private var selectedBackends: [AINRModel: DenoiseBackend] = [:]

    public func prepare(model: AINRModel) async -> String {
        await backend(for: model).rawValue
    }

    public func process(
        data: Data,
        sourceName: String,
        model: AINRModel,
        progress: @escaping @Sendable (AINRProcessingProgress) -> Void
    ) async throws -> Data {
        let sourceURL = try Self.writeTemporaryInput(data: data, sourceName: sourceName)
        defer { try? FileManager.default.removeItem(at: sourceURL) }
        let quality = quality(for: model)
        let backend = await backend(for: model)
        let backendLabel = backend == .neuralEngine ? "ANE" : backend.rawValue
        let result = try await processor.process(
            sourceURL: sourceURL,
            sourceName: sourceName,
            backend: backend,
            quality: quality,
            highOverlap: false
        ) { update in
            progress(.init(
                phase: Self.phase(update.phase),
                completed: update.completedTiles,
                total: update.totalTiles,
                detail: "\(backendLabel) • \(update.phase.rawValue)"
            ))
        }
        defer { try? FileManager.default.removeItem(at: result.url) }
        return try Data(contentsOf: result.url)
    }

    private func quality(for model: AINRModel) -> DenoiseQuality {
        model == .distilled ? .highPerformance : .highQuality
    }

    private func backend(for model: AINRModel) async -> DenoiseBackend {
        if let selected = selectedBackends[model] {
            return selected
        }
        let selected: DenoiseBackend = await processor.neuralEngineSupport(
            for: quality(for: model)
        ) ? .neuralEngine : .gpu
        selectedBackends[model] = selected
        return selected
    }

    private static func phase(_ phase: DenoiseProgress.Phase) -> AINRProcessingProgress.Phase {
        switch phase {
        case .loading: .loading
        case .preparing: .preparing
        case .processing: .processing
        case .encoding: .encoding
        }
    }

    private static func writeTemporaryInput(data: Data, sourceName: String) throws -> URL {
        let fileExtension = (sourceName as NSString).pathExtension.isEmpty
            ? "jpg"
            : (sourceName as NSString).pathExtension
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("AlphaCaptureLab-AINR", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent("\(UUID().uuidString).\(fileExtension)")
        try data.write(to: url, options: .atomic)
        return url
    }
}
