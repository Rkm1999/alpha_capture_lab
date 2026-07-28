import Foundation
import OSLog
import Photos
import UIKit

enum PreviewMode: String, CaseIterable, Identifiable {
    case original = "Original"
    case result = "Result"

    var id: Self { self }
}

private struct ImportedSelection: @unchecked Sendable {
    let url: URL
    let name: String
    let preview: UIImage
    let width: Int
    let height: Int
}

@MainActor
final class DenoiseViewModel: ObservableObject {
    private static let logger = Logger(
        subsystem: "com.ryu.scunetdenoiser.ios",
        category: "Interface"
    )
    @Published var sourcePreview: UIImage?
    @Published var resultPreview: UIImage?
    @Published var sourceName = ""
    @Published var sourceWidth = 0
    @Published var sourceHeight = 0
    @Published var backend: DenoiseBackend {
        didSet { UserDefaults.standard.set(backend.rawValue, forKey: Self.backendKey) }
    }
    @Published var quality: DenoiseQuality {
        didSet { UserDefaults.standard.set(quality.rawValue, forKey: Self.qualityKey) }
    }
    @Published var highOverlap: Bool {
        didSet { UserDefaults.standard.set(highOverlap, forKey: Self.highOverlapKey) }
    }
    @Published var previewMode: PreviewMode = .original
    @Published var progress: DenoiseProgress?
    @Published var status = "Choose a photo"
    @Published var isLoadingImage = false
    @Published var isProcessing = false
    @Published var isSaving = false
    @Published var result: DenoiseResult?
    @Published var results: [DenoiseResult] = []
    @Published var batchIndex = 0
    @Published var batchCount = 0
    @Published var alertMessage: String?
    @Published var processingStartedAt: Date?
    @Published private(set) var neuralEngineSupport: NeuralEngineSupport = .checking

    private static let backendKey = "SCUNetBackend"
    private static let highOverlapKey = "SCUNetHighOverlap"
    private static let qualityKey = "SCUNetQuality"
    private let processor = SCUNetProcessor.shared
    private var sources: [ImportedSelection] = []
    private var importTask: Task<Void, Never>?
    private var processingTask: Task<Void, Never>?
    private var supportTask: Task<Void, Never>?
    private var runID: UUID?
    private var runAfterImport = false

    init() {
        highOverlap = UserDefaults.standard.bool(forKey: Self.highOverlapKey)
        let saved = UserDefaults.standard.string(forKey: Self.backendKey)
        if let requested = CommandLine.arguments.first(where: {
            $0.hasPrefix("--scunet-quality=")
        })?.split(separator: "=", maxSplits: 1).last,
           let requestedQuality = Self.qualityArgument(String(requested)) {
            quality = requestedQuality
        } else {
            quality = DenoiseQuality(
                rawValue: UserDefaults.standard.string(forKey: Self.qualityKey) ?? ""
            ) ?? .highQuality
        }
        if let requested = CommandLine.arguments.first(where: {
            $0.hasPrefix("--scunet-backend=")
        })?.split(separator: "=", maxSplits: 1).last,
           let requestedBackend = DenoiseBackend(rawValue: String(requested)) {
            backend = requestedBackend
        } else {
            backend = DenoiseBackend(rawValue: saved ?? "") ?? .neuralEngine
        }
        if CommandLine.arguments.contains("--scunet-autotest") {
            Task { [weak self] in
                self?.loadTestImage(runWhenReady: true)
            }
        }
        refreshNeuralEngineSupport()
    }

    private static func qualityArgument(_ value: String) -> DenoiseQuality? {
        switch value.lowercased() {
        case "performance", "high-performance":
            return .highPerformance
        case "quality", "high-quality":
            return .highQuality
        default:
            return nil
        }
    }

    var displayedImage: UIImage? {
        if previewMode == .result, let resultPreview { return resultPreview }
        return sourcePreview
    }

    var canRun: Bool {
        !sources.isEmpty
            && !isLoadingImage
            && !isProcessing
            && isBackendAvailable(backend)
    }

    var neuralEngineStatus: String? {
        switch neuralEngineSupport {
        case .checking:
            "Checking Neural Engine compatibility"
        case .available:
            nil
        case .unavailable:
            "Neural Engine is unavailable for this model; using GPU"
        }
    }

    func isBackendAvailable(_ candidate: DenoiseBackend) -> Bool {
        switch candidate {
        case .neuralEngine, .gpuAndNeuralEngine:
            neuralEngineSupport == .available
        case .gpu, .cpu:
            true
        }
    }

    func qualityDidChange() {
        refreshNeuralEngineSupport()
    }

    func validateBackendSelection() {
        if !isBackendAvailable(backend) {
            backend = .gpu
        }
    }

    var overallProgress: Double? {
        guard batchCount > 0, batchIndex > 0 else { return progress?.fraction }
        return (Double(batchIndex - 1) + (progress?.fraction ?? 0)) / Double(batchCount)
    }

    var imageDetails: String {
        guard sourceWidth > 0, sourceHeight > 0 else { return "" }
        let megapixels = Double(sourceWidth * sourceHeight) / 1_000_000
        return "\(sourceWidth) x \(sourceHeight)  |  \(String(format: "%.1f MP", megapixels))"
    }

    func importURL(_ url: URL) {
        importURLs([url])
    }

    func importURLs(_ urls: [URL]) {
        guard !urls.isEmpty else { return }
        beginImport {
            try urls.map { url in
            let imported = try ImageCodec.importSource(url)
            return try Self.selection(url: imported.0, name: imported.1)
            }
        }
    }

    func importPhotoData(_ data: Data, preferredExtension: String) {
        importPhotoData([(data, preferredExtension, nil)])
    }

    func importPhotoData(_ items: [(Data, String, String?)]) {
        guard !items.isEmpty else { return }
        beginImport {
            try items.map { data, fileExtension, originalName in
                let imported = try ImageCodec.importSource(
                    data: data,
                    preferredExtension: fileExtension,
                    originalName: originalName
                )
                return try Self.selection(url: imported.0, name: imported.1)
            }
        }
    }

    func loadTestImage() {
        loadTestImage(runWhenReady: false)
    }

    private func loadTestImage(runWhenReady: Bool) {
        guard let url = Bundle.module.url(
            forResource: "SCUNet-Test-ISO51200",
            withExtension: "JPG",
            subdirectory: "Samples"
        ) else {
            alertMessage = "The bundled test image is missing"
            return
        }
        runAfterImport = runWhenReady
        beginImport {
            [try Self.selection(url: url, name: "Checkerboard test - ISO 25600.JPG")]
        }
    }

    func runDenoise() {
        guard canRun else { return }
        let selectedSources = sources
        processingTask?.cancel()
        let currentRun = UUID()
        runID = currentRun
        isProcessing = true
        processingStartedAt = Date()
        result = nil
        results = []
        batchIndex = 1
        batchCount = selectedSources.count
        resultPreview = nil
        previewMode = .original
        status = "Preparing \(backend.rawValue). First run can take longer."
        let selectedBackend = backend
        let selectedQuality = quality
        let selectedHighOverlap = highOverlap

        processingTask = Task { [weak self] in
            guard let self else { return }
            do {
                var completedBatch: [DenoiseResult] = []
                for (index, source) in selectedSources.enumerated() {
                    try Task.checkCancellation()
                    batchIndex = index + 1
                    status = Self.batchStatus(index + 1, selectedSources.count, "Preparing")
                    let completed = try await processor.process(
                        sourceURL: source.url,
                        sourceName: source.name,
                        backend: selectedBackend,
                        quality: selectedQuality,
                        highOverlap: selectedHighOverlap
                    ) { [weak self] update in
                        Task { @MainActor in
                            guard let self, self.runID == currentRun else { return }
                            self.apply(update)
                        }
                    }
                    completedBatch.append(completed)
                }
                try Task.checkCancellation()
                guard let completed = completedBatch.last else { return }
                let preview = try ImageCodec.previewImage(completed.url)
                guard runID == currentRun else { return }
                result = completed
                results = completedBatch
                resultPreview = preview
                previewMode = .result
                progress = nil
                isProcessing = false
                processingStartedAt = nil
                status = completedBatch.count == 1
                    ? "Completed in \(Self.duration(completed.elapsedSeconds)) · \(completed.quality.rawValue)"
                    : "Completed \(completedBatch.count) images · \(completed.quality.rawValue)"
            } catch is CancellationError {
                guard runID == currentRun else { return }
                progress = nil
                isProcessing = false
                processingStartedAt = nil
                batchIndex = 0
                batchCount = 0
                status = "Canceled"
            } catch {
                guard runID == currentRun else { return }
                Self.logger.error("Denoising failed: \(error.localizedDescription, privacy: .public)")
                progress = nil
                isProcessing = false
                processingStartedAt = nil
                batchIndex = 0
                batchCount = 0
                status = "Denoising failed"
                alertMessage = error.localizedDescription
            }
        }
    }

    func cancelProcessing() {
        processingTask?.cancel()
        status = "Canceling after the current tile"
    }

    func saveResultToPhotos() {
        let completed = results
        guard !completed.isEmpty, !isSaving else { return }
        isSaving = true
        status = "Saving to Photos"
        Task { [weak self] in
            guard let self else { return }
            do {
                let authorization = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
                guard authorization == .authorized || authorization == .limited else {
                    throw SCUNetError(message: "Photo library access was not granted")
                }
                try await PHPhotoLibrary.shared().performChanges {
                    for result in completed {
                        let request = PHAssetCreationRequest.forAsset()
                        let options = PHAssetResourceCreationOptions()
                        options.originalFilename = result.url.lastPathComponent
                        request.addResource(
                            with: .photo,
                            fileURL: result.url,
                            options: options
                        )
                    }
                }
                isSaving = false
                status = completed.count == 1
                    ? "Saved to Photos"
                    : "Saved \(completed.count) images to Photos"
            } catch {
                isSaving = false
                status = "Save failed"
                alertMessage = error.localizedDescription
            }
        }
    }

    func report(_ error: Error) {
        alertMessage = error.localizedDescription
    }

    private func refreshNeuralEngineSupport() {
        supportTask?.cancel()
        neuralEngineSupport = .checking
        let selectedQuality = quality
        supportTask = Task { [weak self] in
            guard let self else { return }
            let supported = await processor.neuralEngineSupport(for: selectedQuality)
            guard !Task.isCancelled, quality == selectedQuality else { return }
            neuralEngineSupport = supported ? .available : .unavailable
            validateBackendSelection()
        }
    }

    private func beginImport(
        operation: @escaping @Sendable () throws -> [ImportedSelection]
    ) {
        importTask?.cancel()
        processingTask?.cancel()
        runID = nil
        isLoadingImage = true
        isProcessing = false
        processingStartedAt = nil
        batchIndex = 0
        batchCount = 0
        progress = nil
        status = "Loading photo"
        importTask = Task { [weak self] in
            guard let self else { return }
            do {
                let imported = try await Task.detached(
                    priority: .userInitiated,
                    operation: operation
                ).value
                try Task.checkCancellation()
                guard let first = imported.first else {
                    throw SCUNetError(message: "No images were selected")
                }
                sources = imported
                sourceName = imported.count == 1
                    ? first.name
                    : "\(first.name) + \(imported.count - 1) more"
                sourceWidth = first.width
                sourceHeight = first.height
                sourcePreview = first.preview
                result = nil
                results = []
                resultPreview = nil
                previewMode = .original
                isLoadingImage = false
                status = "Ready"
                if runAfterImport {
                    runAfterImport = false
                    runDenoise()
                }
            } catch is CancellationError {
                runAfterImport = false
                isLoadingImage = false
            } catch {
                runAfterImport = false
                isLoadingImage = false
                status = "Could not load photo"
                alertMessage = error.localizedDescription
            }
        }
    }

    private func apply(_ update: DenoiseProgress) {
        progress = update
        switch update.phase {
        case .loading:
            status = Self.batchStatus(batchIndex, batchCount, update.phase.rawValue)
        case .preparing:
            status = Self.batchStatus(batchIndex, batchCount, "Preparing \(backend.rawValue)")
        case .processing:
            status = Self.batchStatus(
                batchIndex, batchCount,
                "Tile \(update.completedTiles) of \(update.totalTiles)"
            )
        case .encoding:
            status = Self.batchStatus(batchIndex, batchCount, update.phase.rawValue)
        }
    }

    private nonisolated static func selection(url: URL, name: String) throws -> ImportedSelection {
        let preview = try ImageCodec.previewImage(url)
        guard let dimensions = ImageCodec.imageDimensions(url) else {
            throw SCUNetError(message: "The image dimensions could not be read")
        }
        return ImportedSelection(
            url: url,
            name: name,
            preview: preview,
            width: dimensions.0,
            height: dimensions.1
        )
    }

    private static func duration(_ seconds: TimeInterval) -> String {
        if seconds < 60 { return String(format: "%.1f sec", seconds) }
        let minutes = Int(seconds) / 60
        return "\(minutes)m \(Int(seconds) % 60)s"
    }

    private static func batchStatus(_ current: Int, _ total: Int, _ detail: String) -> String {
        total <= 1 ? detail : "Image \(current) of \(total) · \(detail)"
    }
}
