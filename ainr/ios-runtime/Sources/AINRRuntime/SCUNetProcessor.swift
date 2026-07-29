import CoreML
import Foundation
import ImageIO
import OSLog
import UIKit
import UniformTypeIdentifiers

public enum DenoiseBackend: String, CaseIterable, Identifiable, Sendable {
    case gpu = "GPU"
    case neuralEngine = "Neural Engine"
    case gpuAndNeuralEngine = "GPU + Neural Engine"
    case cpu = "CPU"

    public var id: Self { self }

    var computeUnits: MLComputeUnits {
        switch self {
        case .neuralEngine: .cpuAndNeuralEngine
        case .gpu: .cpuAndGPU
        case .gpuAndNeuralEngine: .all
        case .cpu: .cpuOnly
        }
    }

    public var pickerLabel: String {
        switch self {
        case .gpu: "GPU"
        case .neuralEngine: "ANE"
        case .gpuAndNeuralEngine: "GPU + ANE"
        case .cpu: "CPU"
        }
    }
}

public enum DenoiseQuality: String, CaseIterable, Identifiable, Sendable {
    case highPerformance = "High Performance"
    case highQuality = "High Quality"

    public var id: Self { self }

    public var pickerLabel: String {
        switch self {
        case .highPerformance: "Performance · LiteDenoise"
        case .highQuality: "Quality · 16-bit"
        }
    }

    var modelSuffix: String {
        switch self {
        case .highPerformance: "int8"
        case .highQuality: "fp16"
        }
    }
}

public enum NeuralEngineSupport: Equatable, Sendable {
    case checking
    case available
    case unavailable
}

public struct DenoiseProgress: Sendable {
    public enum Phase: String, Sendable {
        case loading = "Loading full-resolution image"
        case preparing = "Preparing model"
        case processing = "Denoising"
        case encoding = "Encoding JPEG"
    }

    public let phase: Phase
    public let completedTiles: Int
    public let totalTiles: Int
    public let elapsedSeconds: TimeInterval

    public var fraction: Double? {
        guard phase == .processing, totalTiles > 0 else { return nil }
        return Double(completedTiles) / Double(totalTiles)
    }
}

public struct DenoiseResult: Sendable {
    public let url: URL
    public let width: Int
    public let height: Int
    public let elapsedSeconds: TimeInterval
    public let backendLabel: String
    public let quality: DenoiseQuality
}

public struct SCUNetError: LocalizedError, Sendable {
    public let message: String
    public init(message: String) { self.message = message }
    public var errorDescription: String? { message }
}

private final class ModelSession: @unchecked Sendable {
    private static let logger = Logger(
        subsystem: "com.ryu.scunetdenoiser.ios",
        category: "ComputePlan"
    )
    let backend: DenoiseBackend
    let inputChannels: Int
    private let model: MLModel
    private let inputArray: MLMultiArray
    let inputTensorCount: Int
    private let outputTensorCount: Int

    private init(
        model: MLModel,
        backend: DenoiseBackend,
        tileSize: Int,
        inputChannels: Int
    ) throws {
        self.model = model
        self.backend = backend
        self.inputChannels = inputChannels
        inputTensorCount = inputChannels * tileSize * tileSize
        outputTensorCount = 3 * tileSize * tileSize
        inputArray = try MLMultiArray(
            shape: [
                1,
                NSNumber(value: inputChannels),
                NSNumber(value: tileSize),
                NSNumber(value: tileSize),
            ],
            dataType: .float32
        )
    }

    static func load(
        modelURL: URL,
        modelKey: String,
        tileSize: Int,
        inputChannels: Int,
        backend: DenoiseBackend
    ) async throws -> ModelSession {
        let compiledURL = try await compiledModelURL(for: modelURL, modelKey: modelKey)
        let configuration = MLModelConfiguration()
        configuration.computeUnits = backend.computeUnits
        configuration.modelDisplayName = "SCUNet \(tileSize) \(backend.rawValue)"
        configuration.allowLowPrecisionAccumulationOnGPU = false
        if #available(iOS 17.4, *),
           !CommandLine.arguments.contains("--scunet-no-hints") {
            var hints = MLOptimizationHints()
            hints.reshapeFrequency = .infrequent
            if #available(iOS 18.0, *),
               CommandLine.arguments.contains("--scunet-fast-prediction") {
                hints.specializationStrategy = .fastPrediction
            }
            configuration.optimizationHints = hints
        }
        if CommandLine.arguments.contains("--scunet-compute-plan") {
            try await logComputePlan(contentsOf: compiledURL, configuration: configuration)
        }
        let model = try await MLModel.load(
            contentsOf: compiledURL,
            configuration: configuration
        )
        return try ModelSession(
            model: model,
            backend: backend,
            tileSize: tileSize,
            inputChannels: inputChannels
        )
    }

    static func supportsNeuralEngine(
        modelURL: URL,
        modelKey: String
    ) async -> Bool {
        let hasNeuralEngine = MLComputeDevice.allComputeDevices.contains { device in
            if case .neuralEngine = device { return true }
            return false
        }
        guard hasNeuralEngine else { return false }

        do {
            let compiledURL = try await compiledModelURL(
                for: modelURL,
                modelKey: modelKey
            )
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .cpuAndNeuralEngine

            guard #available(iOS 17.4, *) else {
                _ = try await MLModel.load(
                    contentsOf: compiledURL,
                    configuration: configuration
                )
                return true
            }

            let plan = try await MLComputePlan.load(
                contentsOf: compiledURL,
                configuration: configuration
            )
            guard case .program(let program) = plan.modelStructure else {
                return false
            }

            func blockUsesNeuralEngine(
                _ block: MLModelStructure.Program.Block
            ) -> Bool {
                for operation in block.operations where operation.operatorName != "const" {
                    if let usage = plan.deviceUsage(for: operation),
                       case .neuralEngine = usage.preferred {
                        return true
                    }
                    if operation.blocks.contains(where: blockUsesNeuralEngine) {
                        return true
                    }
                }
                return false
            }

            return program.functions.values.contains {
                blockUsesNeuralEngine($0.block)
            }
        } catch {
            Self.logger.error(
                "SCUNET_ANE_CHECK failed: \(error.localizedDescription, privacy: .public)"
            )
            return false
        }
    }

    struct RunResult: Sendable {
        let output: [Float]
        let inputCopySeconds: TimeInterval
        let predictionSeconds: TimeInterval
        let outputCopySeconds: TimeInterval
    }

    func run(_ input: [Float]) async throws -> RunResult {
        guard input.count == inputTensorCount else {
            throw SCUNetError(message: "SCUNet received an invalid input tile")
        }
        let inputStarted = ProcessInfo.processInfo.systemUptime
        _ = input.withUnsafeBytes { bytes in
            memcpy(inputArray.dataPointer, bytes.baseAddress, bytes.count)
        }
        let inputSeconds = ProcessInfo.processInfo.systemUptime - inputStarted
        let provider = try MLDictionaryFeatureProvider(dictionary: ["input": inputArray])
        let predictionStarted = ProcessInfo.processInfo.systemUptime
        let prediction = try await model.prediction(from: provider)
        let predictionSeconds = ProcessInfo.processInfo.systemUptime - predictionStarted
        guard let outputArray = prediction.featureValue(for: "output")?.multiArrayValue,
              outputArray.count == outputTensorCount else {
            throw SCUNetError(message: "Core ML returned an invalid SCUNet output")
        }

        let outputStarted = ProcessInfo.processInfo.systemUptime
        let output: [Float]
        switch outputArray.dataType {
        case .float32:
            let values = outputArray.dataPointer.bindMemory(
                to: Float.self,
                capacity: outputTensorCount
            )
            output = Array(UnsafeBufferPointer(start: values, count: outputTensorCount))
        case .float16:
            let values = outputArray.dataPointer.bindMemory(
                to: Float16.self,
                capacity: outputTensorCount
            )
            output = (0..<outputTensorCount).map { Float(values[$0]) }
        default:
            throw SCUNetError(message: "Core ML returned an unsupported output type")
        }
        let outputSeconds = ProcessInfo.processInfo.systemUptime - outputStarted
        guard let invalid = output.firstIndex(where: { !$0.isFinite }) else {
            return RunResult(
                output: output,
                inputCopySeconds: inputSeconds,
                predictionSeconds: predictionSeconds,
                outputCopySeconds: outputSeconds
            )
        }
        throw SCUNetError(
            message: "Core ML returned a non-finite value at output index \(invalid)"
        )
    }

    private static func compiledModelURL(
        for sourceURL: URL,
        modelKey: String
    ) async throws -> URL {
        let fileManager = FileManager.default
        let supportDirectory = try fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let cacheDirectory = supportDirectory.appendingPathComponent(
            "SCUNetModels",
            isDirectory: true
        )
        try fileManager.createDirectory(
            at: cacheDirectory,
            withIntermediateDirectories: true
        )
        let cachedURL = cacheDirectory.appendingPathComponent(
            "\(modelKey).mlmodelc",
            isDirectory: true
        )
        let readyURL = cacheDirectory.appendingPathComponent(
            "\(modelKey).ready"
        )
        if fileManager.fileExists(atPath: cachedURL.path),
           fileManager.fileExists(atPath: readyURL.path) {
            return cachedURL
        }

        try? fileManager.removeItem(at: cachedURL)
        try? fileManager.removeItem(at: readyURL)
        let temporaryURL = try await MLModel.compileModel(at: sourceURL)
        let stagingURL = cacheDirectory.appendingPathComponent(
            "scunet-\(UUID().uuidString).mlmodelc",
            isDirectory: true
        )
        do {
            try fileManager.copyItem(at: temporaryURL, to: stagingURL)
            try fileManager.moveItem(at: stagingURL, to: cachedURL)
            try Data().write(to: readyURL, options: .atomic)
        } catch {
            try? fileManager.removeItem(at: stagingURL)
            try? fileManager.removeItem(at: cachedURL)
            try? fileManager.removeItem(at: readyURL)
            throw error
        }
        return cachedURL
    }

    private static func logComputePlan(
        contentsOf modelURL: URL,
        configuration: MLModelConfiguration
    ) async throws {
        guard #available(iOS 17.4, *) else { return }
        let plan = try await MLComputePlan.load(
            contentsOf: modelURL,
            configuration: configuration
        )
        guard case .program(let program) = plan.modelStructure else {
            Self.logger.notice("SCUNET_PLAN unsupported model structure")
            return
        }

        var operationCounts: [String: Int] = [:]
        var deviceCounts: [String: Int] = [:]
        var costByDevice: [String: Double] = [:]
        var operatorDevices: [String: [String: Int]] = [:]
        var deviceRuns = 0
        var previousDevice: String?

        func visit(_ block: MLModelStructure.Program.Block) {
            for operation in block.operations {
                let operatorName = operation.operatorName
                operationCounts[operatorName, default: 0] += 1
                if operatorName != "const", let usage = plan.deviceUsage(for: operation) {
                    let device = deviceName(usage.preferred)
                    deviceCounts[device, default: 0] += 1
                    operatorDevices[operatorName, default: [:]][device, default: 0] += 1
                    if previousDevice != device {
                        deviceRuns += 1
                        previousDevice = device
                    }
                    if let cost = plan.estimatedCost(of: operation) {
                        costByDevice[device, default: 0] += cost.weight
                    }
                }
                for nested in operation.blocks { visit(nested) }
            }
        }
        for function in program.functions.values { visit(function.block) }

        let summary = "SCUNET_PLAN operations=\(operationCounts.values.reduce(0, +)) device_runs=\(deviceRuns) devices=\(formatted(deviceCounts)) costs=\(formatted(costByDevice))"
        Self.logger.notice("\(summary, privacy: .public)")
        for operatorName in operatorDevices.keys.sorted() {
            guard let devices = operatorDevices[operatorName] else { continue }
            let detail = "SCUNET_PLAN_OP \(operatorName)=\(formatted(devices))"
            Self.logger.notice("\(detail, privacy: .public)")
        }
    }

    @available(iOS 17.4, *)
    private static func deviceName(_ device: MLComputeDevice) -> String {
        switch device {
        case .cpu: "CPU"
        case .gpu: "GPU"
        case .neuralEngine: "ANE"
        @unknown default: "Unknown"
        }
    }

    private static func formatted<T>(_ values: [String: T]) -> String {
        values.keys.sorted().map { "\($0):\(values[$0]!)" }.joined(separator: ",")
    }
}

private enum ExecutionSessions {
    case single(ModelSession)
    case combined(neuralEngine: ModelSession, gpu: ModelSession)

    var label: String {
        switch self {
        case .single(let session):
            "\(session.backend.rawValue) (Core ML)"
        case .combined:
            "GPU + Neural Engine (Core ML)"
        }
    }
}

private struct InferredTile: Sendable {
    let index: Int
    let row: Int
    let column: Int
    let output: [Float]
}

private struct BandInference: Sendable {
    var tiles: [InferredTile]
    let backend: DenoiseBackend
    let wallSeconds: TimeInterval
    let prepareSeconds: TimeInterval
    let inputCopySeconds: TimeInterval
    let predictionSeconds: TimeInterval
    let outputCopySeconds: TimeInterval
}

private actor TileCompletionCounter {
    private var completed = 0
    private let total: Int
    private let started: Date
    private let progress: @Sendable (DenoiseProgress) -> Void

    init(
        total: Int,
        started: Date,
        progress: @escaping @Sendable (DenoiseProgress) -> Void
    ) {
        self.total = total
        self.started = started
        self.progress = progress
    }

    func finishTile() {
        completed += 1
        if completed == 1 || completed == total || completed.isMultiple(of: 2) {
            progress(.init(
                phase: .processing,
                completedTiles: completed,
                totalTiles: total,
                elapsedSeconds: Date().timeIntervalSince(started)
            ))
        }
    }
}

private actor TileIndexScheduler {
    private var nextFirstIndex = 0
    private var nextLastIndex: Int

    init(tileCount: Int) {
        nextLastIndex = tileCount - 1
    }

    func claimIndex(fromStart: Bool) -> Int? {
        guard nextFirstIndex <= nextLastIndex else { return nil }
        if fromStart {
            defer { nextFirstIndex += 1 }
            return nextFirstIndex
        }
        defer { nextLastIndex -= 1 }
        return nextLastIndex
    }
}

private final class NoiseStrengthWorkspace {
    var luma = [Float](repeating: 0, count: 192 * 192)
    var residual = [Float](repeating: 0, count: 192 * 192)
    var gradient = [Float](repeating: 0, count: 192 * 192)
    var selection = [Float](repeating: 0, count: 192 * 192)
    var opponentRed = [Float](repeating: 0, count: 192 * 192)
    var opponentBlue = [Float](repeating: 0, count: 192 * 192)
    var horizontalRed = [Float](repeating: 0, count: 192 * 192)
    var horizontalBlue = [Float](repeating: 0, count: 192 * 192)
}

public actor SCUNetProcessor {
    public static let shared = SCUNetProcessor()

    private static let logger = Logger(
        subsystem: "com.ryu.scunetdenoiser.ios",
        category: "Denoise"
    )
    private static let tile: Int = {
        guard let value = argumentValue(prefix: "--scunet-tile=").flatMap(Int.init),
              [192, 256, 384, 448].contains(value) else { return 192 }
        return value
    }()
    private static let modelVariant = argumentValue(prefix: "--scunet-model=") ?? "optimized"
    private static let padding = 8
    private static let core = tile - padding * 2
    private static let plane = tile * tile
    private static let benchmarkTileLimit: Int? = argumentValue(
        prefix: "--scunet-benchmark-tiles="
    ).flatMap(Int.init)

    private struct SessionKey: Hashable {
        let backend: DenoiseBackend
        let quality: DenoiseQuality
    }

    private var sessions: [SessionKey: ModelSession] = [:]

    public func neuralEngineSupport(for quality: DenoiseQuality) async -> Bool {
        guard let resource = Self.modelResource(for: quality) else {
            return false
        }
        return await ModelSession.supportsNeuralEngine(
            modelURL: resource.url,
            modelKey: "\(resource.name)_v1"
        )
    }

    public func process(
        sourceURL: URL,
        sourceName: String,
        backend: DenoiseBackend,
        quality: DenoiseQuality,
        highOverlap: Bool,
        progress: @escaping @Sendable (DenoiseProgress) -> Void
    ) async throws -> DenoiseResult {
        let started = Date()
        let startedUptime = ProcessInfo.processInfo.systemUptime
        var modelLoadSeconds: TimeInterval = 0
        var decodeSeconds: TimeInterval = 0
        var prepareSeconds: TimeInterval = 0
        var inputCopySeconds: TimeInterval = 0
        var predictionSeconds: TimeInterval = 0
        var outputCopySeconds: TimeInterval = 0
        var composeSeconds: TimeInterval = 0
        var encodeSeconds: TimeInterval = 0
        try Task.checkCancellation()
        guard let dimensions = ImageCodec.imageDimensions(sourceURL) else {
            throw SCUNetError(message: "The image dimensions could not be read")
        }
        let stride = highOverlap ? Self.tile / 2 : Self.core
        let columns = Self.divideRoundUp(dimensions.0, stride) + (highOverlap ? 1 : 0)
        let rows = Self.divideRoundUp(dimensions.1, stride) + (highOverlap ? 1 : 0)
        let totalTiles = columns * rows
        progress(.init(
            phase: .preparing,
            completedTiles: 0,
            totalTiles: totalTiles,
            elapsedSeconds: Date().timeIntervalSince(started)
        ))

        Self.logger.notice("Preparing \(backend.rawValue, privacy: .public) model session")
        let modelStarted = ProcessInfo.processInfo.systemUptime
        let execution = try await executionSessions(for: backend, quality: quality)
        modelLoadSeconds = ProcessInfo.processInfo.systemUptime - modelStarted
        let backendLabel = execution.label
        Self.logger.notice("Model session ready on \(backendLabel, privacy: .public)")

        progress(.init(
            phase: .loading,
            completedTiles: 0,
            totalTiles: totalTiles,
            elapsedSeconds: Date().timeIntervalSince(started)
        ))
        try Task.checkCancellation()
        let decodeStarted = ProcessInfo.processInfo.systemUptime
        let source = try ImageCodec.decodeFullImage(sourceURL)
        decodeSeconds = ProcessInfo.processInfo.systemUptime - decodeStarted
        guard source.width == dimensions.0, source.height == dimensions.1 else {
            throw SCUNetError(message: "The decoded image dimensions changed unexpectedly")
        }
        try Task.checkCancellation()

        var destination = [UInt8](
            repeating: 0,
            count: source.width * source.height * 3
        )
        var previousRow: [[Float]]?
        var completed = 0

        if highOverlap {
            let highOverlapSessions: [ModelSession]
            switch execution {
            case .single(let selected): highOverlapSessions = [selected]
            case .combined(let neuralEngine, _): highOverlapSessions = [neuralEngine]
            }
            let selectedSession = highOverlapSessions[0]
            var input = [Float](
                repeating: 0,
                count: selectedSession.inputTensorCount
            )
            let noiseWorkspace = Self.conditionedWorkspace(input)
            var band = [Float](repeating: 0, count: source.width * Self.tile * 3)
            let weights = (0..<Self.tile).map { index -> Float in
                let value = sin(Double.pi * (Double(index) + 0.5) / Double(Self.tile))
                return Float(value * value)
            }
            var bandOrigin = -stride
            for row in 0..<rows {
                for column in 0..<columns {
                    try Task.checkCancellation()
                    let startX = column * stride - stride
                    let startY = row * stride - stride
                    let prepareStarted = ProcessInfo.processInfo.systemUptime
                    Self.prepareTile(source.rgba, width: source.width, height: source.height,
                        destination: &input, startX: startX, startY: startY,
                        noiseWorkspace: noiseWorkspace)
                    prepareSeconds += ProcessInfo.processInfo.systemUptime - prepareStarted
                    let session = highOverlapSessions[completed % highOverlapSessions.count]
                    let run = try await session.run(input)
                    inputCopySeconds += run.inputCopySeconds
                    predictionSeconds += run.predictionSeconds
                    outputCopySeconds += run.outputCopySeconds
                    let composeStarted = ProcessInfo.processInfo.systemUptime
                    Self.addWeightedTile(run.output, to: &band, bandOrigin: bandOrigin,
                        width: source.width, height: source.height,
                        startX: startX, startY: startY, weights: weights)
                    composeSeconds += ProcessInfo.processInfo.systemUptime - composeStarted
                    completed += 1
                    if completed == 1 || completed == totalTiles || completed.isMultiple(of: 2) {
                        progress(.init(phase: .processing, completedTiles: completed,
                            totalTiles: totalTiles, elapsedSeconds: Date().timeIntervalSince(started)))
                    }
                }
                Self.flushWeightedBand(&band, into: &destination, width: source.width,
                    height: source.height, bandOrigin: bandOrigin, rowCount: stride)
                bandOrigin += stride
            }
            Self.flushWeightedBand(&band, into: &destination, width: source.width,
                height: source.height, bandOrigin: bandOrigin, rowCount: Self.tile)
        } else {
        switch execution {
        case .single(let session):
            var input = [Float](repeating: 0, count: session.inputTensorCount)
            let noiseWorkspace = Self.conditionedWorkspace(input)
            for row in 0..<rows {
                try Task.checkCancellation()
                var currentRow = [[Float]]()
                currentRow.reserveCapacity(columns)
                let coreY = row * Self.core

                for column in 0..<columns {
                    try Task.checkCancellation()
                    let coreX = column * Self.core
                    let prepareStarted = ProcessInfo.processInfo.systemUptime
                    Self.prepareTile(
                        source.rgba,
                        width: source.width,
                        height: source.height,
                        destination: &input,
                        startX: coreX - Self.padding,
                        startY: coreY - Self.padding,
                        noiseWorkspace: noiseWorkspace
                    )
                    prepareSeconds += ProcessInfo.processInfo.systemUptime - prepareStarted

                    let tileStarted = Date()
                    let run = try await session.run(input)
                    let tileSeconds = Date().timeIntervalSince(tileStarted)
                    inputCopySeconds += run.inputCopySeconds
                    predictionSeconds += run.predictionSeconds
                    outputCopySeconds += run.outputCopySeconds
                    if completed < 3 || (completed + 1).isMultiple(of: 20) {
                        Self.logger.notice(
                            "Tile \(completed + 1) completed on \(backendLabel, privacy: .public) in \(tileSeconds, format: .fixed(precision: 3)) seconds"
                        )
                    }

                    let composeStarted = ProcessInfo.processInfo.systemUptime
                    Self.composeTile(
                        run.output,
                        row: row,
                        column: column,
                        destination: &destination,
                        width: source.width,
                        height: source.height,
                        currentRow: &currentRow,
                        previousRow: previousRow
                    )
                    composeSeconds += ProcessInfo.processInfo.systemUptime - composeStarted
                    completed += 1
                    if completed == 1 || completed == totalTiles || completed.isMultiple(of: 2) {
                        progress(.init(
                            phase: .processing,
                            completedTiles: completed,
                            totalTiles: totalTiles,
                            elapsedSeconds: Date().timeIntervalSince(started)
                        ))
                    }
                    if let limit = Self.benchmarkTileLimit,
                       completed >= min(limit, totalTiles) {
                        Self.logProfile(
                            quality: quality,
                            completedTiles: completed,
                            totalSeconds: ProcessInfo.processInfo.systemUptime - startedUptime,
                            modelLoadSeconds: modelLoadSeconds,
                            decodeSeconds: decodeSeconds,
                            prepareSeconds: prepareSeconds,
                            inputCopySeconds: inputCopySeconds,
                            predictionSeconds: predictionSeconds,
                            outputCopySeconds: outputCopySeconds,
                            composeSeconds: composeSeconds,
                            encodeSeconds: encodeSeconds
                        )
                        throw CancellationError()
                    }
                }
                previousRow = currentRow
            }

        case .combined(let neuralEngine, let gpu):
            let counter = TileCompletionCounter(
                total: totalTiles,
                started: started,
                progress: progress
            )
            let tileScheduler = TileIndexScheduler(tileCount: totalTiles)
            let inferenceStarted = ProcessInfo.processInfo.systemUptime
            async let aneResult = Self.inferBand(
                tilesFromStart: true,
                columns: columns,
                source: source,
                session: neuralEngine,
                tileScheduler: tileScheduler,
                completionCounter: counter
            )
            async let gpuResult = Self.inferBand(
                tilesFromStart: false,
                columns: columns,
                source: source,
                session: gpu,
                tileScheduler: tileScheduler,
                completionCounter: counter
            )
            var (aneBand, gpuBand) = try await (aneResult, gpuResult)
            let inferenceWallSeconds = ProcessInfo.processInfo.systemUptime - inferenceStarted
            prepareSeconds = aneBand.prepareSeconds + gpuBand.prepareSeconds
            inputCopySeconds = aneBand.inputCopySeconds + gpuBand.inputCopySeconds
            outputCopySeconds = aneBand.outputCopySeconds + gpuBand.outputCopySeconds
            predictionSeconds = inferenceWallSeconds
            completed = aneBand.tiles.count + gpuBand.tiles.count
            Self.logCombinedProfile(
                neuralEngine: aneBand,
                gpu: gpuBand,
                inferenceWallSeconds: inferenceWallSeconds,
                totalTiles: totalTiles
            )

            var currentRow = [[Float]]()
            currentRow.reserveCapacity(columns)
            var activeRow = -1
            var orderedTiles = aneBand.tiles + gpuBand.tiles
            orderedTiles.sort { $0.index < $1.index }
            for tile in orderedTiles {
                try Task.checkCancellation()
                if tile.row != activeRow {
                    currentRow = []
                    currentRow.reserveCapacity(columns)
                    activeRow = tile.row
                }
                let composeStarted = ProcessInfo.processInfo.systemUptime
                Self.composeTile(
                    tile.output,
                    row: tile.row,
                    column: tile.column,
                    destination: &destination,
                    width: source.width,
                    height: source.height,
                    currentRow: &currentRow,
                    previousRow: previousRow
                )
                composeSeconds += ProcessInfo.processInfo.systemUptime - composeStarted
                if tile.column == columns - 1 {
                    previousRow = currentRow
                }
            }
            orderedTiles.removeAll(keepingCapacity: false)
            aneBand.tiles.removeAll(keepingCapacity: false)
            gpuBand.tiles.removeAll(keepingCapacity: false)
        }
        }

        try Task.checkCancellation()
        progress(.init(
            phase: .encoding,
            completedTiles: totalTiles,
            totalTiles: totalTiles,
            elapsedSeconds: Date().timeIntervalSince(started)
        ))
        let outputURL = try ImageCodec.outputURL(for: sourceName)
        let encodeStarted = ProcessInfo.processInfo.systemUptime
        try ImageCodec.encodeJPEG(
            destination,
            width: source.width,
            height: source.height,
            metadataSourceURL: sourceURL,
            destinationURL: outputURL
        )
        encodeSeconds = ProcessInfo.processInfo.systemUptime - encodeStarted
        Self.logProfile(
            quality: quality,
            completedTiles: completed,
            totalSeconds: ProcessInfo.processInfo.systemUptime - startedUptime,
            modelLoadSeconds: modelLoadSeconds,
            decodeSeconds: decodeSeconds,
            prepareSeconds: prepareSeconds,
            inputCopySeconds: inputCopySeconds,
            predictionSeconds: predictionSeconds,
            outputCopySeconds: outputCopySeconds,
            composeSeconds: composeSeconds,
            encodeSeconds: encodeSeconds
        )
        Self.logger.notice(
            "Completed \(totalTiles) tiles on \(backendLabel, privacy: .public) in \(Date().timeIntervalSince(started), format: .fixed(precision: 1)) seconds"
        )
        return DenoiseResult(
            url: outputURL,
            width: source.width,
            height: source.height,
            elapsedSeconds: Date().timeIntervalSince(started),
            backendLabel: backendLabel,
            quality: quality
        )
    }

    private func executionSessions(
        for backend: DenoiseBackend,
        quality: DenoiseQuality
    ) async throws -> ExecutionSessions {
        if backend == .gpuAndNeuralEngine {
            let neuralEngine = try await modelSession(for: .neuralEngine, quality: quality)
            let gpu = try await modelSession(for: .gpu, quality: quality)
            return .combined(neuralEngine: neuralEngine, gpu: gpu)
        }
        return .single(try await modelSession(for: backend, quality: quality))
    }

    private func modelSession(
        for backend: DenoiseBackend,
        quality: DenoiseQuality
    ) async throws -> ModelSession {
        let sessionKey = SessionKey(backend: backend, quality: quality)
        if let existing = sessions[sessionKey] { return existing }
        guard let resource = Self.modelResource(for: quality) else {
            throw SCUNetError(message: "SCUNet model is missing from the app")
        }

        let created = try await ModelSession.load(
            modelURL: resource.url,
            modelKey: "\(resource.name)_v1",
            tileSize: Self.tile,
            inputChannels: quality == .highPerformance ? 5 : 3,
            backend: backend
        )
        sessions[sessionKey] = created
        return created
    }

    private static func modelResource(
        for quality: DenoiseQuality
    ) -> (name: String, url: URL)? {
        let resourceName: String
        if Self.tile == 192, quality == .highPerformance {
            resourceName = "litedenoise_192_w24"
        } else if Self.modelVariant == "baseline", Self.tile == 192, quality == .highQuality {
            resourceName = "scunet_192_baseline_fp16"
        } else {
            resourceName = "scunet_\(Self.tile)_\(quality.modelSuffix)"
        }
        guard let modelURL = Bundle.module.url(
            forResource: resourceName,
            withExtension: "mlpackage",
            subdirectory: "Models"
        ) else {
            return nil
        }
        return (resourceName, modelURL)
    }

    private static func inferBand(
        tilesFromStart: Bool,
        columns: Int,
        source: ImageCodec.FullImage,
        session: ModelSession,
        tileScheduler: TileIndexScheduler,
        completionCounter: TileCompletionCounter
    ) async throws -> BandInference {
        let wallStarted = ProcessInfo.processInfo.systemUptime
        var prepareSeconds: TimeInterval = 0
        var inputCopySeconds: TimeInterval = 0
        var predictionSeconds: TimeInterval = 0
        var outputCopySeconds: TimeInterval = 0
        var input = [Float](repeating: 0, count: session.inputTensorCount)
        let noiseWorkspace = conditionedWorkspace(input)
        var tiles = [InferredTile]()
        tiles.reserveCapacity(512)

        while let index = await tileScheduler.claimIndex(fromStart: tilesFromStart) {
            try Task.checkCancellation()
            let row = index / columns
            let column = index % columns
            let coreX = column * core
            let coreY = row * core
            let prepareStarted = ProcessInfo.processInfo.systemUptime
            prepareTile(
                source.rgba,
                width: source.width,
                height: source.height,
                destination: &input,
                startX: coreX - padding,
                startY: coreY - padding,
                noiseWorkspace: noiseWorkspace
            )
            prepareSeconds += ProcessInfo.processInfo.systemUptime - prepareStarted

            let run = try await session.run(input)
            inputCopySeconds += run.inputCopySeconds
            predictionSeconds += run.predictionSeconds
            outputCopySeconds += run.outputCopySeconds
            tiles.append(.init(
                index: index,
                row: row,
                column: column,
                output: run.output
            ))
            if tiles.count <= 3 || tiles.count.isMultiple(of: 50) {
                logger.notice(
                    "Combined tile \(index + 1) completed on \(session.backend.rawValue, privacy: .public) in \(run.predictionSeconds, format: .fixed(precision: 3)) seconds"
                )
            }
            await completionCounter.finishTile()
        }

        return BandInference(
            tiles: tiles,
            backend: session.backend,
            wallSeconds: ProcessInfo.processInfo.systemUptime - wallStarted,
            prepareSeconds: prepareSeconds,
            inputCopySeconds: inputCopySeconds,
            predictionSeconds: predictionSeconds,
            outputCopySeconds: outputCopySeconds
        )
    }

    private static func composeTile(
        _ output: [Float],
        row: Int,
        column: Int,
        destination: inout [UInt8],
        width: Int,
        height: Int,
        currentRow: inout [[Float]],
        previousRow: [[Float]]?
    ) {
        let coreX = column * core
        let coreY = row * core
        copyCore(
            output,
            destination: &destination,
            width: width,
            height: height,
            destinationX: coreX,
            destinationY: coreY
        )
        if let left = currentRow.last {
            blendHorizontal(
                left,
                output,
                destination: &destination,
                width: width,
                height: height,
                seamX: coreX,
                startY: coreY
            )
        }
        if let top = previousRow?[column] {
            blendVertical(
                top,
                output,
                destination: &destination,
                width: width,
                height: height,
                startX: coreX,
                seamY: coreY
            )
        }
        if column > 0,
           let topLeft = previousRow?[column - 1],
           let top = previousRow?[column],
           let left = currentRow.last {
            blendCorner(
                topLeft,
                top,
                left,
                output,
                destination: &destination,
                width: width,
                height: height,
                seamX: coreX,
                seamY: coreY
            )
        }
        currentRow.append(output)
    }

    private static func addWeightedTile(
        _ tile: [Float],
        to band: inout [Float],
        bandOrigin: Int,
        width: Int,
        height: Int,
        startX: Int,
        startY: Int,
        weights: [Float]
    ) {
        let x0 = max(0, startX)
        let x1 = min(width, startX + Self.tile)
        let y0 = max(0, startY)
        let y1 = min(height, startY + Self.tile)
        guard x0 < x1, y0 < y1 else { return }
        for y in y0..<y1 {
            let tileY = y - startY
            let bandY = y - bandOrigin
            let wy = weights[tileY]
            for x in x0..<x1 {
                let tileIndex = tileY * Self.tile + x - startX
                let bandIndex = (bandY * width + x) * 3
                let weight = wy * weights[x - startX]
                band[bandIndex] += tile[tileIndex] * weight
                band[bandIndex + 1] += tile[Self.plane + tileIndex] * weight
                band[bandIndex + 2] += tile[2 * Self.plane + tileIndex] * weight
            }
        }
    }

    private static func flushWeightedBand(
        _ band: inout [Float],
        into destination: inout [UInt8],
        width: Int,
        height: Int,
        bandOrigin: Int,
        rowCount: Int
    ) {
        let y0 = max(0, bandOrigin)
        let y1 = min(height, bandOrigin + rowCount)
        if y0 < y1 {
            for y in y0..<y1 {
                let sourceOffset = (y - bandOrigin) * width * 3
                let destinationOffset = y * width * 3
                for index in 0..<(width * 3) {
                    destination[destinationOffset + index] = UInt8(clamping:
                        Int((band[sourceOffset + index] * 255).rounded()))
                }
            }
        }
        let shift = min(rowCount, Self.tile) * width * 3
        let bandCount = band.count
        let retained = bandCount - shift
        band.withUnsafeMutableBufferPointer { values in
            guard let base = values.baseAddress else { return }
            if retained > 0 {
                memmove(base, base.advanced(by: shift), retained * MemoryLayout<Float>.stride)
            }
            memset(base.advanced(by: retained), 0,
                (bandCount - retained) * MemoryLayout<Float>.stride)
        }
    }

    private static func prepareTile(
        _ source: [UInt8],
        width: Int,
        height: Int,
        destination: inout [Float],
        startX: Int,
        startY: Int,
        noiseWorkspace: NoiseStrengthWorkspace?
    ) {
        for y in 0..<tile {
            let sourceY = reflect(startY + y, limit: height)
            for x in 0..<tile {
                let sourceX = reflect(startX + x, limit: width)
                let sourceIndex = (sourceY * width + sourceX) * 4
                let tileIndex = y * tile + x
                destination[tileIndex] = Float(source[sourceIndex]) / 255
                destination[plane + tileIndex] = Float(source[sourceIndex + 1]) / 255
                destination[plane * 2 + tileIndex] = Float(source[sourceIndex + 2]) / 255
            }
        }
        if let noiseWorkspace {
            estimateNoiseStrengthAndFill(
                destination: &destination,
                workspace: noiseWorkspace
            )
        }
    }

    private static func conditionedWorkspace(
        _ input: [Float]
    ) -> NoiseStrengthWorkspace? {
        input.count == 5 * plane || input.count == 7 * plane
            ? NoiseStrengthWorkspace()
            : nil
    }

    private static func estimateNoiseStrengthAndFill(
        destination: inout [Float],
        workspace: NoiseStrengthWorkspace
    ) {
        precondition(destination.count == 5 * plane || destination.count == 7 * plane)
        let sigmaMinimum: Float = 0.0015
        let sigmaMaximum: Float = 0.07
        let shadowLimit: Float = 0.5
        let madNormalizer: Float = 0.67448975

        for index in 0..<plane {
            workspace.luma[index] = 0.2126 * destination[index]
                + 0.7152 * destination[plane + index]
                + 0.0722 * destination[2 * plane + index]
        }
        for y in 0..<tile {
            for x in 0..<tile {
                let index = y * tile + x
                var sum: Float = 0
                for offsetY in -1...1 {
                    let sourceY = y + offsetY
                    guard sourceY >= 0, sourceY < tile else { continue }
                    for offsetX in -1...1 {
                        let sourceX = x + offsetX
                        if sourceX >= 0, sourceX < tile {
                            sum += workspace.luma[sourceY * tile + sourceX]
                        }
                    }
                }
                workspace.residual[index] = abs(workspace.luma[index] - sum / 9)
                let horizontalX = x < tile - 1 ? x : tile - 2
                let verticalY = y < tile - 1 ? y : tile - 2
                let horizontal = abs(
                    workspace.luma[y * tile + horizontalX + 1]
                        - workspace.luma[y * tile + horizontalX]
                )
                let vertical = abs(
                    workspace.luma[(verticalY + 1) * tile + x]
                        - workspace.luma[verticalY * tile + x]
                )
                workspace.gradient[index] = 0.5 * (horizontal + vertical)
            }
        }

        var shadowCount = 0
        for index in 0..<plane where workspace.luma[index] < shadowLimit {
            workspace.selection[shadowCount] = workspace.gradient[index]
            shadowCount += 1
        }
        let useAll = shadowCount < 64
        let gradientCount = useAll ? plane : shadowCount
        if useAll {
            workspace.selection.replaceSubrange(
                0..<plane,
                with: workspace.gradient
            )
        }
        let gradientLimit = quantile(
            &workspace.selection,
            count: gradientCount,
            fraction: 0.4
        )

        var candidateCount = 0
        for index in 0..<plane {
            let isShadow = useAll || workspace.luma[index] < shadowLimit
            if isShadow, workspace.gradient[index] <= gradientLimit {
                workspace.selection[candidateCount] = workspace.residual[index]
                candidateCount += 1
            }
        }
        if candidateCount < 64 {
            workspace.selection.replaceSubrange(
                0..<plane,
                with: workspace.residual
            )
            candidateCount = plane
        }
        let sigma = quantile(
            &workspace.selection,
            count: candidateCount,
            fraction: 0.5
        ) / madNormalizer
        let clampedSigma = max(sigmaMinimum, sigma)
        let strength = max(
            0,
            min(
                1,
                (log(clampedSigma) - log(sigmaMinimum))
                    / (log(sigmaMaximum) - log(sigmaMinimum))
            )
        )
        destination.replaceSubrange(
            (3 * plane)..<(4 * plane),
            with: repeatElement(strength, count: plane)
        )
        let gatePosition = max(0, min(1, (strength - 0.35) / (0.75 - 0.35)))
        let gate = gatePosition * gatePosition * (3 - 2 * gatePosition)
        if destination.count == 5 * plane {
            destination.replaceSubrange(
                (4 * plane)..<(5 * plane),
                with: repeatElement(gate, count: plane)
            )
            return
        }
        for index in 0..<plane {
            let shadow = (shadowLimit - workspace.luma[index]) / shadowLimit
            destination[4 * plane + index] = max(0, min(1, shadow))
            workspace.opponentRed[index] = destination[index]
                - destination[plane + index]
            workspace.opponentBlue[index] = destination[2 * plane + index]
                - destination[plane + index]
        }
        for y in 0..<tile {
            let row = y * tile
            var redSum = 5 * workspace.opponentRed[row]
            var blueSum = 5 * workspace.opponentBlue[row]
            for x in 1...4 {
                redSum += workspace.opponentRed[row + x]
                blueSum += workspace.opponentBlue[row + x]
            }
            for x in 0..<tile {
                let index = row + x
                workspace.horizontalRed[index] = redSum
                workspace.horizontalBlue[index] = blueSum
                let added = min(tile - 1, x + 5)
                let removed = max(0, x - 4)
                redSum += workspace.opponentRed[row + added]
                    - workspace.opponentRed[row + removed]
                blueSum += workspace.opponentBlue[row + added]
                    - workspace.opponentBlue[row + removed]
            }
        }
        for x in 0..<tile {
            var redSum = 5 * workspace.horizontalRed[x]
            var blueSum = 5 * workspace.horizontalBlue[x]
            for y in 1...4 {
                redSum += workspace.horizontalRed[y * tile + x]
                blueSum += workspace.horizontalBlue[y * tile + x]
            }
            for y in 0..<tile {
                let index = y * tile + x
                let chroma = (
                    abs(workspace.opponentRed[index] - redSum / 81)
                        + abs(workspace.opponentBlue[index] - blueSum / 81)
                ) * 0.5 / 0.04
                destination[5 * plane + index] = max(0, min(1, chroma))
                let added = min(tile - 1, y + 5) * tile + x
                let removed = max(0, y - 4) * tile + x
                redSum += workspace.horizontalRed[added]
                    - workspace.horizontalRed[removed]
                blueSum += workspace.horizontalBlue[added]
                    - workspace.horizontalBlue[removed]
            }
        }
        destination.replaceSubrange(
            (6 * plane)..<(7 * plane),
            with: repeatElement(gate, count: plane)
        )
    }

    private static func quantile(
        _ values: inout [Float],
        count: Int,
        fraction: Float
    ) -> Float {
        values[0..<count].sort()
        let position = Float(count - 1) * fraction
        let lower = Int(position.rounded(.down))
        let upper = min(count - 1, lower + 1)
        let weight = position - Float(lower)
        return values[lower] + weight * (values[upper] - values[lower])
    }

    private static func copyCore(
        _ tileOutput: [Float],
        destination: inout [UInt8],
        width: Int,
        height: Int,
        destinationX: Int,
        destinationY: Int
    ) {
        let copyWidth = min(core, width - destinationX)
        let copyHeight = min(core, height - destinationY)
        for y in 0..<copyHeight {
            let tileRow = (padding + y) * tile + padding
            let imageRow = ((destinationY + y) * width + destinationX) * 3
            for x in 0..<copyWidth {
                let sourceIndex = tileRow + x
                let destinationIndex = imageRow + x * 3
                destination[destinationIndex] = byte(tileOutput[sourceIndex])
                destination[destinationIndex + 1] = byte(tileOutput[plane + sourceIndex])
                destination[destinationIndex + 2] = byte(tileOutput[plane * 2 + sourceIndex])
            }
        }
    }

    private static func blendHorizontal(
        _ left: [Float],
        _ right: [Float],
        destination: inout [UInt8],
        width: Int,
        height: Int,
        seamX: Int,
        startY: Int
    ) {
        let span = padding * 2
        let endY = min(height, startY + core)
        for y in max(0, startY)..<endY {
            let tileY = padding + y - startY
            for x in (seamX - padding)..<min(seamX + padding, width) where x >= 0 {
                let offset = x - (seamX - padding)
                let weight = Float(offset) + 0.5
                blendPixel(
                    destination: &destination,
                    width: width,
                    x: x,
                    y: y,
                    first: left,
                    second: right,
                    firstIndex: tileY * tile + core + offset,
                    secondIndex: tileY * tile + offset,
                    weight: weight / Float(span)
                )
            }
        }
    }

    private static func blendVertical(
        _ top: [Float],
        _ bottom: [Float],
        destination: inout [UInt8],
        width: Int,
        height: Int,
        startX: Int,
        seamY: Int
    ) {
        let span = padding * 2
        let endX = min(width, startX + core)
        for y in (seamY - padding)..<min(seamY + padding, height) where y >= 0 {
            let offset = y - (seamY - padding)
            let weight = (Float(offset) + 0.5) / Float(span)
            for x in max(0, startX)..<endX {
                let tileX = padding + x - startX
                blendPixel(
                    destination: &destination,
                    width: width,
                    x: x,
                    y: y,
                    first: top,
                    second: bottom,
                    firstIndex: (core + offset) * tile + tileX,
                    secondIndex: offset * tile + tileX,
                    weight: weight
                )
            }
        }
    }

    private static func blendCorner(
        _ topLeft: [Float],
        _ top: [Float],
        _ left: [Float],
        _ current: [Float],
        destination: inout [UInt8],
        width: Int,
        height: Int,
        seamX: Int,
        seamY: Int
    ) {
        let span = padding * 2
        for y in (seamY - padding)..<min(seamY + padding, height) where y >= 0 {
            let offsetY = y - (seamY - padding)
            let weightY = (Float(offsetY) + 0.5) / Float(span)
            for x in (seamX - padding)..<min(seamX + padding, width) where x >= 0 {
                let offsetX = x - (seamX - padding)
                let weightX = (Float(offsetX) + 0.5) / Float(span)
                let topLeftIndex = (core + offsetY) * tile + core + offsetX
                let topIndex = (core + offsetY) * tile + offsetX
                let leftIndex = offsetY * tile + core + offsetX
                let currentIndex = offsetY * tile + offsetX
                let destinationIndex = (y * width + x) * 3
                for channel in 0..<3 {
                    let channelOffset = channel * plane
                    let topValue = lerp(
                        topLeft[channelOffset + topLeftIndex],
                        top[channelOffset + topIndex],
                        weightX
                    )
                    let bottomValue = lerp(
                        left[channelOffset + leftIndex],
                        current[channelOffset + currentIndex],
                        weightX
                    )
                    destination[destinationIndex + channel] = byte(
                        lerp(topValue, bottomValue, weightY)
                    )
                }
            }
        }
    }

    private static func blendPixel(
        destination: inout [UInt8],
        width: Int,
        x: Int,
        y: Int,
        first: [Float],
        second: [Float],
        firstIndex: Int,
        secondIndex: Int,
        weight: Float
    ) {
        let destinationIndex = (y * width + x) * 3
        for channel in 0..<3 {
            let offset = channel * plane
            destination[destinationIndex + channel] = byte(
                lerp(first[offset + firstIndex], second[offset + secondIndex], weight)
            )
        }
    }

    private static func reflect(_ value: Int, limit: Int) -> Int {
        guard limit > 1 else { return 0 }
        var reflected = value
        while reflected < 0 || reflected >= limit {
            reflected = reflected < 0 ? -reflected : 2 * limit - 2 - reflected
        }
        return reflected
    }

    private static func byte(_ value: Float) -> UInt8 {
        guard value.isFinite else { return 0 }
        return UInt8(max(0, min(255, Int((value * 255).rounded()))))
    }

    private static func lerp(_ first: Float, _ second: Float, _ weight: Float) -> Float {
        first + (second - first) * weight
    }

    private static func divideRoundUp(_ value: Int, _ divisor: Int) -> Int {
        (value + divisor - 1) / divisor
    }

    private static func argumentValue(prefix: String) -> String? {
        CommandLine.arguments.first(where: { $0.hasPrefix(prefix) })
            .map { String($0.dropFirst(prefix.count)) }
    }

    private static func logCombinedProfile(
        neuralEngine: BandInference,
        gpu: BandInference,
        inferenceWallSeconds: TimeInterval,
        totalTiles: Int
    ) {
        let aneTiles = neuralEngine.tiles.count
        let gpuTiles = gpu.tiles.count
        let boundaryIndex = (neuralEngine.tiles.map(\.index).max() ?? -1) + 1
        let aneMean = aneTiles > 0
            ? neuralEngine.predictionSeconds * 1_000 / Double(aneTiles)
            : 0
        let gpuMean = gpuTiles > 0
            ? gpu.predictionSeconds * 1_000 / Double(gpuTiles)
            : 0
        let overlap = inferenceWallSeconds > 0
            ? (neuralEngine.predictionSeconds + gpu.predictionSeconds) / inferenceWallSeconds
            : 0
        let detail = String(
            format: "SCUNET_COMBINED boundary=%d/%d ane_tiles=%d ane_wall=%.3f ane_prediction=%.3f ane_mean_ms=%.3f gpu_tiles=%d gpu_wall=%.3f gpu_prediction=%.3f gpu_mean_ms=%.3f concurrent_wall=%.3f overlap=%.3f",
            boundaryIndex,
            totalTiles,
            aneTiles,
            neuralEngine.wallSeconds,
            neuralEngine.predictionSeconds,
            aneMean,
            gpuTiles,
            gpu.wallSeconds,
            gpu.predictionSeconds,
            gpuMean,
            inferenceWallSeconds,
            overlap
        )
        logger.notice("\(detail, privacy: .public)")
    }

    private static func logProfile(
        quality: DenoiseQuality,
        completedTiles: Int,
        totalSeconds: TimeInterval,
        modelLoadSeconds: TimeInterval,
        decodeSeconds: TimeInterval,
        prepareSeconds: TimeInterval,
        inputCopySeconds: TimeInterval,
        predictionSeconds: TimeInterval,
        outputCopySeconds: TimeInterval,
        composeSeconds: TimeInterval,
        encodeSeconds: TimeInterval
    ) {
        let measured = prepareSeconds + inputCopySeconds + predictionSeconds
            + outputCopySeconds + composeSeconds
        let otherSeconds = max(
            0,
            totalSeconds - modelLoadSeconds - decodeSeconds - measured - encodeSeconds
        )
        let meanPredictionMilliseconds = completedTiles > 0
            ? predictionSeconds * 1_000 / Double(completedTiles)
            : 0
        let detail = String(
            format: "SCUNET_PROFILE tile=%d variant=%@ weights=%@ completed=%d total=%.3f model=%.3f decode=%.3f prepare=%.3f input_copy=%.3f prediction=%.3f prediction_mean_ms=%.3f output_copy=%.3f compose=%.3f encode=%.3f other=%.3f",
            tile,
            modelVariant,
            quality.modelSuffix,
            completedTiles,
            totalSeconds,
            modelLoadSeconds,
            decodeSeconds,
            prepareSeconds,
            inputCopySeconds,
            predictionSeconds,
            meanPredictionMilliseconds,
            outputCopySeconds,
            composeSeconds,
            encodeSeconds,
            otherSeconds
        )
        logger.notice("\(detail, privacy: .public)")
    }
}

public enum ImageCodec {
    struct FullImage: Sendable {
        let rgba: [UInt8]
        let width: Int
        let height: Int
    }

    static func decodeFullImage(_ url: URL) throws -> FullImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil)
                as? [CFString: Any],
              let rawWidth = properties[kCGImagePropertyPixelWidth] as? Int,
              let rawHeight = properties[kCGImagePropertyPixelHeight] as? Int else {
            throw SCUNetError(message: "The selected image could not be decoded")
        }
        let maximumDimension = max(rawWidth, rawHeight)
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maximumDimension,
            kCGImageSourceShouldCacheImmediately: true,
        ]
        guard let image = CGImageSourceCreateThumbnailAtIndex(
            source,
            0,
            options as CFDictionary
        ) else {
            throw SCUNetError(message: "The full-resolution image could not be decoded")
        }
        let width = image.width
        let height = image.height
        guard width > 0,
              height > 0,
              width <= Int.max / height / 4 else {
            throw SCUNetError(message: "The image dimensions are invalid")
        }

        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        let bitmapInfo = CGBitmapInfo.byteOrder32Big.rawValue |
            CGImageAlphaInfo.premultipliedLast.rawValue
        guard let context = CGContext(
            data: &rgba,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: bitmapInfo
        ) else {
            throw SCUNetError(message: "Could not create the image buffer")
        }
        context.interpolationQuality = .none
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return FullImage(rgba: rgba, width: width, height: height)
    }

    public static func previewImage(_ url: URL, maximumDimension: Int = 1800) throws -> UIImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else {
            throw SCUNetError(message: "The selected image could not be opened")
        }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maximumDimension,
            kCGImageSourceShouldCacheImmediately: true,
        ]
        guard let image = CGImageSourceCreateThumbnailAtIndex(
            source,
            0,
            options as CFDictionary
        ) else {
            throw SCUNetError(message: "The image preview could not be decoded")
        }
        return UIImage(cgImage: image)
    }

    public static func imageDimensions(_ url: URL) -> (Int, Int)? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil)
                as? [CFString: Any],
              let width = properties[kCGImagePropertyPixelWidth] as? Int,
              let height = properties[kCGImagePropertyPixelHeight] as? Int else {
            return nil
        }
        let orientation = (properties[kCGImagePropertyOrientation] as? Int) ?? 1
        return [5, 6, 7, 8].contains(orientation) ? (height, width) : (width, height)
    }

    static func encodeJPEG(
        _ rgb: [UInt8],
        width: Int,
        height: Int,
        metadataSourceURL: URL,
        destinationURL: URL
    ) throws {
        guard rgb.count == width * height * 3,
              let provider = CGDataProvider(data: Data(rgb) as CFData),
              let image = CGImage(
                width: width,
                height: height,
                bitsPerComponent: 8,
                bitsPerPixel: 24,
                bytesPerRow: width * 3,
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                provider: provider,
                decode: nil,
                shouldInterpolate: false,
                intent: .defaultIntent
              ),
              let destination = CGImageDestinationCreateWithURL(
                destinationURL as CFURL,
                UTType.jpeg.identifier as CFString,
                1,
                nil
              ) else {
            throw SCUNetError(message: "Could not create the JPEG output")
        }
        var options = (
            CGImageSourceCreateWithURL(metadataSourceURL as CFURL, nil)
                .flatMap { CGImageSourceCopyPropertiesAtIndex($0, 0, nil) }
                as? [CFString: Any]
        ) ?? [:]
        options[kCGImageDestinationLossyCompressionQuality] = 0.95
        options[kCGImagePropertyOrientation] = 1
        options[kCGImagePropertyPixelWidth] = width
        options[kCGImagePropertyPixelHeight] = height
        var exif = options[kCGImagePropertyExifDictionary] as? [CFString: Any] ?? [:]
        exif[kCGImagePropertyExifPixelXDimension] = width
        exif[kCGImagePropertyExifPixelYDimension] = height
        options[kCGImagePropertyExifDictionary] = exif
        CGImageDestinationAddImage(destination, image, options as CFDictionary)
        guard CGImageDestinationFinalize(destination) else {
            throw SCUNetError(message: "JPEG encoding failed")
        }
    }

    static func outputURL(for sourceName: String) throws -> URL {
        let directory = try outputDirectory()
        let sourceStem = (sourceName as NSString).deletingPathExtension
        let sanitized = sourceStem
            .replacingOccurrences(
                of: "[^A-Za-z0-9._ -]",
                with: "_",
                options: .regularExpression
            )
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let stem = sanitized.isEmpty ? "image" : sanitized
        let destination = directory.appendingPathComponent("\(stem)_denoised.jpg")
        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        return destination
    }

    public static func importSource(_ url: URL) throws -> (URL, String) {
        let didAccess = url.startAccessingSecurityScopedResource()
        defer { if didAccess { url.stopAccessingSecurityScopedResource() } }
        let directory = try inputDirectory()
        let rawName = url.lastPathComponent.isEmpty ? "image" : url.lastPathComponent
        let destination = directory.appendingPathComponent(
            "\(UUID().uuidString)-\(rawName)"
        )
        try FileManager.default.copyItem(at: url, to: destination)
        return (destination, rawName)
    }

    public static func importSource(
        data: Data,
        preferredExtension: String,
        originalName: String? = nil
    ) throws -> (URL, String) {
        let directory = try inputDirectory()
        let suppliedName = originalName.map {
            ($0 as NSString).lastPathComponent
        }
        let fileName = suppliedName.flatMap { $0.isEmpty ? nil : $0 }
            ?? "Photo-\(UUID().uuidString).\(preferredExtension)"
        let destination = directory.appendingPathComponent(fileName)
        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        try data.write(to: destination, options: .atomic)
        return (destination, fileName)
    }

    private static func inputDirectory() throws -> URL {
        try directory(named: "SCUNet Inputs")
    }

    private static func outputDirectory() throws -> URL {
        try directory(named: "SCUNet Outputs")
    }

    private static func directory(named name: String) throws -> URL {
        let root = FileManager.default.urls(
            for: .cachesDirectory,
            in: .userDomainMask
        )[0]
        let directory = root.appendingPathComponent(name, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        return directory
    }
}
