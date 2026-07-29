import Combine
import CoreImage
import Foundation
import OSLog
import AINRRuntime
import UIKit

@MainActor
final class CameraController: ObservableObject {
    private static let logger = Logger(subsystem: "com.ryu.remotecapture.ios", category: "camera-session")
    @Published var phase: ConnectionPhase = .disconnected
    @Published var liveViewImage: UIImage?
    @Published private(set) var lutPreviewImage: UIImage?
    @Published var computationalPreview: UIImage?
    @Published var photos: [SavedPhoto] = []
    @Published var selectedMode: CaptureMode = .photo
    @Published var settings: [CameraSettingID: CameraSetting] = [:]
    @Published var downloadProgress: Double?
    @Published var pendingDownloads = 0
    @Published var ainrProgress: AINRProcessingProgress?
    @Published var statusMessage = "Join the camera Wi-Fi, then connect."
    @Published var zoomPosition: Int?
    @Published var zoomSetting: String?
    @Published var zoomBoxCount: Int?
    @Published var zoomBoxIndex: Int?
    @Published var cameraStatus: String?
    @Published var stackFrameCount = 0
    @Published var stackTargetCount = 0
    @Published var isStackRendering = false
    @Published var pairedCameras: [PairedCamera] = []
    @Published var isLiveViewRunning = false
    @Published var isContinuousCaptureActive = false
    @Published var failedPhotoPublicationIDs = Set<String>()
    @Published var lutImportMessage: String?
    @Published var cameraHost: String { didSet { UserDefaults.standard.set(cameraHost, forKey: Self.hostKey) } }

    var preferences = AppPreferences()
    let lutLibrary = LUTLibrary()
    private let store = PhotoStore()
    private let computationalSessionStore = ComputationalSessionStore()
    private let downloadWorker = RemoteDownloadWorker()
    private let locationProvider = LocationProvider()
    private var api: SonyCameraAPI?
    private var eventAPI: SonyCameraAPI?
    private var availableAPIs = Set<String>()
    private var eventTask: Task<Void, Never>?
    private var queueTask: Task<Void, Never>?
    private let liveView = LiveViewStream()
    private var knownRemoteURLs = Set<String>()
    private var downloadQueue: [QueuedRemoteCapture] = []
    private var stackData: [Data] = []
    private var stackMode: CaptureMode?
    private var modeDriveValues: [CaptureMode: String] = [:]
    private var modeSettingValues: [CaptureMode: [CameraSettingID: String]] = [:]
    private var liveViewTimer: Task<Void, Never>?
    private var autoConnectTask: Task<Void, Never>?
    private var connectionMonitorTask: Task<Void, Never>?
    private var settingsRefreshTask: Task<Void, Never>?
    private var capabilitiesRefreshTask: Task<Void, Never>?
    private var remoteModeRecoveryTask: Task<Void, Never>?
    private var zoomTask: Task<Void, Never>?
    private var zoomCommandActive = false
    private var downloadRetryCycles: [String: Int] = [:]
    private var modeTransitionTask: Task<Void, Never>?
    private var liveNDCompletionTask: Task<Void, Never>?
    private var lastLiveViewURL: URL?
    private var acceptingComputationalFrames = false
    private var closedLiveNDSession = false
    private var liveNDCaptureURLCount = 0
    private var stackPreviewGeneration = 0
    private var stackProcessingSnapshot: QueuedRemoteCapture?
    private var observations = Set<AnyCancellable>()

    var canContinuousCapture: Bool { availableAPIs.contains("startContShooting") && availableAPIs.contains("stopContShooting") }
    var canZoom: Bool { availableAPIs.contains("actZoom") }
    var selectedLUTName: String { preferences.lutSelection.identifier }
    var isPhotoContinuousDrive: Bool {
        selectedMode == .photo && canContinuousCapture &&
            !(settings[.drive]?.current.localizedCaseInsensitiveContains("single") ?? true)
    }
    var isPanoramaSessionActive: Bool {
        selectedMode == .panorama && acceptingComputationalFrames && stackMode == .panorama
    }
    var canFinishPanorama: Bool {
        isPanoramaSessionActive && stackFrameCount >= 2 && pendingDownloads == 0 &&
            !isStackRendering
    }

    init() {
        cameraHost = UserDefaults.standard.string(forKey: Self.hostKey) ?? "192.168.122.1"
        pairedCameras = (try? UserDefaults.standard.data(forKey: "pairedCameras").flatMap { try JSONDecoder().decode([PairedCamera].self, from: $0) }) ?? []
        preferences.objectWillChange.sink { [weak self] _ in self?.objectWillChange.send() }.store(in: &observations)
        lutLibrary.objectWillChange.sink { [weak self] _ in self?.objectWillChange.send() }.store(in: &observations)
        liveView.onFrame = { [weak self] data in
            Task { @MainActor [weak self] in
                guard let self, let source = CIImage(data: data) else { return }
                if self.lutPreviewImage == nil {
                    self.lutPreviewImage = ImageProcessor.shared.preview(source, maxDimension: 240)
                }
                self.liveViewImage = ImageProcessor.shared.preview(
                    ImageProcessor.shared.applyLUT(source, selection: self.preferences.lutSelection, library: self.lutLibrary),
                    maxDimension: 1000
                )
            }
        }
        liveView.onFailure = { [weak self] error in
            Task { @MainActor [weak self] in
                guard let self, self.phase == .connected else { return }
                self.liveViewTimer?.cancel()
                self.liveViewImage = nil
                self.isLiveViewRunning = false
                self.lastLiveViewURL = nil
                self.statusMessage = "Live view paused. Capture monitoring remains active."
                Self.logger.info(
                    "Live view interrupted without ending camera session: \(error.localizedDescription, privacy: .public)"
                )
            }
        }
        Task {
            await restoreComputationalSession()
            await reloadGallery()
        }
        startAutoConnectMonitor()
    }

    deinit {
        eventTask?.cancel()
        queueTask?.cancel()
        liveViewTimer?.cancel()
        autoConnectTask?.cancel()
        connectionMonitorTask?.cancel()
        settingsRefreshTask?.cancel()
        capabilitiesRefreshTask?.cancel()
        remoteModeRecoveryTask?.cancel()
        zoomTask?.cancel()
        modeTransitionTask?.cancel()
        liveNDCompletionTask?.cancel()
        liveView.stop()
    }

    func connect() {
        guard phase != .connecting else { return }
        phase = .connecting
        statusMessage = "Contacting camera at \(cameraHost)..."
        Task {
            do {
                let endpoint = try await SonyCameraDiscovery.cameraEndpoint(host: cameraHost)
                let api = SonyCameraAPI(endpoint: endpoint)
                let eventAPI = SonyCameraAPI(endpoint: endpoint)
                let apis = try await api.startRemoteModeIfNeeded()
                guard apis.contains("startLiveview") || apis.contains("startLiveviewWithSize") else {
                    throw ControllerError.message("This camera mode does not expose live view.")
                }
                try await api.setPostviewSize(preferences.downloadQuality, availableAPIs: apis)
                var loadedSettings = await api.settings(availableAPIs: apis)
                if liveNDDriveOverrideActive,
                   let previousDrive = liveNDPreviousDrive,
                   loadedSettings[.drive]?.options.contains(previousDrive) == true {
                    try? await api.setSetting(.drive, value: previousDrive)
                    loadedSettings[.drive]?.current = previousDrive
                    clearLiveNDDriveOverride()
                }
                let liveURL = try await api.startLiveView(
                    quality: preferences.liveViewQuality,
                    availableAPIs: apis
                )
                let eventVersion = (try? await eventAPI.negotiateEventVersion()) ?? "1.2"
                let baselineEvent = (try? await eventAPI.event(longPolling: false))
                    ?? CameraEventSnapshot()
                self.api = api
                self.eventAPI = eventAPI
                availableAPIs = apis
                settings = loadedSettings
                phase = .connected
                rememberConnectedCamera()
                statusMessage = Self.sessionStatus(controlCount: loadedSettings.count)
                lastLiveViewURL = liveURL
                liveView.start(url: liveURL)
                isLiveViewRunning = true
                armLiveViewTimeout()
                knownRemoteURLs.formUnion(baselineEvent.urls.map(\.absoluteString))
                applyCameraEvent(baselineEvent, enqueueURLs: false)
                Self.logger.info("Physical-shutter event listener negotiated version \(eventVersion)")
                startEventMonitor()
                startConnectionMonitor()
                startSettingsRefresh()
                if acceptingComputationalFrames, stackMode == .liveND {
                    scheduleLiveNDCompletion()
                }
            } catch {
                phase = .failed(Self.friendly(error))
                statusMessage = "Check that the iPhone is on the camera Wi-Fi."
            }
        }
    }

    func disconnect() {
        eventTask?.cancel(); eventTask = nil
        queueTask?.cancel(); queueTask = nil
        connectionMonitorTask?.cancel(); connectionMonitorTask = nil
        settingsRefreshTask?.cancel(); settingsRefreshTask = nil
        capabilitiesRefreshTask?.cancel(); capabilitiesRefreshTask = nil
        remoteModeRecoveryTask?.cancel(); remoteModeRecoveryTask = nil
        zoomTask?.cancel(); zoomTask = nil
        modeTransitionTask?.cancel(); modeTransitionTask = nil
        liveNDCompletionTask?.cancel(); liveNDCompletionTask = nil
        liveViewTimer?.cancel(); liveView.stop()
        liveViewImage = nil; lutPreviewImage = nil; isLiveViewRunning = false; lastLiveViewURL = nil
        isContinuousCaptureActive = false
        let oldAPI = api
        api = nil; eventAPI = nil; availableAPIs = []; knownRemoteURLs.removeAll(); downloadQueue.removeAll()
        phase = .disconnected; statusMessage = "Disconnected"
        Task { await oldAPI?.stopLiveView() }
    }

    func selectMode(_ mode: CaptureMode) {
        guard mode != selectedMode else { return }
        modeTransitionTask?.cancel()
        modeSettingValues[selectedMode] = settings.mapValues(\.current)
        if let drive = settings[.drive]?.current { modeDriveValues[selectedMode] = drive }
        selectedMode = mode
        resetStack()
        guard phase == .connected else { return }
        modeTransitionTask = Task {
            if let saved = modeSettingValues[mode] {
                let order: [CameraSettingID] = [.exposureMode, .aperture, .shutterSpeed, .iso, .exposureCompensation, .burstSpeed, .drive]
                for id in order where saved[id] != nil && settings[id]?.options.contains(saved[id]!) == true {
                    guard !Task.isCancelled, selectedMode == mode else { return }
                    try? await api?.setSetting(id, value: saved[id]!)
                    guard !Task.isCancelled, selectedMode == mode else { return }
                    settings[id]?.current = saved[id]!
                }
            }
            guard !Task.isCancelled, selectedMode == mode else { return }
            await configureDrive(for: mode)
        }
    }

    func importLUTs(_ urls: [URL]) {
        do {
            let result = try lutLibrary.importFiles(urls)
            guard let first = result.imported.first else { throw LUTError.noValidCube }
            preferences.lutSelection.identifier = first.id
            let message = result.summary
            statusMessage = message
            lutImportMessage = message
        } catch {
            let message = "LUT import failed: \(Self.friendly(error))"
            statusMessage = message
            lutImportMessage = message
        }
    }

    func capture() {
        guard let api, phase == .connected else { return }
        switch selectedMode {
        case .panorama:
            if !isPanoramaSessionActive {
                startPanoramaSession()
                return
            }
            statusMessage = "Capturing panorama frame..."
            Task {
                do {
                    try await api.setPostviewSize(
                        preferences.downloadQuality,
                        availableAPIs: availableAPIs
                    )
                    enqueue(try await api.takePicture())
                } catch {
                    statusMessage = "Capture failed: \(Self.friendly(error))"
                }
            }
        case .photo, .composite:
            statusMessage = selectedMode == .photo ? "Capturing..." : "Capturing frame..."
            Task {
                do {
                    if selectedMode != .photo {
                        if !acceptingComputationalFrames || stackMode != selectedMode {
                            acceptingComputationalFrames = true
                            closedLiveNDSession = false
                            stackMode = selectedMode
                            stackTargetCount = 0
                            stackProcessingSnapshot = processingSnapshot(
                                url: URL(fileURLWithPath: "\(selectedMode.rawValue).jpg"),
                                mode: selectedMode
                            )
                            try await persistNewComputationalSession()
                        }
                        await ensureSingleDrive()
                    }
                    try await api.setPostviewSize(preferences.downloadQuality, availableAPIs: availableAPIs)
                    enqueue(try await api.takePicture())
                } catch { statusMessage = "Capture failed: \(Self.friendly(error))" }
            }
        case .liveND: startLiveND()
        }
    }

    func finishStack() {
        if selectedMode == .panorama {
            guard canFinishPanorama else {
                statusMessage = stackFrameCount < 2
                    ? "Take at least two panorama frames"
                    : "Wait for panorama frames to finish downloading"
                return
            }
        } else {
            guard selectedMode == .composite, !stackData.isEmpty else { return }
        }
        Task { await finalizeStack(mode: selectedMode) }
    }

    func cancelPanoramaSession() {
        guard isPanoramaSessionActive, !isStackRendering else { return }
        resetStack()
        statusMessage = "Panorama discarded"
    }

    func resetStack() {
        liveNDCompletionTask?.cancel()
        liveNDCompletionTask = nil
        liveNDCaptureURLCount = 0
        stackPreviewGeneration += 1
        stackData.removeAll(); stackMode = nil; stackFrameCount = 0; stackTargetCount = 0; computationalPreview = nil
        stackProcessingSnapshot = nil
        acceptingComputationalFrames = false
        Task { try? await computationalSessionStore.clear() }
    }

    private func startPanoramaSession() {
        guard selectedMode == .panorama, phase == .connected,
              !acceptingComputationalFrames, !isStackRendering else { return }
        statusMessage = "Starting panorama..."
        Task {
            do {
                try await preparePanoramaDrive()
                acceptingComputationalFrames = true
                closedLiveNDSession = false
                stackMode = .panorama
                stackTargetCount = 0
                stackProcessingSnapshot = processingSnapshot(
                    url: URL(fileURLWithPath: "Panorama.jpg"),
                    mode: .panorama
                )
                try await persistNewComputationalSession()
                statusMessage = "Panorama ready. Use either shutter to add frames."
            } catch {
                resetStack()
                statusMessage = "Panorama could not start: \(Self.friendly(error))"
            }
        }
    }

    func startPhotoBurst() {
        guard selectedMode == .photo, canContinuousCapture, !isContinuousCaptureActive,
              let api else { return }
        isContinuousCaptureActive = true
        Task {
            do {
                try await api.startContinuousShooting()
                statusMessage = "Burst"
            } catch {
                isContinuousCaptureActive = false
                statusMessage = "Burst failed: \(Self.friendly(error))"
            }
        }
    }

    func stopPhotoBurst() {
        guard isContinuousCaptureActive, let api else { return }
        isContinuousCaptureActive = false
        Task {
            do {
                try await api.stopContinuousShooting()
                statusMessage = "Burst complete"
            } catch {
                statusMessage = "Could not stop burst: \(Self.friendly(error))"
            }
        }
    }

    func setSetting(_ id: CameraSettingID, value: String) {
        guard let api else { return }
        Task {
            do {
                try await api.setSetting(id, value: value)
                settings[id]?.current = value
                modeSettingValues[selectedMode, default: [:]][id] = value
                if id == .drive { modeDriveValues[selectedMode] = value }
                if id == .exposureMode || id == .drive {
                    try? await Task.sleep(for: .milliseconds(250))
                    await refreshCapabilitiesAndSettings(updateStatus: false)
                }
            } catch { statusMessage = "Setting failed: \(Self.friendly(error))" }
        }
    }

    func zoom(to target: Int) {
        guard let api, canZoom else { return }
        zoomTask?.cancel()
        zoomTask = Task {
            zoomCommandActive = true
            var activeDirection: String?
            defer {
                zoomCommandActive = false
                Task { @MainActor [weak self] in
                    if let activeDirection {
                        try? await api.zoom(direction: activeDirection, movement: "stop")
                    }
                    guard let self else { return }
                    self.zoomTask = nil
                }
            }
            do {
                if zoomPosition == nil {
                    applyCameraEvent(try await api.event(longPolling: false))
                }
                guard var position = zoomPosition else {
                    throw ControllerError.message("This camera does not report zoom position.")
                }
                let target = min(100, max(0, target))
                var stalledAttempts = 0
                var steps = 0
                while abs(position - target) > 2 {
                    try Task.checkCancellation()
                    guard steps < 24 else {
                        throw ControllerError.message("Zoom target did not converge.")
                    }
                    steps += 1
                    let direction = position < target ? "in" : "out"
                    let previous = position
                    let distance = abs(position - target)
                    if distance > 8 {
                        let pulse = min(0.45, max(0.12, Double(distance) * 0.0045))
                        let clock = ContinuousClock()
                        let deadline = clock.now.advanced(by: .seconds(pulse))
                        activeDirection = direction
                        try await api.zoom(direction: direction, movement: "start")
                        try await clock.sleep(until: deadline)
                        try await api.zoom(direction: direction, movement: "stop")
                        activeDirection = nil
                    } else {
                        try await api.zoom(direction: direction, movement: "1shot")
                    }
                    for _ in 0..<12 {
                        try Task.checkCancellation()
                        if let reported = zoomPosition, reported != previous {
                            position = reported
                            break
                        }
                        try await Task.sleep(for: .milliseconds(50))
                    }
                    if position == previous,
                       let event = try? await api.event(longPolling: false) {
                        applyCameraEvent(event)
                        position = zoomPosition ?? position
                    }
                    if position == previous {
                        stalledAttempts += 1
                        guard stalledAttempts < 3 else {
                            throw ControllerError.message("Zoom did not move after three attempts.")
                        }
                        try await Task.sleep(for: .milliseconds(150))
                    } else {
                        stalledAttempts = 0
                    }
                }
            } catch is CancellationError {
                return
            } catch {
                statusMessage = Self.friendly(error)
                if let event = try? await api.event(longPolling: false) {
                    applyCameraEvent(event)
                }
            }
        }
    }

    func refreshZoomState() {
        guard let api, phase == .connected else { return }
        Task {
            if let event = try? await api.event(longPolling: false) {
                applyCameraEvent(event)
            }
        }
    }

    func applyLiveViewQuality() {
        guard let api else { return }
        Task { try? await api.setLiveViewQuality(preferences.liveViewQuality, availableAPIs: availableAPIs) }
    }

    func reloadGallery() async { photos = await store.load() }
    func setAutoConnect(_ camera: PairedCamera, enabled: Bool) {
        guard let index = pairedCameras.firstIndex(where: { $0.id == camera.id }) else { return }
        pairedCameras[index].autoConnect = enabled; persistPairedCameras()
    }
    func delete(_ photo: SavedPhoto) { Task { try? await store.delete(photo); await reloadGallery() } }
    func replace(
        _ photo: SavedPhoto,
        data: Data,
        denoiseModel: AINRModel?,
        denoiseStrength: Double?
    ) {
        Task {
            try? await store.replace(
                photo,
                data: data,
                denoiseModel: denoiseModel,
                denoiseStrength: denoiseStrength
            )
            await reloadGallery()
        }
    }

    func prepareAINRModel() {
        let model = preferences.denoiseModel
        statusMessage = "Preparing \(model.rawValue)..."
        Task {
            let backend = await AINRService.shared.prepare(model: model)
            statusMessage = "\(model.rawValue) ready on \(backend)"
        }
    }

    func retryPhotoPublishing() {
        let retry = photos.filter { failedPhotoPublicationIDs.contains($0.id) }
        for photo in retry {
            Task { await publishToPhotos(photo) }
        }
    }

    func saveEditedCopy(
        from photo: SavedPhoto,
        data: Data,
        originalData: Data,
        lut: LUTSelection?,
        denoiseModel: AINRModel?,
        denoiseStrength: Double?,
        geometry: EditorGeometry?,
        fileExtension: String
    ) async throws -> SavedPhoto {
        let saved = try await store.save(
            data,
            originalData: originalData,
            kind: photo.kind,
            lut: lut,
            denoiseModel: denoiseModel,
            denoiseStrength: denoiseStrength,
            iso: photo.iso,
            geometry: geometry,
            derivedFromID: photo.id,
            originalFilename: photo.originalFilename ?? photo.url.lastPathComponent,
            filenameSuffix: "_edited",
            extension: fileExtension
        )
        photos.insert(saved, at: 0)
        Task { await publishToPhotos(saved) }
        return saved
    }

    func resumeLiveView() {
        guard phase == .connected, !isLiveViewRunning else { return }
        if let lastLiveViewURL {
            liveView.start(url: lastLiveViewURL)
            isLiveViewRunning = true
            statusMessage = "Live view resumed"
            armLiveViewTimeout()
            return
        }
        guard let api else { return }
        Task {
            do {
                let url = try await api.startLiveView(
                    quality: preferences.liveViewQuality,
                    availableAPIs: availableAPIs
                )
                lastLiveViewURL = url
                liveView.start(url: url)
                isLiveViewRunning = true
                armLiveViewTimeout()
            } catch {
                statusMessage = "Live view failed: \(Self.friendly(error))"
            }
        }
    }

    private func startLiveND() {
        guard let api else { return }
        let target = 1 << preferences.liveNDStops
        stackData.removeAll(); stackMode = .liveND; stackFrameCount = 0; stackTargetCount = target
        liveNDCaptureURLCount = 0
        liveNDCompletionTask?.cancel()
        acceptingComputationalFrames = true
        closedLiveNDSession = false
        statusMessage = "Live ND: capturing \(target) frames"
        Task {
            do {
                stackProcessingSnapshot = processingSnapshot(
                    url: URL(fileURLWithPath: "Live ND.jpg"),
                    mode: .liveND
                )
                try await persistNewComputationalSession()
                try await api.setPostviewSize(preferences.downloadQuality, availableAPIs: availableAPIs)
                if canContinuousCapture {
                    await configureDrive(for: .liveND)
                    let duration = liveNDBurstDuration(
                        shutterValue: settings[.shutterSpeed]?.current ?? "1/60",
                        requiredFrames: target
                    )
                    let clock = ContinuousClock()
                    let deadline = clock.now.advanced(by: .seconds(duration))
                    do {
                        try await api.startContinuousShooting()
                        try await clock.sleep(until: deadline)
                        try await api.stopContinuousShooting()
                    } catch {
                        try? await api.stopContinuousShooting()
                        throw error
                    }
                    scheduleLiveNDCompletion()
                } else {
                    await ensureSingleDrive()
                    for _ in 0..<target { enqueue(try await api.takePicture()) }
                }
            } catch { statusMessage = "Live ND failed: \(Self.friendly(error))" }
        }
    }

    private func startEventMonitor() {
        eventTask?.cancel()
        guard let eventAPI else { return }
        eventTask = Task {
            while !Task.isCancelled {
                do {
                    let event = try await eventAPI.event(longPolling: true)
                    applyCameraEvent(event)
                } catch is CancellationError { return }
                catch {
                    guard !Task.isCancelled else { return }
                    Self.logger.debug("Event polling interrupted: \(error.localizedDescription, privacy: .public)")
                }
            }
        }
    }

    private func startConnectionMonitor() {
        connectionMonitorTask?.cancel()
        guard let api else { return }
        connectionMonitorTask = Task {
            var consecutiveFailures = 0
            while !Task.isCancelled, phase == .connected {
                try? await Task.sleep(for: .seconds(15))
                guard !Task.isCancelled, phase == .connected else { return }
                do {
                    _ = try await api.availableAPIs(timeout: 5)
                    consecutiveFailures = 0
                } catch is CameraRPCError {
                    // The camera replied. Busy-state API errors are not a disconnect.
                    consecutiveFailures = 0
                } catch {
                    consecutiveFailures += 1
                    if consecutiveFailures >= 2 {
                        connectionWasLost(error)
                        return
                    }
                }
            }
        }
    }

    private func startSettingsRefresh() {
        settingsRefreshTask?.cancel()
        settingsRefreshTask = Task {
            while !Task.isCancelled, phase == .connected {
                try? await Task.sleep(for: .seconds(15))
                guard !Task.isCancelled, phase == .connected else { return }
                guard !zoomCommandActive else { continue }
                await refreshCapabilitiesAndSettings(updateStatus: false)
            }
        }
    }

    private func applyCameraEvent(
        _ event: CameraEventSnapshot,
        enqueueURLs: Bool = true
    ) {
        if let status = event.status { cameraStatus = status }
        let availableAPIsChanged = event.availableAPIs.map { $0 != availableAPIs } ?? false
        if let apis = event.availableAPIs { availableAPIs = apis }
        let previousMode = settings[.exposureMode]?.current
        let previousDrive = settings[.drive]?.current
        for (id, value) in event.settingValues {
            if settings[id] != nil {
                settings[id]?.current = value
            } else {
                settings[id] = CameraSetting(id: id, current: value, options: [value], writable: false)
            }
        }
        if let position = event.zoomPosition { zoomPosition = min(100, max(0, position)) }
        if let setting = event.zoomSetting { zoomSetting = setting }
        if let count = event.zoomBoxCount { zoomBoxCount = count }
        if let index = event.zoomBoxIndex { zoomBoxIndex = index }
        if !event.urls.isEmpty {
            Self.logger.info("Camera event supplied \(event.urls.count) downloadable image URL(s)")
        }
        if enqueueURLs { event.urls.forEach(enqueue) }

        let modeChanged = event.settingValues[.exposureMode].map { $0 != previousMode } ?? false
        let driveChanged = event.settingValues[.drive].map { $0 != previousDrive } ?? false
        if availableAPIs.contains("startRecMode") {
            recoverRemoteModeAfterPhysicalCapture()
            return
        }
        if modeChanged || driveChanged || availableAPIsChanged {
            if capabilitiesRefreshTask == nil {
                capabilitiesRefreshTask = Task {
                    await refreshCapabilitiesAndSettings(updateStatus: false)
                    capabilitiesRefreshTask = nil
                }
            }
        }
    }

    private func recoverRemoteModeAfterPhysicalCapture() {
        guard remoteModeRecoveryTask == nil, let api, phase == .connected else { return }
        remoteModeRecoveryTask = Task {
            defer { remoteModeRecoveryTask = nil }
            do {
                Self.logger.info("Camera left remote mode; restoring it to recover pending capture")
                availableAPIs = try await api.startRemoteModeIfNeeded()
                if let pendingEvent = try? await eventAPI?.event(longPolling: false) {
                    applyCameraEvent(pendingEvent)
                }
                statusMessage = pendingDownloads > 0
                    ? "Downloading captured image..."
                    : Self.sessionStatus(controlCount: settings.count)
            } catch {
                Self.logger.debug(
                    "Remote-mode recovery deferred: \(error.localizedDescription, privacy: .public)"
                )
            }
        }
    }

    private func connectionWasLost(_ error: Error) {
        guard phase == .connected else { return }
        disconnect()
        statusMessage = "Camera disconnected. Rejoin its Wi-Fi to reconnect."
        Self.logger.info("Camera connection ended: \(error.localizedDescription, privacy: .public)")
    }

    private func refreshCapabilitiesAndSettings(updateStatus: Bool = true) async {
        guard let api, phase == .connected else { return }
        do {
            let refreshedAPIs = try await api.availableAPIs()
            availableAPIs.formUnion(refreshedAPIs)
            let refreshedSettings = await api.settings(availableAPIs: availableAPIs)
            settings.merge(refreshedSettings) { _, new in new }
            if updateStatus {
                statusMessage = Self.sessionStatus(controlCount: settings.count)
            }
        } catch {
            Self.logger.debug("Settings refresh failed: \(error.localizedDescription, privacy: .public)")
        }
    }

    private func enqueue(_ url: URL) {
        guard knownRemoteURLs.insert(url.absoluteString).inserted else { return }
        // A physical shutter event belongs to a computational capture only after
        // that capture's explicit session has been armed.
        let mode = acceptingComputationalFrames ? (stackMode ?? .photo) : .photo
        if mode == .liveND { liveNDCaptureURLCount += 1 }
        if let snapshot = stackProcessingSnapshot, mode != .photo {
            downloadQueue.append(snapshot.withURL(url))
        } else {
            downloadQueue.append(processingSnapshot(url: url, mode: mode))
        }
        pendingDownloads = downloadQueue.count
        guard queueTask == nil else { return }
        queueTask = Task {
            while !downloadQueue.isEmpty, !Task.isCancelled {
                let next = downloadQueue.removeFirst(); pendingDownloads = downloadQueue.count + 1
                do {
                    try await importRemotePhoto(next)
                    downloadRetryCycles.removeValue(forKey: next.url.absoluteString)
                } catch {
                    let key = next.url.absoluteString
                    let cycle = downloadRetryCycles[key, default: 0] + 1
                    downloadRetryCycles[key] = cycle
                    if cycle < 3, phase == .connected {
                        statusMessage = "Download paused. Retrying shortly..."
                        downloadQueue.append(next)
                        try? await Task.sleep(for: .seconds(2))
                    } else {
                        knownRemoteURLs.remove(key)
                        downloadRetryCycles.removeValue(forKey: key)
                        statusMessage = "Download failed: \(Self.friendly(error))"
                    }
                }
                pendingDownloads = downloadQueue.count
            }
            queueTask = nil; downloadProgress = nil
        }
    }

    private static func sessionStatus(controlCount: Int) -> String {
        controlCount == 1
            ? "Remote session active • 1 control"
            : "Remote session active • \(controlCount) controls"
    }

    private func importRemotePhoto(_ capture: QueuedRemoteCapture) async throws {
        statusMessage = "Downloading image..."
        Self.logger.info("Starting download for \(capture.url.absoluteString, privacy: .public)")
        var lastError: Error?
        for attempt in 1...5 {
            do {
                let data = try await download(capture)
                if acceptingComputationalFrames,
                   let mode = stackMode,
                   mode == capture.mode {
                    if stackProcessingSnapshot == nil { stackProcessingSnapshot = capture }
                    if mode == .liveND, closedLiveNDSession { return }
                    if mode == .panorama {
                        let candidate = stackData + [data]
                        do {
                            let preview = try await Task.detached {
                                try ComputationalCapture.panoramaPreview(candidate)
                            }.value
                            let restored = try await computationalSessionStore.append(
                                data,
                                remoteURL: capture.url
                            )
                            stackData = restored.frames
                            stackFrameCount = stackData.count
                            computationalPreview = ImageProcessor.shared.preview(preview)
                            statusMessage = "\(stackFrameCount) panorama frame\(stackFrameCount == 1 ? "" : "s") ready"
                        } catch {
                            statusMessage = "Frame not added: \(Self.friendly(error))"
                        }
                        downloadProgress = nil
                        return
                    }
                    let restored = try await computationalSessionStore.append(
                        data,
                        remoteURL: capture.url
                    )
                    stackData = restored.frames
                    stackFrameCount = stackData.count
                    if mode == .liveND, stackData.count >= stackTargetCount { await finalizeStack(mode: .liveND) }
                    else if mode == .composite || mode == .panorama { await updateStackPreview(mode: mode) }
                } else if capture.mode != .photo {
                    statusMessage = "Ignored a source frame from a completed \(capture.mode.rawValue) capture"
                } else {
                    let iso = capture.iso
                    let selection = capture.lut
                    let shouldBake = selection.identifier != "Original"
                    let shouldDenoise = capture.autoDenoise == .always ||
                        (capture.autoDenoise == .isoThreshold &&
                            (iso ?? 0) >= capture.denoiseThreshold)
                    let restoredResult = shouldDenoise
                        ? await automaticDenoise(
                            data,
                            sourceName: capture.url.lastPathComponent,
                            model: capture.denoiseModel
                        )
                        : (data, nil)
                    guard let restored = CIImage(
                        data: restoredResult.0,
                        options: [.applyOrientationProperty: true]
                    ) else { throw ControllerError.message("Camera returned invalid image data.") }
                    let processed = shouldBake ? ImageProcessor.shared.applyLUT(restored, selection: selection, library: lutLibrary) : restored
                    let location = capture.geotagging ? await locationProvider.location() : nil
                    guard let encoded = ImageProcessor.shared.encode(
                        processed,
                        format: capture.outputFormat,
                        location: location,
                        sourceData: data
                    ) else { throw ProcessingError.invalidImage }
                    let photo = try await store.save(encoded.0, originalData: (shouldBake || restoredResult.1 != nil) ? data : nil,
                                                     lut: shouldBake ? selection : nil,
                                                     denoiseModel: restoredResult.1,
                                                     denoiseStrength: restoredResult.1 == nil ? nil : 1,
                                                     iso: iso,
                                                     originalFilename: capture.url.lastPathComponent,
                                                     extension: encoded.1)
                    photos.insert(photo, at: 0)
                    Task { await publishToPhotos(photo) }
                    await refreshCapabilitiesAndSettings(updateStatus: false)
                    statusMessage = "Saved \(photo.id)"
                }
                Self.logger.info("Completed download for \(capture.url.absoluteString, privacy: .public)")
                downloadProgress = nil; return
            } catch {
                lastError = error; downloadProgress = nil
                if attempt < 5 {
                    statusMessage = "Download interrupted. Retrying \(attempt + 1)/5..."
                    try? await Task.sleep(for: .seconds(min(attempt * 2, 6)))
                }
            }
        }
        throw lastError ?? URLError(.cannotLoadFromNetwork)
    }

    private func updateStackPreview(mode: CaptureMode) async {
        statusMessage = "Aligning \(stackData.count) frames..."
        let sources = stackData
        stackPreviewGeneration += 1
        let generation = stackPreviewGeneration
        do {
            let result = try await Task.detached {
                mode == .composite
                    ? try ComputationalCapture.composite(sources)
                    : try ComputationalCapture.panoramaPreview(sources)
            }.value
            guard generation == stackPreviewGeneration else { return }
            computationalPreview = ImageProcessor.shared.preview(result)
            statusMessage = "\(stackData.count) frames ready"
        } catch { statusMessage = Self.friendly(error) }
    }

    private func finalizeStack(mode: CaptureMode) async {
        let sources = stackData
        guard !sources.isEmpty else { return }
        let processing = stackProcessingSnapshot ?? processingSnapshot(
            url: URL(fileURLWithPath: "\(mode.rawValue).jpg"),
            mode: mode
        )
        if mode != .panorama {
            acceptingComputationalFrames = false
        }
        if mode == .liveND { closedLiveNDSession = true }
        isStackRendering = true
        defer { isStackRendering = false }
        statusMessage = "Rendering \(mode.rawValue)..."
        do {
            let result: CIImage
            var usedPanoramaPreview = false
            do {
                result = try await Task.detached {
                    switch mode {
                    case .liveND: try ComputationalCapture.liveND(sources)
                    case .composite: try ComputationalCapture.composite(sources)
                    case .panorama: try ComputationalCapture.panorama(sources)
                    case .photo: throw ProcessingError.noFrames
                    }
                }.value
            } catch {
                guard mode == .panorama,
                      let preview = computationalPreview,
                      let previewImage = CIImage(image: preview) else {
                    throw error
                }
                result = previewImage
                usedPanoramaPreview = true
            }
            let selection = processing.lut
            guard let original = ImageProcessor.shared.jpeg(result) else {
                throw ProcessingError.invalidImage
            }
            let iso = processing.iso
            let shouldDenoise = processing.autoDenoise == .always ||
                (processing.autoDenoise == .isoThreshold &&
                    (iso ?? 0) >= processing.denoiseThreshold)
            let restoredResult = shouldDenoise
                ? await automaticDenoise(
                    original,
                    sourceName: "\(mode.rawValue).jpg",
                    model: processing.denoiseModel
                )
                : (original, nil)
            guard let restored = CIImage(data: restoredResult.0) else {
                throw ProcessingError.invalidImage
            }
            let shouldBake = selection.identifier != "Original"
            let baked = shouldBake
                ? ImageProcessor.shared.applyLUT(restored, selection: selection, library: lutLibrary)
                : restored
            let location = processing.geotagging ? await locationProvider.location() : nil
            guard let display = ImageProcessor.shared.encode(
                baked,
                format: processing.outputFormat,
                location: location,
                sourceData: original
            ) else { throw ProcessingError.invalidImage }
            let photo = try await store.save(
                display.0,
                originalData: (shouldBake || restoredResult.1 != nil) ? original : nil,
                                             kind: mode, sourceData: sources,
                lut: shouldBake ? selection : nil,
                denoiseModel: restoredResult.1,
                denoiseStrength: restoredResult.1 == nil ? nil : 1,
                iso: iso,
                extension: display.1
            )
            photos.insert(photo, at: 0)
            Task { await publishToPhotos(photo) }
            statusMessage = usedPanoramaPreview
                ? "Saved Panorama Preview; full render exceeded safe resources"
                : "Saved \(mode.rawValue)"
            try? await computationalSessionStore.clear()
            resetStack()
        } catch { statusMessage = "Render failed: \(Self.friendly(error))" }
    }

    private func download(_ capture: QueuedRemoteCapture) async throws -> Data {
        let timeout: TimeInterval = capture.quality == .original ? 300 : 90
        return try await downloadWorker.download(capture.url, timeout: timeout) { [weak self] fraction in
            Task { @MainActor [weak self] in self?.downloadProgress = fraction }
        }
    }

    private func automaticDenoise(
        _ data: Data,
        sourceName: String,
        model requestedModel: AINRModel? = nil
    ) async -> (Data, AINRModel?) {
        do {
            let model = requestedModel ?? preferences.denoiseModel
            let output = try await AINRService.shared.process(
                data: data,
                sourceName: sourceName,
                model: model
            ) { [weak self] progress in
                Task { @MainActor [weak self] in self?.ainrProgress = progress }
            }
            ainrProgress = nil
            return (output, model)
        } catch is CancellationError {
            ainrProgress = nil
            return (data, nil)
        } catch {
            ainrProgress = nil
            statusMessage = "AINR failed; saved the untouched image. \(Self.friendly(error))"
            return (data, nil)
        }
    }

    private func configureDrive(for mode: CaptureMode) async {
        guard !Task.isCancelled, selectedMode == mode else { return }
        if mode == .liveND {
            rememberLiveNDDriveOverride()
            if let burst = settings[.drive]?.options.first(where: {
                !$0.localizedCaseInsensitiveContains("single")
            }) {
                if settings[.drive]?.current != burst {
                    await applySetting(.drive, value: burst)
                    try? await Task.sleep(for: .milliseconds(250))
                    guard !Task.isCancelled, selectedMode == mode else { return }
                    await refreshCapabilitiesAndSettings(updateStatus: false)
                }
            }
            guard !Task.isCancelled, selectedMode == mode else { return }
            if let fastest = highestBurstSpeed(in: settings[.burstSpeed]?.options ?? []) {
                await applySetting(.burstSpeed, value: fastest)
            }
        } else if mode == .composite || mode == .panorama {
            await ensureSingleDrive()
        } else if mode == .photo, liveNDDriveOverrideActive, let previous = liveNDPreviousDrive {
            await applySetting(.drive, value: previous)
            clearLiveNDDriveOverride()
        } else if let saved = modeDriveValues[mode] {
            await applySetting(.drive, value: saved)
        }
    }

    private var liveNDDriveOverrideActive: Bool {
        UserDefaults.standard.bool(forKey: Self.liveNDDriveOverrideActiveKey)
    }

    private var liveNDPreviousDrive: String? {
        UserDefaults.standard.string(forKey: Self.liveNDPreviousDriveKey)
    }

    private func rememberLiveNDDriveOverride() {
        guard !liveNDDriveOverrideActive, let drive = settings[.drive]?.current else { return }
        UserDefaults.standard.set(drive, forKey: Self.liveNDPreviousDriveKey)
        UserDefaults.standard.set(true, forKey: Self.liveNDDriveOverrideActiveKey)
    }

    private func clearLiveNDDriveOverride() {
        UserDefaults.standard.removeObject(forKey: Self.liveNDPreviousDriveKey)
        UserDefaults.standard.set(false, forKey: Self.liveNDDriveOverrideActiveKey)
    }

    private func ensureSingleDrive() async {
        guard let single = settings[.drive]?.options.first(where: { $0.localizedCaseInsensitiveContains("single") }) else { return }
        if settings[.drive]?.current != single {
            try? await api?.setSetting(.drive, value: single); settings[.drive]?.current = single
        }
    }

    private func preparePanoramaDrive() async throws {
        guard let api else {
            throw ControllerError.message("Connect to the camera before starting panorama.")
        }
        guard let drive = settings[.drive] else { return }
        guard let single = drive.options.first(where: {
            $0.localizedCaseInsensitiveContains("single")
        }) else {
            throw ControllerError.message("Set the camera drive mode to Single before starting panorama.")
        }
        if drive.current != single {
            try await api.setSetting(.drive, value: single)
            let refreshed = await api.settings(availableAPIs: availableAPIs)
            settings.merge(refreshed) { _, new in new }
        }
        guard settings[.drive]?.current.localizedCaseInsensitiveContains("single") == true else {
            throw ControllerError.message("Camera did not enter Single drive mode for panorama.")
        }
    }

    private func applySetting(_ id: CameraSettingID, value: String) async {
        do {
            try await api?.setSetting(id, value: value)
            settings[id]?.current = value
            modeSettingValues[selectedMode, default: [:]][id] = value
            if id == .drive { modeDriveValues[selectedMode] = value }
        } catch {
            statusMessage = "Setting failed: \(Self.friendly(error))"
        }
    }

    private func armLiveViewTimeout() {
        liveViewTimer?.cancel()
        guard preferences.liveViewTimeoutMinutes > 0 else { return }
        liveViewTimer = Task {
            try? await Task.sleep(for: .seconds(preferences.liveViewTimeoutMinutes * 60))
            guard !Task.isCancelled else { return }
            liveView.stop(); liveViewImage = nil; isLiveViewRunning = false
            statusMessage = "Live view paused to save camera battery."
        }
    }

    private func publishToPhotos(_ photo: SavedPhoto) async {
        do {
            let identifier = try await PhotoLibraryPublisher.shared.publish(photo)
            try await store.updatePhotoLibraryIdentifier(identifier, for: photo.id)
            if let index = photos.firstIndex(where: { $0.id == photo.id }) {
                photos[index].photoLibraryIdentifier = identifier
            }
            failedPhotoPublicationIDs.remove(photo.id)
        } catch {
            failedPhotoPublicationIDs.insert(photo.id)
            statusMessage = "Saved in Alpha Capture Lab. Photos: \(Self.friendly(error))"
        }
    }

    private func processingSnapshot(url: URL, mode: CaptureMode) -> QueuedRemoteCapture {
        QueuedRemoteCapture(
            url: url,
            mode: mode,
            quality: preferences.downloadQuality,
            outputFormat: preferences.outputFormat,
            geotagging: preferences.geotagging,
            lut: preferences.lutSelection,
            autoDenoise: preferences.autoDenoise,
            denoiseThreshold: preferences.denoiseISOThreshold,
            denoiseModel: preferences.denoiseModel,
            iso: Int(settings[.iso]?.current ?? "")
        )
    }

    private func persistNewComputationalSession() async throws {
        guard let capture = stackProcessingSnapshot, let mode = stackMode else {
            throw ComputationalSessionError.missingSession
        }
        try await computationalSessionStore.begin(
            ComputationalSessionState(
                id: UUID(),
                mode: mode,
                targetCount: stackTargetCount,
                startedAt: Date(),
                quality: capture.quality,
                outputFormat: capture.outputFormat,
                geotagging: capture.geotagging,
                lut: capture.lut,
                autoDenoise: capture.autoDenoise,
                denoiseThreshold: capture.denoiseThreshold,
                denoiseModel: capture.denoiseModel,
                iso: capture.iso,
                remoteURLs: [],
                frameNames: []
            )
        )
    }

    private func restoreComputationalSession() async {
        do {
            guard let restored = try await computationalSessionStore.load() else { return }
            let state = restored.state
            selectedMode = state.mode
            stackMode = state.mode
            stackTargetCount = state.targetCount
            stackData = restored.frames
            stackFrameCount = restored.frames.count
            acceptingComputationalFrames = true
            closedLiveNDSession = false
            liveNDCaptureURLCount = state.remoteURLs.count
            knownRemoteURLs.formUnion(state.remoteURLs)
            stackProcessingSnapshot = QueuedRemoteCapture(
                url: URL(fileURLWithPath: "\(state.mode.rawValue).jpg"),
                mode: state.mode,
                quality: state.quality,
                outputFormat: state.outputFormat,
                geotagging: state.geotagging,
                lut: state.lut,
                autoDenoise: state.autoDenoise,
                denoiseThreshold: state.denoiseThreshold,
                denoiseModel: state.denoiseModel,
                iso: state.iso
            )
            statusMessage = restored.frames.isEmpty
                ? "Waiting to resume \(state.mode.rawValue)"
                : "Restored \(restored.frames.count) \(state.mode.rawValue) source frames"
            if state.mode == .liveND,
               state.targetCount > 0,
               restored.frames.count >= state.targetCount {
                await finalizeStack(mode: .liveND)
            } else if state.mode == .liveND {
                scheduleLiveNDCompletion()
            } else if !restored.frames.isEmpty,
                      state.mode == .composite || state.mode == .panorama {
                await updateStackPreview(mode: state.mode)
            }
        } catch {
            statusMessage = "Could not restore computational capture: \(Self.friendly(error))"
        }
    }

    private func scheduleLiveNDCompletion() {
        liveNDCompletionTask?.cancel()
        guard acceptingComputationalFrames, stackMode == .liveND, stackTargetCount > 0 else {
            return
        }
        liveNDCompletionTask = Task {
            defer { liveNDCompletionTask = nil }
            for attempt in 1...3 {
                do {
                    try await Task.sleep(for: .seconds(5))
                    try Task.checkCancellation()
                    guard acceptingComputationalFrames,
                          stackMode == .liveND,
                          !closedLiveNDSession,
                          let api else { return }
                    let missing = max(0, stackTargetCount - liveNDCaptureURLCount)
                    guard missing > 0 else { return }
                    statusMessage = "Live ND: collecting \(missing) missing frame\(missing == 1 ? "" : "s")"
                    if canContinuousCapture {
                        let duration = liveNDBurstDuration(
                            shutterValue: settings[.shutterSpeed]?.current ?? "1/60",
                            requiredFrames: missing
                        )
                        let clock = ContinuousClock()
                        let deadline = clock.now.advanced(by: .seconds(duration))
                        do {
                            try await api.startContinuousShooting()
                            try await clock.sleep(until: deadline)
                            try await api.stopContinuousShooting()
                        } catch {
                            try? await api.stopContinuousShooting()
                            throw error
                        }
                    } else {
                        await ensureSingleDrive()
                        for _ in 0..<missing {
                            enqueue(try await api.takePicture())
                        }
                    }
                } catch is CancellationError {
                    return
                } catch {
                    Self.logger.debug(
                        "Live ND completion attempt \(attempt) failed: \(error.localizedDescription, privacy: .public)"
                    )
                }
            }
            if acceptingComputationalFrames, stackMode == .liveND {
                let missing = max(0, stackTargetCount - liveNDCaptureURLCount)
                if missing > 0 {
                    statusMessage = "Live ND is waiting for \(missing) missing frame\(missing == 1 ? "" : "s")"
                }
            }
        }
    }

    private func rememberConnectedCamera() {
        let camera = PairedCamera(host: cameraHost, name: "Camera \(cameraHost)", autoConnect: pairedCameras.first { $0.host == cameraHost }?.autoConnect ?? false, lastConnected: Date())
        pairedCameras.removeAll { $0.host == cameraHost }; pairedCameras.insert(camera, at: 0); persistPairedCameras()
    }

    private func persistPairedCameras() { UserDefaults.standard.set(try? JSONEncoder().encode(pairedCameras), forKey: "pairedCameras") }

    private func startAutoConnectMonitor() {
        autoConnectTask?.cancel()
        autoConnectTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                if self.phase == .disconnected || { if case .failed = self.phase { true } else { false } }() {
                    for camera in self.pairedCameras where camera.autoConnect {
                        if await Self.cameraReachable(host: camera.host) {
                            self.cameraHost = camera.host; self.connect(); break
                        }
                    }
                }
                try? await Task.sleep(for: .seconds(12))
            }
        }
    }

    private nonisolated static func cameraReachable(host: String) async -> Bool {
        guard let url = URL(string: "http://\(host):64321/scalarwebapi_dd.xml") else { return false }
        var request = URLRequest(url: url); request.timeoutInterval = 2
        let configuration = URLSessionConfiguration.ephemeral; configuration.waitsForConnectivity = false
        return (try? await URLSession(configuration: configuration).data(for: request))
            .map { ($0.1 as? HTTPURLResponse)?.statusCode == 200 } ?? false
    }

    private static func friendly(_ error: Error) -> String {
        (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
    }
    private static let hostKey = "camera_host"
    private static let liveNDDriveOverrideActiveKey = "live_nd_drive_override_active"
    private static let liveNDPreviousDriveKey = "live_nd_previous_drive"
}

private struct QueuedRemoteCapture {
    let url: URL
    let mode: CaptureMode
    let quality: DownloadQuality
    let outputFormat: OutputFormat
    let geotagging: Bool
    let lut: LUTSelection
    let autoDenoise: AutoDenoiseMode
    let denoiseThreshold: Int
    let denoiseModel: AINRModel
    let iso: Int?

    func withURL(_ url: URL) -> QueuedRemoteCapture {
        QueuedRemoteCapture(
            url: url,
            mode: mode,
            quality: quality,
            outputFormat: outputFormat,
            geotagging: geotagging,
            lut: lut,
            autoDenoise: autoDenoise,
            denoiseThreshold: denoiseThreshold,
            denoiseModel: denoiseModel,
            iso: iso
        )
    }
}

private actor RemoteDownloadWorker {
    func download(
        _ url: URL,
        timeout: TimeInterval,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> Data {
        var request = URLRequest(url: url)
        request.timeoutInterval = timeout
        let configuration = URLSessionConfiguration.ephemeral
        configuration.waitsForConnectivity = false
        configuration.timeoutIntervalForResource = timeout
        let (bytes, response) = try await URLSession(configuration: configuration).bytes(for: request)
        guard let http = response as? HTTPURLResponse,
              (200..<300).contains(http.statusCode) else {
            throw URLError(.badServerResponse)
        }
        let expected = response.expectedContentLength
        var data = Data()
        if expected > 0 { data.reserveCapacity(Int(expected)) }
        var received: Int64 = 0
        var nextUpdate: Int64 = 65_536
        for try await byte in bytes {
            try Task.checkCancellation()
            data.append(byte)
            received += 1
            if received >= nextUpdate {
                if expected > 0 {
                    progress(min(1, Double(received) / Double(expected)))
                }
                nextUpdate += 65_536
            }
        }
        progress(1)
        return data
    }
}

private enum ControllerError: LocalizedError {
    case message(String)
    var errorDescription: String? { if case .message(let value) = self { value } else { nil } }
}
