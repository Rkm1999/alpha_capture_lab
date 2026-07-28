import Combine
import CoreImage
import Foundation
import OSLog
import UIKit

@MainActor
final class CameraController: ObservableObject {
    private static let logger = Logger(subsystem: "com.ryu.remotecapture.ios", category: "camera-session")
    @Published var phase: ConnectionPhase = .disconnected
    @Published var liveViewImage: UIImage?
    @Published var computationalPreview: UIImage?
    @Published var photos: [SavedPhoto] = []
    @Published var selectedMode: CaptureMode = .photo
    @Published var settings: [CameraSettingID: CameraSetting] = [:]
    @Published var downloadProgress: Double?
    @Published var pendingDownloads = 0
    @Published var statusMessage = "Join the camera Wi-Fi, then connect."
    @Published var zoomPosition: Int?
    @Published var zoomSetting: String?
    @Published var cameraStatus: String?
    @Published var stackFrameCount = 0
    @Published var stackTargetCount = 0
    @Published var pairedCameras: [PairedCamera] = []
    @Published var cameraHost: String { didSet { UserDefaults.standard.set(cameraHost, forKey: Self.hostKey) } }

    var preferences = AppPreferences()
    let lutLibrary = LUTLibrary()
    private let store = PhotoStore()
    private let locationProvider = LocationProvider()
    private var api: SonyCameraAPI?
    private var availableAPIs = Set<String>()
    private var eventTask: Task<Void, Never>?
    private var queueTask: Task<Void, Never>?
    private let liveView = LiveViewStream()
    private var knownRemoteURLs = Set<String>()
    private var downloadQueue: [URL] = []
    private var stackData: [Data] = []
    private var stackMode: CaptureMode?
    private var modeDriveValues: [CaptureMode: String] = [:]
    private var modeSettingValues: [CaptureMode: [CameraSettingID: String]] = [:]
    private var liveViewTimer: Task<Void, Never>?
    private var autoConnectTask: Task<Void, Never>?
    private var observations = Set<AnyCancellable>()

    var canContinuousCapture: Bool { availableAPIs.contains("startContShooting") && availableAPIs.contains("stopContShooting") }
    var canZoom: Bool { availableAPIs.contains("actZoom") }
    var selectedLUTName: String { preferences.lutSelection.identifier }

    init() {
        cameraHost = UserDefaults.standard.string(forKey: Self.hostKey) ?? "192.168.122.1"
        pairedCameras = (try? UserDefaults.standard.data(forKey: "pairedCameras").flatMap { try JSONDecoder().decode([PairedCamera].self, from: $0) }) ?? []
        preferences.objectWillChange.sink { [weak self] _ in self?.objectWillChange.send() }.store(in: &observations)
        lutLibrary.objectWillChange.sink { [weak self] _ in self?.objectWillChange.send() }.store(in: &observations)
        liveView.onFrame = { [weak self] data in
            Task { @MainActor [weak self] in
                guard let self, let source = CIImage(data: data) else { return }
                self.liveViewImage = ImageProcessor.shared.preview(
                    ImageProcessor.shared.applyLUT(source, selection: self.preferences.lutSelection, library: self.lutLibrary),
                    maxDimension: 1000
                )
            }
        }
        liveView.onFailure = { [weak self] error in
            Task { @MainActor [weak self] in
                guard let self, self.phase == .connected else { return }
                self.statusMessage = "Live view stopped: \(error.localizedDescription)"
            }
        }
        Task { await reloadGallery() }
        startAutoConnectMonitor()
    }

    deinit { eventTask?.cancel(); queueTask?.cancel(); liveViewTimer?.cancel(); autoConnectTask?.cancel(); liveView.stop() }

    func connect() {
        guard phase != .connecting else { return }
        phase = .connecting
        statusMessage = "Contacting camera at \(cameraHost)..."
        Task {
            do {
                let endpoint = try await SonyCameraDiscovery.cameraEndpoint(host: cameraHost)
                let api = SonyCameraAPI(endpoint: endpoint)
                let apis = try await api.startRemoteModeIfNeeded()
                guard apis.contains("startLiveview") || apis.contains("startLiveviewWithSize") else {
                    throw ControllerError.message("This camera mode does not expose live view.")
                }
                try await api.setPostviewSize(preferences.downloadQuality, availableAPIs: apis)
                let loadedSettings = await api.settings(availableAPIs: apis)
                let liveURL = try await api.startLiveView(availableAPIs: apis)
                self.api = api
                availableAPIs = apis
                settings = loadedSettings
                phase = .connected
                rememberConnectedCamera()
                statusMessage = "Remote session active"
                liveView.start(url: liveURL)
                armLiveViewTimeout()
                startEventMonitor()
            } catch {
                phase = .failed(Self.friendly(error))
                statusMessage = "Check that the iPhone is on the camera Wi-Fi."
            }
        }
    }

    func disconnect() {
        eventTask?.cancel(); eventTask = nil
        queueTask?.cancel(); queueTask = nil
        liveViewTimer?.cancel(); liveView.stop()
        liveViewImage = nil
        let oldAPI = api
        api = nil; availableAPIs = []; knownRemoteURLs.removeAll(); downloadQueue.removeAll()
        phase = .disconnected; statusMessage = "Disconnected"
        Task { await oldAPI?.stopLiveView() }
    }

    func selectMode(_ mode: CaptureMode) {
        guard mode != selectedMode else { return }
        modeSettingValues[selectedMode] = settings.mapValues(\.current)
        if let drive = settings[.drive]?.current { modeDriveValues[selectedMode] = drive }
        selectedMode = mode
        resetStack()
        guard phase == .connected else { return }
        Task {
            if let saved = modeSettingValues[mode] {
                let order: [CameraSettingID] = [.exposureMode, .aperture, .shutterSpeed, .iso, .exposureCompensation, .burstSpeed, .drive]
                for id in order where saved[id] != nil && settings[id]?.options.contains(saved[id]!) == true {
                    try? await api?.setSetting(id, value: saved[id]!); settings[id]?.current = saved[id]!
                }
            }
            await configureDrive(for: mode)
        }
    }

    func capture() {
        guard let api, phase == .connected else { return }
        switch selectedMode {
        case .photo, .composite, .panorama:
            statusMessage = selectedMode == .photo ? "Capturing..." : "Capturing frame..."
            Task {
                do {
                    if selectedMode != .photo { await ensureSingleDrive() }
                    try await api.setPostviewSize(preferences.downloadQuality, availableAPIs: availableAPIs)
                    enqueue(try await api.takePicture())
                } catch { statusMessage = "Capture failed: \(Self.friendly(error))" }
            }
        case .liveND: startLiveND()
        }
    }

    func finishStack() {
        guard selectedMode == .composite || selectedMode == .panorama, !stackData.isEmpty else { return }
        Task { await finalizeStack(mode: selectedMode) }
    }

    func resetStack() {
        stackData.removeAll(); stackMode = nil; stackFrameCount = 0; stackTargetCount = 0; computationalPreview = nil
    }

    func setSetting(_ id: CameraSettingID, value: String) {
        guard let api else { return }
        Task {
            do {
                try await api.setSetting(id, value: value)
                settings[id]?.current = value
                modeSettingValues[selectedMode, default: [:]][id] = value
                if id == .drive { modeDriveValues[selectedMode] = value }
            } catch { statusMessage = "Setting failed: \(Self.friendly(error))" }
        }
    }

    func zoom(to target: Int) {
        guard let api, canZoom else { return }
        Task {
            for attempt in 0..<3 {
                guard let current = zoomPosition, abs(current - target) > 2 else { return }
                let direction = current < target ? "in" : "out"
                try? await api.zoom(direction: direction, movement: "start")
                let deadline = Date().addingTimeInterval(8)
                while Date() < deadline, let position = zoomPosition,
                      (direction == "in" ? position < target : position > target) {
                    try? await Task.sleep(for: .milliseconds(100))
                }
                try? await api.zoom(direction: direction, movement: "stop")
                if let current = zoomPosition, abs(current - target) <= 3 { return }
                if attempt < 2 { try? await Task.sleep(for: .milliseconds(350)) }
            }
            statusMessage = "Zoom target was not reached; position reset."
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
    func replace(_ photo: SavedPhoto, data: Data) { Task { try? await store.replace(photo, data: data); await reloadGallery() } }

    private func startLiveND() {
        guard let api else { return }
        let target = 1 << preferences.liveNDStops
        stackData.removeAll(); stackMode = .liveND; stackFrameCount = 0; stackTargetCount = target
        statusMessage = "Live ND: capturing \(target) frames"
        Task {
            do {
                try await api.setPostviewSize(preferences.downloadQuality, availableAPIs: availableAPIs)
                if canContinuousCapture {
                    await configureDrive(for: .liveND)
                    try await api.startContinuousShooting()
                    let exposure = Self.shutterSeconds(settings[.shutterSpeed]?.current ?? "1/60")
                    try await Task.sleep(for: .seconds(max(0.35, exposure * Double(target + 1))))
                    try await api.stopContinuousShooting()
                } else {
                    await ensureSingleDrive()
                    for _ in 0..<target { enqueue(try await api.takePicture()) }
                }
            } catch { statusMessage = "Live ND failed: \(Self.friendly(error))" }
        }
    }

    private func startEventMonitor() {
        eventTask?.cancel()
        guard let api else { return }
        eventTask = Task {
            while !Task.isCancelled {
                do {
                    let event = try await api.event(longPolling: true)
                    if let status = event.status { cameraStatus = status }
                    if let position = event.zoomPosition { zoomPosition = position }
                    if let setting = event.zoomSetting { zoomSetting = setting }
                    event.urls.forEach(enqueue)
                } catch is CancellationError { return }
                catch { if !Task.isCancelled { try? await Task.sleep(for: .seconds(1)) } }
            }
        }
    }

    private func enqueue(_ url: URL) {
        guard knownRemoteURLs.insert(url.absoluteString).inserted else { return }
        downloadQueue.append(url); pendingDownloads = downloadQueue.count
        guard queueTask == nil else { return }
        queueTask = Task {
            while !downloadQueue.isEmpty, !Task.isCancelled {
                let next = downloadQueue.removeFirst(); pendingDownloads = downloadQueue.count + 1
                do { try await importRemotePhoto(next) }
                catch { statusMessage = "Download failed: \(Self.friendly(error))" }
                pendingDownloads = downloadQueue.count
            }
            queueTask = nil; downloadProgress = nil
        }
    }

    private func importRemotePhoto(_ url: URL) async throws {
        statusMessage = "Downloading image..."
        var lastError: Error?
        for attempt in 1...5 {
            do {
                let data = try await download(url)
                guard let source = CIImage(data: data, options: [.applyOrientationProperty: true]) else { throw ControllerError.message("Camera returned invalid image data.") }
                if selectedMode != .photo || stackMode != nil {
                    let mode = stackMode ?? selectedMode
                    stackMode = mode; stackData.append(data); stackFrameCount = stackData.count
                    if mode == .liveND, stackData.count >= stackTargetCount { await finalizeStack(mode: .liveND) }
                    else if mode == .composite || mode == .panorama { await updateStackPreview(mode: mode) }
                } else {
                    let iso = Int(settings[.iso]?.current ?? "")
                    let selection = preferences.lutSelection
                    let shouldBake = preferences.bakeLUTIntoCapture && selection.identifier != "Original"
                    let shouldDenoise = preferences.autoDenoise == .always ||
                        (preferences.autoDenoise == .isoThreshold && (iso ?? 0) >= preferences.denoiseISOThreshold)
                    let restored = shouldDenoise
                        ? (try await RawRefineryProcessor.shared.process(source, iso: iso ?? 100, denoiseStrength: 0.65, sharpenStrength: 0))
                        : source
                    let processed = shouldBake ? ImageProcessor.shared.applyLUT(restored, selection: selection, library: lutLibrary) : restored
                    let location = preferences.geotagging ? await locationProvider.location() : nil
                    guard let encoded = ImageProcessor.shared.encode(processed, format: preferences.outputFormat, location: location) else { throw ProcessingError.invalidImage }
                    let photo = try await store.save(encoded.0, originalData: (shouldBake || shouldDenoise) ? data : nil,
                                                     lut: shouldBake ? selection : nil, iso: iso, extension: encoded.1)
                    photos.insert(photo, at: 0)
                    statusMessage = "Saved \(photo.id)"
                }
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
        do {
            let result = try await Task.detached { mode == .composite ? try ComputationalCapture.composite(sources) : try ComputationalCapture.panorama(sources) }.value
            computationalPreview = ImageProcessor.shared.preview(result)
            statusMessage = "\(stackData.count) frames ready"
        } catch { statusMessage = Self.friendly(error) }
    }

    private func finalizeStack(mode: CaptureMode) async {
        let sources = stackData
        guard !sources.isEmpty else { return }
        statusMessage = "Rendering \(mode.rawValue)..."
        do {
            let result = try await Task.detached {
                switch mode {
                case .liveND: try ComputationalCapture.liveND(sources)
                case .composite: try ComputationalCapture.composite(sources)
                case .panorama: try ComputationalCapture.panorama(sources)
                case .photo: throw ProcessingError.noFrames
                }
            }.value
            let selection = preferences.lutSelection
            let baked = preferences.bakeLUTIntoCapture ? ImageProcessor.shared.applyLUT(result, selection: selection, library: lutLibrary) : result
            let location = preferences.geotagging ? await locationProvider.location() : nil
            guard let display = ImageProcessor.shared.encode(baked, format: preferences.outputFormat, location: location),
                  let original = ImageProcessor.shared.jpeg(result) else { throw ProcessingError.invalidImage }
            let photo = try await store.save(display.0, originalData: preferences.bakeLUTIntoCapture ? original : nil,
                                             kind: mode, sourceData: sources,
                                             lut: preferences.bakeLUTIntoCapture ? selection : nil, extension: display.1)
            photos.insert(photo, at: 0); statusMessage = "Saved \(mode.rawValue)"
            resetStack()
        } catch { statusMessage = "Render failed: \(Self.friendly(error))" }
    }

    private func download(_ url: URL) async throws -> Data {
        var request = URLRequest(url: url); request.timeoutInterval = preferences.downloadQuality == .original ? 300 : 90
        let (bytes, response) = try await URLSession.shared.bytes(for: request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else { throw URLError(.badServerResponse) }
        let expected = response.expectedContentLength
        var data = Data(); if expected > 0 { data.reserveCapacity(Int(expected)) }
        var received: Int64 = 0
        for try await byte in bytes {
            data.append(byte); received += 1
            if expected > 0, received % 65_536 == 0 { downloadProgress = min(1, Double(received) / Double(expected)) }
        }
        downloadProgress = 1; return data
    }

    private func configureDrive(for mode: CaptureMode) async {
        if mode == .liveND {
            if let burst = settings[.drive]?.options.first(where: { !$0.localizedCaseInsensitiveContains("single") }) { setSetting(.drive, value: burst) }
            if let fastest = settings[.burstSpeed]?.options.last { setSetting(.burstSpeed, value: fastest) }
        } else if mode == .composite || mode == .panorama { await ensureSingleDrive() }
        else if let saved = modeDriveValues[mode] { setSetting(.drive, value: saved) }
    }

    private func ensureSingleDrive() async {
        guard let single = settings[.drive]?.options.first(where: { $0.localizedCaseInsensitiveContains("single") }) else { return }
        if settings[.drive]?.current != single {
            try? await api?.setSetting(.drive, value: single); settings[.drive]?.current = single
        }
    }

    private func armLiveViewTimeout() {
        liveViewTimer?.cancel()
        guard preferences.liveViewTimeoutMinutes > 0 else { return }
        liveViewTimer = Task {
            try? await Task.sleep(for: .seconds(preferences.liveViewTimeoutMinutes * 60))
            guard !Task.isCancelled else { return }
            liveView.stop(); liveViewImage = nil; statusMessage = "Live view paused to save camera battery."
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

    private static func shutterSeconds(_ value: String) -> Double {
        if value.uppercased() == "BULB" { return 1 }
        let parts = value.split(separator: "/")
        if parts.count == 2, let a = Double(parts[0]), let b = Double(parts[1]), b != 0 { return a / b }
        return Double(value.replacingOccurrences(of: "\"", with: "")) ?? 1 / 60
    }

    private static func friendly(_ error: Error) -> String {
        (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
    }
    private static let hostKey = "camera_host"
}

private enum ControllerError: LocalizedError {
    case message(String)
    var errorDescription: String? { if case .message(let value) = self { value } else { nil } }
}
