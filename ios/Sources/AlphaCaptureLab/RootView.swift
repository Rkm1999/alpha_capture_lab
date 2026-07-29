import CoreImage
import AINRRuntime
import SwiftUI
import UniformTypeIdentifiers
import Vision

struct RootView: View {
    @EnvironmentObject private var camera: CameraController
    @State private var showGallery = false

    var body: some View {
        CameraView(showGallery: $showGallery)
            .fullScreenCover(isPresented: $showGallery) {
                GalleryView(onClose: { showGallery = false })
            }
            .alert(
                "LUT Import",
                isPresented: Binding(
                    get: { camera.lutImportMessage != nil },
                    set: { if !$0 { camera.lutImportMessage = nil } }
                )
            ) {
                Button("OK") { camera.lutImportMessage = nil }
            } message: {
                Text(camera.lutImportMessage ?? "")
            }
    }
}

private struct CameraView: View {
    @EnvironmentObject private var camera: CameraController
    @Binding var showGallery: Bool
    @State private var showSettings = false
    @State private var showLUTImporter = false
    @State private var showLUTEditor = false
    @State private var activePrimaryBar: PrimaryBar?

    var body: some View {
        NavigationStack {
            ZStack {
                Color(white: 0.055).ignoresSafeArea()
                switch camera.phase {
                case .connected:
                    connectedWorkspace
                case .connecting:
                    ConnectionProgressView(
                        title: "Connecting",
                        detail: camera.statusMessage,
                        cancel: camera.disconnect
                    )
                case .disconnected, .failed:
                    SetupView(showGallery: $showGallery)
                }
            }
            .navigationTitle(camera.phase == .connected ? "SonyImagingDevice" : "Alpha Capture Lab")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Circle().fill(camera.phase == .connected ? .green : .secondary).frame(width: 9, height: 9)
                        .accessibilityLabel(camera.phase.title)
                }
                if camera.phase == .connected {
                    ToolbarItem(placement: .topBarTrailing) {
                        HStack(spacing: 18) {
                            Button { showSettings = true } label: { Image(systemName: "gearshape") }
                            Button { camera.disconnect() } label: { Image(systemName: "xmark.circle") }
                                .accessibilityLabel("Disconnect")
                        }
                    }
                }
            }
            .sheet(isPresented: $showSettings) { SettingsView(camera: camera) }
            .sheet(isPresented: $showLUTEditor) { LUTManagerView(camera: camera) }
            .fileImporter(isPresented: $showLUTImporter, allowedContentTypes: [.cubeLUT, .zip], allowsMultipleSelection: true) { result in
                switch result {
                case .success(let urls): camera.importLUTs(urls)
                case .failure(let error): camera.statusMessage = "LUT import failed: \(error.localizedDescription)"
                }
            }
        }
    }

    private var connectedWorkspace: some View {
        VStack(spacing: 0) {
            preview
            modePicker
            status
            settingGrid
                .frame(maxHeight: .infinity, alignment: .top)
            primaryControls
            LUTStrip(
                camera: camera,
                showGallery: $showGallery,
                showImporter: $showLUTImporter,
                showEditor: $showLUTEditor
            )
        }
    }

    private var modePicker: some View {
        HStack(spacing: 0) {
            ForEach(CaptureMode.allCases) { mode in
                Button { camera.selectMode(mode) } label: {
                    VStack(spacing: 5) {
                        Text(mode.rawValue).font(.caption.weight(camera.selectedMode == mode ? .semibold : .regular))
                        Rectangle().fill(camera.selectedMode == mode ? Color.white : .clear).frame(height: 2)
                    }.frame(maxWidth: .infinity)
                }.buttonStyle(.plain)
            }
        }.frame(height: 42)
    }

    private var preview: some View {
        ZStack(alignment: .topLeading) {
            Color(white: 0.07)
            if let image = camera.computationalPreview ?? camera.liveViewImage {
                Image(uiImage: image).resizable().scaledToFit().frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                VStack(spacing: 9) {
                    Image(systemName: "camera.viewfinder").font(.system(size: 32, weight: .light))
                    Text(camera.isLiveViewRunning ? "Starting live view" : "Live view paused").font(.caption)
                    if !camera.isLiveViewRunning {
                        Button("Start live view", action: camera.resumeLiveView)
                            .buttonStyle(.bordered)
                    }
                }.foregroundStyle(.secondary).frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            if camera.selectedMode != .photo, camera.stackFrameCount > 0 {
                Text(camera.stackTargetCount > 0 ? "\(camera.stackFrameCount) / \(camera.stackTargetCount)" : "\(camera.stackFrameCount) frames")
                    .font(.caption.monospacedDigit().weight(.semibold)).padding(.horizontal, 9).padding(.vertical, 6)
                    .background(.black.opacity(0.68)).padding(10)
            }
            if camera.selectedMode == .liveND {
                Text(liveNDTime).font(.caption.monospacedDigit()).padding(7).background(.black.opacity(0.68))
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topTrailing).padding(10)
            }
        }.aspectRatio(3 / 2, contentMode: .fit)
    }

    private var liveNDTime: String {
        let frames = 1 << camera.preferences.liveNDStops
        let duration = liveNDBurstDuration(
            shutterValue: camera.settings[.shutterSpeed]?.current ?? "1/60",
            requiredFrames: frames
        )
        return String(format: "ND %.2fs", duration)
    }

    private var status: some View {
        VStack(spacing: 5) {
            HStack {
                Text(camera.statusMessage).lineLimit(1)
                Spacer()
                if camera.pendingDownloads > 0 { Text("\(camera.pendingDownloads) queued") }
            }.font(.caption2).foregroundStyle(.secondary)
            if let progress = camera.downloadProgress { ProgressView(value: progress).tint(.white) }
            if let progress = camera.ainrProgress {
                HStack {
                    Text("Denoising")
                    Spacer()
                    Text(progress.total > 0 ? "\(progress.completed)/\(progress.total)" : progress.detail)
                }
                .font(.caption2)
                .foregroundStyle(.secondary)
                if let fraction = progress.fraction {
                    ProgressView(value: fraction).tint(.white)
                } else {
                    ProgressView().tint(.white)
                }
            }
        }.padding(.horizontal, 14).padding(.vertical, 7)
    }

    private var settingGrid: some View {
        let items = photoSettingGridOrder(
            settings: camera.settings,
            exposureMode: camera.settings[.exposureMode]?.current
        )
        return Group {
            if items.isEmpty {
                Text("Reduced controls reported by the camera. Live view and shutter remain available.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 14)
            } else {
                LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 5), count: 3), spacing: 5) {
                    ForEach(Array(items.prefix(camera.selectedMode == .liveND ? 5 : 6))) { id in
                        SettingCell(camera: camera, id: id)
                    }
                    if camera.selectedMode == .liveND {
                        Menu {
                            ForEach(1...5, id: \.self) { stops in Button("\(stops) stop • \(1 << stops) frames") { camera.preferences.liveNDStops = stops } }
                        } label: { SettingLabel(title: "Frames", value: "\(1 << camera.preferences.liveNDStops)") }
                    }
                }
            }
        }.padding(.horizontal, 10)
    }

    private var primarySettingID: CameraSettingID? {
        exposurePrioritySettingID(for: camera.settings[.exposureMode]?.current)
    }

    private var primaryControls: some View {
        VStack(spacing: 8) {
            if !camera.isPanoramaSessionActive,
               activePrimaryBar == .exposure,
               let id = primarySettingID,
               let setting = camera.settings[id],
               setting.options.count > 1 {
                ValueBar(title: "\(camera.settings[.exposureMode]?.current ?? "P") • \(id.label)", setting: setting) {
                    camera.setSetting(id, value: $0)
                }
            }
            if !camera.isPanoramaSessionActive,
               activePrimaryBar == .zoom,
               camera.canZoom {
                ZoomValueBar(camera: camera)
            }
            HStack(alignment: .center, spacing: 14) {
                if camera.isPanoramaSessionActive {
                    Button(role: .destructive, action: camera.cancelPanoramaSession) {
                        Image(systemName: "xmark")
                            .font(.title3.weight(.semibold))
                            .frame(width: 54, height: 54)
                    }
                    .buttonStyle(.plain)
                    .frame(maxWidth: .infinity)
                    .accessibilityLabel("Discard panorama")
                } else {
                    Button {
                        activePrimaryBar = activePrimaryBar == .exposure ? nil : .exposure
                    } label: {
                        SettingLabel(
                            title: normalizedExposureMode(camera.settings[.exposureMode]?.current),
                            value: primarySettingID.flatMap { camera.settings[$0]?.current } ?? "--"
                        )
                    }
                    .buttonStyle(.plain)
                    .frame(maxWidth: .infinity)
                    .contextMenu {
                        if let mode = camera.settings[.exposureMode] {
                            ForEach(mode.options, id: \.self) { value in
                                Button(value) { camera.setSetting(.exposureMode, value: value) }
                            }
                        }
                    }
                }

                ShutterControl(
                    icon: camera.selectedMode == .photo ||
                        (camera.selectedMode == .panorama && camera.isPanoramaSessionActive)
                        ? "camera.fill" : "play.fill",
                    supportsHold: camera.isPhotoContinuousDrive,
                    tap: camera.capture,
                    startHold: camera.startPhotoBurst,
                    stopHold: camera.stopPhotoBurst
                )
                .disabled(camera.phase != .connected || camera.isStackRendering)
                .opacity(camera.phase == .connected && !camera.isStackRendering ? 1 : 0.35)
                .accessibilityLabel(
                    camera.selectedMode == .panorama
                        ? (camera.isPanoramaSessionActive ? "Add panorama frame" : "Start panorama")
                        : (camera.selectedMode == .liveND ? "Capture Live ND" : "Take photo")
                )

                if camera.isPanoramaSessionActive {
                    Button(action: camera.finishStack) {
                        Image(systemName: "checkmark")
                            .font(.title3.weight(.semibold))
                            .frame(width: 54, height: 54)
                    }
                    .buttonStyle(.plain)
                    .frame(maxWidth: .infinity)
                    .disabled(!camera.canFinishPanorama)
                    .accessibilityLabel("Finish panorama")
                } else if camera.canZoom {
                    Button {
                        activePrimaryBar = activePrimaryBar == .zoom ? nil : .zoom
                        if activePrimaryBar == .zoom { camera.refreshZoomState() }
                    } label: {
                        SettingLabel(title: camera.zoomSetting ?? "Zoom", value: camera.zoomPosition.map { "\($0)%" } ?? "--")
                    }
                    .buttonStyle(.plain)
                    .frame(maxWidth: .infinity)
                } else { SettingLabel(title: "Zoom", value: "N/A").frame(maxWidth: .infinity) }
            }
            if camera.selectedMode == .composite {
                HStack {
                    Button("Reset", role: .destructive) { camera.resetStack() }
                    Spacer()
                    Button("Finish", systemImage: "checkmark") { camera.finishStack() }.disabled(camera.stackFrameCount == 0)
                }.font(.subheadline).padding(.horizontal, 18)
            }
        }.padding(.horizontal, 10).padding(.bottom, 7)
    }

}

private enum PrimaryBar {
    case exposure
    case zoom
}

private struct ShutterControl: View {
    let icon: String
    let supportsHold: Bool
    let tap: () -> Void
    let startHold: () -> Void
    let stopHold: () -> Void
    @State private var holding = false
    @State private var suppressTap = false

    var body: some View {
        Button {
            if suppressTap {
                suppressTap = false
            } else {
                tap()
            }
        } label: {
            ZStack {
                Circle().stroke(.mint, lineWidth: 3).frame(width: 68, height: 68)
                Image(systemName: icon).font(.title2).foregroundStyle(.mint)
            }
        }
        .onLongPressGesture(
            minimumDuration: 0.22,
            maximumDistance: 50,
            pressing: { pressed in
                if !pressed, holding {
                    holding = false
                    stopHold()
                }
            },
            perform: {
                guard supportsHold else { return }
                suppressTap = true
                holding = true
                startHold()
            }
        )
    }
}

private struct SetupView: View {
    @EnvironmentObject private var camera: CameraController
    @Binding var showGallery: Bool

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Connect a Sony camera").font(.title2.weight(.semibold))
                    Text("Start the camera's remote application, then join its DIRECT Wi-Fi network.")
                        .foregroundStyle(.secondary)
                }
                Button { showGallery = true } label: {
                    Label("Gallery", systemImage: "photo.on.rectangle")
                        .frame(maxWidth: .infinity, minHeight: 42)
                }
                .buttonStyle(.bordered)

                if !camera.pairedCameras.isEmpty {
                    Text("Paired cameras").font(.headline)
                    ForEach(camera.pairedCameras) { paired in
                        VStack(alignment: .leading, spacing: 10) {
                            HStack {
                                VStack(alignment: .leading) {
                                    Text(paired.name).font(.subheadline.weight(.semibold))
                                    Text(paired.autoConnect ? "Starts when iPhone joins this Wi-Fi" : "Manual connection")
                                        .font(.caption).foregroundStyle(.secondary)
                                }
                                Spacer()
                                Toggle("Auto-connect", isOn: Binding(
                                    get: {
                                        camera.pairedCameras.first { $0.id == paired.id }?.autoConnect ?? false
                                    },
                                    set: { camera.setAutoConnect(paired, enabled: $0) }
                                )).labelsHidden()
                            }
                            Button("Connect") {
                                camera.cameraHost = paired.host
                                camera.connect()
                            }
                            .buttonStyle(.borderedProminent)
                            .frame(maxWidth: .infinity)
                        }
                        .padding(12)
                        .background(Color(white: 0.095))
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                    }
                }

                Button {
                    camera.connect()
                } label: {
                    Label("Find camera", systemImage: "magnifyingglass")
                        .frame(maxWidth: .infinity, minHeight: 42)
                }
                .buttonStyle(.borderedProminent)

                SetupStep(number: "1", title: "Start remote control on the camera",
                          detail: "Open the camera's remote-control application.")
                SetupStep(number: "2", title: "Join the camera Wi-Fi",
                          detail: "Use iOS Settings to join the DIRECT network, then return here.")
                if case .failed(let error) = camera.phase {
                    Text(error).font(.caption).foregroundStyle(.red)
                }
            }
            .padding(24)
        }
    }
}

private struct SetupStep: View {
    let number: String
    let title: String
    let detail: String

    var body: some View {
        HStack(alignment: .top, spacing: 14) {
            Text(number).font(.headline).frame(width: 32, height: 32)
                .background(.mint.opacity(0.25)).clipShape(Circle())
            VStack(alignment: .leading, spacing: 4) {
                Text(title).font(.subheadline.weight(.semibold))
                Text(detail).font(.caption).foregroundStyle(.secondary)
            }
        }
    }
}

private struct ConnectionProgressView: View {
    let title: String
    let detail: String
    let cancel: () -> Void

    var body: some View {
        VStack(spacing: 14) {
            ProgressView().controlSize(.large)
            Text(title).font(.title3.weight(.semibold))
            Text(detail).font(.caption).foregroundStyle(.secondary)
            Button("Cancel", role: .cancel, action: cancel)
        }
    }
}

private struct SettingCell: View {
    @ObservedObject var camera: CameraController
    let id: CameraSettingID
    var body: some View {
        Menu {
            if let setting = camera.settings[id] {
                ForEach(setting.options, id: \.self) { value in Button(value) { camera.setSetting(id, value: value) } }
            }
        } label: { SettingLabel(title: id.label, value: camera.settings[id]?.current ?? "--") }
    }
}

private struct SettingLabel: View {
    let title: String
    let value: String
    var body: some View {
        VStack(spacing: 2) {
            Text(title.uppercased()).font(.system(size: 9)).foregroundStyle(.secondary).lineLimit(1)
            Text(value).font(.caption.weight(.semibold)).lineLimit(1).minimumScaleFactor(0.7)
        }.frame(maxWidth: .infinity, minHeight: 39).background(Color(white: 0.11)).clipShape(RoundedRectangle(cornerRadius: 4))
    }
}

private struct ValueBar: View {
    let title: String
    let setting: CameraSetting
    let changed: (String) -> Void
    @State private var index: Double = 0
    var body: some View {
        VStack(spacing: 2) {
            HStack { Text(title); Spacer(); Text(setting.options[safe: Int(index.rounded())] ?? setting.current) }.font(.caption2)
            Slider(value: $index, in: 0...Double(max(1, setting.options.count - 1)), step: 1) { editing in
                if !editing, let value = setting.options[safe: Int(index.rounded())] { changed(value) }
            }.tint(.white)
        }.onAppear { index = Double(setting.options.firstIndex(of: setting.current) ?? 0) }
            .onChange(of: setting.current) { _, value in index = Double(setting.options.firstIndex(of: value) ?? 0) }
    }
}

private struct ZoomValueBar: View {
    @ObservedObject var camera: CameraController
    @State private var target = 0.0
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(camera.zoomSetting ?? "Zoom")
                Spacer()
                Text("\(Int(target))%")
            }.font(.caption2)
            ZStack {
                GeometryReader { proxy in
                    HStack(spacing: 1) {
                        Rectangle().fill(.green.opacity(0.7)).frame(width: proxy.size.width * 0.58)
                        Rectangle().fill(.yellow.opacity(0.7)).frame(width: proxy.size.width * 0.24)
                        Rectangle().fill(.red.opacity(0.7))
                    }
                }.frame(height: 4)
                Slider(value: $target, in: 0...100, step: 1) { editing in
                    guard !editing else { return }
                    let transitions = [58.0, 82.0]
                    if let nearest = transitions.min(by: { abs($0 - target) < abs($1 - target) }),
                       abs(nearest - target) <= 3 {
                        target = nearest
                    }
                    camera.zoom(to: Int(target))
                }
            }
        }
        .onAppear { target = Double(camera.zoomPosition ?? 0) }
        .onChange(of: camera.zoomPosition) { _, position in
            if let position { target = Double(position) }
        }
    }
}

private struct LUTStrip: View {
    @ObservedObject var camera: CameraController
    @Binding var showGallery: Bool
    @Binding var showImporter: Bool
    @Binding var showEditor: Bool
    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text("LUT • \(camera.selectedLUTName) • \(Int(camera.preferences.lutSelection.strength * 100))%")
                .font(.caption2).foregroundStyle(.secondary).padding(.horizontal, 12)
            ScrollViewReader { proxy in
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 7) {
                    Button { showGallery = true } label: {
                        if let latest = camera.photos.first {
                            LocalPhoto(url: latest.url)
                                .frame(width: 54, height: 42)
                                .clipped()
                        } else {
                            Image(systemName: "photo.on.rectangle")
                                .frame(width: 54, height: 42)
                        }
                    }
                    .accessibilityLabel("Open gallery")
                        ForEach(camera.lutLibrary.identifiers, id: \.self) { id in
                        LUTFilmstripItem(
                            camera: camera,
                            identifier: id,
                            selected: camera.preferences.lutSelection.identifier == id
                        ) {
                            camera.preferences.lutSelection.identifier = id
                        }
                        .id(id)
                    }
                    Button { showEditor = true } label: { Image(systemName: "slider.horizontal.3").frame(width: 34, height: 32) }
                    Button { showImporter = true } label: { Image(systemName: "plus").frame(width: 34, height: 32) }
                    }.padding(.horizontal, 10)
                }
                .onChange(of: camera.preferences.lutSelection.identifier) { _, identifier in
                    withAnimation { proxy.scrollTo(identifier, anchor: .center) }
                }
            }
        }.padding(.bottom, 5)
    }
}

private struct LUTFilmstripItem: View {
    @ObservedObject var camera: CameraController
    let identifier: String
    let selected: Bool
    let action: () -> Void
    @State private var thumbnail: UIImage?

    private var title: String {
        camera.lutLibrary.lut(id: identifier)?.title ?? identifier
    }

    var body: some View {
        Button(action: action) {
            ZStack(alignment: .bottom) {
                Color(white: 0.13)
                if let thumbnail {
                    Image(uiImage: thumbnail)
                        .resizable()
                        .scaledToFill()
                } else {
                    Image(systemName: "camera.filters")
                        .foregroundStyle(.secondary)
                }
                Text(title)
                    .font(.system(size: 9, weight: .semibold))
                    .lineLimit(1)
                    .padding(.horizontal, 4)
                    .frame(maxWidth: .infinity, minHeight: 16)
                    .background(.black.opacity(0.72))
                    .foregroundStyle(.white)
            }
            .frame(width: 68, height: 48)
            .clipped()
            .overlay {
                RoundedRectangle(cornerRadius: 3)
                    .stroke(selected ? Color.mint : .clear, lineWidth: 3)
            }
            .clipShape(RoundedRectangle(cornerRadius: 3))
        }
        .buttonStyle(.plain)
        .task(id: "\(identifier)-\(camera.lutPreviewImage == nil ? 0 : 1)") {
            guard let sourceImage = camera.lutPreviewImage,
                  let source = CIImage(image: sourceImage) else {
                thumbnail = nil
                return
            }
            let selection = LUTSelection(identifier: identifier, strength: 1)
            let rendered = ImageProcessor.shared.applyLUT(
                source,
                selection: selection,
                library: camera.lutLibrary
            )
            thumbnail = ImageProcessor.shared.preview(rendered, maxDimension: 180)
        }
        .accessibilityLabel("\(title) LUT")
        .accessibilityAddTraits(selected ? .isSelected : [])
    }
}

private struct LUTManagerView: View {
    @Environment(\.dismiss) private var dismiss
    @ObservedObject var camera: CameraController
    var body: some View {
        NavigationStack {
            Form {
                Section("Selected LUT") {
                    Text(camera.selectedLUTName)
                    Slider(value: $camera.preferences.lutSelection.strength, in: 0...1)
                }
                Section("Imported") { ForEach(camera.lutLibrary.imported) { lut in Text(lut.title) } }
            }.navigationTitle("LUTs").toolbar { Button("Done") { dismiss() } }
        }
    }
}

private struct SettingsView: View {
    @Environment(\.dismiss) private var dismiss
    @ObservedObject var camera: CameraController
    var body: some View {
        NavigationStack {
            Form {
                Section("Camera") {
                    TextField("Camera address", text: $camera.cameraHost).textInputAutocapitalization(.never).autocorrectionDisabled()
                    Picker("Live-view quality", selection: $camera.preferences.liveViewQuality) {
                        ForEach(LiveViewQuality.allCases) { Text($0.label).tag($0) }
                    }.onChange(of: camera.preferences.liveViewQuality) { _, _ in camera.applyLiveViewQuality() }
                    Picker("Turn off live view", selection: $camera.preferences.liveViewTimeoutMinutes) {
                        Text("Never").tag(0); Text("1 minute").tag(1); Text("3 minutes").tag(3)
                        Text("5 minutes").tag(5); Text("10 minutes").tag(10); Text("30 minutes").tag(30)
                    }
                }
                Section("Images") {
                    Picker("Download quality", selection: $camera.preferences.downloadQuality) { ForEach(DownloadQuality.allCases) { Text($0.rawValue).tag($0) } }
                    Picker("Save format", selection: $camera.preferences.outputFormat) { ForEach(OutputFormat.allCases) { Text($0.rawValue).tag($0) } }
                    Toggle("Geotag downloaded images", isOn: $camera.preferences.geotagging)
                    if !camera.failedPhotoPublicationIDs.isEmpty {
                        Button("Retry \(camera.failedPhotoPublicationIDs.count) Photos export(s)") {
                            camera.retryPhotoPublishing()
                        }
                    }
                }
                Section("Automatic denoise") {
                    Picker("Run denoise", selection: $camera.preferences.autoDenoise) { ForEach(AutoDenoiseMode.allCases) { Text($0.rawValue).tag($0) } }
                    if camera.preferences.autoDenoise == .isoThreshold {
                        Stepper("ISO \(camera.preferences.denoiseISOThreshold)+", value: $camera.preferences.denoiseISOThreshold, in: 400...51200, step: 400)
                    }
                    if camera.preferences.autoDenoise != .off {
                        Picker("Denoise model", selection: $camera.preferences.denoiseModel) {
                            ForEach(AINRModel.allCases) { Text($0.rawValue).tag($0) }
                        }
                        .onChange(of: camera.preferences.denoiseModel) { _, _ in
                            camera.prepareAINRModel()
                        }
                        if let progress = camera.ainrProgress {
                            if let fraction = progress.fraction {
                                ProgressView(value: fraction)
                            } else {
                                ProgressView()
                            }
                            Text("\(progress.detail) \(progress.total > 0 ? "\(progress.completed)/\(progress.total)" : "")")
                                .font(.caption)
                        }
                    }
                    Text("Distilled is faster for automatic processing. SCUNet prioritizes quality. The untouched original remains available in Edit.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Section("Connection") { Text(camera.phase.title); Text(camera.cameraHost).foregroundStyle(.secondary) }
                if !camera.pairedCameras.isEmpty {
                    Section("Paired cameras") {
                        ForEach(camera.pairedCameras) { paired in
                            HStack {
                                VStack(alignment: .leading) { Text(paired.name); Text(paired.host).font(.caption).foregroundStyle(.secondary) }
                                Spacer()
                                Toggle("Auto-connect", isOn: Binding(
                                    get: { camera.pairedCameras.first { $0.id == paired.id }?.autoConnect ?? false },
                                    set: { camera.setAutoConnect(paired, enabled: $0) }
                                )).labelsHidden()
                            }
                        }
                    }
                }
            }.navigationTitle("Settings").toolbar { Button("Done") { dismiss() } }
        }
    }
}

private struct GalleryView: View {
    @EnvironmentObject private var camera: CameraController
    let onClose: () -> Void
    @State private var selected: SavedPhoto?
    private let columns = Array(repeating: GridItem(.flexible(), spacing: 2), count: 3)

    var body: some View {
        NavigationStack {
            Group {
                if camera.photos.isEmpty { ContentUnavailableView("No Photos", systemImage: "photo.on.rectangle", description: Text("Captured images appear here.")) }
                else {
                    ScrollView { LazyVGrid(columns: columns, spacing: 2) {
                        ForEach(camera.photos) { photo in
                            Button { selected = photo } label: {
                                Color.clear
                                    .aspectRatio(1, contentMode: .fit)
                                    .overlay {
                                        LocalPhoto(url: photo.url)
                                            .scaledToFill()
                                            .clipped()
                                    }
                                    .clipped()
                                    .overlay(alignment: .bottomLeading) {
                                    if photo.kind != .photo { Text(photo.kind.rawValue).font(.system(size: 9).weight(.bold)).padding(4).background(.black.opacity(0.7)) }
                                }
                            }
                            .buttonStyle(.plain)
                        }
                    }}
                }
            }
            .navigationTitle("Gallery")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button(action: onClose) { Image(systemName: "xmark") }
                        .accessibilityLabel("Close gallery")
                }
            }
            .task { await camera.reloadGallery() }
            .fullScreenCover(item: $selected) { photo in
                GalleryDetail(
                    camera: camera,
                    photos: camera.photos,
                    initialID: photo.id,
                    onClose: { selected = nil }
                )
            }
        }
        .preferredColorScheme(.dark)
    }
}

private struct LocalPhoto: View {
    let url: URL
    var body: some View {
        if let image = UIImage(contentsOfFile: url.path) { Image(uiImage: image).resizable().scaledToFill() }
        else { Color(white: 0.1).overlay { Image(systemName: "exclamationmark.triangle") } }
    }
}

private struct GalleryDetail: View {
    @ObservedObject var camera: CameraController
    let photos: [SavedPhoto]
    let initialID: String
    let onClose: () -> Void
    @State private var selectedID: String
    @State private var editing = false
    @State private var showingSources = false
    @State private var controlsVisible = true

    init(camera: CameraController, photos: [SavedPhoto], initialID: String, onClose: @escaping () -> Void) {
        self.camera = camera
        self.photos = photos
        self.initialID = initialID
        self.onClose = onClose
        _selectedID = State(initialValue: initialID)
    }

    private var photo: SavedPhoto {
        photos.first { $0.id == selectedID } ?? photos.first!
    }

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            TabView(selection: $selectedID) {
                ForEach(photos) { item in
                    ZoomablePhotoPage(
                        url: item.url,
                        onTap: { withAnimation(.easeInOut(duration: 0.16)) { controlsVisible.toggle() } },
                        onDismiss: onClose
                    )
                    .tag(item.id)
                }
            }
            .tabViewStyle(.page(indexDisplayMode: .never))
            .ignoresSafeArea()

            if controlsVisible {
                VStack {
                    HStack {
                        VStack(alignment: .leading, spacing: 3) {
                            Text(photo.kind.rawValue).font(.headline)
                            HStack(spacing: 8) {
                                if let lut = photo.lutIdentifier {
                                    Text("\(lut) \(Int((photo.lutStrength ?? 1) * 100))%")
                                }
                                if photo.denoiseModel != nil {
                                    Text("Denoised \(Int((photo.denoiseStrength ?? 1) * 100))%")
                                }
                            }
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        }
                        Spacer()
                        Button { editing = true } label: { Image(systemName: "slider.horizontal.3") }
                            .accessibilityLabel("Edit photo")
                        if !photo.sourceURLs.isEmpty {
                            Button { showingSources = true } label: {
                                Image(systemName: "square.stack.3d.up")
                            }
                            .accessibilityLabel("Source frames")
                        }
                        Button(action: onClose) { Image(systemName: "xmark") }
                            .accessibilityLabel("Back to gallery")
                    }
                    .font(.title3)
                    .padding()
                    .background(.black.opacity(0.72))
                    Spacer()
                    Text(pageLabel).font(.caption.monospacedDigit())
                        .padding(.horizontal, 10).padding(.vertical, 5)
                        .background(.black.opacity(0.7)).clipShape(Capsule())
                        .padding(.bottom, 18)
                }
                .transition(.opacity)
            }
        }
        .fullScreenCover(isPresented: $editing) {
            PhotoEditor(
                camera: camera,
                photo: photo,
                onClose: { editing = false }
            )
        }
            .sheet(isPresented: $showingSources) { SourceFramesView(urls: photo.sourceURLs) }
    }

    private var pageLabel: String {
        let index = photos.firstIndex(where: { $0.id == selectedID }) ?? 0
        return "\(index + 1) / \(photos.count)"
    }
}

private struct ZoomablePhotoPage: View {
    let url: URL
    let onTap: () -> Void
    let onDismiss: () -> Void
    @State private var scale = 1.0
    @State private var lastScale = 1.0
    @State private var offset = CGSize.zero
    @State private var lastOffset = CGSize.zero
    @State private var dismissOffset = 0.0

    var body: some View {
        GeometryReader { proxy in
            ZStack {
                Color.black.opacity(max(0.2, 1 - abs(dismissOffset) / max(proxy.size.height, 1)))
                LocalPhoto(url: url)
                    .scaledToFit()
                    .scaleEffect(scale)
                    .offset(x: offset.width, y: offset.height + dismissOffset)
            }
            .contentShape(Rectangle())
            .onTapGesture(count: 2) {
                withAnimation(.spring(response: 0.25)) {
                    scale = scale > 1 ? 1 : 2
                    lastScale = scale
                    if scale == 1 { offset = .zero; lastOffset = .zero }
                }
            }
            .onTapGesture(perform: onTap)
            .gesture(
                MagnificationGesture()
                    .onChanged { value in scale = min(max(lastScale * value, 1), 5) }
                    .onEnded { _ in
                        if scale < 1.02 {
                            scale = 1; offset = .zero; lastOffset = .zero
                        }
                        lastScale = scale
                    }
            )
            .simultaneousGesture(
                DragGesture(minimumDistance: 8)
                    .onChanged { value in
                        if scale > 1.01 {
                            offset = constrained(
                                CGSize(
                                    width: lastOffset.width + value.translation.width,
                                    height: lastOffset.height + value.translation.height
                                ),
                                viewport: proxy.size
                            )
                        } else if abs(value.translation.height) > abs(value.translation.width) {
                            dismissOffset = value.translation.height
                        }
                    }
                    .onEnded { value in
                        if scale > 1.01 {
                            offset = constrained(offset, viewport: proxy.size)
                            lastOffset = offset
                        } else if abs(value.translation.height) > 130 ||
                                    abs(value.predictedEndTranslation.height) > 320 {
                            onDismiss()
                        } else {
                            withAnimation(.spring(response: 0.25)) { dismissOffset = 0 }
                        }
                    }
            )
        }
    }

    private func constrained(_ value: CGSize, viewport: CGSize) -> CGSize {
        let maxX = viewport.width * (scale - 1) / 2
        let maxY = viewport.height * (scale - 1) / 2
        return CGSize(
            width: min(max(value.width, -maxX), maxX),
            height: min(max(value.height, -maxY), maxY)
        )
    }
}

private struct SourceFramesView: View {
    let urls: [URL]
    private let columns = Array(repeating: GridItem(.flexible(), spacing: 2), count: 3)

    var body: some View {
        NavigationStack {
            ScrollView {
                LazyVGrid(columns: columns, spacing: 2) {
                    ForEach(urls, id: \.self) { url in
                        Color.clear
                            .aspectRatio(1, contentMode: .fit)
                            .overlay {
                                LocalPhoto(url: url)
                                    .scaledToFill()
                                    .clipped()
                            }
                            .clipped()
                    }
                }
            }
            .navigationTitle("Source Frames")
        }
    }
}

#if false
private struct PhotoEditor: View {
    @ObservedObject var camera: CameraController
    let photo: SavedPhoto
    @State private var edits: EditParameters
    @State private var denoiseModel: AINRModel
    @State private var preview: UIImage?
    @State private var originalPreview: CIImage?
    @State private var denoisedPreview: CIImage?
    @State private var tool: EditTool = .exposure
    @State private var renderTask: Task<Void, Never>?
    @State private var denoiseTask: Task<Void, Never>?
    @State private var saveTask: Task<Void, Never>?
    @State private var saving = false
    @State private var confirmCancel = false

    init(camera: CameraController, photo: SavedPhoto) {
        self.camera = camera; self.photo = photo
        var initial = EditParameters(
            lut: LUTSelection(
                identifier: photo.lutIdentifier ?? "Original",
                strength: photo.lutStrength ?? 1
            )
        )
        initial.denoise = photo.denoiseStrength ?? 0
        _edits = State(initialValue: initial)
        _denoiseModel = State(
            initialValue: photo.denoiseModel ?? camera.preferences.denoiseModel
        )
    }

    var body: some View {
        ZStack { Color.black.ignoresSafeArea(); VStack(spacing: 8) {
            HStack {
                Button("Cancel") { requestClose() }
                Spacer()
                Text("Edit").font(.headline)
                Spacer()
                Button(saving ? "Saving..." : "Save") { save() }.disabled(saving)
            }.padding(.horizontal)
            ZStack(alignment: .bottom) {
                Group {
                    if let preview {
                        Image(uiImage: preview).resizable().scaledToFit()
                    } else {
                        ProgressView()
                    }
                }
                if let progress = camera.ainrProgress {
                    VStack(alignment: .leading, spacing: 5) {
                        Text("\(progress.detail) \(progress.total > 0 ? "\(progress.completed)/\(progress.total)" : "")")
                            .font(.caption)
                        if let fraction = progress.fraction {
                            ProgressView(value: fraction)
                        } else {
                            ProgressView()
                        }
                    }
                    .padding(10)
                    .background(.black.opacity(0.72))
                }
            }.frame(maxWidth: .infinity, maxHeight: .infinity)
            if tool == .denoise {
                Picker("Denoise model", selection: $denoiseModel) {
                    ForEach(AINRModel.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                .padding(.horizontal)
                .onChange(of: denoiseModel) { _, _ in
                    denoisedPreview = nil
                    prepareDenoisePreview()
                }
            }
            slider.padding(.horizontal)
            ScrollView(.horizontal, showsIndicators: false) { HStack {
                ForEach(EditTool.allCases) { item in Button { tool = item } label: { Label(item.label, systemImage: item.icon).labelStyle(.iconOnly).frame(width: 38, height: 34).background(tool == item ? .white : Color(white: 0.14)).foregroundStyle(tool == item ? .black : .white).clipShape(RoundedRectangle(cornerRadius: 4)) } }
            }.padding(.horizontal) }
            ScrollView(.horizontal, showsIndicators: false) { HStack(spacing: 7) {
                ForEach(camera.lutLibrary.identifiers, id: \.self) { id in Button { edits.lut.identifier = id; schedulePreview() } label: { Text(camera.lutLibrary.lut(id: id)?.title ?? id).font(.caption).padding(.horizontal, 10).frame(height: 34).background(edits.lut.identifier == id ? .white : Color(white: 0.14)).foregroundStyle(edits.lut.identifier == id ? .black : .white).clipShape(RoundedRectangle(cornerRadius: 4)) } }
            }.padding(.horizontal) }.padding(.bottom, 8)
        }}
        .onAppear { loadPreview() }
        .onDisappear {
            renderTask?.cancel()
            denoiseTask?.cancel()
        }
        .alert("Cancel denoising?", isPresented: $confirmCancel) {
            Button("Keep processing", role: .cancel) {}
            Button("Cancel processing", role: .destructive) {
                denoiseTask?.cancel()
                saveTask?.cancel()
                camera.ainrProgress = nil
                dismiss()
            }
        } message: {
            Text("The current processing will stop and no unfinished edit will be saved.")
        }
    }

    @ViewBuilder private var slider: some View {
        let binding = binding(for: tool)
        HStack {
            Text(tool.label).font(.caption).frame(width: 70, alignment: .leading)
            Slider(value: binding, in: tool.range).onChange(of: binding.wrappedValue) { _, value in
                schedulePreview()
                if tool == .denoise, value > 0, denoisedPreview == nil {
                    prepareDenoisePreview()
                }
            }
            Text(tool.format(binding.wrappedValue)).font(.caption.monospacedDigit()).frame(width: 45)
        }
    }

    private func binding(for tool: EditTool) -> Binding<Double> {
        switch tool {
        case .exposure: $edits.exposure
        case .contrast: $edits.contrast
        case .saturation: $edits.saturation
        case .highlights: $edits.highlights
        case .shadows: $edits.shadows
        case .warmth: $edits.warmth
        case .sharpen: $edits.sharpen
        case .denoise: $edits.denoise
        case .lutStrength: $edits.lut.strength
        }
    }

    private func sourceImage() -> CIImage? { CIImage(contentsOf: photo.originalURL ?? photo.url, options: [.applyOrientationProperty: true]) }

    private func loadPreview() {
        guard let source = sourceImage(),
              let data = ImageProcessor.shared.previewJPEG(source),
              let reduced = CIImage(data: data) else { return }
        originalPreview = reduced
        schedulePreview()
        if edits.denoise > 0 { prepareDenoisePreview() }
    }

    private func schedulePreview() {
        renderTask?.cancel(); let edits = edits; let library = camera.lutLibrary
        guard let original = originalPreview else { return }
        let denoised = denoisedPreview
        renderTask = Task {
            try? await Task.sleep(for: .milliseconds(35))
            guard !Task.isCancelled else { return }
            let base = if edits.denoise > 0, let denoised {
                ImageProcessor.shared.blend(original, denoised, strength: edits.denoise)
            } else {
                original
            }
            var remaining = edits
            remaining.denoise = 0
            preview = ImageProcessor.shared.preview(
                ImageProcessor.shared.render(base, edits: remaining, library: library)
            )
        }
    }

    private func prepareDenoisePreview() {
        guard edits.denoise > 0,
              denoisedPreview == nil,
              let original = originalPreview,
              let data = ImageProcessor.shared.jpeg(original) else { return }
        denoiseTask?.cancel()
        let model = denoiseModel
        denoiseTask = Task {
            do {
                let output = try await AINRService.shared.process(
                    data: data,
                    sourceName: "editor-preview.jpg",
                    model: model
                ) { progress in
                    Task { @MainActor in camera.ainrProgress = progress }
                }
                try Task.checkCancellation()
                guard model == denoiseModel, let image = CIImage(data: output) else { return }
                denoisedPreview = image
                camera.ainrProgress = nil
                schedulePreview()
            } catch {
                camera.ainrProgress = nil
            }
        }
    }

    private func requestClose() {
        if saving || denoiseTask != nil && denoisedPreview == nil {
            confirmCancel = true
        } else {
            dismiss()
        }
    }

    private func save() {
        guard let sourceURL = Optional(photo.originalURL ?? photo.url),
              let sourceData = try? Data(contentsOf: sourceURL),
              let original = CIImage(data: sourceData, options: [.applyOrientationProperty: true])
        else { return }
        saving = true
        let requested = edits
        let model = denoiseModel
        saveTask = Task {
            do {
                let denoisedData = if requested.denoise > 0 {
                    try await AINRService.shared.process(
                        data: sourceData,
                        sourceName: sourceURL.lastPathComponent,
                        model: model
                    ) { progress in
                        Task { @MainActor in camera.ainrProgress = progress }
                    }
                } else {
                    sourceData
                }
                guard let denoised = CIImage(
                    data: denoisedData,
                    options: [.applyOrientationProperty: true]
                ) else { throw ProcessingError.invalidImage }
                let blended = requested.denoise > 0
                    ? ImageProcessor.shared.blend(original, denoised, strength: requested.denoise)
                    : original
                let restored = try await RawRefineryProcessor.shared.process(
                    blended,
                    iso: photo.iso ?? 100,
                    sharpenStrength: requested.sharpen
                )
                var remaining = requested; remaining.denoise = 0; remaining.sharpen = 0
                let rendered = ImageProcessor.shared.render(restored, edits: remaining, library: camera.lutLibrary)
                if let data = ImageProcessor.shared.jpeg(rendered) {
                    camera.replace(
                        photo,
                        data: data,
                        denoiseModel: requested.denoise > 0 ? model : nil,
                        denoiseStrength: requested.denoise > 0 ? requested.denoise : nil
                    )
                }
                camera.ainrProgress = nil
                saving = false
                dismiss()
            } catch {
                camera.ainrProgress = nil
                saving = false
            }
        }
    }
}

private enum EditTool: String, CaseIterable, Identifiable {
    case exposure, contrast, saturation, highlights, shadows, warmth, sharpen, denoise, lutStrength
    var id: String { rawValue }
    var label: String { rawValue == "lutStrength" ? "LUT" : rawValue.capitalized }
    var icon: String { ["exposure":"sun.max", "contrast":"circle.lefthalf.filled", "saturation":"drop", "highlights":"sun.max.fill", "shadows":"moon.fill", "warmth":"thermometer.medium", "sharpen":"camera.filters", "denoise":"sparkles", "lutStrength":"slider.horizontal.below.rectangle"] [rawValue]! }
    var range: ClosedRange<Double> { switch self { case .exposure: -3...3; case .contrast: 0.5...1.5; case .saturation: 0...2; case .highlights: 0...1; case .shadows: -1...1; case .warmth: -1...1; default: 0...1 } }
    func format(_ value: Double) -> String { String(format: "%.2f", value) }
}
#endif

private enum EditorSource: String, CaseIterable, Identifiable {
    case processed = "Processed"
    case original = "Original"
    var id: Self { self }
}

private enum EditorTool: String, CaseIterable, Identifiable {
    case adjust = "Adjust"
    case lut = "LUT"
    case denoise = "Denoise"
    case crop = "Crop"
    case rotate = "Rotate"
    case perspective = "Perspective"

    var id: Self { self }
    var icon: String {
        switch self {
        case .adjust: "slider.horizontal.3"
        case .lut: "paintpalette"
        case .denoise: "wand.and.stars"
        case .crop: "crop"
        case .rotate: "rotate.right"
        case .perspective: "skew"
        }
    }
}

private struct EditorSnapshot: Equatable {
    var edits: EditParameters
    var denoiseModel: AINRModel
    var denoiseEnabled: Bool
}

private struct PhotoEditor: View {
    @ObservedObject var camera: CameraController
    let photo: SavedPhoto
    let onClose: () -> Void

    @State private var source: EditorSource = .processed
    @State private var edits = EditParameters()
    @State private var denoiseModel: AINRModel
    @State private var denoiseEnabled = false
    @State private var sourceImages: [EditorSource: CIImage] = [:]
    @State private var previewSources: [EditorSource: CIImage] = [:]
    @State private var sourceData: [EditorSource: Data] = [:]
    @State private var preview: UIImage?
    @State private var denoisedPreview: CIImage?
    @State private var tool: EditorTool = .adjust
    @State private var panelExpanded = true
    @State private var comparing = false
    @State private var comparison = 0.5
    @State private var pendingCrop: NormalizedCrop?
    @State private var undoStack: [EditorSnapshot] = []
    @State private var redoStack: [EditorSnapshot] = []
    @State private var gestureStart: EditorSnapshot?
    @State private var renderTask: Task<Void, Never>?
    @State private var denoiseTask: Task<Void, Never>?
    @State private var saveTask: Task<Void, Never>?
    @State private var saving = false
    @State private var confirmCancel = false

    init(camera: CameraController, photo: SavedPhoto, onClose: @escaping () -> Void) {
        self.camera = camera
        self.photo = photo
        self.onClose = onClose
        _denoiseModel = State(initialValue: photo.denoiseModel ?? camera.preferences.denoiseModel)
    }

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            VStack(spacing: 0) {
                editorHeader
                EditorCanvas(
                    original: previewSources[source].flatMap { ImageProcessor.shared.preview($0) },
                    edited: preview,
                    compare: comparing,
                    comparison: $comparison,
                    activeTool: tool,
                    geometry: previewGeometry,
                    onGeometryChanged: setGeometryFromCanvas,
                    onGeometryGestureEnded: commitGesture
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .clipped()
                editorPanel
            }
            if let progress = camera.ainrProgress {
                VStack {
                    Spacer()
                    HStack(spacing: 10) {
                        ProgressView(value: progress.fraction)
                        Text(progress.total > 0
                             ? "\(progress.detail) \(progress.completed)/\(progress.total)"
                             : progress.detail)
                            .font(.caption)
                    }
                    .padding(12)
                    .background(.black.opacity(0.82))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                    .padding(.bottom, panelExpanded ? 330 : 50)
                }
            }
        }
        .preferredColorScheme(.dark)
        .task { loadSources() }
        .onDisappear {
            renderTask?.cancel()
            denoiseTask?.cancel()
            saveTask?.cancel()
        }
        .alert("Cancel processing?", isPresented: $confirmCancel) {
            Button("Keep processing", role: .cancel) {}
            Button("Cancel processing", role: .destructive) {
                denoiseTask?.cancel()
                saveTask?.cancel()
                camera.ainrProgress = nil
                onClose()
            }
        } message: {
            Text("The active denoise or save operation will stop and no unfinished edit will be saved.")
        }
    }

    private var editorHeader: some View {
        HStack(spacing: 18) {
            Text(photo.kind.rawValue).font(.headline)
            Spacer()
            Button(action: undo) { Image(systemName: "arrow.uturn.backward") }
                .disabled(undoStack.isEmpty || saving)
                .accessibilityLabel("Undo")
            Button(action: redo) { Image(systemName: "arrow.uturn.forward") }
                .disabled(redoStack.isEmpty || saving)
                .accessibilityLabel("Redo")
            Button { comparing.toggle() } label: {
                Image(systemName: "rectangle.split.2x1")
            }
            .accessibilityLabel("Compare")
            Button(action: save) { Image(systemName: "checkmark") }
                .disabled(saving || preview == nil)
                .accessibilityLabel("Save copy")
            Button(action: requestClose) { Image(systemName: "xmark") }
                .accessibilityLabel("Close editor")
        }
        .font(.title3)
        .padding(.horizontal, 16)
        .frame(height: 58)
        .background(Color(white: 0.08))
    }

    private var editorPanel: some View {
        VStack(spacing: 0) {
            Capsule().fill(.secondary).frame(width: 42, height: 5).padding(.vertical, 10)
                .contentShape(Rectangle().inset(by: -14))
                .onTapGesture { withAnimation(.easeInOut(duration: 0.18)) { panelExpanded.toggle() } }
                .gesture(
                    DragGesture(minimumDistance: 8).onEnded { value in
                        withAnimation(.easeInOut(duration: 0.18)) {
                            panelExpanded = value.translation.height < 0
                        }
                    }
                )
            if panelExpanded {
                sourceSelector
                attribution
                toolSelector
                toolControls.frame(minHeight: 130, maxHeight: 180)
            }
        }
        .background(Color(white: 0.075))
    }

    private var sourceSelector: some View {
        Picker("Editing source", selection: $source) {
            Text("Processed").tag(EditorSource.processed)
            if sourceImages[.original] != nil { Text("Original").tag(EditorSource.original) }
        }
        .pickerStyle(.segmented)
        .padding(.horizontal, 16)
        .onChange(of: source) { _, value in switchSource(value) }
    }

    @ViewBuilder private var attribution: some View {
        if source == .processed, photo.lutIdentifier != nil || photo.denoiseModel != nil {
            HStack(spacing: 8) {
                if let lut = photo.lutIdentifier {
                    Text("\(lut) \(Int((photo.lutStrength ?? 1) * 100))%")
                }
                if photo.denoiseModel != nil {
                    Text("Denoised \(Int((photo.denoiseStrength ?? 1) * 100))%")
                }
                Spacer()
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            .padding(.horizontal, 18)
            .padding(.top, 8)
        }
    }

    private var toolSelector: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(EditorTool.allCases) { item in
                    Button {
                        if tool == .crop, item != .crop { pendingCrop = nil }
                        tool = item
                    } label: {
                        Label(item.rawValue, systemImage: item.icon)
                            .font(.caption)
                            .padding(.horizontal, 10)
                            .frame(height: 38)
                    }
                    .buttonStyle(.plain)
                    .background(tool == item ? Color.accentColor.opacity(0.35) : Color(white: 0.12))
                    .clipShape(RoundedRectangle(cornerRadius: 5))
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
        }
    }

    @ViewBuilder private var toolControls: some View {
        switch tool {
        case .adjust:
            ScrollView {
                editorSlider("Exposure", value: $edits.exposure, range: -3...3)
                editorSlider("Contrast", value: $edits.contrast, range: 0.5...1.5)
                editorSlider("Saturation", value: $edits.saturation, range: 0...2)
            }
        case .lut:
            VStack(spacing: 8) {
                editorSlider("Strength", value: $edits.lut.strength, range: 0...1)
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(camera.lutLibrary.identifiers, id: \.self) { identifier in
                            Button {
                                recordChange {
                                    edits.lut.identifier = identifier
                                }
                                schedulePreview()
                            } label: {
                                EditorLUTTile(
                                    identifier: identifier,
                                    title: camera.lutLibrary.lut(id: identifier)?.title ?? identifier,
                                    source: previewSources[source],
                                    library: camera.lutLibrary,
                                    selected: edits.lut.identifier == identifier
                                )
                            }
                            .buttonStyle(.plain)
                        }
                    }.padding(.horizontal, 16)
                }
            }
        case .denoise:
            VStack(spacing: 9) {
                HStack {
                    Toggle("Denoise", isOn: Binding(
                        get: { denoiseEnabled },
                        set: { setDenoiseEnabled($0) }
                    ))
                    .toggleStyle(.button)
                    Spacer()
                    Picker("Model", selection: $denoiseModel) {
                        ForEach(AINRModel.allCases) { Text($0.rawValue).tag($0) }
                    }
                    .pickerStyle(.segmented)
                    .frame(maxWidth: 240)
                    .onChange(of: denoiseModel) { _, _ in
                        denoisedPreview = nil
                        if denoiseEnabled { prepareDenoisePreview() }
                    }
                }
                .padding(.horizontal, 16)
                if denoiseEnabled {
                    editorSlider("Amount", value: $edits.denoise, range: 0...1)
                }
            }
        case .crop:
            cropControls
        case .rotate:
            VStack(spacing: 8) {
                HStack {
                    Button("Rotate left", systemImage: "rotate.left") {
                        recordChange { edits.geometry.quarterTurns -= 1 }
                        schedulePreview()
                    }
                    Button("Rotate right", systemImage: "rotate.right") {
                        recordChange { edits.geometry.quarterTurns += 1 }
                        schedulePreview()
                    }
                    Button("Auto", systemImage: "wand.and.stars") { autoGeometry(false) }
                    Button("Reset") {
                        recordChange {
                            edits.geometry.quarterTurns = 0
                            edits.geometry.straightenDegrees = 0
                        }
                        schedulePreview()
                    }
                }.font(.caption)
                editorSlider("Straighten", value: $edits.geometry.straightenDegrees, range: -15...15)
            }
        case .perspective:
            HStack {
                Text("Drag the four corners on the image.").font(.caption).foregroundStyle(.secondary)
                Spacer()
                Button("Auto") { autoGeometry(true) }
                Button("Reset") {
                    recordChange { edits.geometry.perspective = EditorGeometry.identityPerspective }
                    schedulePreview()
                }
            }
            .padding(.horizontal, 16)
        }
    }

    private var cropControls: some View {
        VStack(spacing: 10) {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack {
                    ForEach([
                        ("Original", 0.0),
                        ("1:1", 1.0),
                        ("4:3", 4.0 / 3.0),
                        ("3:2", 3.0 / 2.0),
                        ("16:9", 16.0 / 9.0),
                    ], id: \.0) { label, aspect in
                        Button(label) {
                            let base = previewSources[source]?.extent ?? .zero
                            pendingCrop = crop(for: aspect, imageAspect: base.width / max(base.height, 1))
                        }.buttonStyle(.bordered)
                    }
                }.padding(.horizontal, 16)
            }
            HStack {
                Button("Reset") {
                    recordChange { edits.geometry.crop = .init() }
                    pendingCrop = nil
                    schedulePreview()
                }
                Spacer()
                Button("Cancel") { pendingCrop = nil; tool = .adjust }
                Button("Apply") {
                    guard let pendingCrop else { return }
                    recordChange { edits.geometry.crop = pendingCrop }
                    self.pendingCrop = nil
                    tool = .adjust
                    schedulePreview()
                }
                .buttonStyle(.borderedProminent)
                .disabled(pendingCrop == nil || pendingCrop == edits.geometry.crop)
            }.padding(.horizontal, 16)
        }
    }

    private func editorSlider(
        _ label: String,
        value: Binding<Double>,
        range: ClosedRange<Double>
    ) -> some View {
        HStack {
            Text(label).font(.caption).frame(width: 76, alignment: .leading)
            Slider(value: value, in: range) { editing in
                if editing {
                    if gestureStart == nil { gestureStart = snapshot }
                } else {
                    commitGesture()
                }
            }
            .onChange(of: value.wrappedValue) { _, _ in schedulePreview() }
            Text(String(format: "%.2f", value.wrappedValue))
                .font(.caption.monospacedDigit()).frame(width: 46)
        }
        .padding(.horizontal, 16)
    }

    private var snapshot: EditorSnapshot {
        .init(edits: edits, denoiseModel: denoiseModel, denoiseEnabled: denoiseEnabled)
    }

    private var previewGeometry: EditorGeometry {
        var geometry = edits.geometry
        if let pendingCrop { geometry.crop = pendingCrop }
        return geometry
    }

    private func loadSources() {
        guard sourceImages.isEmpty else { return }
        if let processedData = try? Data(contentsOf: photo.url),
           let processed = CIImage(data: processedData, options: [.applyOrientationProperty: true]) {
            sourceData[.processed] = processedData
            sourceImages[.processed] = processed
            previewSources[.processed] = reducedPreview(processed)
        }
        if let originalURL = photo.originalURL,
           let originalData = try? Data(contentsOf: originalURL),
           let original = CIImage(data: originalData, options: [.applyOrientationProperty: true]) {
            sourceData[.original] = originalData
            sourceImages[.original] = original
            previewSources[.original] = reducedPreview(original)
        }
        schedulePreview()
    }

    private func switchSource(_ newSource: EditorSource) {
        renderTask?.cancel()
        denoiseTask?.cancel()
        denoisedPreview = nil
        edits = EditParameters()
        denoiseEnabled = false
        if newSource == .original {
            edits.lut = LUTSelection(
                identifier: photo.lutIdentifier ?? "Original",
                strength: photo.lutStrength ?? 1
            )
            if photo.denoiseModel != nil {
                denoiseModel = photo.denoiseModel ?? denoiseModel
                denoiseEnabled = true
                edits.denoise = photo.denoiseStrength ?? 1
                prepareDenoisePreview()
            }
        }
        undoStack.removeAll()
        redoStack.removeAll()
        schedulePreview()
    }

    private func schedulePreview() {
        renderTask?.cancel()
        let edits = edits
        guard let original = previewSources[source] else { return }
        let denoised = denoisedPreview
        let enabled = denoiseEnabled
        let library = camera.lutLibrary
        renderTask = Task {
            try? await Task.sleep(for: .milliseconds(25))
            guard !Task.isCancelled else { return }
            let base = if enabled, let denoised {
                ImageProcessor.shared.blend(original, denoised, strength: edits.denoise)
            } else {
                original
            }
            var remaining = edits
            remaining.denoise = 0
            preview = ImageProcessor.shared.preview(
                ImageProcessor.shared.render(base, edits: remaining, library: library)
            )
        }
    }

    private func setDenoiseEnabled(_ enabled: Bool) {
        recordChange {
            denoiseEnabled = enabled
            edits.denoise = enabled ? max(edits.denoise, 0.6) : 0
        }
        if enabled { prepareDenoisePreview() } else { schedulePreview() }
    }

    private func prepareDenoisePreview() {
        guard denoiseEnabled, denoisedPreview == nil,
              let original = previewSources[source],
              let jpeg = ImageProcessor.shared.jpeg(original) else { return }
        denoiseTask?.cancel()
        let requestedSource = source
        let requestedModel = denoiseModel
        denoiseTask = Task {
            do {
                let output = try await AINRService.shared.process(
                    data: jpeg,
                    sourceName: photo.url.lastPathComponent,
                    model: requestedModel
                ) { progress in
                    Task { @MainActor in camera.ainrProgress = progress }
                }
                guard !Task.isCancelled, source == requestedSource,
                      denoiseModel == requestedModel,
                      let image = CIImage(data: output, options: [.applyOrientationProperty: true])
                else { return }
                denoisedPreview = image
                camera.ainrProgress = nil
                schedulePreview()
            } catch {
                camera.ainrProgress = nil
            }
        }
    }

    private func recordChange(_ change: () -> Void) {
        let before = snapshot
        change()
        if before != snapshot {
            undoStack.append(before)
            if undoStack.count > 50 { undoStack.removeFirst() }
            redoStack.removeAll()
        }
    }

    private func commitGesture() {
        guard let gestureStart else { return }
        self.gestureStart = nil
        if gestureStart != snapshot {
            undoStack.append(gestureStart)
            if undoStack.count > 50 { undoStack.removeFirst() }
            redoStack.removeAll()
        }
    }

    private func undo() {
        guard let previous = undoStack.popLast() else { return }
        redoStack.append(snapshot)
        install(previous)
    }

    private func redo() {
        guard let next = redoStack.popLast() else { return }
        undoStack.append(snapshot)
        install(next)
    }

    private func install(_ value: EditorSnapshot) {
        edits = value.edits
        denoiseModel = value.denoiseModel
        denoiseEnabled = value.denoiseEnabled
        denoisedPreview = nil
        if denoiseEnabled { prepareDenoisePreview() }
        schedulePreview()
    }

    private func setGeometryFromCanvas(_ geometry: EditorGeometry) {
        if gestureStart == nil { gestureStart = snapshot }
        edits.geometry = geometry
        schedulePreview()
    }

    private func autoGeometry(_ includePerspective: Bool) {
        guard let image = previewSources[source],
              let cg = CIContext().createCGImage(image, from: image.extent) else { return }
        Task {
            let request = VNDetectRectanglesRequest()
            request.maximumObservations = 1
            request.minimumConfidence = 0.55
            request.minimumSize = 0.2
            try? VNImageRequestHandler(cgImage: cg).perform([request])
            guard let rectangle = request.results?.first else { return }
            recordChange {
                if includePerspective {
                    edits.geometry.perspective = [
                        .init(x: rectangle.topLeft.x, y: 1 - rectangle.topLeft.y),
                        .init(x: rectangle.topRight.x, y: 1 - rectangle.topRight.y),
                        .init(x: rectangle.bottomRight.x, y: 1 - rectangle.bottomRight.y),
                        .init(x: rectangle.bottomLeft.x, y: 1 - rectangle.bottomLeft.y),
                    ]
                } else {
                    let dx = rectangle.topRight.x - rectangle.topLeft.x
                    let dy = rectangle.topRight.y - rectangle.topLeft.y
                    edits.geometry.straightenDegrees = -atan2(dy, dx) * 180 / .pi
                }
            }
            schedulePreview()
        }
    }

    private func crop(for aspect: Double, imageAspect: Double) -> NormalizedCrop {
        guard aspect > 0, imageAspect > 0 else { return .init() }
        if imageAspect > aspect {
            let width = aspect / imageAspect
            return .init(left: (1 - width) / 2, top: 0, right: (1 + width) / 2, bottom: 1)
        }
        let height = imageAspect / aspect
        return .init(left: 0, top: (1 - height) / 2, right: 1, bottom: (1 + height) / 2)
    }

    private func setGeometryFromCanvas(_ pointIndex: Int, point: NormalizedPoint) {
        var geometry = edits.geometry
        guard geometry.perspective.indices.contains(pointIndex) else { return }
        geometry.perspective[pointIndex] = .init(
            x: min(max(point.x, 0), 1),
            y: min(max(point.y, 0), 1)
        )
        setGeometryFromCanvas(geometry)
    }

    private func requestClose() {
        if saving || denoiseTask != nil && denoisedPreview == nil {
            confirmCancel = true
        } else {
            onClose()
        }
    }

    private func reducedPreview(_ image: CIImage) -> CIImage {
        guard let data = ImageProcessor.shared.previewJPEG(image, maxDimension: 1280),
              let preview = CIImage(data: data, options: [.applyOrientationProperty: true]) else {
            return image
        }
        return preview
    }

    private func save() {
        guard let selectedData = sourceData[source],
              let selectedImage = sourceImages[source] else { return }
        saving = true
        let requested = edits
        let requestedModel = denoiseModel
        let requestedDenoise = denoiseEnabled
        let library = camera.lutLibrary
        let retainedOriginal: Data
        if let originalURL = photo.originalURL,
           let originalData = try? Data(contentsOf: originalURL) {
            retainedOriginal = originalData
        } else {
            retainedOriginal = selectedData
        }
        saveTask = Task {
            do {
                let denoisedData = if requestedDenoise {
                    try await AINRService.shared.process(
                        data: selectedData,
                        sourceName: photo.url.lastPathComponent,
                        model: requestedModel
                    ) { progress in
                        Task { @MainActor in camera.ainrProgress = progress }
                    }
                } else {
                    selectedData
                }
                guard let denoised = CIImage(
                    data: denoisedData,
                    options: [.applyOrientationProperty: true]
                ) else { throw ProcessingError.invalidImage }
                let base = requestedDenoise
                    ? ImageProcessor.shared.blend(selectedImage, denoised, strength: requested.denoise)
                    : selectedImage
                var remaining = requested
                remaining.denoise = 0
                let rendered = ImageProcessor.shared.render(base, edits: remaining, library: library)
                guard let encoded = ImageProcessor.shared.encode(
                    rendered,
                    format: camera.preferences.outputFormat,
                    sourceData: selectedData
                ) else { throw ProcessingError.invalidImage }
                let resultLUT = requested.lut.identifier == "Original"
                    ? (source == .processed ? photo.lutIdentifier.map {
                        LUTSelection(identifier: $0, strength: photo.lutStrength ?? 1)
                    } : nil)
                    : requested.lut
                let resultDenoiseModel = requestedDenoise
                    ? requestedModel
                    : (source == .processed ? photo.denoiseModel : nil)
                let resultDenoiseStrength = requestedDenoise
                    ? requested.denoise
                    : (source == .processed ? photo.denoiseStrength : nil)
                _ = try await camera.saveEditedCopy(
                    from: photo,
                    data: encoded.0,
                    originalData: retainedOriginal,
                    lut: resultLUT,
                    denoiseModel: resultDenoiseModel,
                    denoiseStrength: resultDenoiseStrength,
                    geometry: requested.geometry.hasChanges ? requested.geometry : nil,
                    fileExtension: encoded.1
                )
                camera.ainrProgress = nil
                saving = false
                onClose()
            } catch is CancellationError {
                camera.ainrProgress = nil
                saving = false
            } catch {
                camera.ainrProgress = nil
                saving = false
            }
        }
    }
}

private struct EditorCanvas: View {
    let original: UIImage?
    let edited: UIImage?
    let compare: Bool
    @Binding var comparison: Double
    let activeTool: EditorTool
    let geometry: EditorGeometry
    let onGeometryChanged: (EditorGeometry) -> Void
    let onGeometryGestureEnded: () -> Void
    @State private var scale = 1.0
    @State private var lastScale = 1.0
    @State private var offset = CGSize.zero
    @State private var lastOffset = CGSize.zero

    var body: some View {
        GeometryReader { proxy in
            ZStack {
                Color.black
                if let edited {
                    Image(uiImage: edited).resizable().scaledToFit()
                        .scaleEffect(scale).offset(offset)
                } else {
                    ProgressView()
                }
                if compare, let original {
                    Image(uiImage: original).resizable().scaledToFit()
                        .scaleEffect(scale).offset(offset)
                        .mask(alignment: .leading) {
                            Rectangle().frame(width: proxy.size.width * comparison)
                        }
                    Rectangle().fill(.white).frame(width: 2)
                        .position(x: proxy.size.width * comparison, y: proxy.size.height / 2)
                    Circle().fill(.white).frame(width: 24, height: 24)
                        .overlay { Image(systemName: "arrow.left.and.right").font(.caption).foregroundStyle(.black) }
                        .position(x: proxy.size.width * comparison, y: proxy.size.height / 2)
                }
                if activeTool == .perspective {
                    perspectiveOverlay(size: proxy.size)
                }
                if activeTool == .crop {
                    cropOverlay(size: proxy.size)
                }
            }
            .contentShape(Rectangle())
            .gesture(
                MagnificationGesture()
                    .onChanged { value in scale = min(max(lastScale * value, 1), 5) }
                    .onEnded { _ in
                        lastScale = scale
                        if scale == 1 { offset = .zero; lastOffset = .zero }
                    }
            )
            .simultaneousGesture(
                DragGesture(minimumDistance: 4)
                    .onChanged { value in
                        if compare, scale <= 1.01 {
                            comparison = min(max(value.location.x / max(proxy.size.width, 1), 0), 1)
                        } else if scale > 1.01 {
                            offset = CGSize(
                                width: lastOffset.width + value.translation.width,
                                height: lastOffset.height + value.translation.height
                            )
                        }
                    }
                    .onEnded { _ in lastOffset = offset }
            )
        }
    }

    @ViewBuilder private func perspectiveOverlay(size: CGSize) -> some View {
        ForEach(Array(geometry.perspective.enumerated()), id: \.offset) { index, point in
            Circle().fill(.mint).frame(width: 22, height: 22)
                .overlay { Circle().stroke(.black, lineWidth: 1) }
                .position(x: point.x * size.width, y: point.y * size.height)
                .gesture(
                    DragGesture()
                        .onChanged { value in
                            var updated = geometry
                            updated.perspective[index] = .init(
                                x: min(max(value.location.x / max(size.width, 1), 0), 1),
                                y: min(max(value.location.y / max(size.height, 1), 0), 1)
                            )
                            onGeometryChanged(updated)
                        }
                        .onEnded { _ in onGeometryGestureEnded() }
                )
        }
    }

    @ViewBuilder private func cropOverlay(size: CGSize) -> some View {
        let crop = geometry.crop.normalized()
        let rect = CGRect(
            x: crop.left * size.width,
            y: crop.top * size.height,
            width: (crop.right - crop.left) * size.width,
            height: (crop.bottom - crop.top) * size.height
        )
        Path { path in path.addRect(rect) }
            .stroke(.white, style: StrokeStyle(lineWidth: 2, dash: [7, 5]))
        Path { path in
            path.addRect(CGRect(origin: .zero, size: size))
            path.addRect(rect)
        }
        .fill(.black.opacity(0.45), style: FillStyle(eoFill: true))
    }
}

private struct EditorLUTTile: View {
    let identifier: String
    let title: String
    let source: CIImage?
    let library: LUTLibrary
    let selected: Bool
    @State private var thumbnail: UIImage?

    var body: some View {
        ZStack(alignment: .bottomLeading) {
            if let thumbnail {
                Image(uiImage: thumbnail).resizable().scaledToFill()
            } else {
                Color(white: 0.12).overlay { ProgressView().controlSize(.small) }
            }
            Text(title).font(.caption2).lineLimit(1)
                .padding(3).frame(maxWidth: .infinity, alignment: .leading)
                .background(.black.opacity(0.6))
        }
        .frame(width: 92, height: 64)
        .clipped()
        .overlay { RoundedRectangle(cornerRadius: 4).stroke(selected ? .mint : .secondary, lineWidth: selected ? 2 : 1) }
        .clipShape(RoundedRectangle(cornerRadius: 4))
        .task(id: "\(identifier)-\(source?.extent.width ?? 0)") {
            guard let source else { return }
            let selection = LUTSelection(identifier: identifier, strength: 1)
            thumbnail = ImageProcessor.shared.preview(
                ImageProcessor.shared.applyLUT(source, selection: selection, library: library),
                maxDimension: 220
            )
        }
    }
}

private extension UTType {
    static let cubeLUT = UTType(filenameExtension: "cube") ?? .data
}
private extension Collection { subscript(safe index: Index) -> Element? { indices.contains(index) ? self[index] : nil } }
