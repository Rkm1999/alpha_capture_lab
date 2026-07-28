import CoreImage
import SwiftUI
import UniformTypeIdentifiers

struct RootView: View {
    @EnvironmentObject private var camera: CameraController
    @State private var page: AppPage = .camera
    var body: some View {
        TabView(selection: $page) {
            CameraView().tabItem { Label("Camera", systemImage: "camera") }.tag(AppPage.camera)
            GalleryView().tabItem { Label("Gallery", systemImage: "photo.on.rectangle") }.tag(AppPage.gallery)
        }
        .tint(.white)
    }
}

private enum AppPage { case camera, gallery }

private struct CameraView: View {
    @EnvironmentObject private var camera: CameraController
    @State private var showSettings = false
    @State private var showLUTImporter = false
    @State private var showLUTEditor = false

    var body: some View {
        NavigationStack {
            ZStack {
                Color.black.ignoresSafeArea()
                VStack(spacing: 0) {
                    modePicker
                    preview
                    status
                    settingGrid
                    primaryControls
                    LUTStrip(camera: camera, showImporter: $showLUTImporter, showEditor: $showLUTEditor)
                }
            }
            .navigationTitle("Alpha Capture Lab")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Circle().fill(camera.phase == .connected ? .green : .secondary).frame(width: 9, height: 9)
                        .accessibilityLabel(camera.phase.title)
                }
                ToolbarItemGroup(placement: .topBarTrailing) {
                    Button { showSettings = true } label: { Image(systemName: "gearshape") }
                    Button(camera.phase == .connected ? "Disconnect" : "Connect") {
                        camera.phase == .connected ? camera.disconnect() : camera.connect()
                    }.disabled(camera.phase == .connecting)
                }
            }
            .sheet(isPresented: $showSettings) { SettingsView(camera: camera) }
            .sheet(isPresented: $showLUTEditor) { LUTManagerView(camera: camera) }
            .fileImporter(isPresented: $showLUTImporter, allowedContentTypes: [.cubeLUT, .zip], allowsMultipleSelection: true) { result in
                if case .success(let urls) = result { try? camera.lutLibrary.importFiles(urls) }
            }
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
                    Text(camera.phase == .connecting ? "Connecting" : "Live view unavailable").font(.caption)
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
        return "ND\(camera.preferences.liveNDStops)  •  \(frames) frames"
    }

    private var status: some View {
        VStack(spacing: 5) {
            HStack {
                Text(camera.statusMessage).lineLimit(1)
                Spacer()
                if camera.pendingDownloads > 0 { Text("\(camera.pendingDownloads) queued") }
            }.font(.caption2).foregroundStyle(.secondary)
            if let progress = camera.downloadProgress { ProgressView(value: progress).tint(.white) }
        }.padding(.horizontal, 14).padding(.vertical, 7)
    }

    private var settingGrid: some View {
        let hidden = primarySettingID
        let items = CameraSettingID.allCases.filter { $0 != .exposureMode && $0 != hidden && camera.settings[$0] != nil }
        return LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 5), count: 3), spacing: 5) {
            ForEach(Array(items.prefix(6))) { id in SettingCell(camera: camera, id: id) }
            if camera.selectedMode == .liveND, items.count < 6 {
                Menu {
                    ForEach(1...5, id: \.self) { stops in Button("\(stops) stop • \(1 << stops) frames") { camera.preferences.liveNDStops = stops } }
                } label: { SettingLabel(title: "Frames", value: "\(1 << camera.preferences.liveNDStops)") }
            }
        }.padding(.horizontal, 10)
    }

    private var primarySettingID: CameraSettingID? {
        switch camera.settings[.exposureMode]?.current.uppercased() {
        case "A", "APERTURE": .aperture
        case "S", "SHUTTER": .shutterSpeed
        case "M", "MANUAL": .shutterSpeed
        default: .exposureCompensation
        }
    }

    private var primaryControls: some View {
        VStack(spacing: 8) {
            if let id = primarySettingID, let setting = camera.settings[id], setting.options.count > 1 {
                ValueBar(title: "\(camera.settings[.exposureMode]?.current ?? "P") • \(id.label)", setting: setting) {
                    camera.setSetting(id, value: $0)
                }
            }
            HStack(alignment: .center, spacing: 14) {
                Menu {
                    if let mode = camera.settings[.exposureMode] {
                        ForEach(mode.options, id: \.self) { value in Button(value) { camera.setSetting(.exposureMode, value: value) } }
                    }
                } label: {
                    SettingLabel(title: "Exposure", value: camera.settings[.exposureMode]?.current ?? "--")
                }.frame(maxWidth: .infinity)

                Button(action: camera.capture) {
                    ZStack { Circle().stroke(.white, lineWidth: 3).frame(width: 68, height: 68); Circle().fill(.white).frame(width: 56, height: 56) }
                }.disabled(camera.phase != .connected).opacity(camera.phase == .connected ? 1 : 0.35)
                    .accessibilityLabel(camera.selectedMode == .liveND ? "Capture Live ND" : "Take photo")

                if camera.canZoom {
                    Menu { ZoomTargetView(camera: camera) } label: {
                        SettingLabel(title: camera.zoomSetting ?? "Zoom", value: camera.zoomPosition.map { "\($0)%" } ?? "--")
                    }.frame(maxWidth: .infinity)
                } else { SettingLabel(title: "Zoom", value: "N/A").frame(maxWidth: .infinity) }
            }
            if camera.selectedMode == .composite || camera.selectedMode == .panorama {
                HStack {
                    Button("Reset", role: .destructive) { camera.resetStack() }
                    Spacer()
                    Button("Finish", systemImage: "checkmark") { camera.finishStack() }.disabled(camera.stackFrameCount == 0)
                }.font(.subheadline).padding(.horizontal, 18)
            }
        }.padding(.horizontal, 10).padding(.bottom, 7)
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

private struct ZoomTargetView: View {
    @ObservedObject var camera: CameraController
    @State private var target = 0.0
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Zoom target \(Int(target))%").font(.caption)
            ZStack {
                GeometryReader { proxy in
                    HStack(spacing: 1) {
                        Rectangle().fill(.green.opacity(0.7)).frame(width: proxy.size.width * 0.58)
                        Rectangle().fill(.yellow.opacity(0.7)).frame(width: proxy.size.width * 0.24)
                        Rectangle().fill(.red.opacity(0.7))
                    }
                }.frame(height: 4)
                Slider(value: $target, in: 0...100, step: 1)
            }
            Button("Set zoom") { camera.zoom(to: Int(target)) }
        }.padding().frame(width: 280).onAppear { target = Double(camera.zoomPosition ?? 0) }
    }
}

private struct LUTStrip: View {
    @ObservedObject var camera: CameraController
    @Binding var showImporter: Bool
    @Binding var showEditor: Bool
    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text("LUT • \(camera.selectedLUTName) • \(Int(camera.preferences.lutSelection.strength * 100))%")
                .font(.caption2).foregroundStyle(.secondary).padding(.horizontal, 12)
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 7) {
                    ForEach(camera.lutLibrary.identifiers, id: \.self) { id in
                        Button { camera.preferences.lutSelection.identifier = id } label: {
                            Text(camera.lutLibrary.lut(id: id)?.title ?? id).font(.caption).lineLimit(1)
                                .padding(.horizontal, 10).frame(height: 32)
                                .background(camera.preferences.lutSelection.identifier == id ? Color.white : Color(white: 0.13))
                                .foregroundStyle(camera.preferences.lutSelection.identifier == id ? .black : .white)
                                .clipShape(RoundedRectangle(cornerRadius: 4))
                        }
                    }
                    Button { showEditor = true } label: { Image(systemName: "slider.horizontal.3").frame(width: 34, height: 32) }
                    Button { showImporter = true } label: { Image(systemName: "plus").frame(width: 34, height: 32) }
                }.padding(.horizontal, 10)
            }
        }.padding(.bottom, 5)
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
                    Toggle("Apply LUT to downloaded photos", isOn: $camera.preferences.bakeLUTIntoCapture)
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
                        Text("Never").tag(0); Text("1 minute").tag(1); Text("2 minutes").tag(2); Text("5 minutes").tag(5); Text("10 minutes").tag(10)
                    }
                }
                Section("Images") {
                    Picker("Download quality", selection: $camera.preferences.downloadQuality) { ForEach(DownloadQuality.allCases) { Text($0.rawValue).tag($0) } }
                    Picker("Save format", selection: $camera.preferences.outputFormat) { ForEach(OutputFormat.allCases) { Text($0.rawValue).tag($0) } }
                    Toggle("Geotag downloaded images", isOn: $camera.preferences.geotagging)
                }
                Section("Automatic denoise") {
                    Picker("Run denoise", selection: $camera.preferences.autoDenoise) { ForEach(AutoDenoiseMode.allCases) { Text($0.rawValue).tag($0) } }
                    if camera.preferences.autoDenoise == .isoThreshold {
                        Stepper("ISO \(camera.preferences.denoiseISOThreshold)+", value: $camera.preferences.denoiseISOThreshold, in: 400...51200, step: 400)
                    }
                    Text("RawRefinery Light denoise and Deep Sharpen use ONNX Runtime with Core ML acceleration and CPU fallback.")
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
                                LocalPhoto(url: photo.url).aspectRatio(1, contentMode: .fill).clipped().overlay(alignment: .bottomLeading) {
                                    if photo.kind != .photo { Text(photo.kind.rawValue).font(.system(size: 9).weight(.bold)).padding(4).background(.black.opacity(0.7)) }
                                }
                            }
                        }
                    }}
                }
            }.navigationTitle("Gallery").task { await camera.reloadGallery() }
                .fullScreenCover(item: $selected) { PhotoDetail(camera: camera, photo: $0) }
        }
    }
}

private struct LocalPhoto: View {
    let url: URL
    var body: some View {
        if let image = UIImage(contentsOfFile: url.path) { Image(uiImage: image).resizable().scaledToFill() }
        else { Color(white: 0.1).overlay { Image(systemName: "exclamationmark.triangle") } }
    }
}

private struct PhotoDetail: View {
    @Environment(\.dismiss) private var dismiss
    @ObservedObject var camera: CameraController
    let photo: SavedPhoto
    @State private var editing = false
    @State private var showingSources = false
    var body: some View {
        ZStack(alignment: .topTrailing) {
            Color.black.ignoresSafeArea()
            LocalPhoto(url: photo.url).scaledToFit().frame(maxWidth: .infinity, maxHeight: .infinity)
            HStack {
                Button { editing = true } label: { Image(systemName: "slider.horizontal.3") }
                if !photo.sourceURLs.isEmpty { Button { showingSources = true } label: { Image(systemName: "square.stack.3d.up") } }
                Button(role: .destructive) { camera.delete(photo); dismiss() } label: { Image(systemName: "trash") }
                Button { dismiss() } label: { Image(systemName: "xmark.circle.fill") }
            }.font(.system(size: 24)).padding()
        }.fullScreenCover(isPresented: $editing) { PhotoEditor(camera: camera, photo: photo) }
            .sheet(isPresented: $showingSources) { SourceFramesView(urls: photo.sourceURLs) }
    }
}

private struct SourceFramesView: View {
    let urls: [URL]
    let columns = [GridItem(.adaptive(minimum: 110), spacing: 2)]
    var body: some View { NavigationStack { ScrollView { LazyVGrid(columns: columns, spacing: 2) { ForEach(urls, id: \.self) { LocalPhoto(url: $0).aspectRatio(1, contentMode: .fill).clipped() } } }.navigationTitle("Source Frames") } }
}

private struct PhotoEditor: View {
    @Environment(\.dismiss) private var dismiss
    @ObservedObject var camera: CameraController
    let photo: SavedPhoto
    @State private var edits: EditParameters
    @State private var preview: UIImage?
    @State private var tool: EditTool = .exposure
    @State private var renderTask: Task<Void, Never>?
    @State private var saving = false

    init(camera: CameraController, photo: SavedPhoto) {
        self.camera = camera; self.photo = photo
        _edits = State(initialValue: EditParameters(lut: LUTSelection(identifier: photo.lutIdentifier ?? "Original", strength: photo.lutStrength ?? 1)))
    }

    var body: some View {
        ZStack { Color.black.ignoresSafeArea(); VStack(spacing: 8) {
            HStack { Button("Cancel") { dismiss() }; Spacer(); Text("Edit").font(.headline); Spacer(); Button(saving ? "Saving..." : "Save") { save() }.disabled(saving) }.padding(.horizontal)
            Group { if let preview { Image(uiImage: preview).resizable().scaledToFit() } else { ProgressView() } }.frame(maxWidth: .infinity, maxHeight: .infinity)
            slider.padding(.horizontal)
            ScrollView(.horizontal, showsIndicators: false) { HStack {
                ForEach(EditTool.allCases) { item in Button { tool = item } label: { Label(item.label, systemImage: item.icon).labelStyle(.iconOnly).frame(width: 38, height: 34).background(tool == item ? .white : Color(white: 0.14)).foregroundStyle(tool == item ? .black : .white).clipShape(RoundedRectangle(cornerRadius: 4)) } }
            }.padding(.horizontal) }
            ScrollView(.horizontal, showsIndicators: false) { HStack(spacing: 7) {
                ForEach(camera.lutLibrary.identifiers, id: \.self) { id in Button { edits.lut.identifier = id; schedulePreview() } label: { Text(camera.lutLibrary.lut(id: id)?.title ?? id).font(.caption).padding(.horizontal, 10).frame(height: 34).background(edits.lut.identifier == id ? .white : Color(white: 0.14)).foregroundStyle(edits.lut.identifier == id ? .black : .white).clipShape(RoundedRectangle(cornerRadius: 4)) } }
            }.padding(.horizontal) }.padding(.bottom, 8)
        }}.onAppear { schedulePreview() }.onDisappear { renderTask?.cancel() }
    }

    @ViewBuilder private var slider: some View {
        let binding = binding(for: tool)
        HStack { Text(tool.label).font(.caption).frame(width: 70, alignment: .leading); Slider(value: binding, in: tool.range).onChange(of: binding.wrappedValue) { _, _ in schedulePreview() }; Text(tool.format(binding.wrappedValue)).font(.caption.monospacedDigit()).frame(width: 45) }
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
    private func schedulePreview() {
        renderTask?.cancel(); let edits = edits; let library = camera.lutLibrary
        guard let source = sourceImage() else { return }
        renderTask = Task { try? await Task.sleep(for: .milliseconds(35)); guard !Task.isCancelled else { return }; preview = ImageProcessor.shared.preview(ImageProcessor.shared.render(source, edits: edits, library: library)) }
    }
    private func save() {
        guard let source = sourceImage() else { return }; saving = true
        let requested = edits
        Task {
            do {
                let restored = try await RawRefineryProcessor.shared.process(
                    source, iso: photo.iso ?? 100,
                    denoiseStrength: requested.denoise, sharpenStrength: requested.sharpen
                )
                var remaining = requested; remaining.denoise = 0; remaining.sharpen = 0
                let rendered = ImageProcessor.shared.render(restored, edits: remaining, library: camera.lutLibrary)
                if let data = ImageProcessor.shared.jpeg(rendered) { camera.replace(photo, data: data) }
                saving = false; dismiss()
            } catch {
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

private extension UTType {
    static let cubeLUT = UTType(filenameExtension: "cube") ?? .data
}
private extension Collection { subscript(safe index: Index) -> Element? { indices.contains(index) ? self[index] : nil } }
