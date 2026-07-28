import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @EnvironmentObject private var viewModel: DenoiseViewModel
    @State private var showPhotoPicker = false
    @State private var showFileImporter = false
    @State private var showCancelConfirmation = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                preview
                controlPanel
            }
            .background(Color.black.ignoresSafeArea())
            .navigationTitle("SCUNet Denoiser")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { importToolbar }
            .fileImporter(
                isPresented: $showFileImporter,
                allowedContentTypes: [.image],
                allowsMultipleSelection: true
            ) { result in
                if case .success(let urls) = result, !urls.isEmpty {
                    viewModel.importURLs(urls)
                } else if case .failure(let error) = result {
                    viewModel.report(error)
                }
            }
            .sheet(isPresented: $showPhotoPicker) {
                PhotoLibraryPicker { result in
                    showPhotoPicker = false
                    switch result {
                    case .success(let photos):
                        guard !photos.isEmpty else { return }
                        viewModel.importPhotoData(photos.map {
                            ($0.data, $0.fileExtension, $0.originalName)
                        })
                    case .failure(let error):
                        viewModel.report(error)
                    }
                }
            }
            .alert(
                "SCUNet Denoiser",
                isPresented: Binding(
                    get: { viewModel.alertMessage != nil },
                    set: { if !$0 { viewModel.alertMessage = nil } }
                )
            ) {
                Button("OK", role: .cancel) { viewModel.alertMessage = nil }
            } message: {
                Text(viewModel.alertMessage ?? "")
            }
        }
    }

    private var preview: some View {
        ZStack {
            Color(white: 0.055)
            if let image = viewModel.displayedImage {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if viewModel.isLoadingImage {
                ProgressView()
                    .controlSize(.large)
            } else {
                VStack(spacing: 16) {
                    Image(systemName: "photo.badge.plus")
                        .font(.system(size: 42, weight: .light))
                        .foregroundStyle(.secondary)
                    HStack(spacing: 10) {
                        Button {
                            showPhotoPicker = true
                        } label: {
                            Label("Choose Photos", systemImage: "photo")
                        }
                        .buttonStyle(.borderedProminent)
                        Button(action: viewModel.loadTestImage) {
                            Label("Test Image", systemImage: "testtube.2")
                        }
                        .buttonStyle(.bordered)
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .clipped()
    }

    private var controlPanel: some View {
        VStack(spacing: 11) {
            if viewModel.result != nil {
                Picker("Preview", selection: $viewModel.previewMode) {
                    ForEach(PreviewMode.allCases) { mode in
                        Text(mode.rawValue).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
            }

            HStack(spacing: 8) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(viewModel.sourceName.isEmpty ? "No photo selected" : viewModel.sourceName)
                        .font(.subheadline.weight(.semibold))
                        .lineLimit(1)
                    Text(viewModel.imageDetails)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer(minLength: 8)
                Text(viewModel.status)
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.trailing)
                    .lineLimit(2)
            }

            Picker("Processor", selection: $viewModel.backend) {
                ForEach(DenoiseBackend.allCases) { backend in
                    Text(backend.pickerLabel)
                        .tag(backend)
                        .disabled(
                            !viewModel.isBackendAvailable(backend)
                                || (viewModel.highOverlap
                                    && backend == .gpuAndNeuralEngine)
                        )
                }
            }
            .pickerStyle(.segmented)
            .disabled(viewModel.isProcessing)
            .onChange(of: viewModel.backend) { _, selected in
                if !viewModel.isBackendAvailable(selected)
                    || (viewModel.highOverlap && selected == .gpuAndNeuralEngine) {
                    viewModel.backend = .neuralEngine
                    viewModel.validateBackendSelection()
                }
            }

            if let status = viewModel.neuralEngineStatus {
                Text(status)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            Picker("Model quality", selection: $viewModel.quality) {
                ForEach(DenoiseQuality.allCases) { quality in
                    Text(quality.pickerLabel).tag(quality)
                }
            }
            .pickerStyle(.segmented)
            .disabled(viewModel.isProcessing)
            .onChange(of: viewModel.quality) { _, _ in
                viewModel.qualityDidChange()
            }

            Toggle("High overlap", isOn: $viewModel.highOverlap)
                .font(.subheadline)
                .disabled(viewModel.isProcessing)
                .onChange(of: viewModel.highOverlap) { _, enabled in
                    if enabled && viewModel.backend == .gpuAndNeuralEngine {
                        viewModel.backend = .neuralEngine
                    }
                }

            if viewModel.isProcessing {
                VStack(spacing: 5) {
                    if let fraction = viewModel.overallProgress {
                        ProgressView(value: fraction)
                    } else {
                        ProgressView()
                    }
                    HStack {
                        Text(progressLabel)
                        Spacer()
                        TimelineView(.periodic(from: .now, by: 1)) { context in
                            Text(elapsedLabel(at: context.date))
                        }
                    }
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.secondary)
                }
            }

            HStack(spacing: 12) {
                if viewModel.isProcessing {
                    Button(role: .cancel) {
                        showCancelConfirmation = true
                    } label: {
                        Label("Cancel", systemImage: "xmark")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .alert(
                        "Cancel denoising?",
                        isPresented: $showCancelConfirmation
                    ) {
                        Button("Continue", role: .cancel) {}
                        Button("Cancel denoising", role: .destructive) {
                            viewModel.cancelProcessing()
                        }
                    } message: {
                        Text("The current image and remaining batch will not be processed.")
                    }
                } else {
                    Button(action: viewModel.runDenoise) {
                        Label("Denoise", systemImage: "wand.and.stars")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!viewModel.canRun)
                }

                if viewModel.result != nil {
                    Button(action: viewModel.saveResultToPhotos) {
                        Image(systemName: viewModel.isSaving ? "hourglass" : "square.and.arrow.down")
                            .frame(width: 30)
                    }
                    .buttonStyle(.bordered)
                    .disabled(viewModel.isSaving)
                    .accessibilityLabel("Save to Photos")

                    ShareLink(items: viewModel.results.map(\.url)) {
                        Image(systemName: "square.and.arrow.up")
                            .frame(width: 30)
                    }
                    .buttonStyle(.bordered)
                    .accessibilityLabel("Share result")
                }
            }
        }
        .padding(.horizontal, 14)
        .padding(.top, 12)
        .padding(.bottom, 10)
        .background(Color(white: 0.09))
    }

    @ToolbarContentBuilder
    private var importToolbar: some ToolbarContent {
        ToolbarItemGroup(placement: .topBarTrailing) {
            Button {
                showPhotoPicker = true
            } label: {
                Image(systemName: "photo.badge.plus")
            }
            .accessibilityLabel("Choose photos")
            Button { showFileImporter = true } label: {
                Image(systemName: "folder")
            }
            .accessibilityLabel("Choose from Files")
            Button(action: viewModel.loadTestImage) {
                Image(systemName: "testtube.2")
            }
            .accessibilityLabel("Load ISO 51200 test image")
        }
    }

    private var progressLabel: String {
        guard let progress = viewModel.progress else { return "Preparing" }
        if progress.phase == .processing {
            return "\(progress.completedTiles) / \(progress.totalTiles) tiles"
        }
        return progress.phase.rawValue
    }

    private func elapsedLabel(at date: Date) -> String {
        let seconds: TimeInterval
        if let started = viewModel.processingStartedAt {
            seconds = date.timeIntervalSince(started)
        } else if let elapsed = viewModel.progress?.elapsedSeconds {
            seconds = elapsed
        } else {
            return ""
        }
        if seconds < 60 { return String(format: "%.0fs", seconds) }
        return "\(Int(seconds) / 60)m \(Int(seconds) % 60)s"
    }

}
