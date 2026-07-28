import Combine
import Foundation

@MainActor
final class AppPreferences: ObservableObject {
    @Published var downloadQuality: DownloadQuality { didSet { save() } }
    @Published var liveViewQuality: LiveViewQuality { didSet { save() } }
    @Published var outputFormat: OutputFormat { didSet { save() } }
    @Published var geotagging: Bool { didSet { save() } }
    @Published var liveViewTimeoutMinutes: Int { didSet { save() } }
    @Published var autoDenoise: AutoDenoiseMode { didSet { save() } }
    @Published var denoiseISOThreshold: Int { didSet { save() } }
    @Published var bakeLUTIntoCapture: Bool { didSet { save() } }
    @Published var lutSelection: LUTSelection { didSet { save() } }
    @Published var liveNDStops: Int { didSet { save() } }

    private let defaults = UserDefaults.standard
    private var loading = true

    init() {
        downloadQuality = DownloadQuality(rawValue: defaults.string(forKey: "downloadQuality") ?? "") ?? .original
        liveViewQuality = LiveViewQuality(rawValue: defaults.string(forKey: "liveViewQuality") ?? "") ?? .standard
        outputFormat = OutputFormat(rawValue: defaults.string(forKey: "outputFormat") ?? "") ?? .jpeg
        geotagging = defaults.bool(forKey: "geotagging")
        liveViewTimeoutMinutes = defaults.object(forKey: "liveViewTimeoutMinutes") as? Int ?? 0
        autoDenoise = AutoDenoiseMode(rawValue: defaults.string(forKey: "autoDenoise") ?? "") ?? .off
        denoiseISOThreshold = defaults.object(forKey: "denoiseISOThreshold") as? Int ?? 3200
        bakeLUTIntoCapture = defaults.bool(forKey: "bakeLUTIntoCapture")
        lutSelection = (try? defaults.data(forKey: "lutSelection").flatMap { try JSONDecoder().decode(LUTSelection.self, from: $0) }) ?? LUTSelection()
        liveNDStops = defaults.object(forKey: "liveNDStops") as? Int ?? 3
        loading = false
    }

    private func save() {
        guard !loading else { return }
        defaults.set(downloadQuality.rawValue, forKey: "downloadQuality")
        defaults.set(liveViewQuality.rawValue, forKey: "liveViewQuality")
        defaults.set(outputFormat.rawValue, forKey: "outputFormat")
        defaults.set(geotagging, forKey: "geotagging")
        defaults.set(liveViewTimeoutMinutes, forKey: "liveViewTimeoutMinutes")
        defaults.set(autoDenoise.rawValue, forKey: "autoDenoise")
        defaults.set(denoiseISOThreshold, forKey: "denoiseISOThreshold")
        defaults.set(bakeLUTIntoCapture, forKey: "bakeLUTIntoCapture")
        defaults.set(try? JSONEncoder().encode(lutSelection), forKey: "lutSelection")
        defaults.set(liveNDStops, forKey: "liveNDStops")
    }
}
