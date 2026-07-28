import Foundation
import OSLog

actor SonyCameraAPI {
    private static let logger = Logger(subsystem: "com.ryu.remotecapture.ios", category: "camera-rpc")
    private let endpoint: URL
    private let session: URLSession
    private var requestID = 1
    private var eventVersion = "1.2"

    init(endpoint: URL) {
        self.endpoint = endpoint
        let configuration = URLSessionConfiguration.default
        configuration.timeoutIntervalForRequest = 15
        configuration.timeoutIntervalForResource = 120
        // Camera access points intentionally have no internet route. Waiting for
        // general connectivity can otherwise leave a local request suspended.
        configuration.waitsForConnectivity = false
        self.session = URLSession(configuration: configuration)
    }

    func availableAPIs() async throws -> Set<String> {
        let result = try await call("getAvailableApiList")
        return Set((result.first as? [String]) ?? [])
    }

    func startRemoteModeIfNeeded() async throws -> Set<String> {
        var apis = try await availableAPIs()
        if apis.contains("startRecMode") {
            _ = try await call("startRecMode")
            for _ in 0..<8 {
                try await Task.sleep(for: .milliseconds(350))
                apis = try await availableAPIs()
                if !apis.contains("startRecMode") { break }
            }
        }
        return apis
    }

    func setPostviewSize(_ quality: DownloadQuality, availableAPIs: Set<String>) async throws {
        guard availableAPIs.contains("setPostviewImageSize") else { return }
        _ = try await call("setPostviewImageSize", params: [quality.sonyValue])
    }

    func startLiveView(availableAPIs: Set<String>) async throws -> URL {
        let result: [Any]
        if availableAPIs.contains("startLiveviewWithSize") {
            result = try await call("startLiveviewWithSize", params: ["L"])
        } else {
            result = try await call("startLiveview")
        }
        guard let value = result.first as? String, let url = URL(string: value) else {
            throw URLError(.badServerResponse)
        }
        return url
    }

    func stopLiveView() async {
        _ = try? await call("stopLiveview")
    }

    func settings(availableAPIs: Set<String>) async -> [CameraSettingID: CameraSetting] {
        var output: [CameraSettingID: CameraSetting] = [:]
        let definitions: [(CameraSettingID, String, String, String)] = [
            (.aperture, "getAvailableFNumber", "fNumber", "fNumberCandidates"),
            (.shutterSpeed, "getAvailableShutterSpeed", "shutterSpeed", "shutterSpeedCandidates"),
            (.iso, "getAvailableIsoSpeedRate", "isoSpeedRate", "isoSpeedRateCandidates"),
        ]
        for (id, method, currentKey, candidatesKey) in definitions where availableAPIs.contains(method) {
            if let setting = try? await stringSetting(id, method, currentKey, candidatesKey) { output[id] = setting }
        }
        if availableAPIs.contains("getAvailableExposureMode"),
           let result = try? await call("getAvailableExposureMode"),
           let current = result.first as? String, let options = result.dropFirst().first as? [String] {
            output[.exposureMode] = CameraSetting(id: .exposureMode, current: current, options: options)
        }
        if availableAPIs.contains("getAvailableContShootingMode"),
           let result = try? await call("getAvailableContShootingMode"),
           let object = result.first as? [String: Any],
           let current = object["contShootingMode"] as? String {
            output[.drive] = CameraSetting(id: .drive, current: current, options: object["candidate"] as? [String] ?? [current])
        }
        if availableAPIs.contains("getAvailableContShootingSpeed"),
           let result = try? await call("getAvailableContShootingSpeed"),
           let object = result.first as? [String: Any],
           let current = object["contShootingSpeed"] as? String {
            output[.burstSpeed] = CameraSetting(id: .burstSpeed, current: current, options: object["candidate"] as? [String] ?? [current])
        }
        if availableAPIs.contains("getAvailableExposureCompensation"),
           let result = try? await call("getAvailableExposureCompensation"), result.count >= 3,
           let current = result[0] as? Int, let upper = result[1] as? Int, let lower = result[2] as? Int {
            output[.exposureCompensation] = CameraSetting(id: .exposureCompensation, current: String(current), options: Array(lower...upper).map(String.init))
        }
        return output
    }

    func setSetting(_ id: CameraSettingID, value: String) async throws {
        switch id {
        case .drive: _ = try await call("setContShootingMode", params: [["contShootingMode": value]])
        case .burstSpeed: _ = try await call("setContShootingSpeed", params: [["contShootingSpeed": value]])
        case .exposureMode: _ = try await call("setExposureMode", params: [value])
        case .aperture: _ = try await call("setFNumber", params: [value])
        case .shutterSpeed: _ = try await call("setShutterSpeed", params: [value])
        case .iso: _ = try await call("setIsoSpeedRate", params: [value])
        case .exposureCompensation: _ = try await call("setExposureCompensation", params: [Int(value) ?? 0])
        }
    }

    func setLiveViewQuality(_ quality: LiveViewQuality, availableAPIs: Set<String>) async throws {
        guard availableAPIs.contains("setLiveviewSize") else { return }
        _ = try await call("setLiveviewSize", params: [quality.rawValue])
    }

    func startContinuousShooting() async throws { _ = try await call("startContShooting") }
    func stopContinuousShooting() async throws { _ = try await call("stopContShooting", timeout: 30) }

    func zoom(direction: String, movement: String) async throws {
        _ = try await call("actZoom", params: [direction, movement])
    }

    func takePicture() async throws -> URL {
        let result = try await call("actTakePicture", timeout: 120)
        guard let urls = result.first as? [String], let value = urls.first,
              let url = URL(string: value) else {
            throw URLError(.badServerResponse)
        }
        return url
    }

    func event(longPolling: Bool) async throws -> CameraEventSnapshot {
        let result: [Any]
        do {
            result = try await call("getEvent", params: [longPolling], version: eventVersion, timeout: longPolling ? 40 : 15)
        } catch is CameraRPCError where eventVersion == "1.2" {
            eventVersion = "1.0"
            result = try await call("getEvent", params: [longPolling], version: eventVersion, timeout: longPolling ? 40 : 15)
        }
        var snapshot = CameraEventSnapshot()
        for value in result {
            guard let event = value as? [String: Any], let type = event["type"] as? String else { continue }
            if type == "takePicture", let values = event["takePictureUrl"] as? [String] {
                snapshot.urls.append(contentsOf: values.compactMap(URL.init(string:)))
            }
            if type == "contShooting", let values = event["contShootingUrl"] as? [[String: Any]] {
                snapshot.urls.append(contentsOf: values.compactMap { item in
                    (item["postviewUrl"] as? String).flatMap(URL.init(string:))
                })
            }
            if type == "cameraStatus" { snapshot.status = event["cameraStatus"] as? String }
            if type == "zoomInformation" {
                snapshot.zoomPosition = event["zoomPosition"] as? Int
                snapshot.zoomBoxCount = event["zoomNumberBox"] as? Int
                snapshot.zoomBoxIndex = event["zoomIndexCurrentBox"] as? Int ?? event["zoomIndexCurrent"] as? Int
            }
            if type == "zoomSetting" { snapshot.zoomSetting = event["zoom"] as? String }
        }
        return snapshot
    }

    private func call(
        _ method: String,
        params: [Any] = [],
        version: String = "1.0",
        timeout: TimeInterval = 15
    ) async throws -> [Any] {
        let id = requestID
        requestID += 1
        Self.logger.debug("RPC start: \(method, privacy: .public) id=\(id)")
        let body: [String: Any] = [
            "method": method,
            "params": params,
            "id": id,
            "version": version,
        ]
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.timeoutInterval = timeout
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, response): (Data, URLResponse)
        do {
            (data, response) = try await session.data(for: request)
        } catch {
            Self.logger.error("RPC failed: \(method, privacy: .public): \(error.localizedDescription, privacy: .public)")
            throw error
        }
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw URLError(.badServerResponse)
        }
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw URLError(.cannotParseResponse)
        }
        if let error = json["error"] as? [Any], let code = error.first as? Int {
            let message = error.dropFirst().first as? String ?? "Request failed"
            Self.logger.error("RPC camera error: \(method, privacy: .public) code=\(code) message=\(message, privacy: .public)")
            throw CameraRPCError(code: code, message: message)
        }
        guard let result = json["result"] as? [Any] else {
            throw URLError(.cannotParseResponse)
        }
        Self.logger.debug("RPC complete: \(method, privacy: .public) id=\(id)")
        return result
    }

    private func stringSetting(
        _ id: CameraSettingID,
        _ method: String,
        _ currentKey: String,
        _ candidatesKey: String
    ) async throws -> CameraSetting? {
        let result = try await call(method)
        guard let current = result.first as? String else { return nil }
        return CameraSetting(id: id, current: current, options: result.dropFirst().first as? [String] ?? [current])
    }
}
