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

    func availableAPIs(timeout: TimeInterval = 15) async throws -> Set<String> {
        let result = try await call("getAvailableApiList", timeout: timeout)
        return Set((result.first as? [String]) ?? [])
    }

    func negotiateEventVersion() async throws -> String {
        do {
            let versions = try await call("getVersions")
                .first as? [String] ?? []
            guard versions.contains("1.2") else {
                eventVersion = "1.0"
                return eventVersion
            }
            let methods = try await call("getMethodTypes", params: ["1.2"])
            let supportsExtendedEvents = methods.contains { value in
                guard let definition = value as? [Any], definition.count >= 4 else {
                    return false
                }
                return definition[0] as? String == "getEvent" &&
                    definition[3] as? String == "1.2"
            }
            eventVersion = supportsExtendedEvents ? "1.2" : "1.0"
        } catch let error as CameraRPCError where error.code == 12 || error.code == 14 {
            eventVersion = "1.0"
        }
        return eventVersion
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

    func startLiveView(
        quality: LiveViewQuality = .high,
        availableAPIs: Set<String>
    ) async throws -> URL {
        let result: [Any]
        if availableAPIs.contains("startLiveviewWithSize") {
            result = try await call("startLiveviewWithSize", params: [quality.rawValue])
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
        for id in CameraSettingID.allCases {
            if let value = await setting(id, availableAPIs: availableAPIs) {
                output[id] = value
            }
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
            result = try await call("getEvent", params: [longPolling], version: eventVersion, timeout: longPolling ? 75 : 15)
        } catch let error as CameraRPCError where error.code == 2 {
            // Sony reports an idle long-poll expiry as RPC error 2. It does not
            // mean event version 1.2 is unsupported.
            return CameraEventSnapshot()
        } catch let error as CameraRPCError
            where eventVersion == "1.2" && (error.code == 12 || error.code == 14) {
            eventVersion = "1.0"
            result = try await call("getEvent", params: [longPolling], version: eventVersion, timeout: longPolling ? 75 : 15)
        }
        return Self.parseEventResult(result)
    }

    static func parseEventResult(_ result: [Any]) -> CameraEventSnapshot {
        var snapshot = CameraEventSnapshot()
        func visit(_ value: Any) {
            if let values = value as? [Any] {
                values.forEach(visit)
                return
            }
            guard let event = value as? [String: Any] else { return }
            defer { event.values.forEach(visit) }
            guard let type = event["type"] as? String else { return }
            if type == "takePicture", let values = event["takePictureUrl"] as? [String] {
                snapshot.urls.append(contentsOf: values.compactMap(URL.init(string:)))
            }
            if type == "contShooting", let values = event["contShootingUrl"] as? [[String: Any]] {
                snapshot.urls.append(contentsOf: values.compactMap { item in
                    (item["postviewUrl"] as? String).flatMap(URL.init(string:))
                })
            }
            if type == "cameraStatus" { snapshot.status = event["cameraStatus"] as? String }
            if type == "availableApiList", let names = event["names"] as? [String] {
                snapshot.availableAPIs = Set(names)
            }
            let settingEvent: (CameraSettingID, [String])? = switch type {
            case "exposureMode": (.exposureMode, ["exposureMode"])
            case "fNumber": (.aperture, ["currentFNumber", "fNumber"])
            case "shutterSpeed": (.shutterSpeed, ["currentShutterSpeed", "shutterSpeed"])
            case "isoSpeedRate": (.iso, ["currentIsoSpeedRate", "isoSpeedRate"])
            case "exposureCompensation":
                (.exposureCompensation, ["currentExposureCompensation", "exposureCompensation"])
            case "contShootingMode": (.drive, ["contShootingMode"])
            case "contShootingSpeed": (.burstSpeed, ["contShootingSpeed"])
            default: nil
            }
            if let (id, keys) = settingEvent,
               let settingValue = keys.lazy.compactMap({ Self.stringValue(event[$0]) }).first {
                snapshot.settingValues[id] = settingValue
            }
            if type == "zoomInformation" {
                snapshot.zoomPosition = Self.intValue(event["zoomPosition"])
                snapshot.zoomBoxCount = Self.intValue(event["zoomNumberBox"])
                snapshot.zoomBoxIndex = Self.intValue(event["zoomIndexCurrentBox"])
                    ?? Self.intValue(event["zoomIndexCurrent"])
            }
            if type == "zoomSetting" { snapshot.zoomSetting = event["zoom"] as? String }
        }
        result.forEach(visit)
        return snapshot
    }

    static func intValue(_ value: Any?) -> Int? {
        switch value {
        case let value as Int: value
        case let value as NSNumber: value.intValue
        case let value as String: Int(value)
        default: nil
        }
    }

    private static func stringValue(_ value: Any?) -> String? {
        switch value {
        case let value as String: value
        case let value as Int: String(value)
        case let value as NSNumber: value.stringValue
        default: nil
        }
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

    private func setting(
        _ id: CameraSettingID,
        availableAPIs: Set<String>
    ) async -> CameraSetting? {
        let timeout: TimeInterval = 4
        switch id {
        case .aperture:
            return try? await stringSetting(id, "getAvailableFNumber", availableAPIs, timeout)
        case .shutterSpeed:
            return try? await stringSetting(id, "getAvailableShutterSpeed", availableAPIs, timeout)
        case .iso:
            return try? await stringSetting(id, "getAvailableIsoSpeedRate", availableAPIs, timeout)
        case .exposureMode:
            if availableAPIs.contains("getAvailableExposureMode"),
               let result = try? await call("getAvailableExposureMode", timeout: timeout),
               let current = result.first as? String {
                return CameraSetting(
                    id: id,
                    current: current,
                    options: result.dropFirst().first as? [String] ?? [current],
                    writable: availableAPIs.contains("setExposureMode")
                )
            }
            guard availableAPIs.contains("getExposureMode"),
                  let result = try? await call("getExposureMode", timeout: timeout),
                  let current = result.first as? String else { return nil }
            return CameraSetting(id: id, current: current, options: [current], writable: false)
        case .drive, .burstSpeed:
            let isDrive = id == .drive
            let method = isDrive ? "getAvailableContShootingMode" : "getAvailableContShootingSpeed"
            guard availableAPIs.contains(method),
                  let result = try? await call(method, timeout: timeout),
                  let object = result.first as? [String: Any] else { return nil }
            let key = isDrive ? "contShootingMode" : "contShootingSpeed"
            guard let current = object[key] as? String else { return nil }
            return CameraSetting(
                id: id,
                current: current,
                options: object["candidate"] as? [String] ?? [current]
            )
        case .exposureCompensation:
            guard availableAPIs.contains("getAvailableExposureCompensation"),
                  let result = try? await call("getAvailableExposureCompensation", timeout: timeout),
                  result.count >= 3,
                  let current = result[0] as? Int,
                  let upper = result[1] as? Int,
                  let lower = result[2] as? Int else { return nil }
            return CameraSetting(
                id: id,
                current: String(current),
                options: Array(lower...upper).map(String.init)
            )
        }
    }

    private func stringSetting(
        _ id: CameraSettingID,
        _ method: String,
        _ availableAPIs: Set<String>,
        _ timeout: TimeInterval
    ) async throws -> CameraSetting? {
        guard availableAPIs.contains(method) else { return nil }
        let result = try await call(method, timeout: timeout)
        guard let current = result.first as? String else { return nil }
        return CameraSetting(id: id, current: current, options: result.dropFirst().first as? [String] ?? [current])
    }
}
