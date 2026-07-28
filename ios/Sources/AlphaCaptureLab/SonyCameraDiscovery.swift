import Foundation
import Network
import OSLog

enum SonyCameraDiscovery {
    private static let logger = Logger(subsystem: "com.ryu.remotecapture.ios", category: "camera-discovery")

    static func cameraEndpoint(host: String) async throws -> URL {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.waitsForConnectivity = false
        configuration.timeoutIntervalForRequest = 4
        configuration.timeoutIntervalForResource = 6
        let session = URLSession(configuration: configuration)

        if let advertisedURL = await SSDPDiscovery.discover(cameraHost: host) {
            Self.logger.info("SSDP advertised description: \(advertisedURL.absoluteString, privacy: .public)")
            do {
                var request = URLRequest(url: advertisedURL)
                request.timeoutInterval = 5
                let (data, response) = try await session.data(for: request)
                let status = (response as? HTTPURLResponse)?.statusCode ?? 0
                Self.logger.info("SSDP description response: status=\(status) bytes=\(data.count)")
                if let endpoint = DeviceDescriptionParser.cameraEndpoint(data: data, relativeTo: advertisedURL) {
                    Self.logger.info("Discovered camera endpoint: \(endpoint.absoluteString, privacy: .public)")
                    return endpoint
                }
            } catch {
                Self.logger.error("SSDP description failed: \(error.localizedDescription, privacy: .public)")
            }
        } else {
            Self.logger.info("Camera did not answer unicast SSDP")
        }

        // Sony's legacy cameras normally advertise dd.xml on 64321. Trying the
        // known description endpoints avoids multicast entitlement requirements.
        let descriptions = [64321, 61000].flatMap { port in
            ["scalarwebapi_dd.xml", "dd.xml", "DmsDesc.xml", "sony/desc", ""].compactMap {
                URL(string: "http://\(host):\(port)/\($0)")
            }
        }

        for url in descriptions {
            do {
                Self.logger.info("Trying camera description: \(url.absoluteString, privacy: .public)")
                var request = URLRequest(url: url)
                request.timeoutInterval = 4
                let (data, response) = try await session.data(for: request)
                guard let http = response as? HTTPURLResponse else { continue }
                Self.logger.info("Description response: port=\(url.port ?? 0) path=\(url.path, privacy: .public) status=\(http.statusCode) bytes=\(data.count)")
                if let endpoint = DeviceDescriptionParser.cameraEndpoint(data: data, relativeTo: url) {
                    Self.logger.info("Discovered camera endpoint: \(endpoint.absoluteString, privacy: .public)")
                    return endpoint
                }
            } catch {
                Self.logger.debug("Description unavailable at \(url.absoluteString, privacy: .public): \(error.localizedDescription, privacy: .public)")
            }
        }

        guard let fallback = URL(string: "http://\(host):8080/sony/camera") else {
            throw URLError(.badURL)
        }
        Self.logger.info("Using legacy fallback endpoint: \(fallback.absoluteString, privacy: .public)")
        return fallback
    }
}

private enum SSDPDiscovery {
    static func discover(cameraHost: String) async -> URL? {
        var destinations = [cameraHost]
        let octets = cameraHost.split(separator: ".")
        if octets.count == 4 {
            destinations.append("\(octets[0]).\(octets[1]).\(octets[2]).255")
        }
        destinations.append("239.255.255.250")
        for destination in destinations {
            if let location = await discover(destination: destination) { return location }
        }
        return nil
    }

    private static func discover(destination: String) async -> URL? {
        await withCheckedContinuation { continuation in
            let attempt = Attempt(continuation: continuation)
            let connection = NWConnection(host: NWEndpoint.Host(destination), port: 1900, using: .udp)
            attempt.connection = connection
            connection.stateUpdateHandler = { state in
                switch state {
                case .ready:
                    let request = [
                        "M-SEARCH * HTTP/1.1",
                        "HOST: 239.255.255.250:1900",
                        "MAN: \"ssdp:discover\"",
                        "MX: 1",
                        "ST: urn:schemas-sony-com:service:ScalarWebAPI:1",
                        "",
                        "",
                    ].joined(separator: "\r\n")
                    connection.send(content: request.data(using: .ascii), completion: .contentProcessed { error in
                        guard error == nil else {
                            attempt.finish(nil)
                            return
                        }
                        connection.receiveMessage { data, _, _, _ in
                            let response = data.flatMap { String(data: $0, encoding: .ascii) }
                            attempt.finish(response.flatMap(Self.location))
                        }
                    })
                case .failed, .cancelled:
                    attempt.finish(nil)
                default:
                    break
                }
            }
            connection.start(queue: .global(qos: .userInitiated))
            DispatchQueue.global().asyncAfter(deadline: .now() + 2.5) {
                attempt.finish(nil)
            }
        }
    }

    private static func location(response: String) -> URL? {
        for line in response.components(separatedBy: .newlines) {
            let parts = line.split(separator: ":", maxSplits: 1)
            guard parts.count == 2,
                  parts[0].trimmingCharacters(in: .whitespaces).lowercased() == "location" else { continue }
            return URL(string: parts[1].trimmingCharacters(in: .whitespacesAndNewlines))
        }
        return nil
    }

    private final class Attempt: @unchecked Sendable {
        private let lock = NSLock()
        private var continuation: CheckedContinuation<URL?, Never>?
        var connection: NWConnection?

        init(continuation: CheckedContinuation<URL?, Never>) {
            self.continuation = continuation
        }

        func finish(_ result: URL?) {
            let pending = lock.withLock { () -> CheckedContinuation<URL?, Never>? in
                defer { continuation = nil }
                return continuation
            }
            guard let pending else { return }
            connection?.cancel()
            pending.resume(returning: result)
        }
    }
}

private final class DeviceDescriptionParser: NSObject, XMLParserDelegate {
    private var currentElement = ""
    private var currentText = ""
    private var serviceType: String?
    private var actionURL: String?
    private var cameraActionURL: String?

    static func cameraEndpoint(data: Data, relativeTo descriptionURL: URL) -> URL? {
        let delegate = DeviceDescriptionParser()
        let parser = XMLParser(data: data)
        parser.delegate = delegate
        guard parser.parse(), let value = delegate.cameraActionURL,
              let base = URL(string: value, relativeTo: descriptionURL)?.absoluteURL else { return nil }
        if base.path.lowercased().hasSuffix("/camera") { return base }
        return base.appendingPathComponent("camera")
    }

    func parser(
        _ parser: XMLParser,
        didStartElement elementName: String,
        namespaceURI: String?,
        qualifiedName qName: String?,
        attributes attributeDict: [String: String] = [:]
    ) {
        currentElement = elementName.split(separator: ":").last.map(String.init) ?? elementName
        currentText = ""
        if currentElement == "X_ScalarWebAPI_Service" {
            serviceType = nil
            actionURL = nil
        }
    }

    func parser(_ parser: XMLParser, foundCharacters string: String) {
        currentText += string
    }

    func parser(
        _ parser: XMLParser,
        didEndElement elementName: String,
        namespaceURI: String?,
        qualifiedName qName: String?
    ) {
        let element = elementName.split(separator: ":").last.map(String.init) ?? elementName
        let value = currentText.trimmingCharacters(in: .whitespacesAndNewlines)
        if element == "X_ScalarWebAPI_ServiceType" { serviceType = value }
        if element == "X_ScalarWebAPI_ActionList_URL" { actionURL = value }
        if element == "X_ScalarWebAPI_Service",
           serviceType?.lowercased() == "camera",
           let actionURL {
            cameraActionURL = actionURL
        }
        currentElement = ""
        currentText = ""
    }
}
