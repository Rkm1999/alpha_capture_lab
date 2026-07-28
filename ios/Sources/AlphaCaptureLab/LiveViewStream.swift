import Foundation
import OSLog

final class LiveViewStream: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    private static let logger = Logger(subsystem: "com.ryu.remotecapture.ios", category: "live-view")
    private var session: URLSession?
    private var buffer = Data()
    private let lock = NSLock()
    var onFrame: (@Sendable (Data) -> Void)?
    var onFailure: (@Sendable (Error) -> Void)?

    func start(url: URL) {
        stop()
        let configuration = URLSessionConfiguration.default
        configuration.timeoutIntervalForRequest = 15
        configuration.timeoutIntervalForResource = .infinity
        configuration.waitsForConnectivity = false
        let session = URLSession(configuration: configuration, delegate: self, delegateQueue: nil)
        self.session = session
        Self.logger.info("Starting live-view stream: \(url.absoluteString, privacy: .public)")
        session.dataTask(with: url).resume()
    }

    func stop() {
        session?.invalidateAndCancel()
        session = nil
        lock.withLock { buffer.removeAll(keepingCapacity: false) }
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        let frames = lock.withLock { () -> [Data] in
            buffer.append(data)
            return parseFrames()
        }
        frames.forEach { onFrame?($0) }
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: Error?
    ) {
        if let error, (error as? URLError)?.code != .cancelled { onFailure?(error) }
        if let error, (error as? URLError)?.code != .cancelled {
            Self.logger.error("Live-view stream failed: \(error.localizedDescription, privacy: .public)")
        }
    }

    private func parseFrames() -> [Data] {
        var frames: [Data] = []
        while true {
            guard let start = buffer.firstIndex(of: 0xff) else {
                buffer.removeAll(keepingCapacity: true)
                break
            }
            if start > 0 { buffer.removeSubrange(0..<start) }
            guard buffer.count >= 136 else { break }
            let type = buffer[1]
            let payloadHeader = 8
            guard buffer[payloadHeader] == 0x24,
                  buffer[payloadHeader + 1] == 0x35,
                  buffer[payloadHeader + 2] == 0x68,
                  buffer[payloadHeader + 3] == 0x79 else {
                buffer.removeFirst()
                continue
            }
            let payloadSize = Int(buffer[payloadHeader + 4]) << 16
                | Int(buffer[payloadHeader + 5]) << 8
                | Int(buffer[payloadHeader + 6])
            let paddingSize = Int(buffer[payloadHeader + 7])
            guard payloadSize <= 20 * 1024 * 1024 else {
                buffer.removeFirst()
                continue
            }
            let total = 136 + payloadSize + paddingSize
            guard buffer.count >= total else { break }
            if type == 0x01 {
                frames.append(buffer.subdata(in: 136..<(136 + payloadSize)))
            }
            buffer.removeSubrange(0..<total)
        }
        return frames
    }
}
