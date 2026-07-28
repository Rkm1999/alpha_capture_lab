import Foundation
import XCTest
@testable import AlphaCaptureLab

final class AlphaCaptureLabTests: XCTestCase {
func testCubeLUTParsesExpectedDimension() throws {
    let values = (0..<8).map { index in
        "\((index >> 0) & 1) \((index >> 1) & 1) \((index >> 2) & 1)"
    }.joined(separator: "\n")
    let lut = try CubeLUT.parse("LUT_3D_SIZE 2\n\(values)", title: "Test")
    XCTAssertEqual(lut.dimension, 2)
    XCTAssertEqual(lut.title, "Test")
}

func testSavedPhotoMetadataRoundTrip() throws {
    let photo = SavedPhoto(
        id: "image.jpg", url: URL(fileURLWithPath: "/image.jpg"), capturedAt: Date(timeIntervalSince1970: 12),
        kind: .liveND, lutIdentifier: "Cinema", lutStrength: 0.7, iso: 3200
    )
    let decoded = try JSONDecoder().decode(SavedPhoto.self, from: JSONEncoder().encode(photo))
    XCTAssertEqual(decoded, photo)
}
}
