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
        let geometry = EditorGeometry(
            crop: .init(left: 0.1, top: 0.2, right: 0.9, bottom: 0.8),
            quarterTurns: 1,
            straightenDegrees: 1.5
        )
        let photo = SavedPhoto(
            id: "image.jpg",
            url: URL(fileURLWithPath: "/image.jpg"),
            capturedAt: Date(timeIntervalSince1970: 12),
            kind: .liveND,
            lutIdentifier: "Cinema",
            lutStrength: 0.7,
            iso: 3200,
            geometry: geometry,
            derivedFromID: "source.jpg",
            photoLibraryIdentifier: "photos-id",
            originalFilename: "DSC0001.JPG"
        )
        let decoded = try JSONDecoder().decode(SavedPhoto.self, from: JSONEncoder().encode(photo))
        XCTAssertEqual(decoded, photo)
    }

    func testLegacySavedPhotoDecodesWithoutNewEditFields() throws {
        let json = """
        {
          "id": "legacy.jpg",
          "url": "file:///legacy.jpg",
          "capturedAt": 12,
          "kind": "Photo",
          "sourceURLs": []
        }
        """.data(using: .utf8)!
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .secondsSince1970
        let photo = try decoder.decode(SavedPhoto.self, from: json)
        XCTAssertNil(photo.geometry)
        XCTAssertNil(photo.derivedFromID)
        XCTAssertNil(photo.photoLibraryIdentifier)
    }

    func testSavedPhotoPreservesCaptureKindWhenDenoiseModelIsUnknown() throws {
        let json = """
        {
          "id": "live-nd.jpg",
          "url": "file:///live-nd.jpg",
          "capturedAt": 12,
          "kind": "Live ND",
          "sourceURLs": [],
          "denoiseModel": "Removed model"
        }
        """.data(using: .utf8)!
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .secondsSince1970
        let photo = try decoder.decode(SavedPhoto.self, from: json)
        XCTAssertEqual(photo.kind, .liveND)
        XCTAssertNil(photo.denoiseModel)
    }

    func testComputationalSessionRestoresPrivateFramesAndDeduplicatesURLs() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ComputationalSessionStore(root: root)
        let state = ComputationalSessionState(
            id: UUID(),
            mode: .liveND,
            targetCount: 2,
            startedAt: Date(),
            quality: .original,
            outputFormat: .jpeg,
            geotagging: false,
            lut: LUTSelection(),
            autoDenoise: .off,
            denoiseThreshold: 6400,
            denoiseModel: .distilled,
            iso: 3200,
            remoteURLs: [],
            frameNames: []
        )
        try await store.begin(state)
        let url = URL(string: "http://camera/frame-1.jpg")!
        _ = try await store.append(Data([1, 2, 3]), remoteURL: url)
        _ = try await store.append(Data([9, 9, 9]), remoteURL: url)
        let restored = try XCTUnwrap(try await store.load())
        XCTAssertEqual(restored.state.mode, .liveND)
        XCTAssertEqual(restored.frames, [Data([1, 2, 3])])
    }

    func testCropNormalizationKeepsMinimumAreaInsideImage() {
        let crop = NormalizedCrop(left: -0.4, top: 0.98, right: 1.5, bottom: 0.99)
            .normalized(minimumSize: 0.1)
        XCTAssertGreaterThanOrEqual(crop.left, 0)
        XCTAssertGreaterThanOrEqual(crop.top, 0)
        XCTAssertLessThanOrEqual(crop.right, 1)
        XCTAssertLessThanOrEqual(crop.bottom, 1)
        XCTAssertGreaterThanOrEqual(crop.right - crop.left, 0.1)
        XCTAssertGreaterThanOrEqual(crop.bottom - crop.top, 0.1)
    }

    func testGeometryNormalizesNegativeQuarterTurns() {
        var geometry = EditorGeometry()
        geometry.quarterTurns = -1
        XCTAssertEqual(geometry.normalizedQuarterTurns, 3)
        XCTAssertTrue(geometry.hasChanges)
    }

    func testExposureModeNormalizationAndPriority() {
        XCTAssertEqual(normalizedExposureMode("Program Auto"), "P")
        XCTAssertEqual(normalizedExposureMode("Aperture Priority"), "A")
        XCTAssertEqual(normalizedExposureMode("Shutter Priority"), "S")
        XCTAssertEqual(normalizedExposureMode("Manual Exposure"), "M")
        XCTAssertEqual(exposurePrioritySettingID(for: "A"), .aperture)
        XCTAssertEqual(exposurePrioritySettingID(for: "M"), .aperture)
        XCTAssertEqual(exposurePrioritySettingID(for: "S"), .shutterSpeed)
        XCTAssertNil(exposurePrioritySettingID(for: "P"))
    }

    func testPhotoGridMatchesAndroidOrder() {
        var settings = Dictionary(
            uniqueKeysWithValues: CameraSettingID.allCases.map {
                ($0, CameraSetting(id: $0, current: "value", options: ["value"]))
            }
        )
        settings[.drive]?.current = "Continuous"
        XCTAssertEqual(
            photoSettingGridOrder(settings: settings, exposureMode: "P"),
            [.aperture, .shutterSpeed, .iso, .exposureCompensation, .drive, .burstSpeed]
        )
        XCTAssertEqual(
            photoSettingGridOrder(settings: settings, exposureMode: "A"),
            [.shutterSpeed, .iso, .exposureCompensation, .drive, .burstSpeed]
        )
        XCTAssertEqual(
            photoSettingGridOrder(settings: settings, exposureMode: "S"),
            [.aperture, .iso, .exposureCompensation, .drive, .burstSpeed]
        )
        XCTAssertEqual(
            photoSettingGridOrder(settings: settings, exposureMode: "M"),
            [.shutterSpeed, .iso, .exposureCompensation, .drive, .burstSpeed]
        )
    }

    func testPhotoGridHidesBurstSpeedInSingleDrive() {
        let settings: [CameraSettingID: CameraSetting] = [
            .drive: CameraSetting(id: .drive, current: "Single", options: ["Single", "Continuous"]),
            .burstSpeed: CameraSetting(id: .burstSpeed, current: "Hi", options: ["Hi", "Low"]),
            .iso: CameraSetting(id: .iso, current: "AUTO", options: ["AUTO"]),
        ]
        XCTAssertEqual(
            photoSettingGridOrder(settings: settings, exposureMode: "A"),
            [.iso, .drive]
        )
    }

    func testHighestBurstSpeedUsesSonyLabelsNotCandidateOrder() {
        XCTAssertEqual(highestBurstSpeed(in: ["Hi", "Mid", "Lo"]), "Hi")
        XCTAssertEqual(highestBurstSpeed(in: ["Slow", "Normal", "Fast"]), "Fast")
        XCTAssertEqual(highestBurstSpeed(in: ["Low", "High"]), "High")
        XCTAssertEqual(highestBurstSpeed(in: ["Unknown A", "Unknown B"]), "Unknown A")
        XCTAssertNil(highestBurstSpeed(in: []))
    }

    func testLiveNDBurstDurationUsesExactExposureFormula() {
        XCTAssertEqual(shutterDurationSeconds("1/8"), 0.125)
        XCTAssertEqual(shutterDurationSeconds("1/8\""), 0.125)
        XCTAssertEqual(liveNDBurstDuration(shutterValue: "1/8", requiredFrames: 2), 0.375)
        XCTAssertEqual(liveNDBurstDuration(shutterValue: "2\"", requiredFrames: 4), 10)
    }

    func testLUTArchiveExtractsEveryNestedCube() throws {
        let encoded = """
        UEsDBBQAAAAIAA2Y/FxeNlk3IwAAAEoAAAANAAAATFVUcy9vbmUuY3ViZQvxDPFxVVDyz0tV4vIJDYk3dokP9oxyVTDiMlAwUDAggQQAUEsDBBQAAAAIAA2Y/FwRq5bXIwAAAEoAAAAPAAAAbmVzdGVkL3R3by5DVUJFC/EM8XFVUAopz1fi8gkNiTd2iQ/2jHJVMOIyVDBUMCSBBABQSwMEFAAAAAgADZj8XOLWiA0IAAAABgAAAAoAAAByZWFkbWUudHh0y0zPyy9KBQBQSwECFAMUAAAACAANmPxcXjZZNyMAAABKAAAADQAAAAAAAAAAAAAAgAEAAAAATFVUcy9vbmUuY3ViZVBLAQIUAxQAAAAIAA2Y/FwRq5bXIwAAAEoAAAAPAAAAAAAAAAAAAACAAU4AAABuZXN0ZWQvdHdvLkNVQkVQSwECFAMUAAAACAANmPxc4taIDQgAAAAGAAAACgAAAAAAAAAAAAAAgAGeAAAAcmVhZG1lLnR4dFBLBQYAAAAAAwADALAAAADOAAAAAAA=
        """
        let archive = try XCTUnwrap(Data(base64Encoded: encoded))
        let entries = try LUTArchive.cubeFiles(in: archive)
        XCTAssertEqual(entries.map(\.0), ["one.cube", "two.CUBE"])
        XCTAssertTrue(String(data: entries[0].1, encoding: .utf8)?.contains("TITLE \"One\"") == true)
        XCTAssertTrue(String(data: entries[1].1, encoding: .utf8)?.contains("TITLE \"Two\"") == true)
    }

    func testCameraEventParsesSettingsAndNumericZoomValues() {
        let result: [Any] = [
            ["type": "exposureMode", "exposureMode": "Aperture Priority"],
            ["type": "fNumber", "currentFNumber": "5.6"],
            ["type": "shutterSpeed", "currentShutterSpeed": "1/125"],
            ["type": "isoSpeedRate", "currentIsoSpeedRate": "ISO 1600"],
            ["type": "exposureCompensation", "currentExposureCompensation": NSNumber(value: -3)],
            ["type": "contShootingMode", "contShootingMode": "Continuous"],
            ["type": "contShootingSpeed", "contShootingSpeed": "Hi"],
            [
                "type": "zoomInformation",
                "zoomPosition": NSNumber(value: 47),
                "zoomNumberBox": "3",
                "zoomIndexCurrent": NSNumber(value: 1),
            ],
            ["type": "availableApiList", "names": ["actZoom", "setFNumber"]],
        ]
        let event = SonyCameraAPI.parseEventResult(result)
        XCTAssertEqual(event.settingValues[.exposureMode], "Aperture Priority")
        XCTAssertEqual(event.settingValues[.aperture], "5.6")
        XCTAssertEqual(event.settingValues[.shutterSpeed], "1/125")
        XCTAssertEqual(event.settingValues[.iso], "ISO 1600")
        XCTAssertEqual(event.settingValues[.exposureCompensation], "-3")
        XCTAssertEqual(event.settingValues[.drive], "Continuous")
        XCTAssertEqual(event.settingValues[.burstSpeed], "Hi")
        XCTAssertEqual(event.zoomPosition, 47)
        XCTAssertEqual(event.zoomBoxCount, 3)
        XCTAssertEqual(event.zoomBoxIndex, 1)
        XCTAssertEqual(event.availableAPIs, ["actZoom", "setFNumber"])
    }
}
