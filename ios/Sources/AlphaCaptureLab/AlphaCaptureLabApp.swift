import SwiftUI

@main
struct AlphaCaptureLabApp: App {
    @StateObject private var camera = CameraController()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(camera)
                .preferredColorScheme(.dark)
                .onOpenURL { url in try? camera.lutLibrary.importFiles([url]) }
        }
    }
}
