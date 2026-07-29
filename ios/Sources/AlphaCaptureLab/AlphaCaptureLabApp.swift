import SwiftUI

@main
struct AlphaCaptureLabApp: App {
    @StateObject private var camera = CameraController()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(camera)
                .preferredColorScheme(.dark)
                .tint(.mint)
                .onOpenURL { url in camera.importLUTs([url]) }
        }
    }
}
