import PhotosUI
import SwiftUI
import AINRRuntime
import UniformTypeIdentifiers

struct PhotoLibraryPicker: UIViewControllerRepresentable {
    typealias ImportedPhoto = (data: Data, fileExtension: String, originalName: String?)

    let onComplete: (Result<[ImportedPhoto], Error>) -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator(parent: self)
    }

    func makeUIViewController(context: Context) -> PHPickerViewController {
        var configuration = PHPickerConfiguration(photoLibrary: .shared())
        configuration.filter = .images
        configuration.selectionLimit = 100
        configuration.selection = .ordered
        let picker = PHPickerViewController(configuration: configuration)
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(
        _ uiViewController: PHPickerViewController,
        context: Context
    ) {}

    final class Coordinator: NSObject, PHPickerViewControllerDelegate {
        private let parent: PhotoLibraryPicker

        init(parent: PhotoLibraryPicker) {
            self.parent = parent
        }

        func picker(
            _ picker: PHPickerViewController,
            didFinishPicking results: [PHPickerResult]
        ) {
            picker.dismiss(animated: true)
            guard !results.isEmpty else {
                parent.onComplete(.success([]))
                return
            }

            Task {
                do {
                    var imported: [ImportedPhoto] = []
                    imported.reserveCapacity(results.count)
                    for result in results {
                        let provider = result.itemProvider
                        let typeIdentifier = provider.registeredTypeIdentifiers.first(where: {
                            UTType($0)?.conforms(to: .image) == true
                        }) ?? UTType.image.identifier
                        let fileExtension = UTType(typeIdentifier)?
                            .preferredFilenameExtension ?? "jpg"
                        let data = try await Self.loadData(
                            provider: provider,
                            typeIdentifier: typeIdentifier
                        )
                        imported.append((
                            data,
                            fileExtension,
                            Self.filename(
                                suggestedName: provider.suggestedName,
                                fileExtension: fileExtension
                            )
                        ))
                    }
                    await MainActor.run {
                        self.parent.onComplete(.success(imported))
                    }
                } catch {
                    await MainActor.run {
                        self.parent.onComplete(.failure(error))
                    }
                }
            }
        }

        private static func loadData(
            provider: NSItemProvider,
            typeIdentifier: String
        ) async throws -> Data {
            try await withCheckedThrowingContinuation { continuation in
                provider.loadDataRepresentation(forTypeIdentifier: typeIdentifier) {
                    data,
                    error in
                    if let data {
                        continuation.resume(returning: data)
                    } else {
                        continuation.resume(throwing: error ?? SCUNetError(
                            message: "A selected photo could not be loaded"
                        ))
                    }
                }
            }
        }

        private static func filename(
            suggestedName: String?,
            fileExtension: String
        ) -> String? {
            guard var name = suggestedName?
                .trimmingCharacters(in: .whitespacesAndNewlines),
                  !name.isEmpty else {
                return nil
            }
            name = (name as NSString).lastPathComponent
            if (name as NSString).pathExtension.isEmpty {
                name += ".\(fileExtension)"
            }
            return name
        }
    }
}
