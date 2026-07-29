package com.ryu.sonyremote.data

import android.content.Context
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import android.net.NetworkRequest
import android.net.Uri
import com.ryu.sonyremote.model.CameraCapabilities
import com.ryu.sonyremote.model.CameraBackend
import com.ryu.sonyremote.model.CameraCaptureEventKind
import com.ryu.sonyremote.model.CameraSettingId
import com.ryu.sonyremote.model.retainDynamicZoomFrom
import com.ryu.sonyremote.model.CapturedImage
import com.ryu.sonyremote.model.ConnectionState
import com.ryu.sonyremote.model.LiveViewFrame
import com.ryu.sonyremote.model.PostviewSizePreference
import com.ryu.sonyremote.model.OutputImageFormat
import com.ryu.sonyremote.processing.EditorGeometry
import com.ryu.sonyremote.model.RemoteCapture
import com.ryu.sonyremote.model.SonyCameraDevice
import com.ryu.sonyremote.network.CameraHttpStream
import com.ryu.sonyremote.network.NetworkHttpTransport
import com.ryu.sonyremote.network.PrivateNetworkPolicy
import com.ryu.sonyremote.network.SsdpCameraDiscovery
import com.ryu.sonyremote.network.WifiNetworkProvider
import com.ryu.sonyremote.protocol.CameraApiException
import com.ryu.sonyremote.protocol.AvContentClient
import com.ryu.sonyremote.protocol.DeviceDescriptionParser
import com.ryu.sonyremote.protocol.LiveViewFrameParser
import com.ryu.sonyremote.protocol.ScalarWebApiClient
import java.io.IOException
import java.net.URI
import java.net.SocketTimeoutException
import java.security.MessageDigest
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull

class SonyCameraRepository(context: Context) {
    private val appContext = context.applicationContext
    private val connectivityManager = appContext.getSystemService(ConnectivityManager::class.java)
    private val networkProvider = WifiNetworkProvider(appContext)
    private val discovery = SsdpCameraDiscovery(appContext)
    private val descriptionParser = DeviceDescriptionParser()
    private val mediaStore = JpegMediaStore(appContext)

    suspend fun loadSavedGallery(): List<SavedMediaItem> = mediaStore.listSaved()
    suspend fun setEditAttribution(
        uri: Uri,
        lutName: String?,
        lutStrength: Float?,
        denoiseModel: String?,
        denoiseStrength: Float?,
        geometry: EditorGeometry? = null,
    ) = mediaStore.setEditAttribution(
        uri, lutName, lutStrength, denoiseModel, denoiseStrength, geometry,
    )
    suspend fun readEditingOriginal(uri: Uri): ByteArray? = withContext(Dispatchers.IO) {
        editingOriginalFile(uri).takeIf { it.isFile }?.readBytes()
    }
    suspend fun discardSavedCapture(uri: Uri) {
        mediaStore.delete(uri)
        withContext(Dispatchers.IO) { editingOriginalFile(uri).delete() }
    }
    private val phoneLocation = PhoneLocationProvider(appContext)
    @Volatile private var geotaggingEnabled = false
    private val commandMutex = Mutex()
    private val repositoryScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    private val _connection = MutableStateFlow<ConnectionState>(ConnectionState.Disconnected)
    val connection: StateFlow<ConnectionState> = _connection.asStateFlow()

    private val _latestFrame = MutableStateFlow<LiveViewFrame?>(null)
    val latestFrame: StateFlow<LiveViewFrame?> = _latestFrame.asStateFlow()

    private val _isStreaming = MutableStateFlow(false)
    val isStreaming: StateFlow<Boolean> = _isStreaming.asStateFlow()

    private val _isRecording = MutableStateFlow(false)
    val isRecording: StateFlow<Boolean> = _isRecording.asStateFlow()

    private val _isContinuousShooting = MutableStateFlow(false)
    val isContinuousShooting: StateFlow<Boolean> = _isContinuousShooting.asStateFlow()

    private val _cameraStatus = MutableStateFlow<String?>(null)
    val cameraStatus: StateFlow<String?> = _cameraStatus.asStateFlow()

    private val _isPhysicalShutterTransferActive = MutableStateFlow(false)
    val isPhysicalShutterTransferActive: StateFlow<Boolean> =
        _isPhysicalShutterTransferActive.asStateFlow()

    private val _physicalShutterCaptures = MutableSharedFlow<SavedCapture>(
        extraBufferCapacity = PHYSICAL_CAPTURE_HANDOFF_BUFFER,
        onBufferOverflow = BufferOverflow.SUSPEND,
    )
    val physicalShutterCaptures: SharedFlow<SavedCapture> =
        _physicalShutterCaptures.asSharedFlow()

    private val _physicalShutterReferences = MutableSharedFlow<PendingRemoteCapture>(
        extraBufferCapacity = PHYSICAL_CAPTURE_EVENT_BUFFER,
        onBufferOverflow = BufferOverflow.SUSPEND,
    )
    val physicalShutterReferences: SharedFlow<PendingRemoteCapture> =
        _physicalShutterReferences.asSharedFlow()

    private val _physicalShutterFailures = MutableSharedFlow<String>(
        extraBufferCapacity = PHYSICAL_CAPTURE_EVENT_BUFFER,
        onBufferOverflow = BufferOverflow.DROP_OLDEST,
    )
    val physicalShutterFailures: SharedFlow<String> =
        _physicalShutterFailures.asSharedFlow()

    private val _continuousCaptureBatches = MutableSharedFlow<ContinuousBatchResult>(
        extraBufferCapacity = PHYSICAL_CAPTURE_EVENT_BUFFER,
        onBufferOverflow = BufferOverflow.SUSPEND,
    )
    val continuousCaptureBatches: SharedFlow<ContinuousBatchResult> =
        _continuousCaptureBatches.asSharedFlow()
    private val _downloadProgress = MutableSharedFlow<CaptureDownloadProgress>(extraBufferCapacity = 32)
    val downloadProgress: SharedFlow<CaptureDownloadProgress> = _downloadProgress.asSharedFlow()

    private val sessions = CurrentSessionGate<CameraSession>()
    private val computationalCaptureImport = AtomicBoolean(false)
    @Volatile private var downloadedImageProcessor:
        suspend (ByteArray) -> DownloadedImageProcessingResult = {
            DownloadedImageProcessingResult(it)
        }
    @Volatile private var outputImageFormat = OutputImageFormat.Jpeg
    @Volatile private var pendingTransport: NetworkHttpTransport? = null
    @Volatile private var activeLiveViewStream: CameraHttpStream? = null
    private var networkCallback: ConnectivityManager.NetworkCallback? = null
    @Volatile private var activeWifiSsid: String? = null

    fun connectedWifiSsid(): String? = activeWifiSsid
    fun currentWifiSsid(): String? = networkProvider.currentSsid()
    suspend fun awaitCurrentWifiSsid(): String? = networkProvider.awaitCurrentSsid()

    suspend fun connect(cameraSsid: String? = null, cameraPassword: String? = null): ConnectionState.Ready {
        disconnect()
        return try {
            val requestedNetwork = if (cameraSsid != null && cameraPassword != null) {
                networkProvider.requestCameraNetwork(cameraSsid, cameraPassword)
                    ?: throw fail("Camera Wi-Fi did not become available")
            } else null
            connectAfterDisconnect(requestedNetwork)
        } catch (error: Throwable) {
            if (error is CancellationException) throw error
            pendingTransport?.cancelActiveRequests()
            pendingTransport = null
            stopNetworkMonitor()
            networkProvider.releaseRequestedNetwork()
            val active = sessions.clear {
                _latestFrame.value = null
                _isStreaming.value = false
                _isRecording.value = false
                _isContinuousShooting.value = false
                _cameraStatus.value = null
            }
            active?.let(::stopPhysicalShutterMonitor)
            active?.transport?.cancelActiveRequests()
            active?.originalImportTransport?.cancelActiveRequests()
            active?.avContentTransport?.cancelActiveRequests()
            val existingFailure = _connection.value as? ConnectionState.Failed
            if (existingFailure != null) {
                throw IOException(existingFailure.message, error)
            }
            throw fail(error.toUserMessage("Could not connect to the Sony camera"))
        }
    }

    private suspend fun connectAfterDisconnect(requestedNetwork: Network? = null): ConnectionState.Ready {
        DiagnosticLog.record("connect_started")
        _connection.value = ConnectionState.WaitingForWifi
        val network = requestedNetwork ?: networkProvider.awaitWifiNetwork()
            ?: throw fail("Join the camera Wi-Fi network, then try again")
        activeWifiSsid = networkProvider.ssid(network)
        _connection.value = ConnectionState.Discovering
        val locations = discovery.discover(network)
        DiagnosticLog.record("discovery_finished", mapOf("responses" to locations.size.toString()))
        if (locations.isEmpty()) throw fail("No Sony camera responded on this Wi-Fi network")

        var lastError: Throwable? = null
        for (location in locations) {
            try {
                _connection.value = ConnectionState.Connecting("Reading camera services")
                val bootstrapTransport = NetworkHttpTransport(network, requireNotNull(location.host))
                pendingTransport = bootstrapTransport
                val xml = bootstrapTransport.getText(location)
                val device = descriptionParser.parse(xml, location)
                val cameraHost = requireNotNull(device.descriptionUri.host)
                val cameraEndpoint = device.endpoint("camera")
                val transport = NetworkHttpTransport(network, cameraHost)
                pendingTransport = transport
                val backend: CameraBackend = ScalarWebApiClient(cameraEndpoint, transport)
                var apis = backend.getAvailableApiList()
                if ("startRecMode" in apis) {
                    backend.startRecMode()
                    for (attempt in 0 until REC_MODE_START_ATTEMPTS) {
                        delay(REC_MODE_START_DELAY_MILLIS)
                        apis = backend.getAvailableApiList()
                        if ("startRecMode" !in apis) break
                    }
                    check("startRecMode" !in apis) { "Camera did not enter remote shooting mode" }
                }
                var capabilities = backend.loadCapabilities().let { loaded ->
                    if (loaded.availableApis.isEmpty()) loaded.copy(availableApis = apis) else loaded
                }
                val contentsTransferAvailable = "avcontent" in device.endpoints &&
                    "getAvailableCameraFunction" in capabilities.availableApis &&
                    runCatching { backend.getAvailableCameraFunctions() }
                        .getOrDefault(emptyList())
                        .any { it.equals(CONTENTS_TRANSFER_FUNCTION, ignoreCase = true) }
                capabilities = capabilities.copy(contentsTransferAvailable = contentsTransferAvailable)
                val eventTransport = NetworkHttpTransport(network, cameraHost)
                val eventBackend: CameraBackend = ScalarWebApiClient(cameraEndpoint, eventTransport)
                val originalImportTransport = NetworkHttpTransport(network, cameraHost)
                val avContentTransport = NetworkHttpTransport(network, cameraHost)
                val connected = CameraSession(
                    network = network,
                    device = device,
                    transport = transport,
                    backend = backend,
                    eventTransport = eventTransport,
                    eventBackend = eventBackend,
                    originalImportTransport = originalImportTransport,
                    avContentTransport = avContentTransport,
                    capabilities = capabilities,
                )
                pendingTransport = null
                sessions.install(connected) { }
                try {
                    monitorNetwork(connected)
                    if (!networkProvider.isAvailable(network)) {
                        markNetworkLostIfCurrent(connected, "connect_validation")
                        throw IOException(NETWORK_LOST_MESSAGE)
                    }
                    preparePhysicalShutterMonitor(connected)
                    check(
                        sessions.withCurrent(connected) {
                            syncRecordingState(connected.capabilities)
                            _connection.value = ConnectionState.Ready(device, connected.capabilities)
                        },
                    ) { NETWORK_LOST_MESSAGE }
                } catch (error: Throwable) {
                    stopNetworkMonitor()
                    if (sessions.invalidate(connected) { }) {
                        stopPhysicalShutterMonitor(connected)
                        connected.transport.cancelActiveRequests()
                        connected.avContentTransport.cancelActiveRequests()
                    }
                    throw error
                }
                DiagnosticLog.record(
                    "camera_connected",
                    mapOf(
                        "model" to device.modelName,
                        "camera_app" to (capabilities.applicationName ?: "unknown"),
                        "camera_app_version" to (capabilities.applicationVersion ?: "unknown"),
                        "available_api_count" to capabilities.availableApis.size.toString(),
                    ),
                )
                return ConnectionState.Ready(device, connected.capabilities)
            } catch (error: Throwable) {
                pendingTransport?.cancelActiveRequests()
                pendingTransport = null
                if (error is CancellationException) throw error
                lastError = error
            }
        }
        throw fail(lastError.toUserMessage("The Sony camera service could not be opened"))
    }

    suspend fun streamLiveView() {
        val active = requireSession()
        check(active.capabilities.canLiveView) { "This camera does not currently offer live view" }
        _isStreaming.value = true
        var cameraStreamStartAttempted = false
        DiagnosticLog.record("liveview_starting")
        try {
            val liveViewUri = commandMutex.withLock {
                cameraStreamStartAttempted = true
                val requestedSize = active.capabilities.liveviewSize
                if (
                    "startLiveviewWithSize" in active.capabilities.availableApis &&
                    requestedSize != null
                ) {
                    active.backend.startLiveviewWithSize(requestedSize)
                } else {
                    active.backend.startLiveview()
                }
            }
            PrivateNetworkPolicy.requireCameraUri(liveViewUri, active.device.descriptionUri.host)
            DiagnosticLog.record("liveview_started")
            withContext(Dispatchers.IO) {
                active.transport.openStream(liveViewUri).use { stream ->
                    activeLiveViewStream = stream
                    val parser = LiveViewFrameParser(stream.input)
                    while (currentCoroutineContext().isActive) {
                        _latestFrame.value = parser.nextJpeg() ?: break
                    }
                }
            }
            handleLiveViewEnd(active, source = "eof")
        } catch (error: Throwable) {
            currentCoroutineContext().ensureActive()
            if (!sessions.isCurrent(active)) return
            if (networkProvider.isAvailable(active.network)) delay(LIVEVIEW_END_GRACE_MILLIS)
            val networkAvailable = networkProvider.isAvailable(active.network)
            val decision = classifyLiveViewEnd(
                coroutineActive = currentCoroutineContext().isActive,
                networkAvailable = networkAvailable,
            )
            DiagnosticLog.record(
                "liveview_failed",
                mapOf(
                    "error" to error.toUserMessage("unknown"),
                    "type" to error.javaClass.simpleName,
                    "network_available" to networkAvailable.toString(),
                    "decision" to decision.name,
                ),
            )
            when (decision) {
                LiveViewEndDecision.IntentionalStop -> currentCoroutineContext().ensureActive()
                LiveViewEndDecision.NetworkLost -> markNetworkLostIfCurrent(active, "liveview_error")
                LiveViewEndDecision.KeepSession -> throw error
            }
        } finally {
            activeLiveViewStream = null
            _latestFrame.value = null
            _isStreaming.value = false
            if (cameraStreamStartAttempted) {
                withContext(NonCancellable) {
                    withTimeoutOrNull(STOP_TIMEOUT_MILLIS) {
                        commandMutex.withLock {
                            runCatching { active.backend.stopLiveview(STOP_TIMEOUT_MILLIS.toInt()) }
                        }
                    }
                }
            }
            DiagnosticLog.record("liveview_stopped")
        }
    }

    suspend fun downloadCapture(
        postviewSize: PostviewSizePreference = PostviewSizePreference.Original,
    ): DownloadedCapture {
        val capture = requestPhotoCapture(postviewSize)
        val downloaded = downloadCapturedPostview(capture)
        awaitPhotoCaptureReady()
        return downloaded
    }

    suspend fun requestPhotoCapture(
        postviewSize: PostviewSizePreference = PostviewSizePreference.Original,
    ): CapturedImage = commandMutex.withLock {
        val active = requireSession()
        check(active.capabilities.canTakePicture) { "Photo capture is unavailable in the current camera mode" }
        val capture = try {
            active.apiCaptureInFlight.set(true)
            active.backend.takePicture(active.capabilities.availableApis, postviewSize).also { captured ->
                active.recentCaptureUris.add(captured.remoteUri)
            }
        } finally {
            active.apiCaptureInFlight.set(false)
        }
        capture
    }

    suspend fun awaitPhotoCaptureReady(): Boolean {
        val active = requireSession()
        val readyForNextCapture = try {
            refreshCapabilitiesUntil(active) { it.canTakePicture }
        } catch (error: Throwable) {
            if (error is CancellationException) throw error
            DiagnosticLog.record(
                "capture_recovery_failed",
                mapOf("error" to error.toUserMessage("unknown")),
            )
            false
        }
        if (!readyForNextCapture) DiagnosticLog.record("capture_recovery_pending")
        return readyForNextCapture
    }

    suspend fun downloadCapturedPostview(capture: CapturedImage): DownloadedCapture {
        val active = requireSession()
        PrivateNetworkPolicy.requireCameraUri(capture.remoteUri, active.device.descriptionUri.host)
        val jpeg = JpegValidator.normalize(
            active.transport.readBytes(capture.remoteUri) { bytes, total ->
                _downloadProgress.tryEmit(CaptureDownloadProgress(capture.remoteUri, bytes, total))
            },
        )
        DiagnosticLog.record(
            "photo_downloaded",
            mapOf(
                "bytes" to jpeg.size.toString(),
                "original_requested" to capture.originalSizeRequested.toString(),
                "postview_size" to (capture.postviewSize ?: "unknown"),
            ),
        )
        return DownloadedCapture(
            jpeg = jpeg,
            originalSizeRequested = capture.originalSizeRequested,
            postviewSize = capture.postviewSize,
            postviewRemoteUri = capture.remoteUri,
            cameraContentId = capture.capturedAtEpochMillis.toString()
                .takeIf { active.capabilities.contentsTransferAvailable && !capture.originalSizeRequested },
        )
    }

    suspend fun capturePhoto(): SavedCapture {
        val capture = downloadCapture(PostviewSizePreference.Original)
        return saveCapture(capture, "SONY")
    }

    suspend fun saveCapture(capture: DownloadedCapture, prefix: String): SavedCapture {
        val sourceJpeg = capture.jpeg
        val processing = if (computationalCaptureImport.get()) {
            DownloadedImageProcessingResult(sourceJpeg)
        } else {
            downloadedImageProcessor(sourceJpeg)
        }
        val jpeg = JpegValidator.normalize(processing.jpeg)
        val savedUri = mediaStore.save(jpeg, prefix, outputImageFormat)
        if (!jpeg.contentEquals(sourceJpeg)) {
            runCatching { mediaStore.copyExif(sourceJpeg, savedUri) }
            runCatching { saveEditingOriginal(savedUri, sourceJpeg) }
                .onFailure { error ->
                    DiagnosticLog.record(
                        "editing_original_save_failed",
                        mapOf("error" to error.toUserMessage("unknown")),
                    )
                }
        }
        applyPhoneLocation(savedUri)
        DiagnosticLog.record(
            "photo_saved",
            mapOf(
                "bytes" to jpeg.size.toString(),
                "prefix" to prefix,
                "source_count" to "1",
            ),
        )
        return SavedCapture(
            uri = savedUri,
            jpeg = jpeg,
            originalSizeRequested = capture.originalSizeRequested,
            postviewRemoteUri = capture.postviewRemoteUri,
            postviewRemoteIsOriginal = capture.originalSizeRequested,
            cameraContentId = capture.cameraContentId,
            appliedEdits = processing.appliedEdits,
        )
    }

    suspend fun saveProcessedJpeg(
        jpeg: ByteArray,
        prefix: String,
        sourceCount: Int,
        metadataSourceJpeg: ByteArray? = null,
        editingOriginalJpeg: ByteArray? = null,
    ): SavedCapture {
        val normalized = JpegValidator.normalize(jpeg)
        val savedUri = mediaStore.save(normalized, prefix, outputImageFormat)
        if (metadataSourceJpeg != null) {
            runCatching { mediaStore.copyExif(metadataSourceJpeg, savedUri) }
                .onFailure { error ->
                    DiagnosticLog.record(
                        "processed_photo_metadata_failed",
                        mapOf("error" to error.toUserMessage("unknown")),
                    )
                }
        }
        if (editingOriginalJpeg != null && !normalized.contentEquals(editingOriginalJpeg)) {
            runCatching { saveEditingOriginal(savedUri, editingOriginalJpeg) }
                .onFailure { error ->
                    DiagnosticLog.record(
                        "editing_original_save_failed",
                        mapOf("error" to error.toUserMessage("unknown")),
                    )
                }
        }
        applyPhoneLocation(savedUri)
        DiagnosticLog.record(
            "processed_photo_saved",
            mapOf(
                "bytes" to normalized.size.toString(),
                "prefix" to prefix,
                "source_count" to sourceCount.toString(),
            ),
        )
        return SavedCapture(
            uri = savedUri,
            jpeg = normalized,
            originalSizeRequested = false,
        )
    }

    fun setGeotaggingEnabled(enabled: Boolean) {
        geotaggingEnabled = enabled
    }

    fun setDownloadedImageProcessor(
        processor: suspend (ByteArray) -> DownloadedImageProcessingResult,
    ) {
        downloadedImageProcessor = processor
    }

    private suspend fun saveEditingOriginal(uri: Uri, jpeg: ByteArray) = withContext(Dispatchers.IO) {
        val destination = editingOriginalFile(uri)
        val temporary = destination.resolveSibling("${destination.name}.tmp")
        temporary.writeBytes(JpegValidator.normalize(jpeg))
        check(temporary.renameTo(destination)) { "Could not retain the editing original" }
    }

    private fun editingOriginalFile(uri: Uri): java.io.File {
        val key = MessageDigest.getInstance("SHA-256")
            .digest(uri.toString().toByteArray(Charsets.UTF_8))
            .joinToString("") { "%02x".format(it) }
        return appContext.getDir(EDITING_ORIGINAL_DIRECTORY, Context.MODE_PRIVATE).resolve("$key.jpg")
    }

    fun setOutputImageFormat(format: OutputImageFormat) {
        outputImageFormat = format
    }

    suspend fun setAutomaticPostviewPreference(preference: PostviewSizePreference) = commandMutex.withLock {
        val active = requireSession()
        active.backend.setPostviewSize(active.capabilities.availableApis, preference)
    }

    private suspend fun applyPhoneLocation(uri: android.net.Uri) {
        if (!geotaggingEnabled) return
        val location = phoneLocation.currentLocation()
        if (location == null) {
            DiagnosticLog.record("geotag_unavailable")
            return
        }
        runCatching { mediaStore.setLocation(uri, location) }
            .onFailure { DiagnosticLog.record("geotag_failed", mapOf("error" to it.toString())) }
    }

    suspend fun importOriginalPostview(remoteUri: URI): SavedCapture {
        val active = requireSession()
        PrivateNetworkPolicy.requireCameraUri(remoteUri, active.device.descriptionUri.host)
        val jpeg = JpegValidator.normalize(active.originalImportTransport.readBytes(remoteUri))
        currentCoroutineContext().ensureActive()
        check(sessions.isCurrent(active)) { "Camera connection ended during Original import" }
        val savedUri = mediaStore.save(jpeg, "SONY_ORIGINAL", outputImageFormat)
        DiagnosticLog.record(
            "original_import_saved",
            mapOf("bytes" to jpeg.size.toString()),
        )
        return SavedCapture(
            uri = savedUri,
            jpeg = jpeg,
            originalSizeRequested = true,
            postviewRemoteUri = remoteUri,
        )
    }

    suspend fun importOriginalFromContentsTransfer(reference: String): SavedCapture = commandMutex.withLock {
        val capturedAt = reference.toLongOrNull() ?: error("Invalid camera content reference")
        val active = requireSession()
        check(active.capabilities.contentsTransferAvailable) {
            "Contents Transfer is unavailable for this camera"
        }
        val endpoint = active.device.endpoints["avcontent"]
            ?: error("Camera did not advertise the Contents Transfer service")
        stopPhysicalShutterMonitor(active)
        var switched = false
        try {
            active.backend.setCameraFunction(CONTENTS_TRANSFER_FUNCTION)
            switched = true
            var original: com.ryu.sonyremote.protocol.CameraContentOriginal? = null
            var lastError: Throwable? = null
            repeat(CONTENTS_TRANSFER_READY_ATTEMPTS) {
                if (original != null) return@repeat
                try {
                    original = AvContentClient(endpoint, active.avContentTransport)
                        .findOriginalNear(capturedAt)
                    return@repeat
                } catch (error: Throwable) {
                    if (error is CancellationException) throw error
                    lastError = error
                    delay(CONTENTS_TRANSFER_READY_DELAY_MILLIS)
                }
            }
            val matched = original ?: throw IOException(
                lastError.toUserMessage("Camera Contents Transfer did not become ready"),
                lastError,
            )
            PrivateNetworkPolicy.requireCameraUri(matched.originalUrl, active.device.descriptionUri.host)
            val jpeg = JpegValidator.normalize(active.originalImportTransport.readBytes(matched.originalUrl))
            currentCoroutineContext().ensureActive()
            val savedUri = mediaStore.save(jpeg, "SONY_ORIGINAL", outputImageFormat)
            DiagnosticLog.record(
                "contents_transfer_original_saved",
                mapOf("file_name" to (matched.fileName ?: "unknown"), "bytes" to jpeg.size.toString()),
            )
            SavedCapture(
                uri = savedUri,
                jpeg = jpeg,
                originalSizeRequested = true,
                postviewRemoteUri = matched.originalUrl,
                cameraContentId = matched.contentUri,
            )
        } finally {
            if (switched && sessions.isCurrent(active)) {
                withContext(NonCancellable) {
                    runCatching { active.backend.setCameraFunction(REMOTE_SHOOTING_FUNCTION) }
                    runCatching {
                        refreshCapabilitiesUntil(
                            active = active,
                            attempts = CONTENTS_TRANSFER_READY_ATTEMPTS,
                        ) { capabilities -> capabilities.canLiveView }
                    }
                    runCatching { preparePhysicalShutterMonitor(active) }
                }
            }
        }
    }

    fun cancelOriginalImportIo() {
        sessions.current()?.originalImportTransport?.cancelActiveRequests()
    }

    fun setComputationalCaptureImport(enabled: Boolean) {
        computationalCaptureImport.set(enabled)
        DiagnosticLog.record(
            "capture_import_policy_changed",
            mapOf("computational" to enabled.toString()),
        )
    }

    suspend fun restoreOriginalPostview() = commandMutex.withLock {
        val active = requireSession()
        val restored = active.backend.setPostviewSize(
            active.capabilities.availableApis,
            PostviewSizePreference.Original,
        )
        if ("getAvailablePostviewImageSize" in active.capabilities.availableApis) {
            check(restored.equals(PostviewSizePreference.Original.wireValue, ignoreCase = true)) {
                "Camera did not restore Original postview size"
            }
        }
        DiagnosticLog.record(
            "postview_restored",
            mapOf("size" to (restored ?: "unknown")),
        )
    }

    suspend fun prepareFastPostview() = commandMutex.withLock {
        val active = requireSession()
        val selected = active.backend.setPostviewSize(
            active.capabilities.availableApis,
            PostviewSizePreference.FastPreview,
        )
        if ("getAvailablePostviewImageSize" in active.capabilities.availableApis) {
            check(selected.equals(PostviewSizePreference.FastPreview.wireValue, ignoreCase = true)) {
                "Camera did not enter 2M postview mode for panorama"
            }
        }
        DiagnosticLog.record(
            "postview_prepared",
            mapOf("size" to (selected ?: "unknown")),
        )
    }

    suspend fun setSetting(id: CameraSettingId, wireValue: String) = commandMutex.withLock {
        val active = requireSession()
        active.backend.setSetting(id, wireValue)
        DiagnosticLog.record("setting_changed", mapOf("setting" to id.name, "value" to wireValue))
        refreshCapabilitiesUntil(active) { capabilities ->
            capabilities.settings[id]?.currentWireValue == wireValue
        }
    }

    suspend fun setLiveviewSize(size: String) = commandMutex.withLock {
        val active = requireSession()
        check(size in active.capabilities.availableLiveviewSizes) { "Unsupported live-view quality" }
        if ("setLiveviewSize" in active.capabilities.availableApis) {
            active.backend.setLiveviewSize(size)
            refreshCapabilitiesUntil(active) { it.liveviewSize == size }
        } else {
            active.capabilities = active.capabilities.copy(liveviewSize = size)
            _connection.value = ConnectionState.Ready(active.device, active.capabilities)
        }
    }

    suspend fun startZoom(direction: String) = commandMutex.withLock {
        val active = requireSession()
        check(active.capabilities.canRemoteZoom) { "Remote zoom is unavailable for this lens" }
        active.backend.actZoom(direction, "start")
    }

    suspend fun stopZoom(direction: String) = commandMutex.withLock {
        val active = requireSession()
        if (active.capabilities.canRemoteZoom) active.backend.actZoom(direction, "stop")
    }

    suspend fun stepZoomAndReadPosition(direction: String): Int = commandMutex.withLock {
        val active = requireSession()
        check(active.capabilities.canRemoteZoom) { "Remote zoom is unavailable for this lens" }
        active.backend.actZoom(direction, "1shot")
        repeat(ZOOM_POSITION_READ_ATTEMPTS) {
            val event = active.backend.getEvent(longPolling = false)
            applyCameraEventState(active, event)
            event.zoomPosition?.let { return@withLock it.coerceIn(0, 100) }
            delay(ZOOM_POSITION_READ_DELAY_MILLIS)
        }
        throw IOException("Camera did not report its new zoom position")
    }

    suspend fun refreshZoomState() = commandMutex.withLock {
        val active = requireSession()
        val event = active.backend.getEvent(longPolling = false)
        applyCameraEventState(active, event)
    }

    suspend fun toggleMovieRecording(): MovieActionResult = commandMutex.withLock {
        val active = requireSession()
        val requestedStop = _isRecording.value
        refreshCapabilities(active)
        if (!active.capabilities.canStartMovie && !active.capabilities.canStopMovie) {
            refreshCapabilitiesUntil(
                active,
                updateRecordingState = false,
                attempts = MOVIE_STATE_REFRESH_ATTEMPTS,
            ) { it.canStartMovie || it.canStopMovie }
        }
        val decision = decideMovieCommand(
            requestedStop = requestedStop,
            canStart = active.capabilities.canStartMovie,
            canStop = active.capabilities.canStopMovie,
        )
        when (decision) {
            MovieCommandDecision.AlreadyRecording -> {
                sessions.withCurrent(active) { _isRecording.value = true }
                return@withLock MovieActionResult.AlreadyRecording
            }

            MovieCommandDecision.AlreadyStopped -> {
                sessions.withCurrent(active) { _isRecording.value = false }
                return@withLock MovieActionResult.AlreadyStopped
            }

            MovieCommandDecision.Start -> {
                active.backend.startMovieRecording()
                DiagnosticLog.record("movie_start_requested")
            }

            MovieCommandDecision.Stop -> {
                active.backend.stopMovieRecording()
                DiagnosticLog.record("movie_stop_requested")
            }

            MovieCommandDecision.Unavailable -> {
                throw IOException("Movie recording is unavailable in this camera mode")
            }
        }
        val recordingAfterCommand = decision == MovieCommandDecision.Start
        val confirmed = refreshCapabilitiesUntil(
            active,
            updateRecordingState = false,
            attempts = MOVIE_STATE_REFRESH_ATTEMPTS,
        ) { capabilities ->
            if (recordingAfterCommand) {
                capabilities.canStopMovie
            } else {
                capabilities.canStartMovie
            }
        }
        if (!confirmed) {
            sessions.withCurrent(active) {
                _isRecording.value = recordingStateFromCapabilities(
                    canStart = active.capabilities.canStartMovie,
                    canStop = active.capabilities.canStopMovie,
                    fallback = true,
                )
            }
            val action = if (recordingAfterCommand) "start" else "stop"
            DiagnosticLog.record("movie_state_unconfirmed", mapOf("action" to action))
            throw IOException("Camera did not confirm movie recording $action")
        }
        sessions.withCurrent(active) { _isRecording.value = recordingAfterCommand }
        DiagnosticLog.record(if (recordingAfterCommand) "movie_started" else "movie_stopped")
        if (recordingAfterCommand) MovieActionResult.Started else MovieActionResult.Stopped
    }

    suspend fun startContinuousShooting(
        postviewPreference: PostviewSizePreference? = null,
    ) = commandMutex.withLock {
        val active = requireSession()
        if (!active.capabilities.canStartContinuousShooting) refreshCapabilities(active)
        check(active.capabilities.canStartContinuousShooting) {
            "Continuous shooting is unavailable in the current camera mode"
        }
        val drive = active.capabilities.settings[CameraSettingId.ContShootingMode]?.currentWireValue
        check(drive.equals("Continuous", ignoreCase = true) ||
            drive.equals("Spd Priority Cont.", ignoreCase = true) ||
            drive.equals("Burst", ignoreCase = true)
        ) { "Select a continuous Drive mode before starting a burst" }
        val postviewSize = postviewPreference?.let { preference ->
            active.backend.setPostviewSize(active.capabilities.availableApis, preference)
        }
        _isContinuousShooting.value = true
        DiagnosticLog.record(
            "continuous_shooting_start_requested",
            mapOf(
                "drive" to (drive ?: "unknown"),
                "postview_size" to (postviewSize ?: "unchanged"),
            ),
        )
        active.backend.startContinuousShooting()
        DiagnosticLog.record("continuous_shooting_started")
    }

    suspend fun stopContinuousShooting() = commandMutex.withLock {
        val active = sessions.current()
        if (active == null) {
            _isContinuousShooting.value = false
            return@withLock
        }
        val canStop = active.capabilities.canStopContinuousShooting ||
            _cameraStatus.value.equals("StillCapturing", ignoreCase = true) ||
            _isContinuousShooting.value
        DiagnosticLog.record(
            "continuous_shooting_stop_requested",
            mapOf(
                "command_sent" to canStop.toString(),
                "camera_status" to (_cameraStatus.value ?: "unknown"),
            ),
        )
        if (canStop) active.backend.stopContinuousShooting(CONTINUOUS_STOP_TIMEOUT_MILLIS)
        _isContinuousShooting.value = false
        DiagnosticLog.record("continuous_shooting_stopped")
    }

    suspend fun refreshCapabilities() {
        refreshCapabilities(requireSession())
    }

    suspend fun awaitLiveViewAvailability(): Boolean = commandMutex.withLock {
        val active = requireSession()
        refreshCapabilities(active)
        if (!active.capabilities.canLiveView && "stopLiveview" in active.capabilities.availableApis) {
            try {
                active.backend.stopLiveview(STOP_TIMEOUT_MILLIS.toInt())
            } catch (error: Throwable) {
                if (error is CancellationException) throw error
                DiagnosticLog.record(
                    "stale_liveview_stop_failed",
                    mapOf("error" to error.toUserMessage("unknown")),
                )
            }
            refreshCapabilitiesUntil(active) { it.canLiveView }
        } else if (!active.capabilities.canLiveView) {
            refreshCapabilitiesUntil(active) { it.canLiveView }
        }
        active.capabilities.canLiveView
    }

    suspend fun pollCameraState() {
        var consecutiveFailures = 0
        while (currentCoroutineContext().isActive) {
            delay(CAPABILITY_POLL_INTERVAL_MILLIS)
            val active = sessions.current() ?: return
            if (!commandMutex.tryLock()) continue
            try {
                val capabilities = mergeStableInfo(
                    active,
                    active.backend.loadCapabilities(includeApplicationInfo = false),
                )
                if (!publishCapabilitiesIfCurrent(active, capabilities)) return
                consecutiveFailures = 0
            } catch (error: Throwable) {
                if (error is CancellationException) throw error
                if (!networkProvider.isAvailable(active.network)) {
                    markNetworkLostIfCurrent(active, "capability_poll")
                    return
                }
                if (error is IOException) {
                    consecutiveFailures++
                } else {
                    consecutiveFailures = 0
                }
                if (consecutiveFailures == 1) {
                    DiagnosticLog.record(
                        "capability_poll_failed",
                        mapOf("error" to error.toUserMessage("unknown")),
                    )
                }
                if (consecutiveFailures >= CAMERA_UNREACHABLE_FAILURE_THRESHOLD) {
                    markCameraUnreachableIfCurrent(active, "capability_poll")
                    return
                }
            } finally {
                commandMutex.unlock()
            }
        }
    }

    suspend fun disconnect() {
        computationalCaptureImport.set(false)
        if (_isContinuousShooting.value) runCatching { stopContinuousShooting() }
        sessions.current()?.let(::stopPhysicalShutterMonitor)
        stopNetworkMonitor()
        networkProvider.releaseRequestedNetwork()
        activeWifiSsid = null
        cancelActiveIo()
        val active = sessions.clear {
            _latestFrame.value = null
            _isStreaming.value = false
            _isRecording.value = false
            _isContinuousShooting.value = false
            _cameraStatus.value = null
            _connection.value = ConnectionState.Disconnected
        }
        if (active != null) {
            withContext(NonCancellable) {
                withTimeoutOrNull(POSTVIEW_RESTORE_TIMEOUT_MILLIS) {
                    runCatching {
                        active.backend.setPostviewSize(
                            active.capabilities.availableApis,
                            PostviewSizePreference.Original,
                        )
                    }
                }
                if ("stopRecMode" in active.capabilities.availableApis) {
                    withTimeoutOrNull(STOP_TIMEOUT_MILLIS) {
                        runCatching { active.backend.stopRecMode(STOP_TIMEOUT_MILLIS.toInt()) }
                    }
                }
            }
        }
        if (active != null) DiagnosticLog.record("camera_disconnected")
    }

    fun close() {
        computationalCaptureImport.set(false)
        val active = sessions.current()
        if (
            active != null &&
            (_isContinuousShooting.value || _cameraStatus.value.equals("StillCapturing", ignoreCase = true))
        ) {
            runBlocking(Dispatchers.IO) {
                withTimeoutOrNull(CONTINUOUS_CLOSE_STOP_TIMEOUT_MILLIS) {
                    runCatching {
                        active.backend.stopContinuousShooting(CONTINUOUS_STOP_TIMEOUT_MILLIS)
                    }
                }
            }
        }
        sessions.current()?.let(::stopPhysicalShutterMonitor)
        stopNetworkMonitor()
        cancelActiveIo()
        sessions.clear { }
        repositoryScope.cancel()
    }

    fun cancelActiveIo() {
        cancelLiveViewIo()
        pendingTransport?.cancelActiveRequests()
        sessions.current()?.transport?.cancelActiveRequests()
        sessions.current()?.originalImportTransport?.cancelActiveRequests()
        sessions.current()?.avContentTransport?.cancelActiveRequests()
    }

    fun cancelLiveViewIo() {
        runCatching { activeLiveViewStream?.close() }
        activeLiveViewStream = null
    }

    private suspend fun refreshCapabilities(active: CameraSession) {
        val capabilities = mergeStableInfo(
            active,
            active.backend.loadCapabilities(includeApplicationInfo = false),
        )
        check(publishCapabilitiesIfCurrent(active, capabilities)) { "Camera connection ended" }
    }

    private suspend fun refreshCapabilitiesUntil(
        active: CameraSession,
        updateRecordingState: Boolean = true,
        attempts: Int = CAPABILITY_REFRESH_ATTEMPTS,
        condition: (CameraCapabilities) -> Boolean,
    ): Boolean {
        var lastError: Throwable? = null
        repeat(attempts) {
            delay(CAPABILITY_REFRESH_DELAY_MILLIS)
            try {
                val capabilities = mergeStableInfo(
                    active,
                    active.backend.loadCapabilities(includeApplicationInfo = false),
                )
                if (!publishCapabilitiesIfCurrent(active, capabilities, updateRecordingState)) {
                    throw IOException("Camera connection ended")
                }
                if (condition(capabilities)) return true
            } catch (error: Throwable) {
                if (error is CancellationException) throw error
                if (!sessions.isCurrent(active)) throw error
                lastError = error
            }
        }
        if (lastError != null && active.capabilities.availableApis.isEmpty()) throw lastError
        return false
    }

    private fun requireSession(): CameraSession =
        requireNotNull(sessions.current()) { "Connect to the camera first" }

    private fun fail(message: String): IOException {
        _connection.value = ConnectionState.Failed(message)
        DiagnosticLog.record("connection_failed", mapOf("error" to message))
        return IOException(message)
    }

    private fun mergeStableInfo(
        active: CameraSession,
        updated: CameraCapabilities,
    ): CameraCapabilities {
        val retainedSettings = active.capabilities.settings.filterKeys { settingId ->
            settingId !in updated.settings && settingId.requiredApis().all(updated.availableApis::contains)
        }
        return updated.copy(
            settings = retainedSettings + updated.settings,
            applicationName = updated.applicationName ?: active.capabilities.applicationName,
            applicationVersion = updated.applicationVersion ?: active.capabilities.applicationVersion,
            contentsTransferAvailable = active.capabilities.contentsTransferAvailable,
            liveviewSize = if ("startLiveviewWithSize" in updated.availableApis) {
                active.capabilities.liveviewSize ?: updated.liveviewSize
            } else {
                updated.liveviewSize ?: active.capabilities.liveviewSize
            },
            availableLiveviewSizes = updated.availableLiveviewSizes.ifEmpty {
                active.capabilities.availableLiveviewSizes
            },
        ).retainDynamicZoomFrom(active.capabilities)
    }

    private fun CameraSettingId.requiredApis(): Set<String> = when (this) {
        CameraSettingId.ShootMode -> setOf("getAvailableShootMode")
        CameraSettingId.ContShootingMode -> setOf("getAvailableContShootingMode")
        CameraSettingId.ContShootingSpeed -> setOf("getAvailableContShootingSpeed")
        CameraSettingId.ExposureMode -> setOf("getExposureMode")
        CameraSettingId.FNumber -> setOf("getAvailableFNumber")
        CameraSettingId.ShutterSpeed -> setOf("getAvailableShutterSpeed")
        CameraSettingId.IsoSpeedRate -> setOf("getAvailableIsoSpeedRate")
        CameraSettingId.ExposureCompensation -> setOf("getAvailableExposureCompensation")
    }

    private fun syncRecordingState(capabilities: CameraCapabilities) {
        when {
            capabilities.canStopMovie -> _isRecording.value = true
            capabilities.canStartMovie -> _isRecording.value = false
        }
    }

    private suspend fun handleLiveViewEnd(active: CameraSession, source: String) {
        if (currentCoroutineContext().isActive) delay(LIVEVIEW_END_GRACE_MILLIS)
        val coroutineActive = currentCoroutineContext().isActive
        val networkAvailable = networkProvider.isAvailable(active.network)
        val decision = classifyLiveViewEnd(coroutineActive, networkAvailable)
        DiagnosticLog.record(
            "liveview_ended",
            mapOf(
                "source" to source,
                "coroutine_active" to coroutineActive.toString(),
                "network_available" to networkAvailable.toString(),
                "decision" to decision.name,
            ),
        )
        when (decision) {
            LiveViewEndDecision.IntentionalStop -> currentCoroutineContext().ensureActive()
            LiveViewEndDecision.KeepSession -> Unit
            LiveViewEndDecision.NetworkLost -> markNetworkLostIfCurrent(active, "liveview_$source")
        }
    }

    private fun publishCapabilitiesIfCurrent(
        active: CameraSession,
        capabilities: CameraCapabilities,
        updateRecordingState: Boolean = true,
    ): Boolean = sessions.withCurrent(active) {
        active.capabilities = capabilities
        if (updateRecordingState) syncRecordingState(capabilities)
        _connection.value = ConnectionState.Ready(active.device, capabilities)
    }

    private fun markNetworkLostIfCurrent(active: CameraSession, source: String): Boolean {
        val invalidated = invalidateSessionIfCurrent(active, NETWORK_LOST_MESSAGE)
        DiagnosticLog.record(
            "network_loss_observed",
            mapOf("source" to source, "invalidated_current_session" to invalidated.toString()),
        )
        if (invalidated) DiagnosticLog.record("wifi_lost", mapOf("source" to source))
        return invalidated
    }

    private fun markCameraUnreachableIfCurrent(active: CameraSession, source: String): Boolean {
        val invalidated = invalidateSessionIfCurrent(active, CAMERA_UNREACHABLE_MESSAGE)
        DiagnosticLog.record(
            "camera_unreachable_observed",
            mapOf("source" to source, "invalidated_current_session" to invalidated.toString()),
        )
        return invalidated
    }

    private fun invalidateSessionIfCurrent(active: CameraSession, message: String): Boolean {
        val invalidated = sessions.invalidate(active) {
            _latestFrame.value = null
            _isStreaming.value = false
            _isRecording.value = false
            _isContinuousShooting.value = false
            _cameraStatus.value = null
            _connection.value = ConnectionState.Failed(message)
        }
        if (invalidated) {
            stopPhysicalShutterMonitor(active)
            active.transport.cancelActiveRequests()
            active.originalImportTransport.cancelActiveRequests()
            active.avContentTransport.cancelActiveRequests()
        }
        return invalidated
    }

    private suspend fun preparePhysicalShutterMonitor(active: CameraSession) {
        if ("getEvent" !in active.capabilities.availableApis) {
            DiagnosticLog.record("physical_shutter_listener_unavailable")
            return
        }
        stopPhysicalShutterMonitor(active)
        val eventVersion = active.eventBackend.negotiateEventVersion()
        val baseline = active.eventBackend.getEvent(longPolling = false)
        baseline.bodyCaptureUris.forEach(active.recentCaptureUris::add)
        applyCameraEventState(active, baseline)
        val postviewSize = active.backend.setPostviewSize(
            active.capabilities.availableApis,
            PostviewSizePreference.Original,
        )
        DiagnosticLog.record(
            "physical_shutter_baseline_ready",
            mapOf(
                "queued_uri_count" to baseline.bodyCaptureUris.size.toString(),
                "event_version" to eventVersion,
                "postview_size" to (postviewSize ?: "unknown"),
            ),
        )
        active.physicalShutterJob = repositoryScope.launch {
            monitorPhysicalShutter(active)
        }
        DiagnosticLog.record("physical_shutter_listener_started")
    }

    private fun stopPhysicalShutterMonitor(active: CameraSession) {
        active.physicalShutterJob?.cancel()
        active.physicalShutterJob = null
        active.eventTransport.cancelActiveRequests()
        _isPhysicalShutterTransferActive.value = false
    }

    private suspend fun monitorPhysicalShutter(active: CameraSession) = coroutineScope {
        val queue = Channel<RemoteCaptureBatch>(capacity = PHYSICAL_CAPTURE_URI_QUEUE_SIZE)
        val queuedUris = ConcurrentHashMap.newKeySet<String>()
        val pendingCount = AtomicInteger(0)
        val worker = launch {
            for (batch in queue) {
                val imported = mutableListOf<SavedCapture>()
                var failedDownloads = 0
                var becameIdle = false
                batch.captures.forEach { capture ->
                    try {
                        val saved = handlePhysicalShutterCapture(active, capture)
                        if (saved == null) failedDownloads++ else imported += saved
                    } finally {
                        queuedUris.remove(capture.postviewUri.captureKey())
                        if (pendingCount.decrementAndGet() == 0) becameIdle = true
                    }
                }
                if (batch.isContinuous && sessions.isCurrent(active)) {
                    val result = ContinuousBatchResult(
                        reportedUrlCount = batch.reportedUrlCount,
                        importedCaptures = imported,
                        failedDownloadCount = failedDownloads,
                    )
                    _continuousCaptureBatches.emit(result)
                    DiagnosticLog.record(
                        "continuous_capture_batch_completed",
                        mapOf(
                            "reported_url_count" to result.reportedUrlCount.toString(),
                            "imported_count" to result.importedCaptures.size.toString(),
                            "failed_download_count" to result.failedDownloadCount.toString(),
                        ),
                    )
                }
                if (becameIdle && sessions.isCurrent(active)) {
                    _isPhysicalShutterTransferActive.value = false
                }
            }
        }
        var consecutiveFailures = 0
        try {
            while (currentCoroutineContext().isActive && sessions.isCurrent(active)) {
                try {
                    val event = active.eventBackend.getEvent(longPolling = true)
                    currentCoroutineContext().ensureActive()
                    if (!sessions.isCurrent(active)) return@coroutineScope
                    applyCameraEventState(active, event)
                    consecutiveFailures = 0
                    if (event.bodyCaptureUris.isNotEmpty()) {
                        DiagnosticLog.record(
                            "physical_shutter_event_received",
                            mapOf(
                                "uri_count" to event.bodyCaptureUris.size.toString(),
                                "event_kinds" to event.captureKinds.joinToString { it.name },
                                "event_version" to event.eventVersion,
                                "camera_status" to (event.cameraStatus ?: "unknown"),
                                "cached_drive" to (
                                    active.capabilities.settings[CameraSettingId.ContShootingMode]
                                        ?.currentWireValue ?: "unknown"
                                    ),
                            ),
                        )
                    }
                    val acceptedCaptures = mutableListOf<RemoteCapture>()
                    event.remoteCaptures.forEach { capture ->
                        val uri = capture.postviewUri
                        if (active.recentCaptureUris.contains(uri)) {
                            DiagnosticLog.record("physical_shutter_capture_duplicate")
                            return@forEach
                        }
                        if (!queuedUris.add(uri.captureKey())) return@forEach
                        if (!computationalCaptureImport.get()) {
                            _physicalShutterReferences.emit(
                                PendingRemoteCapture(capture, _latestFrame.value?.jpeg?.copyOf()),
                            )
                        }
                        pendingCount.incrementAndGet()
                        _isPhysicalShutterTransferActive.value = true
                        try {
                            acceptedCaptures += capture
                        } catch (error: Throwable) {
                            queuedUris.remove(uri.captureKey())
                            if (pendingCount.decrementAndGet() == 0 && sessions.isCurrent(active)) {
                                _isPhysicalShutterTransferActive.value = false
                            }
                            throw error
                        }
                    }
                    if (acceptedCaptures.isNotEmpty()) {
                        try {
                            queue.send(
                                RemoteCaptureBatch(
                                    captures = acceptedCaptures,
                                    isContinuous = CameraCaptureEventKind.Continuous in event.captureKinds,
                                    reportedUrlCount = event.remoteCaptures.size,
                                ),
                            )
                        } catch (error: Throwable) {
                            acceptedCaptures.forEach { capture ->
                                queuedUris.remove(capture.postviewUri.captureKey())
                                pendingCount.decrementAndGet()
                            }
                            if (pendingCount.get() == 0 && sessions.isCurrent(active)) {
                                _isPhysicalShutterTransferActive.value = false
                            }
                            throw error
                        }
                    }
                    if (CameraCaptureEventKind.Continuous in event.captureKinds && acceptedCaptures.isNotEmpty()) {
                        DiagnosticLog.record(
                            "continuous_capture_batch_queued",
                            mapOf("accepted_uri_count" to acceptedCaptures.size.toString()),
                        )
                    }
                } catch (error: Throwable) {
                    if (error is CancellationException) throw error
                    if (!sessions.isCurrent(active)) return@coroutineScope
                    if (error.isNormalEventPollingTimeout()) {
                        consecutiveFailures = 0
                        continue
                    }
                    consecutiveFailures++
                    if (consecutiveFailures == 1 || consecutiveFailures % EVENT_FAILURE_LOG_INTERVAL == 0) {
                        DiagnosticLog.record(
                            "physical_shutter_listener_failed",
                            mapOf(
                                "attempt" to consecutiveFailures.toString(),
                                "error" to error.toUserMessage("unknown"),
                                "type" to error.javaClass.simpleName,
                            ),
                        )
                    }
                    if (!networkProvider.isAvailable(active.network)) {
                        markNetworkLostIfCurrent(active, "physical_shutter_listener")
                        return@coroutineScope
                    }
                    delay(
                        minOf(
                            EVENT_RETRY_MAX_MILLIS,
                            EVENT_RETRY_BASE_MILLIS * consecutiveFailures,
                        ),
                    )
                }
            }
        } finally {
            queue.close()
            worker.cancelAndJoin()
            if (sessions.isCurrent(active)) _isPhysicalShutterTransferActive.value = false
            DiagnosticLog.record("physical_shutter_listener_stopped")
        }
    }

    private fun applyCameraEventState(active: CameraSession, event: com.ryu.sonyremote.model.CameraEvent) {
        event.cameraStatus?.let { status ->
            _cameraStatus.value = status
            _isContinuousShooting.value = status.equals("StillCapturing", ignoreCase = true)
        }
        val eventApis = event.availableApis
        val eventDrive = event.continuousShootingMode
        val eventZoomPosition = event.zoomPosition
        val eventZoomSetting = event.zoomSetting
        val eventZoomBoxCount = event.zoomBoxCount
        val eventZoomBoxIndex = event.zoomBoxIndex
        if (eventApis == null && eventDrive == null && eventZoomPosition == null && eventZoomSetting == null &&
            eventZoomBoxCount == null && eventZoomBoxIndex == null
        ) return
        val settings = if (eventDrive == null) {
            active.capabilities.settings
        } else {
            active.capabilities.settings.toMutableMap().apply {
                this[CameraSettingId.ContShootingMode]?.let { current ->
                    this[CameraSettingId.ContShootingMode] = current.copy(currentWireValue = eventDrive)
                }
            }
        }
        publishCapabilitiesIfCurrent(
            active,
            active.capabilities.copy(
                availableApis = eventApis ?: active.capabilities.availableApis,
                settings = settings,
                zoomPosition = eventZoomPosition?.coerceIn(0, 100) ?: active.capabilities.zoomPosition,
                zoomSetting = eventZoomSetting ?: active.capabilities.zoomSetting,
                zoomBoxCount = eventZoomBoxCount ?: active.capabilities.zoomBoxCount,
                zoomBoxIndex = eventZoomBoxIndex ?: active.capabilities.zoomBoxIndex,
            ),
        )
    }

    private suspend fun handlePhysicalShutterCapture(
        active: CameraSession,
        remoteCapture: RemoteCapture,
    ): SavedCapture? {
        val remoteUri = remoteCapture.postviewUri
        val download = selectRemoteCaptureDownload(remoteCapture, computationalCaptureImport.get())
        val previewFirst = download.previewFirst
        val downloadUri = download.uri
        while (active.apiCaptureInFlight.get()) {
            delay(API_CAPTURE_DEDUPE_WAIT_MILLIS)
            if (!sessions.isCurrent(active)) return null
        }
        if (active.recentCaptureUris.contains(remoteUri)) {
            DiagnosticLog.record("physical_shutter_capture_duplicate")
            return null
        }
        var lastError: Throwable? = null
        repeat(PHYSICAL_CAPTURE_SAVE_ATTEMPTS) { attempt ->
            try {
                _downloadProgress.tryEmit(
                    CaptureDownloadProgress(remoteUri, 0, null, attempt = attempt + 1),
                )
                PrivateNetworkPolicy.requireCameraUri(downloadUri, active.device.descriptionUri.host)
                val sourceJpeg = JpegValidator.normalize(
                    active.eventTransport.readBytes(downloadUri) { bytes, total ->
                        _downloadProgress.tryEmit(CaptureDownloadProgress(remoteUri, bytes, total))
                    },
                )
                val processing = if (computationalCaptureImport.get()) {
                    DownloadedImageProcessingResult(sourceJpeg)
                } else {
                    downloadedImageProcessor(sourceJpeg)
                }
                val jpeg = JpegValidator.normalize(processing.jpeg)
                currentCoroutineContext().ensureActive()
                if (!sessions.isCurrent(active)) return null
                DiagnosticLog.record(
                    "physical_shutter_capture_downloaded",
                    mapOf("bytes" to jpeg.size.toString()),
                )
                val savedUri = mediaStore.save(
                    jpeg,
                    when {
                        computationalCaptureImport.get() -> "SONY_SOURCE"
                        previewFirst -> "SONY_PREVIEW"
                        else -> "SONY"
                    },
                    outputImageFormat,
                )
                if (!jpeg.contentEquals(sourceJpeg)) {
                    runCatching { mediaStore.copyExif(sourceJpeg, savedUri) }
                    runCatching { saveEditingOriginal(savedUri, sourceJpeg) }
                        .onFailure { error ->
                            DiagnosticLog.record(
                                "editing_original_save_failed",
                                mapOf("error" to error.toUserMessage("unknown")),
                            )
                        }
                }
                applyPhoneLocation(savedUri)
                val saved = SavedCapture(
                    uri = savedUri,
                    jpeg = jpeg,
                    originalSizeRequested = !previewFirst && remoteUri.requestsOriginalPostview(),
                    thumbnailRemoteUri = remoteCapture.thumbnailUri,
                    postviewRemoteUri = remoteUri,
                    postviewRemoteIsOriginal = remoteUri.requestsOriginalPostview(),
                    cameraContentId = System.currentTimeMillis().toString().takeIf {
                        active.capabilities.contentsTransferAvailable &&
                            !remoteUri.requestsOriginalPostview()
                    },
                    appliedEdits = processing.appliedEdits,
                )
                active.recentCaptureUris.add(remoteUri)
                currentCoroutineContext().ensureActive()
                if (!sessions.isCurrent(active)) return null
                _physicalShutterCaptures.emit(saved)
                DiagnosticLog.record(
                    "physical_shutter_capture_saved",
                    mapOf(
                        "bytes" to jpeg.size.toString(),
                        "event_emitted" to "true",
                        "original_requested" to saved.originalSizeRequested.toString(),
                        "preview_first" to previewFirst.toString(),
                    ),
                )
                return saved
            } catch (error: Throwable) {
                if (error is CancellationException) throw error
                lastError = error
                if (attempt + 1 < PHYSICAL_CAPTURE_SAVE_ATTEMPTS) {
                    delay(PHYSICAL_CAPTURE_RETRY_MILLIS * (attempt + 1))
                }
            }
        }
        val message = lastError.toUserMessage("Could not transfer the camera shutter photo")
        _downloadProgress.tryEmit(
            CaptureDownloadProgress(remoteUri, 0, null, attempt = PHYSICAL_CAPTURE_SAVE_ATTEMPTS, failed = true),
        )
        _physicalShutterFailures.tryEmit(message)
        DiagnosticLog.record(
            "physical_shutter_capture_failed",
            mapOf(
                "error" to message,
                "type" to (lastError?.javaClass?.simpleName ?: "unknown"),
                "attempts" to PHYSICAL_CAPTURE_SAVE_ATTEMPTS.toString(),
            ),
        )
        return null
    }

    private fun monitorNetwork(active: CameraSession) {
        stopNetworkMonitor()
        val request = NetworkRequest.Builder()
            .addTransportType(NetworkCapabilities.TRANSPORT_WIFI)
            .build()
        val callback = object : ConnectivityManager.NetworkCallback() {
            override fun onLost(lostNetwork: Network) {
                if (lostNetwork != active.network) return
                markNetworkLostIfCurrent(active, "network_callback")
            }
        }
        networkCallback = callback
        connectivityManager.registerNetworkCallback(request, callback)
    }

    private fun stopNetworkMonitor() {
        networkCallback?.let { callback ->
            runCatching { connectivityManager.unregisterNetworkCallback(callback) }
        }
        networkCallback = null
    }

    private fun Throwable?.toUserMessage(fallback: String): String = when (this) {
        null -> fallback
        is java.net.SocketTimeoutException -> "$fallback: the camera timed out"
        is java.net.ConnectException -> "$fallback: connection refused"
        else -> message?.takeIf(String::isNotBlank) ?: fallback
    }

    private data class CameraSession(
        val network: Network,
        val device: SonyCameraDevice,
        val transport: NetworkHttpTransport,
        val backend: CameraBackend,
        val eventTransport: NetworkHttpTransport,
        val eventBackend: CameraBackend,
        val originalImportTransport: NetworkHttpTransport,
        val avContentTransport: NetworkHttpTransport,
        var capabilities: CameraCapabilities,
    ) {
        @Volatile var physicalShutterJob: Job? = null
        val apiCaptureInFlight = AtomicBoolean(false)
        val recentCaptureUris = RecentCaptureUris(PHYSICAL_CAPTURE_DEDUPE_SIZE)
    }

    private companion object {
        const val EDITING_ORIGINAL_DIRECTORY = "editing_originals"
        const val STOP_TIMEOUT_MILLIS = 1_500L
        const val CONTINUOUS_STOP_TIMEOUT_MILLIS = 2_000
        const val CONTINUOUS_CLOSE_STOP_TIMEOUT_MILLIS = 2_500L
        const val POSTVIEW_RESTORE_TIMEOUT_MILLIS = 4_000L
        const val CAPABILITY_REFRESH_ATTEMPTS = 8
        const val MOVIE_STATE_REFRESH_ATTEMPTS = 20
        const val CAPABILITY_REFRESH_DELAY_MILLIS = 250L
        const val CAPABILITY_POLL_INTERVAL_MILLIS = 3_000L
        const val CAMERA_UNREACHABLE_FAILURE_THRESHOLD = 2
        const val PHYSICAL_CAPTURE_HANDOFF_BUFFER = 1
        const val PHYSICAL_CAPTURE_EVENT_BUFFER = 8
        const val PHYSICAL_CAPTURE_DEDUPE_SIZE = 128
        const val PHYSICAL_CAPTURE_URI_QUEUE_SIZE = 32
        const val PHYSICAL_CAPTURE_SAVE_ATTEMPTS = 5
        const val PHYSICAL_CAPTURE_RETRY_MILLIS = 500L
        const val API_CAPTURE_DEDUPE_WAIT_MILLIS = 25L
        const val EVENT_FAILURE_LOG_INTERVAL = 5
        const val EVENT_RETRY_BASE_MILLIS = 500L
        const val EVENT_RETRY_MAX_MILLIS = 5_000L
        const val LIVEVIEW_END_GRACE_MILLIS = 250L
        const val REC_MODE_START_ATTEMPTS = 12
        const val REC_MODE_START_DELAY_MILLIS = 250L
        const val CONTENTS_TRANSFER_READY_ATTEMPTS = 12
        const val CONTENTS_TRANSFER_READY_DELAY_MILLIS = 500L
        const val CONTENTS_TRANSFER_FUNCTION = "Contents Transfer"
        const val ZOOM_POSITION_READ_ATTEMPTS = 5
        const val ZOOM_POSITION_READ_DELAY_MILLIS = 100L
        const val REMOTE_SHOOTING_FUNCTION = "Remote Shooting"
        const val NETWORK_LOST_MESSAGE = "Camera Wi-Fi connection was lost"
        const val CAMERA_UNREACHABLE_MESSAGE = "Camera stopped responding on this Wi-Fi network"
    }
}

internal class RecentCaptureUris(private val capacity: Int) {
    private val values = LinkedHashSet<String>()

    init {
        require(capacity > 0) { "Capture URI capacity must be positive" }
    }

    @Synchronized
    fun add(uri: URI): Boolean {
        val key = uri.normalize().toASCIIString()
        if (!values.add(key)) return false
        if (values.size > capacity) {
            values.remove(values.first())
        }
        return true
    }

    @Synchronized
    fun contains(uri: URI): Boolean = uri.normalize().toASCIIString() in values
}

private fun URI.captureKey(): String = normalize().toASCIIString()

internal fun Throwable.isNormalEventPollingTimeout(): Boolean =
    this is SocketTimeoutException || (this is CameraApiException && code == 2)

internal fun URI.requestsOriginalPostview(): Boolean =
    rawQuery
        .orEmpty()
        .split('&')
        .mapNotNull { parameter ->
            val separator = parameter.indexOf('=')
            if (separator <= 0) return@mapNotNull null
            parameter.substring(0, separator) to parameter.substring(separator + 1)
        }
        .any { (name, value) ->
            name.equals("size", ignoreCase = true) &&
                (value.equals("Origin", ignoreCase = true) ||
                    value.equals("Original", ignoreCase = true))
        }

internal data class RemoteCaptureDownload(
    val uri: URI,
    val previewFirst: Boolean,
)

internal fun selectRemoteCaptureDownload(
    capture: RemoteCapture,
    computational: Boolean,
): RemoteCaptureDownload = RemoteCaptureDownload(capture.postviewUri, previewFirst = false)

internal fun shouldRetainSingleOriginalReference(
    capture: RemoteCapture,
    computational: Boolean,
): Boolean =
    capture.kind == CameraCaptureEventKind.Single &&
        !computational &&
        capture.postviewUri.requestsOriginalPostview()

data class SavedCapture(
    val uri: Uri,
    val jpeg: ByteArray,
    val originalSizeRequested: Boolean,
    val thumbnailRemoteUri: URI? = null,
    val postviewRemoteUri: URI? = null,
    val postviewRemoteIsOriginal: Boolean = originalSizeRequested,
    val cameraContentId: String? = null,
    val appliedEdits: AppliedImageEdits = AppliedImageEdits(),
) {
    val bytes: Int get() = jpeg.size
}

data class AppliedImageEdits(
    val lutName: String? = null,
    val lutStrength: Float? = null,
    val denoiseModel: String? = null,
    val denoiseStrength: Float? = null,
)

data class DownloadedImageProcessingResult(
    val jpeg: ByteArray,
    val appliedEdits: AppliedImageEdits = AppliedImageEdits(),
)

data class ContinuousBatchResult(
    val reportedUrlCount: Int,
    val importedCaptures: List<SavedCapture>,
    val failedDownloadCount: Int,
) {
    init {
        require(reportedUrlCount >= 0)
        require(failedDownloadCount >= 0)
        require(importedCaptures.size + failedDownloadCount <= reportedUrlCount)
    }

    val completedDownloadCount: Int get() = importedCaptures.size + failedDownloadCount
}

data class PendingRemoteCapture(
    val remoteCapture: RemoteCapture,
    val liveViewJpeg: ByteArray?,
)

data class CaptureDownloadProgress(
    val remoteUri: URI,
    val bytesRead: Long,
    val totalBytes: Long?,
    val attempt: Int = 1,
    val failed: Boolean = false,
)

private data class RemoteCaptureBatch(
    val captures: List<RemoteCapture>,
    val isContinuous: Boolean,
    val reportedUrlCount: Int,
)

data class DownloadedCapture(
    val jpeg: ByteArray,
    val originalSizeRequested: Boolean,
    val postviewSize: String?,
    val postviewRemoteUri: URI? = null,
    val cameraContentId: String? = null,
)
