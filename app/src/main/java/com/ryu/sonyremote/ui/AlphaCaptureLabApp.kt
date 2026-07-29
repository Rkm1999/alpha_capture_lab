package com.ryu.sonyremote.ui

import android.content.Intent
import android.Manifest
import android.content.pm.PackageManager
import android.provider.OpenableColumns
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import android.graphics.Bitmap
import androidx.compose.foundation.Image
import androidx.compose.foundation.Canvas
import androidx.compose.animation.core.animate
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.gestures.draggable
import androidx.compose.foundation.gestures.Orientation
import androidx.compose.foundation.gestures.rememberDraggableState
import androidx.compose.foundation.gestures.rememberTransformableState
import androidx.compose.foundation.gestures.transformable
import androidx.compose.foundation.gestures.waitForUpOrCancellation
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items as gridItems
import androidx.compose.foundation.pager.HorizontalPager
import androidx.compose.foundation.pager.rememberPagerState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.FileDownload
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.LinkOff
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Compare
import androidx.compose.material.icons.filled.AutoFixHigh
import androidx.compose.material.icons.filled.Crop
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material.icons.filled.Palette
import androidx.compose.material.icons.filled.PhotoLibrary
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.RotateLeft
import androidx.compose.material.icons.filled.RotateRight
import androidx.compose.material.icons.filled.Search
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material.icons.filled.Transform
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material.icons.automirrored.filled.Undo
import androidx.compose.material.icons.automirrored.filled.Redo
import androidx.compose.material.icons.filled.Wifi
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledIconButton
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Switch
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.runtime.snapshotFlow
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.clipRect
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import com.ryu.sonyremote.processing.AinrDenoiseModel
import com.ryu.sonyremote.processing.AinrProgress
import com.ryu.sonyremote.processing.EditorGeometry
import com.ryu.sonyremote.processing.NormalizedCrop
import com.ryu.sonyremote.processing.NormalizedPoint
import com.ryu.sonyremote.processing.cropPresetForAspect
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.compose.LocalLifecycleOwner
import kotlin.math.roundToInt
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import com.ryu.sonyremote.model.CameraCapabilities
import com.ryu.sonyremote.model.CameraSetting
import com.ryu.sonyremote.model.ConnectionState
import com.ryu.sonyremote.model.SonyCameraDevice
import com.ryu.sonyremote.data.DiagnosticLog
import com.ryu.sonyremote.processing.LutPreset
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.math.abs

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AlphaCaptureLabApp(
    viewModel: CameraViewModel,
    onOpenWifi: () -> Unit,
    onFindCamera: () -> Unit,
    onPairCamera: (String, String, Boolean) -> Unit,
    onOpenAppSettings: () -> Unit,
) {
    val connection by viewModel.connection.collectAsStateWithLifecycle()
    val preview by viewModel.previewBitmap.collectAsStateWithLifecycle()
    val isStreaming by viewModel.isStreaming.collectAsStateWithLifecycle()
    val isPhysicalShutterTransferActive by
        viewModel.isPhysicalShutterTransferActive.collectAsStateWithLifecycle()
    val isContinuousBurstActive by viewModel.isContinuousBurstActive.collectAsStateWithLifecycle()
    val isBusy by viewModel.isBusy.collectAsStateWithLifecycle()
    val message by viewModel.message.collectAsStateWithLifecycle()
    val permissionBlocked by viewModel.permissionBlocked.collectAsStateWithLifecycle()
    val pairedCameras by viewModel.pairedCameras.collectAsStateWithLifecycle()
    val captureMode by viewModel.captureMode.collectAsStateWithLifecycle()
    val liveNdFrames by viewModel.liveNdFrames.collectAsStateWithLifecycle()
    val liveNdStrategy by viewModel.liveNdStrategy.collectAsStateWithLifecycle()
    val captureSession by viewModel.captureSession.collectAsStateWithLifecycle()
    val filmstrip by viewModel.filmstrip.collectAsStateWithLifecycle()
    val lutEditor by viewModel.lutEditor.collectAsStateWithLifecycle()
    val lutCaptureState by viewModel.lutCaptureState.collectAsStateWithLifecycle()
    val lutPreviews by viewModel.lutPreviews.collectAsStateWithLifecycle()
    val zoomTargetPercent by viewModel.zoomTargetPercent.collectAsStateWithLifecycle()
    val geotaggingEnabled by viewModel.geotaggingEnabled.collectAsStateWithLifecycle()
    val automaticPostviewPreference by viewModel.automaticPostviewPreference.collectAsStateWithLifecycle()
    val outputImageFormat by viewModel.outputImageFormat.collectAsStateWithLifecycle()
    val autoDenoiseMode by viewModel.autoDenoiseMode.collectAsStateWithLifecycle()
    val autoDenoiseIsoThreshold by viewModel.autoDenoiseIsoThreshold.collectAsStateWithLifecycle()
    val autoDenoiseModel by viewModel.autoDenoiseModel.collectAsStateWithLifecycle()
    val ainrProgress by viewModel.ainrProgress.collectAsStateWithLifecycle()
    val liveViewTimeoutMinutes by viewModel.liveViewTimeoutMinutes.collectAsStateWithLifecycle()
    val snackbarHostState = remember { SnackbarHostState() }
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()
    val lifecycleOwner = LocalLifecycleOwner.current
    val cubeLauncher = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
            val name = context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
                ?.use { cursor -> if (cursor.moveToFirst()) cursor.getString(0) else null }
                ?: uri.lastPathSegment.orEmpty()
            val text = runCatching {
                context.contentResolver.openInputStream(uri)?.bufferedReader()?.use { it.readText() }
                    ?: error("Could not open LUT")
            }
            text.onSuccess { viewModel.importCubeLut(name, it) }
                .onFailure { viewModel.reportLutImportFailure(it) }
        }
    }
    val locationPermission = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted -> viewModel.setGeotaggingEnabled(granted) }

    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            when (event) {
                Lifecycle.Event.ON_START -> viewModel.onForeground()
                Lifecycle.Event.ON_STOP -> viewModel.onBackground()
                else -> Unit
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose { lifecycleOwner.lifecycle.removeObserver(observer) }
    }
    var showDiagnostics by remember { mutableStateOf(false) }
    var showHistory by remember { mutableStateOf(false) }
    var showSettings by remember { mutableStateOf(false) }

    LaunchedEffect(message) {
        val current = message ?: return@LaunchedEffect
        snackbarHostState.showSnackbar(current.text)
        viewModel.clearMessage()
    }

    val ready = connection as? ConnectionState.Ready
    Scaffold(
        topBar = {
            TopAppBar(
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.background,
                ),
                title = {
                    Column {
                        Text(
                            text = ready?.device?.modelName?.takeUnless { it == "Unknown model" }
                                ?: "Alpha Capture Lab",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.SemiBold,
                        )
                        Text(
                            text = if (ready == null) "Sony camera control" else "Connected",
                            style = MaterialTheme.typography.labelSmall,
                            color = if (ready == null) {
                                MaterialTheme.colorScheme.onSurfaceVariant
                            } else {
                                MaterialTheme.colorScheme.primary
                            },
                        )
                    }
                },
                actions = {
                    if (ready != null) {
                        IconButton(onClick = { showSettings = true }) {
                            Icon(Icons.Default.Settings, contentDescription = "Settings")
                        }
                        IconButton(onClick = { showDiagnostics = true }) {
                            Icon(Icons.Default.Info, contentDescription = "Camera diagnostics")
                        }
                        IconButton(onClick = viewModel::disconnect) {
                            Icon(Icons.Default.LinkOff, contentDescription = "Disconnect")
                        }
                    }
                },
            )
        },
        snackbarHost = { SnackbarHost(snackbarHostState) },
    ) { contentPadding ->
        when (val state = connection) {
            is ConnectionState.Ready -> CameraControlScreen(
                modifier = Modifier.padding(contentPadding),
                state = state,
                preview = preview,
                isStreaming = isStreaming,
                isPhysicalShutterTransferActive = isPhysicalShutterTransferActive,
                isContinuousBurstActive = isContinuousBurstActive,
                isBusy = isBusy,
                captureMode = captureMode,
                liveNdFrames = liveNdFrames,
                liveNdStrategy = liveNdStrategy,
                session = captureSession,
                filmstrip = filmstrip,
                lutState = lutCaptureState,
                lutPreviews = lutPreviews,
                ainrProgress = ainrProgress,
                zoomTargetPercent = zoomTargetPercent,
                onSelectMode = viewModel::selectCaptureMode,
                onSetLiveNdFrames = viewModel::setLiveNdFrames,
                onSetLiveNdStrategy = viewModel::setLiveNdStrategy,
                onPrimaryAction = viewModel::primaryCaptureAction,
                onStartContinuousBurst = viewModel::startContinuousBurst,
                onStopContinuousBurst = viewModel::stopContinuousBurst,
                onFinishPanorama = viewModel::finishPanorama,
                onCancelPanorama = viewModel::cancelPanorama,
                onOpenLut = viewModel::openLutEditor,
                onImportOriginal = viewModel::requestOriginalImport,
                onCancelOriginal = viewModel::cancelOriginalImport,
                onSelectLiveLut = viewModel::selectLiveLut,
                onSetLutIntensity = viewModel::setLutIntensity,
                onSetBakeLut = viewModel::setBakeLutIntoPhotos,
                onAddLiveLut = viewModel::addLiveLut,
                onRemoveLiveLut = viewModel::removeLiveLut,
                onImportCubeLut = { cubeLauncher.launch(arrayOf("*/*")) },
                onSelectImportedLut = viewModel::selectImportedLut,
                onRemoveImportedLut = viewModel::removeImportedLut,
                onOpenHistory = { showHistory = true },
                onRetryLiveView = viewModel::retryLiveView,
                onSetSetting = viewModel::setSetting,
                onStartRemoteZoom = viewModel::startRemoteZoom,
                onStopRemoteZoom = viewModel::stopRemoteZoom,
                onSetRemoteZoomTarget = viewModel::setRemoteZoomTarget,
                onRefreshZoomState = viewModel::refreshZoomState,
            )

            ConnectionState.Disconnected -> SetupScreen(
                modifier = Modifier.padding(contentPadding),
                error = null,
                isBusy = isBusy,
                permissionBlocked = permissionBlocked,
                pairedCameras = pairedCameras,
                onPairCamera = onPairCamera,
                onConnectPaired = viewModel::connectPairedCamera,
                onSetAutoConnect = viewModel::setPairedCameraAutoConnect,
                onForgetPaired = viewModel::forgetPairedCamera,
                onOpenWifi = onOpenWifi,
                onFindCamera = onFindCamera,
                onOpenAppSettings = onOpenAppSettings,
                onOpenGallery = { showHistory = true },
            )

            is ConnectionState.Failed -> SetupScreen(
                modifier = Modifier.padding(contentPadding),
                error = state.message,
                isBusy = isBusy,
                permissionBlocked = permissionBlocked,
                pairedCameras = pairedCameras,
                onPairCamera = onPairCamera,
                onConnectPaired = viewModel::connectPairedCamera,
                onSetAutoConnect = viewModel::setPairedCameraAutoConnect,
                onForgetPaired = viewModel::forgetPairedCamera,
                onOpenWifi = onOpenWifi,
                onFindCamera = onFindCamera,
                onOpenAppSettings = onOpenAppSettings,
                onOpenGallery = { showHistory = true },
            )

            ConnectionState.WaitingForWifi -> ProgressScreen(
                modifier = Modifier.padding(contentPadding),
                title = "Looking for Wi-Fi",
                detail = "Waiting for the camera network",
                onCancel = viewModel::disconnect,
            )

            ConnectionState.Discovering -> ProgressScreen(
                modifier = Modifier.padding(contentPadding),
                title = "Finding camera",
                detail = "Searching the local network",
                onCancel = viewModel::disconnect,
            )

            is ConnectionState.Connecting -> ProgressScreen(
                modifier = Modifier.padding(contentPadding),
                title = "Connecting",
                detail = state.description,
                onCancel = viewModel::disconnect,
            )
        }
    }

    if (showSettings && ready != null) {
        CameraSettingsDialog(
            capabilities = ready.capabilities,
            enabled = !isBusy,
            geotaggingEnabled = geotaggingEnabled,
            postviewPreference = automaticPostviewPreference,
            onSetPostviewPreference = viewModel::setAutomaticPostviewPreference,
            outputImageFormat = outputImageFormat,
            onSetOutputImageFormat = viewModel::setOutputImageFormat,
            autoDenoiseMode = autoDenoiseMode,
            autoDenoiseIsoThreshold = autoDenoiseIsoThreshold,
            autoDenoiseModel = autoDenoiseModel,
            ainrProgress = ainrProgress,
            onSetAutoDenoiseMode = viewModel::setAutoDenoiseMode,
            onSetAutoDenoiseIsoThreshold = viewModel::setAutoDenoiseIsoThreshold,
            onSetAutoDenoiseModel = viewModel::setAutoDenoiseModel,
            liveViewTimeoutMinutes = liveViewTimeoutMinutes,
            onSetLiveViewTimeoutMinutes = viewModel::setLiveViewTimeoutMinutes,
            onSetGeotagging = { enabled ->
                if (!enabled) {
                    viewModel.setGeotaggingEnabled(false)
                } else if (ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION) ==
                    PackageManager.PERMISSION_GRANTED
                ) {
                    viewModel.setGeotaggingEnabled(true)
                } else {
                    locationPermission.launch(Manifest.permission.ACCESS_FINE_LOCATION)
                }
            },
            onSetLiveviewSize = viewModel::setLiveviewSize,
            onDismiss = { showSettings = false },
        )
    }

    if (showDiagnostics && ready != null) {
        DiagnosticsDialog(
            device = ready.device,
            capabilities = ready.capabilities,
            onExport = {
                coroutineScope.launch {
                    val shareIntent = withContext(Dispatchers.IO) {
                        DiagnosticLog.createShareIntent(context)
                    }
                    context.startActivity(Intent.createChooser(shareIntent, "Share diagnostics"))
                }
            },
            onDismiss = { showDiagnostics = false },
        )
    }

    if (showHistory) {
        CaptureHistoryDialog(
            items = filmstrip,
            enabled = true,
            onOpenLut = viewModel::openLutEditor,
            onLoadDetail = viewModel::loadGalleryDetail,
            onImportOriginal = viewModel::requestOriginalImport,
            onCancelOriginal = viewModel::cancelOriginalImport,
            onDismiss = { showHistory = false },
        )
    }
    if (lutEditor != null) {
        LutEditorDialog(
            state = requireNotNull(lutEditor),
            onSelectPreset = viewModel::selectLut,
            onSelectImported = viewModel::selectEditorImportedLut,
            onRequestLutThumbnails = viewModel::requestEditorLutThumbnails,
            onSetIntensity = viewModel::setLutIntensity,
            onSetBasicEdits = viewModel::setBasicEdits,
            onSetDenoiseEnabled = viewModel::setEditorDenoiseEnabled,
            onSetDenoiseStrength = viewModel::setEditorDenoiseStrength,
            onSetDenoiseModel = viewModel::setEditorDenoiseModel,
            onSetSource = viewModel::setEditorSource,
            onSetCrop = viewModel::setEditorCrop,
            onRotateQuarterTurn = viewModel::rotateEditorQuarterTurn,
            onSetStraighten = viewModel::setEditorStraighten,
            onSetPerspective = viewModel::setEditorPerspective,
            onResetGeometry = viewModel::resetEditorGeometry,
            onAutoGeometry = viewModel::autoCorrectEditorGeometry,
            onUndo = viewModel::undoEditor,
            onRedo = viewModel::redoEditor,
            onBeginGesture = viewModel::beginEditorGesture,
            onEndGesture = viewModel::endEditorGesture,
            onSave = viewModel::saveLutCopy,
            onDismiss = viewModel::closeLutEditor,
            ainrProgress = ainrProgress,
        )
    }
}

@Composable
private fun SetupScreen(
    modifier: Modifier,
    error: String?,
    isBusy: Boolean,
    permissionBlocked: Boolean,
    pairedCameras: List<com.ryu.sonyremote.network.PairedCamera>,
    onPairCamera: (String, String, Boolean) -> Unit,
    onConnectPaired: (String) -> Unit,
    onSetAutoConnect: (String, Boolean) -> Unit,
    onForgetPaired: (String) -> Unit,
    onOpenWifi: () -> Unit,
    onFindCamera: () -> Unit,
    onOpenAppSettings: () -> Unit,
    onOpenGallery: () -> Unit,
) {
    var showPairDialog by remember { mutableStateOf(false) }
    Column(
        modifier = modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 24.dp, vertical = 20.dp),
        verticalArrangement = Arrangement.spacedBy(20.dp),
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Connect a Sony camera", style = MaterialTheme.typography.headlineSmall)
            Text(
                "Start the camera's remote application, then join its DIRECT Wi-Fi network.",
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }

        OutlinedButton(
            onClick = onOpenGallery,
            modifier = Modifier.fillMaxWidth().height(52.dp),
        ) {
            Icon(Icons.Default.PhotoLibrary, contentDescription = null)
            Spacer(Modifier.width(10.dp))
            Text("Gallery")
        }

        if (pairedCameras.isNotEmpty()) {
            Text("Paired cameras", style = MaterialTheme.typography.titleMedium)
            pairedCameras.forEach { camera ->
                Surface(
                    modifier = Modifier.fillMaxWidth(),
                    shape = MaterialTheme.shapes.small,
                    border = androidx.compose.foundation.BorderStroke(
                        1.dp,
                        MaterialTheme.colorScheme.outlineVariant,
                    ),
                ) {
                    Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Column(Modifier.weight(1f)) {
                                Text(camera.displayName, style = MaterialTheme.typography.titleSmall)
                                Text(
                                    when {
                                        camera.canRequestWifi && camera.autoConnect -> "Can join Wi-Fi automatically"
                                        camera.autoConnect -> "Starts when Android joins this Wi-Fi"
                                        else -> "Manual connection"
                                    },
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                )
                            }
                            Switch(
                                checked = camera.autoConnect,
                                onCheckedChange = { onSetAutoConnect(camera.id, it) },
                                enabled = !isBusy,
                            )
                        }
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Button(
                                onClick = { onConnectPaired(camera.id) },
                                enabled = !isBusy,
                                modifier = Modifier.weight(1f),
                            ) { Text("Connect") }
                            IconButton(
                                onClick = { onForgetPaired(camera.id) },
                                enabled = !isBusy,
                            ) { Icon(Icons.Default.Delete, contentDescription = "Forget ${camera.displayName}") }
                        }
                    }
                }
            }
        }
        OutlinedButton(
            onClick = { showPairDialog = true },
            enabled = !isBusy,
            modifier = Modifier.fillMaxWidth().height(52.dp),
        ) {
            Icon(Icons.Default.Add, contentDescription = null)
            Spacer(Modifier.width(10.dp))
            Text("Pair camera Wi-Fi")
        }

        if (error != null) {
            StatusBand(text = error, isError = true)
        }
        if (permissionBlocked) {
            StatusBand(
                text = "Nearby devices access is blocked. Allow it in app settings before finding the camera.",
                isError = true,
            )
            OutlinedButton(
                onClick = onOpenAppSettings,
                modifier = Modifier.fillMaxWidth().height(52.dp),
            ) {
                Icon(Icons.Default.Settings, contentDescription = null)
                Spacer(Modifier.width(10.dp))
                Text("Open app permissions")
            }
        }

        SetupStep(
            number = "1",
            title = "Start remote control on the camera",
            detail = "MENU > Application > Application List > Smart Remote Embedded",
        )
        SetupStep(
            number = "2",
            title = "Join the camera Wi-Fi",
            detail = "Use the DIRECT-xxxx SSID and password shown on the camera.",
        )
        OutlinedButton(
            onClick = onOpenWifi,
            modifier = Modifier.fillMaxWidth().height(52.dp),
        ) {
            Icon(Icons.Default.Wifi, contentDescription = null)
            Spacer(Modifier.width(10.dp))
            Text("Open Wi-Fi settings")
        }
        Button(
            onClick = onFindCamera,
            enabled = !isBusy && !permissionBlocked,
            modifier = Modifier.fillMaxWidth().height(52.dp),
        ) {
            if (isBusy) {
                CircularProgressIndicator(Modifier.size(20.dp), strokeWidth = 2.dp)
            } else {
                Icon(Icons.Default.Search, contentDescription = null)
                Spacer(Modifier.width(10.dp))
                Text("Find camera")
            }
        }

        StatusBand(
            text = "Sony ended Smart Remote Control downloads in 2025. Factory Smart Remote Embedded cameras may expose fewer settings and only a 2 MP transfer.",
            isError = false,
        )
    }

    if (showPairDialog) {
        PairCameraDialog(
            onPair = { ssid, password, autoConnect ->
                showPairDialog = false
                onPairCamera(ssid, password, autoConnect)
            },
            onDismiss = { showPairDialog = false },
        )
    }
}

@Composable
private fun PairCameraDialog(
    onPair: (String, String, Boolean) -> Unit,
    onDismiss: () -> Unit,
) {
    var ssid by remember { mutableStateOf("") }
    var password by remember { mutableStateOf("") }
    var autoConnect by remember { mutableStateOf(true) }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Pair camera Wi-Fi") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                OutlinedTextField(
                    value = ssid,
                    onValueChange = { ssid = it },
                    label = { Text("Camera Wi-Fi name") },
                    placeholder = { Text("DIRECT-xxxx") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                )
                OutlinedTextField(
                    value = password,
                    onValueChange = { password = it },
                    label = { Text("Wi-Fi password") },
                    singleLine = true,
                    visualTransformation = PasswordVisualTransformation(),
                    modifier = Modifier.fillMaxWidth(),
                )
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween,
                ) {
                    Text("Connect automatically")
                    Switch(checked = autoConnect, onCheckedChange = { autoConnect = it })
                }
                Text(
                    "Android asks for approval the first time this camera is connected.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        },
        confirmButton = {
            TextButton(
                onClick = { onPair(ssid.trim(), password, autoConnect) },
                enabled = ssid.isNotBlank() && password.length >= 8,
            ) { Text("Pair and connect") }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
    )
}

@Composable
private fun CameraSettingsDialog(
    capabilities: CameraCapabilities,
    enabled: Boolean,
    geotaggingEnabled: Boolean,
    onSetGeotagging: (Boolean) -> Unit,
    postviewPreference: com.ryu.sonyremote.model.PostviewSizePreference,
    onSetPostviewPreference: (com.ryu.sonyremote.model.PostviewSizePreference) -> Unit,
    outputImageFormat: com.ryu.sonyremote.model.OutputImageFormat,
    onSetOutputImageFormat: (com.ryu.sonyremote.model.OutputImageFormat) -> Unit,
    autoDenoiseMode: AutoDenoiseMode,
    autoDenoiseIsoThreshold: Int,
    autoDenoiseModel: AinrDenoiseModel,
    ainrProgress: AinrProgress?,
    onSetAutoDenoiseMode: (AutoDenoiseMode) -> Unit,
    onSetAutoDenoiseIsoThreshold: (Int) -> Unit,
    onSetAutoDenoiseModel: (AinrDenoiseModel) -> Unit,
    liveViewTimeoutMinutes: Int,
    onSetLiveViewTimeoutMinutes: (Int) -> Unit,
    onSetLiveviewSize: (String) -> Unit,
    onDismiss: () -> Unit,
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Settings") },
        text = {
            Column(
                modifier = Modifier.verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Text("Photo download", style = MaterialTheme.typography.titleSmall)
                SingleChoiceSegmentedButtonRow(modifier = Modifier.fillMaxWidth()) {
                    val choices = listOf(
                        com.ryu.sonyremote.model.PostviewSizePreference.Original to "Original",
                        com.ryu.sonyremote.model.PostviewSizePreference.FastPreview to "Reduced",
                    )
                    choices.forEachIndexed { index, (preference, label) ->
                        SegmentedButton(
                            selected = postviewPreference == preference,
                            onClick = { onSetPostviewPreference(preference) },
                            enabled = enabled,
                            shape = SegmentedButtonDefaults.itemShape(index, choices.size),
                            label = { Text(label) },
                        )
                    }
                }
                Text(
                    "Original uses the camera's full postview when available. Reduced uses 2M for faster transfer.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Text("Saved format", style = MaterialTheme.typography.titleSmall)
                SingleChoiceSegmentedButtonRow(modifier = Modifier.fillMaxWidth()) {
                    val formats = listOf(
                        com.ryu.sonyremote.model.OutputImageFormat.Jpeg to "JPEG",
                        com.ryu.sonyremote.model.OutputImageFormat.Webp to "WebP",
                    )
                    formats.forEachIndexed { index, (format, label) ->
                        SegmentedButton(
                            selected = outputImageFormat == format,
                            onClick = { onSetOutputImageFormat(format) },
                            enabled = enabled,
                            shape = SegmentedButtonDefaults.itemShape(index, formats.size),
                            label = { Text(label) },
                        )
                    }
                }
                Text("Automatic denoise", style = MaterialTheme.typography.titleSmall)
                SingleChoiceSegmentedButtonRow(modifier = Modifier.fillMaxWidth()) {
                    AutoDenoiseMode.entries.forEachIndexed { index, mode ->
                        SegmentedButton(
                            selected = autoDenoiseMode == mode,
                            onClick = { onSetAutoDenoiseMode(mode) },
                            enabled = enabled,
                            shape = SegmentedButtonDefaults.itemShape(index, AutoDenoiseMode.entries.size),
                            label = { Text(mode.label) },
                        )
                    }
                }
                if (autoDenoiseMode == AutoDenoiseMode.IsoThreshold) {
                    val isoChoices = listOf(400, 800, 1600, 3200, 6400, 12800, 25600)
                    val selectedIndex = isoChoices.indexOf(autoDenoiseIsoThreshold)
                        .takeIf { it >= 0 } ?: 2
                    Text(
                        "Run at ISO $autoDenoiseIsoThreshold and above",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    Slider(
                        value = selectedIndex.toFloat(),
                        onValueChange = { index ->
                            onSetAutoDenoiseIsoThreshold(isoChoices[index.roundToInt()])
                        },
                        valueRange = 0f..isoChoices.lastIndex.toFloat(),
                        steps = isoChoices.size - 2,
                        enabled = enabled,
                    )
                }
                if (autoDenoiseMode != AutoDenoiseMode.Off) {
                    Text("Denoise model", style = MaterialTheme.typography.titleSmall)
                    SingleChoiceSegmentedButtonRow(modifier = Modifier.fillMaxWidth()) {
                        AinrDenoiseModel.entries.forEachIndexed { index, model ->
                            SegmentedButton(
                                selected = autoDenoiseModel == model,
                                onClick = { onSetAutoDenoiseModel(model) },
                                enabled = enabled,
                                shape = SegmentedButtonDefaults.itemShape(
                                    index,
                                    AinrDenoiseModel.entries.size,
                                ),
                                label = { Text(model.label) },
                            )
                        }
                    }
                    ainrProgress?.let { progress ->
                        val fraction = progress.fraction
                        if (fraction != null) {
                            LinearProgressIndicator(
                                progress = { fraction },
                                modifier = Modifier.fillMaxWidth(),
                            )
                        } else {
                            LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                        }
                        Text(
                            "${progress.phase.name.replace('_', ' ').lowercase().replaceFirstChar(Char::uppercase)} · ${progress.detail}",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    Text(
                        "Distilled is faster for automatic processing. SCUNet prioritizes quality. The untouched original remains available in Edit.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                Text("Live view battery saver", style = MaterialTheme.typography.titleSmall)
                var timeoutMenuOpen by remember { mutableStateOf(false) }
                val timeoutChoices = listOf(
                    0 to "Never turn off",
                    1 to "After 1 minute",
                    3 to "After 3 minutes",
                    5 to "After 5 minutes",
                    10 to "After 10 minutes",
                    30 to "After 30 minutes",
                )
                Box(modifier = Modifier.fillMaxWidth()) {
                    OutlinedButton(
                        onClick = { timeoutMenuOpen = true },
                        enabled = enabled,
                        modifier = Modifier.fillMaxWidth(),
                    ) {
                        Text(timeoutChoices.firstOrNull { it.first == liveViewTimeoutMinutes }?.second ?: "Never turn off")
                        Spacer(Modifier.weight(1f))
                        Icon(Icons.Default.KeyboardArrowDown, contentDescription = null)
                    }
                    DropdownMenu(
                        expanded = timeoutMenuOpen,
                        onDismissRequest = { timeoutMenuOpen = false },
                    ) {
                        timeoutChoices.forEach { (minutes, label) ->
                            DropdownMenuItem(
                                text = { Text(label) },
                                onClick = {
                                    onSetLiveViewTimeoutMinutes(minutes)
                                    timeoutMenuOpen = false
                                },
                                leadingIcon = if (minutes == liveViewTimeoutMinutes) {
                                    { Icon(Icons.Default.Check, contentDescription = null) }
                                } else null,
                            )
                        }
                    }
                }
                Text(
                    "Stops live view without disconnecting the camera. Tap Start live view to resume.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Text("Live view quality", style = MaterialTheme.typography.titleSmall)
                if (capabilities.availableLiveviewSizes.isEmpty()) {
                    Text(
                        "Not available from this camera",
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                } else {
                    SingleChoiceSegmentedButtonRow(modifier = Modifier.fillMaxWidth()) {
                        capabilities.availableLiveviewSizes.forEachIndexed { index, size ->
                            SegmentedButton(
                                selected = capabilities.liveviewSize == size,
                                onClick = { onSetLiveviewSize(size) },
                                enabled = enabled && capabilities.canSetLiveviewSize,
                                shape = SegmentedButtonDefaults.itemShape(
                                    index,
                                    capabilities.availableLiveviewSizes.size,
                                ),
                                label = { Text(liveviewSizeLabel(size)) },
                            )
                        }
                    }
                }
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween,
                ) {
                    Column(Modifier.weight(1f)) {
                        Text("Geotag photos", style = MaterialTheme.typography.titleSmall)
                        Text(
                            "Add the phone's GPS coordinates to saved image EXIF",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    Switch(
                        checked = geotaggingEnabled,
                        onCheckedChange = onSetGeotagging,
                        enabled = enabled,
                    )
                }
            }
        },
        confirmButton = { TextButton(onClick = onDismiss) { Text("Done") } },
    )
}

private fun liveviewSizeLabel(size: String): String = when (size.uppercase()) {
    "L" -> "High"
    "M" -> "Standard"
    else -> size
}

@Composable
private fun SetupStep(number: String, title: String, detail: String) {
    Row(horizontalArrangement = Arrangement.spacedBy(14.dp), verticalAlignment = Alignment.Top) {
        Surface(
            shape = CircleShape,
            color = MaterialTheme.colorScheme.primaryContainer,
            modifier = Modifier.size(32.dp),
        ) {
            Box(contentAlignment = Alignment.Center) {
                Text(number, fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.onPrimaryContainer)
            }
        }
        Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(title, style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
            Text(detail, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

@Composable
private fun ProgressScreen(
    modifier: Modifier,
    title: String,
    detail: String,
    onCancel: () -> Unit,
) {
    Column(
        modifier = modifier.fillMaxSize().padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        CircularProgressIndicator()
        Spacer(Modifier.height(20.dp))
        Text(title, style = MaterialTheme.typography.titleLarge)
        Spacer(Modifier.height(6.dp))
        Text(detail, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Spacer(Modifier.height(24.dp))
        TextButton(onClick = onCancel) {
            Icon(Icons.Default.Close, contentDescription = null)
            Spacer(Modifier.width(6.dp))
            Text("Cancel")
        }
    }
}

@Composable
private fun CameraControlScreen(
    modifier: Modifier,
    state: ConnectionState.Ready,
    preview: Bitmap?,
    isStreaming: Boolean,
    isPhysicalShutterTransferActive: Boolean,
    isContinuousBurstActive: Boolean,
    isBusy: Boolean,
    captureMode: CaptureMode,
    liveNdFrames: Int,
    liveNdStrategy: StackCaptureStrategy,
    session: CaptureSessionUiState,
    filmstrip: List<FilmstripItem>,
    lutState: LutCaptureState,
    lutPreviews: List<LutPreviewItem>,
    ainrProgress: AinrProgress?,
    zoomTargetPercent: Int?,
    onSelectMode: (CaptureMode) -> Unit,
    onSetLiveNdFrames: (Int) -> Unit,
    onSetLiveNdStrategy: (StackCaptureStrategy) -> Unit,
    onPrimaryAction: () -> Unit,
    onStartContinuousBurst: () -> Unit,
    onStopContinuousBurst: () -> Unit,
    onFinishPanorama: () -> Unit,
    onCancelPanorama: () -> Unit,
    onOpenLut: (FilmstripItem) -> Unit,
    onImportOriginal: (String) -> Unit,
    onCancelOriginal: (String) -> Unit,
    onSelectLiveLut: (LutPreset) -> Unit,
    onSetLutIntensity: (Float) -> Unit,
    onSetBakeLut: (Boolean) -> Unit,
    onAddLiveLut: (LutPreset) -> Unit,
    onRemoveLiveLut: (LutPreset) -> Unit,
    onImportCubeLut: () -> Unit,
    onSelectImportedLut: (String) -> Unit,
    onRemoveImportedLut: (String) -> Unit,
    onOpenHistory: () -> Unit,
    onRetryLiveView: () -> Unit,
    onSetSetting: (com.ryu.sonyremote.model.CameraSettingId, String) -> Unit,
    onStartRemoteZoom: (String) -> Unit,
    onStopRemoteZoom: (String) -> Unit,
    onSetRemoteZoomTarget: (Int) -> Unit,
    onRefreshZoomState: () -> Unit,
) {
    val capabilities = state.capabilities
    val visiblePhotoSettings = photoSettingsAfterShutterControls(capabilities)
    val commandBlocked = isBusy || isPhysicalShutterTransferActive
    val controlsBlocked = commandBlocked || isContinuousBurstActive
    val routedPreviews = routeCapturePreviews(preview, session.preview, session.mode)
    val showProcessedPreview = routedPreviews.main === session.preview && session.preview != null
    val visiblePreview = routedPreviews.main
    val pipPreview = routedPreviews.pip
    val liveNdExposure = liveNdExposureLabel(
        capabilities.settings[com.ryu.sonyremote.model.CameraSettingId.ShutterSpeed]?.currentWireValue,
        liveNdFrames,
    )
    val activeLutLabel = if (lutState.isOriginal) {
        "Original"
    } else {
        val name = lutState.importedLut?.label ?: lutState.preset.label
        "$name ${(lutState.intensity * 100).roundToInt()}%"
    }
    val liveStatus = when {
        showProcessedPreview && session.mode == CaptureMode.LiveNd -> {
            "ND ${session.frameCount}/${session.targetFrames ?: liveNdFrames}" +
                (liveNdExposure?.let { " • $it" } ?: "")
        }
        showProcessedPreview && session.mode == CaptureMode.LiveComposite -> "COMPOSITE ${session.frameCount}"
        isContinuousBurstActive -> "BURST"
        captureMode == CaptureMode.LiveNd && liveNdExposure != null -> "ND $liveNdExposure"
        else -> "LIVE"
    }
    Column(
        modifier = modifier.fillMaxSize(),
    ) {
        LiveView(
            preview = visiblePreview,
            pipPreview = pipPreview,
            isStreaming = isStreaming || session.isActive,
            statusLabel = liveStatus,
            lutLabel = activeLutLabel,
            loadingText = if (session.isActive) "Capturing frame" else "Starting live view",
            isImporting = isPhysicalShutterTransferActive,
            processingProgress = session.processingProgress,
            onRetry = onRetryLiveView,
        )
        ainrProgress?.let { progress ->
            Column(
                modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 6.dp),
                verticalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                val fraction = progress.fraction
                if (fraction != null) {
                    LinearProgressIndicator(
                        progress = { fraction },
                        modifier = Modifier.fillMaxWidth(),
                    )
                } else {
                    LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                }
                Text(
                    "Denoising · ${progress.detail}",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
        CaptureModeSelector(
            selected = captureMode,
            enabled = !controlsBlocked && !session.isActive,
            onSelect = onSelectMode,
        )

        Column(
            modifier = Modifier.weight(1f).verticalScroll(rememberScrollState()),
        ) {
            ModeOptions(
                mode = captureMode,
                liveNdFrames = liveNdFrames,
                liveNdStrategy = liveNdStrategy,
                session = session,
                enabled = !controlsBlocked && !session.isActive,
                onSetLiveNdFrames = onSetLiveNdFrames,
                onSetLiveNdStrategy = onSetLiveNdStrategy,
            )

            if (visiblePhotoSettings.isNotEmpty() || captureMode == CaptureMode.LiveNd) {
                val gridItems = visiblePhotoSettings.take(5).map { setting ->
                    @Composable {
                        SettingMenu(
                            setting = setting,
                            enabled = !controlsBlocked && !session.isActive,
                            onSelect = { onSetSetting(setting.id, it) },
                            modifier = Modifier.fillMaxWidth(),
                        )
                    }
                } + if (captureMode == CaptureMode.LiveNd) {
                    listOf<@Composable () -> Unit>({
                        LiveNdFramesMenu(
                            frames = liveNdFrames,
                            enabled = !controlsBlocked && !session.isActive,
                            onSelect = onSetLiveNdFrames,
                            modifier = Modifier.fillMaxWidth(),
                        )
                    })
                } else {
                    emptyList()
                }
                Column(
                    modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 8.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    gridItems.take(6).chunked(3).forEach { rowItems ->
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                        ) {
                            rowItems.forEach { item -> Box(Modifier.weight(1f)) { item() } }
                            repeat(3 - rowItems.size) { Spacer(Modifier.weight(1f)) }
                        }
                    }
                }
            }

            if (capabilities.settings.size < 3) {
                StatusBand(
                    text = "Reduced controls reported by the camera. This is expected with Smart Remote Embedded.",
                    isError = false,
                    modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp),
                )
            }

        }
        HorizontalDivider()
        CaptureControls(
            capabilities = capabilities,
            mode = captureMode,
            liveNdStrategy = liveNdStrategy,
            session = session,
            isBusy = commandBlocked,
            isContinuousBurstActive = isContinuousBurstActive,
            onPrimaryAction = onPrimaryAction,
            onStartContinuousBurst = onStartContinuousBurst,
            onStopContinuousBurst = onStopContinuousBurst,
            onFinishPanorama = onFinishPanorama,
            onCancelPanorama = onCancelPanorama,
            onStartRemoteZoom = onStartRemoteZoom,
            onStopRemoteZoom = onStopRemoteZoom,
            onSetRemoteZoomTarget = onSetRemoteZoomTarget,
            onRefreshZoomState = onRefreshZoomState,
            zoomTargetPercent = zoomTargetPercent,
            onSetPrioritySetting = onSetSetting,
        )
        PhotoBottomArea(
            latest = filmstrip.lastOrNull(),
            state = lutState,
            previews = lutPreviews,
            enabled = !controlsBlocked,
            onOpenHistory = onOpenHistory,
            onSelect = onSelectLiveLut,
            onIntensity = onSetLutIntensity,
            onBake = onSetBakeLut,
            onAdd = onAddLiveLut,
            onRemove = onRemoveLiveLut,
            onImport = onImportCubeLut,
            onSelectImported = onSelectImportedLut,
            onRemoveImported = onRemoveImportedLut,
        )
    }
}

@Composable
private fun CaptureModeSelector(
    selected: CaptureMode,
    enabled: Boolean,
    onSelect: (CaptureMode) -> Unit,
) {
    val modes = CaptureMode.entries
    SingleChoiceSegmentedButtonRow(
        modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 10.dp),
    ) {
        modes.forEachIndexed { index, mode ->
            SegmentedButton(
                selected = selected == mode,
                onClick = { onSelect(mode) },
                enabled = enabled,
                shape = SegmentedButtonDefaults.itemShape(index, modes.size),
                icon = {},
                label = {
                    Text(
                        mode.label,
                        style = MaterialTheme.typography.labelSmall,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                },
            )
        }
    }
}

@Composable
private fun ModeOptions(
    mode: CaptureMode,
    liveNdFrames: Int,
    liveNdStrategy: StackCaptureStrategy,
    session: CaptureSessionUiState,
    enabled: Boolean,
    onSetLiveNdFrames: (Int) -> Unit,
    onSetLiveNdStrategy: (StackCaptureStrategy) -> Unit,
) {
    when (mode) {
        CaptureMode.Photo -> Unit
        CaptureMode.LiveNd -> {
            if (session.mode == mode && session.frameCount > 0) {
                Text(
                    "${session.frameCount} / ${session.targetFrames ?: liveNdFrames} frames",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp),
                )
            }
        }
        CaptureMode.LiveComposite -> {
            if (session.mode == mode && (session.isActive || session.frameCount > 0)) {
                Text(
                    "${session.frameCount} frames",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.padding(horizontal = 16.dp, vertical = 6.dp),
                )
            }
        }
        CaptureMode.Panorama -> {
            if (session.mode == mode && session.isActive) {
                Text(
                    "${session.frameCount} accepted frames",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.padding(horizontal = 16.dp, vertical = 6.dp),
                )
            }
        }
    }
}

@Composable
private fun LiveView(
    preview: Bitmap?,
    pipPreview: Bitmap?,
    isStreaming: Boolean,
    statusLabel: String,
    lutLabel: String,
    loadingText: String,
    isImporting: Boolean,
    processingProgress: Float?,
    onRetry: () -> Unit,
) {
    Box(
        modifier = Modifier.fillMaxWidth().aspectRatio(3f / 2f).background(Color(0xFF101211)),
        contentAlignment = Alignment.Center,
    ) {
        if (preview != null) {
            Image(
                bitmap = preview.asImageBitmap(),
                contentDescription = "Camera live view",
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Fit,
            )
        } else if (isStreaming) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                CircularProgressIndicator(color = Color.White, modifier = Modifier.size(28.dp))
                Spacer(Modifier.height(12.dp))
                Text(loadingText, color = Color.White)
            }
        } else {
            OutlinedButton(
                onClick = onRetry,
                colors = ButtonDefaults.outlinedButtonColors(contentColor = Color.White),
                border = androidx.compose.foundation.BorderStroke(1.dp, Color.White.copy(alpha = 0.7f)),
            ) {
                Text("Start live view")
            }
        }
        if (pipPreview != null) {
            Surface(
                color = Color.Black,
                shape = MaterialTheme.shapes.small,
                tonalElevation = 4.dp,
                modifier = Modifier
                    .align(Alignment.BottomEnd)
                    .padding(10.dp)
                    .width(118.dp)
                    .aspectRatio(3f / 2f)
                    .border(1.dp, Color.White.copy(alpha = 0.8f), MaterialTheme.shapes.small),
            ) {
                Image(
                    bitmap = pipPreview.asImageBitmap(),
                    contentDescription = "Camera live-view picture in picture",
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.Fit,
                )
            }
        }
        if (isStreaming) {
            Surface(
                color = Color.Black.copy(alpha = 0.68f),
                shape = MaterialTheme.shapes.small,
                modifier = Modifier.align(Alignment.TopStart).padding(10.dp),
            ) {
                Row(
                    modifier = Modifier.padding(horizontal = 9.dp, vertical = 6.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Box(Modifier.size(7.dp).background(Color(0xFF79D5BF), CircleShape))
                    Spacer(Modifier.width(7.dp))
                    Text(statusLabel, color = Color.White, style = MaterialTheme.typography.labelSmall)
                }
            }
        }
        Surface(
            color = Color.Black.copy(alpha = 0.68f),
            shape = MaterialTheme.shapes.small,
            modifier = Modifier.align(Alignment.TopEnd).padding(10.dp),
        ) {
            Text(
                "LUT: $lutLabel",
                color = Color.White,
                style = MaterialTheme.typography.labelSmall,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier.padding(horizontal = 9.dp, vertical = 6.dp),
            )
        }
        if (isImporting || processingProgress != null) {
            Surface(
                color = Color.Black.copy(alpha = 0.74f),
                shape = MaterialTheme.shapes.small,
                modifier = Modifier.align(Alignment.TopEnd).padding(10.dp),
            ) {
                Row(
                    modifier = Modifier.padding(horizontal = 9.dp, vertical = 6.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    CircularProgressIndicator(
                        color = Color.White,
                        modifier = Modifier.size(14.dp),
                        strokeWidth = 2.dp,
                    )
                    Spacer(Modifier.width(7.dp))
                    Text(
                        processingProgress?.let { "RENDER ${(it * 100).toInt()}%" } ?: "IMPORTING",
                        color = Color.White,
                        style = MaterialTheme.typography.labelSmall,
                    )
                }
            }
        }
    }
}

@Composable
private fun SettingMenu(
    setting: CameraSetting,
    enabled: Boolean,
    onSelect: (String) -> Unit,
    modifier: Modifier = Modifier,
) {
    var expanded by remember(setting.id) { mutableStateOf(false) }
    val hasAlternatives = setting.isWritable &&
        setting.options.any { it.wireValue != setting.currentWireValue }
    Column(modifier = modifier) {
        Text(
            setting.id.label,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
        )
        Spacer(Modifier.height(4.dp))
        Box {
            OutlinedButton(
                onClick = { expanded = true },
                enabled = enabled && hasAlternatives,
                modifier = Modifier.fillMaxWidth().height(44.dp),
                contentPadding = androidx.compose.foundation.layout.PaddingValues(horizontal = 10.dp),
            ) {
                Text(
                    setting.currentLabel,
                    modifier = Modifier.weight(1f),
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
                if (hasAlternatives) {
                    Icon(Icons.Default.KeyboardArrowDown, contentDescription = null, modifier = Modifier.size(18.dp))
                }
            }
            DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
                setting.options.forEach { option ->
                    DropdownMenuItem(
                        text = { Text(option.label) },
                        onClick = {
                            expanded = false
                            if (option.wireValue != setting.currentWireValue) onSelect(option.wireValue)
                        },
                    )
                }
            }
        }
    }
}

@Composable
private fun LiveNdFramesMenu(
    frames: Int,
    enabled: Boolean,
    onSelect: (Int) -> Unit,
    modifier: Modifier = Modifier,
) {
    var expanded by remember { mutableStateOf(false) }
    Column(modifier = modifier) {
        Text(
            "Frames",
            style = MaterialTheme.typography.labelMedium,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
        )
        Spacer(Modifier.height(4.dp))
        Box {
            OutlinedButton(
                onClick = { expanded = true },
                enabled = enabled,
                modifier = Modifier.fillMaxWidth().height(44.dp),
                contentPadding = PaddingValues(horizontal = 10.dp),
            ) {
                Text(frames.toString(), maxLines = 1, modifier = Modifier.weight(1f))
                Icon(Icons.Default.KeyboardArrowDown, contentDescription = null)
            }
            DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
                listOf(2, 4, 8, 16).forEach { count ->
                    DropdownMenuItem(
                        text = { Text(count.toString()) },
                        onClick = {
                            expanded = false
                            onSelect(count)
                        },
                    )
                }
            }
        }
    }
}

@Composable
private fun ExposurePriorityDragControl(
    capabilities: CameraCapabilities,
    enabled: Boolean,
    onOpen: () -> Unit,
) {
    val mode = capabilities.settings[com.ryu.sonyremote.model.CameraSettingId.ExposureMode]
    val shortMode = mode?.let { exposureModeShortLabel(it.currentLabel) } ?: "-"
    val priorityId = when (shortMode) {
        "A", "M" -> com.ryu.sonyremote.model.CameraSettingId.FNumber
        "S" -> com.ryu.sonyremote.model.CameraSettingId.ShutterSpeed
        else -> null
    }
    val priority = priorityId?.let(capabilities.settings::get)
    val selectable = enabled && priority?.isWritable == true && priority.options.size > 1
    Surface(
        shape = MaterialTheme.shapes.small,
        color = MaterialTheme.colorScheme.surfaceVariant,
        modifier = Modifier
            .size(width = 82.dp, height = 72.dp)
            .clickable(enabled = selectable, onClick = onOpen),
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
        ) {
            Text(shortMode, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            Text(
                priority?.currentLabel ?: mode?.currentLabel.orEmpty(),
                style = MaterialTheme.typography.labelSmall,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
        }
    }
}

@Composable
private fun InlineExposureBar(
    mode: String,
    setting: CameraSetting,
    onSetValue: (String) -> Unit,
    onClose: () -> Unit,
) {
    val currentIndex = setting.options.indexOfFirst { it.wireValue == setting.currentWireValue }
        .coerceAtLeast(0)
    var selectedIndex by remember(setting.currentWireValue) { mutableStateOf(currentIndex.toFloat()) }
    val selected = setting.options[selectedIndex.toInt().coerceIn(setting.options.indices)]
    Surface(color = MaterialTheme.colorScheme.surfaceVariant) {
        Row(
            modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(modifier = Modifier.weight(1f), horizontalAlignment = Alignment.CenterHorizontally) {
                Text("$mode ${setting.id.label}: ${selected.label}", style = MaterialTheme.typography.titleSmall)
                Slider(
                    value = selectedIndex,
                    onValueChange = { selectedIndex = it },
                    onValueChangeFinished = {
                        if (selected.wireValue != setting.currentWireValue) onSetValue(selected.wireValue)
                    },
                    valueRange = 0f..setting.options.lastIndex.toFloat(),
                    steps = (setting.options.size - 2).coerceAtLeast(0),
                )
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                    Text(setting.options.first().label, style = MaterialTheme.typography.labelSmall)
                    Text(setting.options.last().label, style = MaterialTheme.typography.labelSmall)
                }
            }
            IconButton(onClick = onClose) { Icon(Icons.Default.Close, contentDescription = "Close exposure bar") }
        }
    }
}

@Composable
private fun ZoomDragControl(
    available: Boolean,
    enabled: Boolean,
    position: Int?,
    zoomSetting: String?,
    zoomBoxCount: Int?,
    zoomBoxIndex: Int?,
    onOpen: () -> Unit,
) {
    Surface(
        shape = MaterialTheme.shapes.small,
        color = MaterialTheme.colorScheme.surfaceVariant,
        modifier = Modifier
            .size(width = 82.dp, height = 72.dp)
            .clickable(enabled = available && enabled, onClick = onOpen),
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
        ) {
            Text("Zoom", style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.SemiBold)
            Text(
                when {
                    !available -> "Unavailable"
                    position != null -> "$position%"
                    else -> "W  -  T"
                },
                style = MaterialTheme.typography.labelSmall,
                maxLines = 1,
            )
            if (available) {
                Text(
                    activeZoomTypeLabel(zoomSetting, zoomBoxCount, zoomBoxIndex),
                    style = MaterialTheme.typography.labelSmall,
                    maxLines = 1,
                )
            }
        }
    }
}

private fun zoomTypeLabel(setting: String?): String = when {
    setting == null -> "Type unknown"
    setting.contains("clear", ignoreCase = true) -> "Clear Image"
    setting.contains("digital", ignoreCase = true) -> "Digital"
    setting.contains("optical", ignoreCase = true) -> "Optical"
    else -> setting
}

private fun activeZoomTypeLabel(setting: String?, boxCount: Int?, boxIndex: Int?): String {
    val index = boxIndex ?: return zoomTypeLabel(setting)
    if (index <= 0) return "Optical"
    val digitalEnabled = setting?.contains("digital", ignoreCase = true) == true
    return if (digitalEnabled && boxCount != null && index >= boxCount - 1) "Digital" else "Clear Image"
}

@Composable
private fun InlineZoomBar(
    reportedPosition: Int?,
    zoomSetting: String?,
    zoomType: String,
    zoomBoxCount: Int?,
    zoomBoxIndex: Int?,
    activeTarget: Int?,
    onSetTarget: (Int) -> Unit,
    onClose: () -> Unit,
) {
    var target by remember { mutableStateOf((reportedPosition ?: 0).toFloat()) }
    val rangeLabels = zoomRangeLabels(zoomSetting, zoomBoxCount)
    LaunchedEffect(activeTarget, reportedPosition) {
        target = (activeTarget ?: reportedPosition ?: target.toInt()).toFloat()
    }
    Surface(color = MaterialTheme.colorScheme.surfaceVariant) {
        Row(
            modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(modifier = Modifier.weight(1f), horizontalAlignment = Alignment.CenterHorizontally) {
                Text(
                    if (reportedPosition != null) {
                        "Zoom ${reportedPosition.coerceIn(0, 100)}%  •  Target ${target.toInt()}%"
                    } else "Zoom position unavailable",
                    style = MaterialTheme.typography.titleMedium,
                )
                Text(
                    "$zoomType • ${zoomSetting ?: "range unknown"}",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Box(modifier = Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) {
                    Row(
                        modifier = Modifier.fillMaxWidth().padding(horizontal = 10.dp).height(10.dp),
                    ) {
                        rangeLabels.forEachIndexed { index, _ ->
                            Box(
                                Modifier.weight(1f).fillMaxSize().background(
                                    when (index) {
                                        0 -> MaterialTheme.colorScheme.primary
                                        1 -> MaterialTheme.colorScheme.tertiary
                                        else -> MaterialTheme.colorScheme.error
                                    },
                                ),
                            )
                        }
                    }
                    Slider(
                        value = target,
                        onValueChange = { target = it },
                        onValueChangeFinished = {
                            val snapped = snapZoomTarget(target.toInt(), rangeLabels.size)
                            target = snapped.toFloat()
                            onSetTarget(snapped)
                        },
                        enabled = reportedPosition != null && activeTarget == null,
                        valueRange = 0f..100f,
                        steps = 99,
                        colors = SliderDefaults.colors(
                            activeTrackColor = Color.Transparent,
                            inactiveTrackColor = Color.Transparent,
                        ),
                    )
                }
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                    Text("0% Wide")
                    Text("100% Tele")
                }
                Row(Modifier.fillMaxWidth().padding(horizontal = 10.dp, vertical = 2.dp)) {
                    rangeLabels.forEachIndexed { index, label ->
                        Box(Modifier.weight(1f), contentAlignment = Alignment.Center) {
                            Text(
                                label,
                                style = MaterialTheme.typography.labelSmall,
                                fontWeight = if (index == zoomBoxIndex) FontWeight.Bold else FontWeight.Normal,
                                maxLines = 1,
                            )
                        }
                    }
                }
            }
            IconButton(onClick = {
                onClose()
            }) { Icon(Icons.Default.Close, contentDescription = "Close zoom bar") }
        }
    }
}

internal fun snapZoomTarget(target: Int, boxCount: Int, threshold: Int = 3): Int {
    val clamped = target.coerceIn(0, 100)
    if (boxCount <= 1) return clamped
    val boundaries = (1 until boxCount).map { (100f * it / boxCount).toInt() }
    return boundaries.minByOrNull { kotlin.math.abs(it - clamped) }
        ?.takeIf { kotlin.math.abs(it - clamped) <= threshold }
        ?: clamped
}

private fun zoomRangeLabels(setting: String?, reportedCount: Int?): List<String> {
    val configured = when {
        setting?.contains("digital", ignoreCase = true) == true -> listOf("Optical", "Clear Image", "Digital")
        setting?.contains("clear", ignoreCase = true) == true -> listOf("Optical", "Clear Image")
        else -> listOf("Optical")
    }
    val count = reportedCount?.coerceIn(1, 3) ?: configured.size
    return configured.take(count).let { labels ->
        if (labels.size == count) labels else labels + List(count - labels.size) { "Zoom ${labels.size + it + 1}" }
    }
}

@Composable
private fun CaptureControls(
    capabilities: CameraCapabilities,
    mode: CaptureMode,
    liveNdStrategy: StackCaptureStrategy,
    session: CaptureSessionUiState,
    isBusy: Boolean,
    isContinuousBurstActive: Boolean,
    onPrimaryAction: () -> Unit,
    onStartContinuousBurst: () -> Unit,
    onStopContinuousBurst: () -> Unit,
    onFinishPanorama: () -> Unit,
    onCancelPanorama: () -> Unit,
    onStartRemoteZoom: (String) -> Unit,
    onStopRemoteZoom: (String) -> Unit,
    onSetRemoteZoomTarget: (Int) -> Unit,
    onRefreshZoomState: () -> Unit,
    zoomTargetPercent: Int?,
    onSetPrioritySetting: (com.ryu.sonyremote.model.CameraSettingId, String) -> Unit,
) {
    var inlineControl by remember { mutableStateOf<PhotoInlineControl?>(null) }
    val panoramaActive = mode == CaptureMode.Panorama && session.isActive
    val automaticActive = mode in setOf(CaptureMode.LiveNd, CaptureMode.LiveComposite) && session.isActive
    val drive = capabilities.settings[com.ryu.sonyremote.model.CameraSettingId.ContShootingMode]
        ?.currentWireValue
    val continuousPhoto = mode == CaptureMode.Photo &&
        !drive.equals("Single", ignoreCase = true) &&
        (capabilities.canStartContinuousShooting || isContinuousBurstActive)
    val canPrepareContinuousLiveNd = capabilities.settings[
        com.ryu.sonyremote.model.CameraSettingId.ContShootingMode
    ]?.let { setting ->
        setting.isWritable && setting.options.any {
            it.wireValue.equals("Continuous", ignoreCase = true)
        }
    } == true
    val primaryEnabled = when {
        continuousPhoto -> !isBusy
        automaticActive -> !session.isFinishing
        panoramaActive -> !isBusy &&
            capabilities.canTakePicture &&
            session.frameCount < (session.targetFrames ?: Int.MAX_VALUE)
        mode == CaptureMode.Panorama -> !isBusy
        mode == CaptureMode.LiveNd && liveNdStrategy == StackCaptureStrategy.ContinuousBurst ->
            !isBusy && (
                capabilities.canStartContinuousShooting ||
                    canPrepareContinuousLiveNd ||
                    capabilities.canTakePicture
                )
        else -> !isBusy && capabilities.canTakePicture
    }
    if (inlineControl == PhotoInlineControl.Exposure) {
        val exposure = capabilities.settings[com.ryu.sonyremote.model.CameraSettingId.ExposureMode]
        val shortMode = exposure?.let { exposureModeShortLabel(it.currentLabel) }.orEmpty()
        val priorityId = when (shortMode) {
            "A", "M" -> com.ryu.sonyremote.model.CameraSettingId.FNumber
            "S" -> com.ryu.sonyremote.model.CameraSettingId.ShutterSpeed
            else -> null
        }
        val priority = priorityId?.let(capabilities.settings::get)
        if (priority != null && priority.options.size > 1) {
            InlineExposureBar(
                mode = shortMode,
                setting = priority,
                onSetValue = { onSetPrioritySetting(priority.id, it) },
                onClose = { inlineControl = null },
            )
            return
        }
        inlineControl = null
    }
    if (inlineControl == PhotoInlineControl.Zoom) {
        InlineZoomBar(
            reportedPosition = capabilities.zoomPosition,
            zoomSetting = capabilities.zoomSetting,
            zoomType = activeZoomTypeLabel(
                capabilities.zoomSetting,
                capabilities.zoomBoxCount,
                capabilities.zoomBoxIndex,
            ),
            zoomBoxCount = capabilities.zoomBoxCount,
            zoomBoxIndex = capabilities.zoomBoxIndex,
            activeTarget = zoomTargetPercent,
            onSetTarget = onSetRemoteZoomTarget,
            onClose = { inlineControl = null },
        )
        return
    }
    Row(
        modifier = Modifier.fillMaxWidth().padding(horizontal = 24.dp, vertical = 20.dp),
        horizontalArrangement = Arrangement.SpaceEvenly,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        if (!panoramaActive) {
            ExposurePriorityDragControl(
                capabilities = capabilities,
                enabled = !isBusy && !session.isActive,
                onOpen = { inlineControl = PhotoInlineControl.Exposure },
            )
        } else if (panoramaActive) {
            IconButton(
                onClick = onCancelPanorama,
                enabled = !isBusy || session.isFinishing,
                modifier = Modifier.size(54.dp),
            ) {
                Icon(
                    Icons.Default.Close,
                    contentDescription = if (session.isFinishing) "Cancel panorama render" else "Discard panorama",
                )
            }
        } else {
            Spacer(Modifier.size(54.dp))
        }

        val shutterModifier = Modifier
            .size(76.dp)
            .border(3.dp, MaterialTheme.colorScheme.primary, CircleShape)
            .then(
                if (continuousPhoto) {
                    Modifier.pointerInput(primaryEnabled) {
                        awaitEachGesture {
                            awaitFirstDown(requireUnconsumed = false)
                            if (primaryEnabled) {
                                onStartContinuousBurst()
                                try {
                                    waitForUpOrCancellation()
                                } finally {
                                    onStopContinuousBurst()
                                }
                            }
                        }
                    }
                } else {
                    Modifier
                },
            )
        FilledIconButton(
            onClick = if (continuousPhoto) ({}) else onPrimaryAction,
            enabled = primaryEnabled,
            modifier = shutterModifier,
            shape = CircleShape,
            colors = IconButtonDefaults.filledIconButtonColors(
                containerColor = MaterialTheme.colorScheme.surface,
                contentColor = MaterialTheme.colorScheme.primary,
            ),
        ) {
            if ((isBusy && !automaticActive) || session.isFinishing) {
                CircularProgressIndicator(Modifier.size(28.dp), strokeWidth = 3.dp)
            } else {
                val icon = when {
                    isContinuousBurstActive -> Icons.Default.Stop
                    automaticActive -> Icons.Default.Stop
                    mode in setOf(CaptureMode.LiveNd, CaptureMode.LiveComposite) -> Icons.Default.PlayArrow
                    mode == CaptureMode.Panorama && !panoramaActive -> Icons.Default.PlayArrow
                    else -> Icons.Default.CameraAlt
                }
                val description = when {
                    isContinuousBurstActive -> "Release to stop continuous photos"
                    continuousPhoto -> "Hold for continuous photos"
                    automaticActive -> "Finish ${mode.label}"
                    mode == CaptureMode.Panorama && panoramaActive -> "Add panorama frame"
                    mode == CaptureMode.Panorama -> "Start panorama"
                    mode == CaptureMode.Photo -> "Take photo"
                    else -> "Start ${mode.label}"
                }
                Icon(icon, contentDescription = description, modifier = Modifier.size(32.dp))
            }
        }

        if (!panoramaActive) {
            ZoomDragControl(
                available = capabilities.canRemoteZoom,
                enabled = !isBusy && !session.isActive,
                position = capabilities.zoomPosition,
                zoomSetting = capabilities.zoomSetting,
                zoomBoxCount = capabilities.zoomBoxCount,
                zoomBoxIndex = capabilities.zoomBoxIndex,
                onOpen = {
                    inlineControl = PhotoInlineControl.Zoom
                    onRefreshZoomState()
                },
            )
        } else if (panoramaActive) {
            IconButton(
                onClick = onFinishPanorama,
                enabled = session.canFinishPanorama && !isBusy,
                modifier = Modifier.size(54.dp),
            ) {
                Icon(Icons.Default.Check, contentDescription = "Finish panorama")
            }
        } else {
            Spacer(Modifier.size(54.dp))
        }
    }
}

private enum class PhotoInlineControl { Exposure, Zoom }

@Composable
private fun PhotoBottomArea(
    latest: FilmstripItem?,
    state: LutCaptureState,
    previews: List<LutPreviewItem>,
    enabled: Boolean,
    onOpenHistory: () -> Unit,
    onSelect: (LutPreset) -> Unit,
    onIntensity: (Float) -> Unit,
    onBake: (Boolean) -> Unit,
    onAdd: (LutPreset) -> Unit,
    onRemove: (LutPreset) -> Unit,
    onImport: () -> Unit,
    onSelectImported: (String) -> Unit,
    onRemoveImported: (String) -> Unit,
) {
    Row(
        modifier = Modifier.fillMaxWidth().background(MaterialTheme.colorScheme.surfaceContainer),
        verticalAlignment = Alignment.Top,
    ) {
        Surface(
            shape = MaterialTheme.shapes.small,
            color = MaterialTheme.colorScheme.surfaceVariant,
            modifier = Modifier
                .padding(start = 10.dp, top = 4.dp)
                .width(78.dp)
                .height(62.dp)
                .clickable(enabled = latest != null, onClick = onOpenHistory),
        ) {
            Box(contentAlignment = Alignment.Center) {
                if (latest != null) {
                    Image(
                        bitmap = latest.thumbnail.asImageBitmap(),
                        contentDescription = "Open capture history",
                        modifier = Modifier.fillMaxSize(),
                        contentScale = ContentScale.Crop,
                    )
                    Surface(
                        color = Color.Black.copy(alpha = 0.7f),
                        modifier = Modifier.align(Alignment.BottomCenter).fillMaxWidth(),
                    ) {
                        Text(
                            latest.importedCapture?.qualityLabel ?: latest.title,
                            color = Color.White,
                            style = MaterialTheme.typography.labelSmall,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                            modifier = Modifier.padding(horizontal = 4.dp, vertical = 2.dp),
                        )
                    }
                    if (latest.source is CaptureAssetSource.LiveViewPlaceholder) {
                        DownloadProgressOverlay(latest.importedCapture?.downloadFraction)
                    } else if (latest.importedCapture?.originalImportState in setOf(
                            OriginalImportState.Queued,
                            OriginalImportState.Downloading,
                        )
                    ) {
                        CircularProgressIndicator(Modifier.size(20.dp), strokeWidth = 2.dp)
                    }
                } else {
                    Icon(Icons.Default.CameraAlt, contentDescription = "Capture history is empty")
                }
            }
        }
        Box(modifier = Modifier.weight(1f)) {
            LutCaptureControls(
                state, previews, enabled, onSelect, onIntensity, onBake, onAdd, onRemove,
                onImport, onSelectImported, onRemoveImported,
            )
        }
    }
}

@Composable
private fun LutCaptureControls(
    state: LutCaptureState,
    previews: List<LutPreviewItem>,
    enabled: Boolean,
    onSelect: (LutPreset) -> Unit,
    onIntensity: (Float) -> Unit,
    onBake: (Boolean) -> Unit,
    onAdd: (LutPreset) -> Unit,
    onRemove: (LutPreset) -> Unit,
    onImport: () -> Unit,
    onSelectImported: (String) -> Unit,
    onRemoveImported: (String) -> Unit,
) {
    var editorOpen by remember { mutableStateOf(false) }
    Row(
        modifier = Modifier.fillMaxWidth().background(MaterialTheme.colorScheme.surfaceContainer),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        LazyRow(
            modifier = Modifier.weight(1f),
            contentPadding = androidx.compose.foundation.layout.PaddingValues(horizontal = 10.dp),
            horizontalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            items(previews.filter { it.preset in state.visiblePresets }, key = { it.preset.name }) { preview ->
                Surface(
                    shape = MaterialTheme.shapes.small,
                    color = MaterialTheme.colorScheme.surfaceVariant,
                    modifier = Modifier
                        .width(76.dp)
                        .height(62.dp)
                        .border(
                            width = if (!state.importedSelected && state.preset == preview.preset) 2.dp else 0.dp,
                            color = if (!state.importedSelected && state.preset == preview.preset) {
                                MaterialTheme.colorScheme.primary
                            } else {
                                Color.Transparent
                            },
                            shape = MaterialTheme.shapes.small,
                        )
                        .clickable(enabled = enabled) { onSelect(preview.preset) },
                ) {
                    Box(contentAlignment = Alignment.Center) {
                        preview.thumbnail?.let { bitmap ->
                            Image(
                                bitmap = bitmap.asImageBitmap(),
                                contentDescription = "${preview.preset.label} LUT preview",
                                modifier = Modifier.fillMaxSize(),
                                contentScale = ContentScale.Crop,
                            )
                        }
                        when {
                            preview.isLoading -> CircularProgressIndicator(Modifier.size(18.dp), strokeWidth = 2.dp)
                            preview.failed -> Text("Unavailable", style = MaterialTheme.typography.labelSmall)
                        }
                        Surface(
                            color = Color.Black.copy(alpha = 0.68f),
                            modifier = Modifier.align(Alignment.BottomCenter).fillMaxWidth(),
                        ) {
                            Text(
                                preview.preset.label,
                                color = Color.White,
                                style = MaterialTheme.typography.labelSmall,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis,
                                modifier = Modifier.padding(horizontal = 4.dp, vertical = 2.dp),
                            )
                        }
                    }
                }
            }
            items(state.importedLuts, key = { "imported:${it.label}" }) { imported ->
                    Surface(
                        shape = MaterialTheme.shapes.small,
                        modifier = Modifier.width(76.dp).height(62.dp)
                            .border(
                                if (state.selectedImportedLabel == imported.label) 2.dp else 0.dp,
                                MaterialTheme.colorScheme.primary,
                                MaterialTheme.shapes.small,
                            )
                            .clickable(enabled = enabled) { onSelectImported(imported.label) },
                    ) {
                        Box(contentAlignment = Alignment.Center) {
                            imported.thumbnail?.let {
                                Image(it.asImageBitmap(), null, Modifier.fillMaxSize(), contentScale = ContentScale.Crop)
                            }
                            Text(
                                imported.label,
                                color = Color.White,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis,
                                modifier = Modifier.align(Alignment.BottomCenter).fillMaxWidth()
                                    .background(Color.Black.copy(alpha = 0.68f)).padding(4.dp),
                            )
                        }
                    }
            }
        }
        IconButton(
            onClick = { editorOpen = true },
            enabled = enabled,
            modifier = Modifier.size(56.dp),
        ) {
            Icon(Icons.Default.Edit, contentDescription = "Edit LUTs")
        }
    }
    if (editorOpen) {
        LutSettingsDialog(state, onSelect, onIntensity, onBake, onAdd, onRemove, onImport, onSelectImported, onRemoveImported) {
            editorOpen = false
        }
    }
}

@Composable
private fun LutSettingsDialog(
    state: LutCaptureState,
    onSelect: (LutPreset) -> Unit,
    onIntensity: (Float) -> Unit,
    onBake: (Boolean) -> Unit,
    onAdd: (LutPreset) -> Unit,
    onRemove: (LutPreset) -> Unit,
    onImport: () -> Unit,
    onSelectImported: (String) -> Unit,
    onRemoveImported: (String) -> Unit,
    onDismiss: () -> Unit,
) {
    var addMenuOpen by remember { mutableStateOf(false) }
    val available = LutPreset.entries.filterNot { it in state.visiblePresets }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Edit LUTs") },
        text = {
            Column {
                state.importedLuts.forEach { imported ->
                    val selected = state.selectedImportedLabel == imported.label
                    val intensity = state.importedIntensities[imported.label] ?: 1f
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        FilterChip(selected, { onSelectImported(imported.label) }, { Text(imported.label) })
                        Slider(
                            value = intensity,
                            onValueChange = { if (!selected) onSelectImported(imported.label); onIntensity(it) },
                            modifier = Modifier.weight(1f).padding(horizontal = 6.dp),
                        )
                        Text("${(intensity * 100).toInt()}%")
                        IconButton(onClick = { onRemoveImported(imported.label) }) {
                            Icon(Icons.Default.Delete, contentDescription = "Remove ${imported.label}")
                        }
                    }
                }
                state.visiblePresets.filterNot { it == LutPreset.Neutral }.forEach { preset ->
                    val selected = !state.importedSelected && state.preset == preset
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        FilterChip(
                            selected = selected,
                            onClick = { onSelect(preset) },
                            label = { Text(preset.label) },
                        )
                        Slider(
                            value = state.presetIntensities[preset] ?: 1f,
                            onValueChange = {
                                if (!selected) onSelect(preset)
                                onIntensity(it)
                            },
                            modifier = Modifier.weight(1f).padding(horizontal = 6.dp),
                        )
                        Text("${((state.presetIntensities[preset] ?: 1f) * 100).toInt()}%")
                        IconButton(onClick = { onRemove(preset) }) {
                            Icon(Icons.Default.Delete, contentDescription = "Remove ${preset.label}")
                        }
                    }
                }
                Box {
                    OutlinedButton(onClick = onImport) {
                        Icon(Icons.Default.Add, contentDescription = null)
                        Spacer(Modifier.width(6.dp))
                        Text("Add LUT")
                    }
                    DropdownMenu(expanded = addMenuOpen, onDismissRequest = { addMenuOpen = false }) {
                        available.forEach { preset ->
                            DropdownMenuItem(
                                text = { Text(preset.label) },
                                onClick = {
                                    onAdd(preset)
                                    onSelect(preset)
                                    addMenuOpen = false
                                },
                            )
                        }
                    }
                }
            }
        },
        confirmButton = { TextButton(onClick = onDismiss) { Text("Done") } },
    )
}

@Composable
private fun Filmstrip(
    items: List<FilmstripItem>,
    enabled: Boolean,
    onOpenLut: (FilmstripItem) -> Unit,
    onImportOriginal: (String) -> Unit,
    onCancelOriginal: (String) -> Unit,
) {
    val listState = rememberLazyListState()
    LaunchedEffect(items.size) {
        if (items.isNotEmpty()) listState.animateScrollToItem(items.lastIndex)
    }
    HorizontalDivider()
    Column(
        modifier = Modifier.fillMaxWidth().height(108.dp).background(MaterialTheme.colorScheme.surfaceContainer),
    ) {
        Text(
            "Filmstrip",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(start = 12.dp, top = 6.dp, bottom = 4.dp),
        )
        LazyRow(
            state = listState,
            contentPadding = androidx.compose.foundation.layout.PaddingValues(horizontal = 10.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            items(items, key = { it.id }) { item ->
                Surface(
                    shape = MaterialTheme.shapes.small,
                    color = MaterialTheme.colorScheme.surfaceVariant,
                    modifier = Modifier
                        .width(78.dp)
                        .height(76.dp)
                        .clickable(
                            enabled = enabled && item.source !is CaptureAssetSource.LiveViewPlaceholder,
                        ) { onOpenLut(item) },
                ) {
                    Box {
                        Image(
                            bitmap = item.thumbnail.asImageBitmap(),
                            contentDescription = "${item.title} image",
                            modifier = Modifier.fillMaxSize(),
                            contentScale = ContentScale.Crop,
                        )
                        val imported = item.importedCapture
                        if (item.source is CaptureAssetSource.LiveViewPlaceholder) {
                            DownloadProgressOverlay(imported?.downloadFraction)
                        }
                        when (imported?.originalImportState) {
                            OriginalImportState.Available -> IconButton(
                                onClick = { onImportOriginal(item.id) },
                                enabled = enabled,
                                modifier = Modifier.align(Alignment.TopEnd).size(30.dp),
                            ) {
                                Icon(
                                    Icons.Default.FileDownload,
                                    contentDescription = "Import original",
                                    modifier = Modifier.size(18.dp),
                                )
                            }
                            OriginalImportState.Failed -> IconButton(
                                onClick = { onImportOriginal(item.id) },
                                enabled = enabled,
                                modifier = Modifier.align(Alignment.TopEnd).size(30.dp),
                            ) {
                                Icon(
                                    Icons.Default.Refresh,
                                    contentDescription = "Retry original import",
                                    modifier = Modifier.size(18.dp),
                                )
                            }
                            OriginalImportState.Queued,
                            OriginalImportState.Downloading,
                            -> IconButton(
                                onClick = { onCancelOriginal(item.id) },
                                modifier = Modifier.align(Alignment.TopEnd).size(30.dp),
                            ) {
                                if (imported.originalImportState == OriginalImportState.Downloading) {
                                    CircularProgressIndicator(Modifier.size(17.dp), strokeWidth = 2.dp)
                                } else {
                                    Icon(
                                        Icons.Default.Close,
                                        contentDescription = "Cancel original import",
                                        modifier = Modifier.size(18.dp),
                                    )
                                }
                            }
                            OriginalImportState.Imported,
                            OriginalImportState.NotAvailable,
                            null,
                            -> Unit
                        }
                        Surface(
                            color = Color.Black.copy(alpha = 0.72f),
                            modifier = Modifier.align(Alignment.BottomCenter).fillMaxWidth(),
                        ) {
                            Row(
                                modifier = Modifier.padding(horizontal = 5.dp, vertical = 3.dp),
                                verticalAlignment = Alignment.CenterVertically,
                            ) {
                                Text(
                                    imported?.qualityLabel ?: item.title,
                                    color = Color.White,
                                    style = MaterialTheme.typography.labelSmall,
                                    maxLines = 1,
                                    overflow = TextOverflow.Ellipsis,
                                    modifier = Modifier.weight(1f),
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun DownloadProgressOverlay(fraction: Float?) {
    Surface(
        color = Color.Black.copy(alpha = 0.72f),
        shape = CircleShape,
    ) {
        Box(Modifier.size(42.dp), contentAlignment = Alignment.Center) {
            if (fraction == null) {
                CircularProgressIndicator(Modifier.size(34.dp), strokeWidth = 3.dp)
            } else {
                CircularProgressIndicator(
                    progress = { fraction },
                    modifier = Modifier.size(34.dp),
                    strokeWidth = 3.dp,
                )
                Text("${(fraction * 100).toInt()}%", style = MaterialTheme.typography.labelSmall)
            }
        }
    }
}

@Composable
private fun CaptureHistoryDialog(
    items: List<FilmstripItem>,
    enabled: Boolean,
    onOpenLut: (FilmstripItem) -> Unit,
    onLoadDetail: suspend (FilmstripItem) -> Bitmap,
    onImportOriginal: (String) -> Unit,
    onCancelOriginal: (String) -> Unit,
    onDismiss: () -> Unit,
) {
    var selectedId by remember { mutableStateOf<String?>(null) }
    var sourceIds by remember { mutableStateOf<Set<String>>(emptySet()) }
    val selected = items.firstOrNull { it.id == selectedId }
    val visibleItems = if (sourceIds.isEmpty()) {
        items.filterNot { it.kind == CaptureAssetKind.SourceFrame }
    } else {
        items.filter { it.id in sourceIds }
    }
    val detailIds = remember(selectedId, sourceIds) {
        visibleItems.asReversed().map(FilmstripItem::id)
    }
    val detailItems = detailIds.mapNotNull { id -> items.firstOrNull { it.id == id } }
    Dialog(
        onDismissRequest = { if (selected != null) selectedId = null else onDismiss() },
        properties = DialogProperties(usePlatformDefaultWidth = false),
    ) {
        Surface(
            color = if (selected != null) Color.Transparent else Color.Black,
            modifier = Modifier.fillMaxSize(),
        ) {
            if (selected != null) {
                GalleryDetailViewer(
                    items = detailItems,
                    initialSelectedId = selected.id,
                    onLoadDetail = onLoadDetail,
                    onEdit = { item ->
                        selectedId = null
                        onOpenLut(item)
                    },
                    onShowSourceFrames = { ids ->
                        sourceIds = ids.toSet()
                        selectedId = null
                    },
                    onDismiss = { selectedId = null },
                )
            } else {
                Column(Modifier.fillMaxSize()) {
                    Row(
                        modifier = Modifier.fillMaxWidth().padding(horizontal = 8.dp, vertical = 6.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Text(
                            if (sourceIds.isEmpty()) "Gallery" else "Source frames",
                            color = Color.White,
                            style = MaterialTheme.typography.titleLarge,
                            modifier = Modifier.weight(1f),
                        )
                        if (sourceIds.isNotEmpty()) {
                            TextButton(onClick = { sourceIds = emptySet() }) { Text("All photos") }
                        }
                        IconButton(onClick = onDismiss) {
                            Icon(Icons.Default.Close, "Close gallery", tint = Color.White)
                        }
                    }
                    if (items.isEmpty()) {
                        Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                            Text("No captures", color = Color.LightGray)
                        }
                    } else {
                        LazyVerticalGrid(
                            columns = GridCells.Fixed(3),
                            modifier = Modifier.fillMaxSize(),
                            horizontalArrangement = Arrangement.spacedBy(2.dp),
                            verticalArrangement = Arrangement.spacedBy(2.dp),
                        ) {
                            gridItems(visibleItems.reversed(), key = { it.id }) { item ->
                                Box(
                                    Modifier.aspectRatio(1f).clickable { selectedId = item.id },
                                ) {
                                    Image(
                                        item.thumbnail.asImageBitmap(),
                                        item.title,
                                        Modifier.fillMaxSize(),
                                        contentScale = ContentScale.Crop,
                                    )
                                    if (item.source is CaptureAssetSource.LiveViewPlaceholder) {
                                        Box(Modifier.align(Alignment.Center)) {
                                            DownloadProgressOverlay(item.importedCapture?.downloadFraction)
                                        }
                                    }
                                    Text(
                                        item.importedCapture?.qualityLabel ?: item.title,
                                        color = Color.White,
                                        style = MaterialTheme.typography.labelSmall,
                                        maxLines = 1,
                                        modifier = Modifier.align(Alignment.BottomStart).fillMaxWidth()
                                            .background(Color.Black.copy(alpha = 0.6f)).padding(4.dp),
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun GalleryDetailViewer(
    items: List<FilmstripItem>,
    initialSelectedId: String,
    onLoadDetail: suspend (FilmstripItem) -> Bitmap,
    onEdit: (FilmstripItem) -> Unit,
    onShowSourceFrames: (List<String>) -> Unit,
    onDismiss: () -> Unit,
) {
    if (items.isEmpty()) {
        LaunchedEffect(Unit) { onDismiss() }
        return
    }
    val initialPage = remember(items, initialSelectedId) {
        items.indexOfFirst { it.id == initialSelectedId }.coerceAtLeast(0)
    }
    val pagerState = rememberPagerState(initialPage = initialPage) { items.size }
    val coroutineScope = rememberCoroutineScope()
    val density = LocalDensity.current
    var detailBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var detailBitmapId by remember { mutableStateOf<String?>(null) }
    var scale by remember { mutableFloatStateOf(1f) }
    var imageOffset by remember { mutableStateOf(Offset.Zero) }
    var dismissOffset by remember { mutableFloatStateOf(0f) }
    var viewportSize by remember { mutableStateOf(IntSize.Zero) }
    var controlsVisible by remember { mutableStateOf(true) }
    val settledItem = items.getOrNull(pagerState.settledPage) ?: items.first()
    val activeBitmap = detailBitmap.takeIf { detailBitmapId == settledItem.id }
        ?: settledItem.thumbnail

    LaunchedEffect(pagerState) {
        snapshotFlow { pagerState.settledPage }
            .distinctUntilChanged()
            .collect {
                scale = 1f
                imageOffset = Offset.Zero
                dismissOffset = 0f
            }
    }
    LaunchedEffect(settledItem.id, settledItem.source) {
        detailBitmap?.takeUnless(Bitmap::isRecycled)?.recycle()
        detailBitmap = null
        detailBitmapId = null
        if (settledItem.source is CaptureAssetSource.LiveViewPlaceholder) return@LaunchedEffect
        var loaded: Bitmap? = null
        try {
            loaded = runCatching { onLoadDetail(settledItem) }.getOrNull()
            detailBitmap = loaded
            detailBitmapId = loaded?.let { settledItem.id }
        } finally {
            if (loaded != null && detailBitmap !== loaded) loaded.recycle()
        }
    }
    DisposableEffect(Unit) {
        onDispose { detailBitmap?.takeUnless(Bitmap::isRecycled)?.recycle() }
    }

    val transformState = rememberTransformableState { centroid, zoomChange, panChange, _ ->
        val nextScale = clampGalleryScale(scale * zoomChange)
        val appliedZoom = nextScale / scale
        val focalPoint = centroid - Offset(
            viewportSize.width / 2f,
            viewportSize.height / 2f,
        )
        val transformedOffset = (imageOffset - focalPoint) * appliedZoom +
            focalPoint + panChange
        val clamped = clampGalleryOffset(
            x = transformedOffset.x,
            y = transformedOffset.y,
            scale = nextScale,
            viewportWidth = viewportSize.width,
            viewportHeight = viewportSize.height,
            imageWidth = activeBitmap.width,
            imageHeight = activeBitmap.height,
        )
        scale = nextScale
        imageOffset = Offset(clamped.x, clamped.y)
    }
    val dismissVelocity = with(density) { 1_200.dp.toPx() }
    val backdropAlpha = if (viewportSize.height == 0) {
        1f
    } else {
        (1f - dismissOffset / viewportSize.height).coerceIn(0.2f, 1f)
    }

    Surface(
        color = Color.Black.copy(alpha = backdropAlpha),
        modifier = Modifier.fillMaxSize().onSizeChanged { viewportSize = it },
    ) {
        Box(
            Modifier.fillMaxSize().draggable(
                state = rememberDraggableState { delta ->
                    dismissOffset = (dismissOffset + delta).coerceAtLeast(0f)
                },
                orientation = Orientation.Vertical,
                enabled = scale <= 1.01f,
                onDragStopped = { velocity ->
                    if (
                        shouldDismissGalleryDetail(
                            dismissOffset,
                            viewportSize.height,
                            velocity,
                            dismissVelocity,
                        )
                    ) {
                        onDismiss()
                    } else {
                        val start = dismissOffset
                        coroutineScope.launch {
                            animate(start, 0f) { value, _ -> dismissOffset = value }
                        }
                    }
                },
            ),
        ) {
            HorizontalPager(
                state = pagerState,
                key = { items[it].id },
                userScrollEnabled = scale <= 1.01f && dismissOffset == 0f,
                modifier = Modifier.fillMaxSize().graphicsLayer {
                    translationY = dismissOffset
                    val dismissScale = 1f - (dismissOffset / viewportSize.height.coerceAtLeast(1)) * 0.05f
                    scaleX = dismissScale.coerceAtLeast(0.95f)
                    scaleY = dismissScale.coerceAtLeast(0.95f)
                },
            ) { page ->
                val item = items[page]
                val bitmap = detailBitmap.takeIf {
                    detailBitmapId == item.id && page == pagerState.settledPage
                } ?: item.thumbnail
                Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    Image(
                        bitmap = bitmap.asImageBitmap(),
                        contentDescription = "${item.title} detail",
                        modifier = Modifier.fillMaxSize()
                            .graphicsLayer {
                                if (page == pagerState.settledPage) {
                                    scaleX = scale
                                    scaleY = scale
                                    translationX = imageOffset.x
                                    translationY = imageOffset.y
                                }
                            }
                            .transformable(
                                state = transformState,
                                enabled = page == pagerState.settledPage,
                                canPan = { scale > 1.01f },
                            )
                            .pointerInput(item.id, scale) {
                                detectTapGestures(
                                    onTap = { controlsVisible = !controlsVisible },
                                    onDoubleTap = {
                                        scale = if (scale > 1.01f) 1f else 2.5f
                                        imageOffset = Offset.Zero
                                    },
                                )
                            },
                        contentScale = ContentScale.Fit,
                    )
                    if (
                        page == pagerState.settledPage &&
                        detailBitmapId != item.id &&
                        item.source !is CaptureAssetSource.LiveViewPlaceholder
                    ) {
                        CircularProgressIndicator(color = Color.White)
                    }
                    if (item.source is CaptureAssetSource.LiveViewPlaceholder) {
                        DownloadProgressOverlay(item.importedCapture?.downloadFraction)
                    }
                }
            }
            if (controlsVisible) {
                Row(
                    modifier = Modifier.fillMaxWidth()
                        .background(Color.Black.copy(alpha = 0.68f))
                        .padding(start = 12.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Column(Modifier.weight(1f)) {
                        Text(
                            settledItem.importedCapture?.qualityLabel ?: settledItem.title,
                            color = Color.White,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                        )
                        Text(
                            "${pagerState.settledPage + 1} / ${items.size}",
                            color = Color.LightGray,
                            style = MaterialTheme.typography.labelSmall,
                        )
                    }
                    if (settledItem.source !is CaptureAssetSource.LiveViewPlaceholder) {
                        IconButton(onClick = { onEdit(settledItem) }) {
                            Icon(Icons.Default.Palette, "Edit photo", tint = Color.White)
                        }
                    }
                    if (settledItem.relatedSourceIds.isNotEmpty()) {
                        TextButton(onClick = {
                            onShowSourceFrames(settledItem.relatedSourceIds)
                        }) {
                            Text("Source frames", color = Color.White)
                        }
                    }
                    IconButton(onClick = onDismiss) {
                        Icon(Icons.Default.Close, "Back to gallery", tint = Color.White)
                    }
                }
            }
        }
    }
}

@Composable
private fun LutEditorDialog(
    state: LutEditorUiState,
    onSelectPreset: (LutPreset) -> Unit,
    onSelectImported: (String) -> Unit,
    onRequestLutThumbnails: () -> Unit,
    onSetIntensity: (Float) -> Unit,
    onSetBasicEdits: (Float, Float, Float) -> Unit,
    onSetDenoiseEnabled: (Boolean) -> Unit,
    onSetDenoiseStrength: (Float) -> Unit,
    onSetDenoiseModel: (AinrDenoiseModel) -> Unit,
    onSetSource: (EditorImageSource) -> Unit,
    onSetCrop: (NormalizedCrop) -> Unit,
    onRotateQuarterTurn: (Boolean) -> Unit,
    onSetStraighten: (Float) -> Unit,
    onSetPerspective: (List<NormalizedPoint>) -> Unit,
    onResetGeometry: (EditorTool) -> Unit,
    onAutoGeometry: (Boolean) -> Unit,
    onUndo: () -> Unit,
    onRedo: () -> Unit,
    onBeginGesture: () -> Unit,
    onEndGesture: () -> Unit,
    onSave: () -> Unit,
    onDismiss: () -> Unit,
    ainrProgress: AinrProgress?,
) {
    var confirmCancel by remember(state.item.id) { mutableStateOf(false) }
    var controlsExpanded by remember(state.item.id) { mutableStateOf(true) }
    var activeTool by remember(state.item.id) { mutableStateOf(EditorTool.Adjust) }
    var pendingCrop by remember(state.item.id, state.source) {
        mutableStateOf(state.geometry.crop)
    }
    var cropRatioLabel by remember(state.item.id, state.source) {
        mutableStateOf("Free")
    }
    var panelDrag by remember(state.item.id) { mutableFloatStateOf(0f) }
    var compareMode by remember(state.item.id) { mutableStateOf(false) }
    var comparisonFraction by remember(state.item.id) { mutableFloatStateOf(0.5f) }
    var scale by remember(state.item.id) { mutableFloatStateOf(1f) }
    var imageOffset by remember(state.item.id) { mutableStateOf(Offset.Zero) }
    var previewSize by remember(state.item.id) { mutableStateOf(IntSize.Zero) }
    val requestDismiss = {
        if (state.isProcessing || ainrProgress != null) confirmCancel = true else onDismiss()
    }
    val canSave = state.hasEdits && !state.isProcessing
    val activeBitmap = if (
        activeTool == EditorTool.Crop && !state.geometry.crop.isFullFrame
    ) {
        state.item.thumbnail
    } else {
        state.preview ?: state.originalPreview ?: state.item.thumbnail
    }
    val transformState = rememberTransformableState { centroid, zoomChange, panChange, _ ->
        val nextScale = clampGalleryScale(scale * zoomChange)
        val appliedZoom = nextScale / scale
        val focalPoint = centroid - Offset(
            previewSize.width / 2f,
            previewSize.height / 2f,
        )
        val transformedOffset = (imageOffset - focalPoint) * appliedZoom +
            focalPoint + panChange
        val clamped = clampGalleryOffset(
            x = transformedOffset.x,
            y = transformedOffset.y,
            scale = nextScale,
            viewportWidth = previewSize.width,
            viewportHeight = previewSize.height,
            imageWidth = activeBitmap.width,
            imageHeight = activeBitmap.height,
        )
        scale = nextScale
        imageOffset = Offset(clamped.x, clamped.y)
    }
    LaunchedEffect(compareMode) {
        scale = 1f
        imageOffset = Offset.Zero
        comparisonFraction = 0.5f
    }
    LaunchedEffect(state.geometry.crop, state.source, activeTool) {
        if (activeTool != EditorTool.Crop) {
            pendingCrop = state.geometry.crop
            cropRatioLabel = "Free"
        }
    }
    LaunchedEffect(previewSize, activeBitmap.width, activeBitmap.height, scale) {
        val clamped = clampGalleryOffset(
            x = imageOffset.x,
            y = imageOffset.y,
            scale = scale,
            viewportWidth = previewSize.width,
            viewportHeight = previewSize.height,
            imageWidth = activeBitmap.width,
            imageHeight = activeBitmap.height,
        )
        imageOffset = Offset(clamped.x, clamped.y)
    }
    Dialog(
        onDismissRequest = requestDismiss,
        properties = DialogProperties(usePlatformDefaultWidth = false),
    ) {
        Surface(
            color = MaterialTheme.colorScheme.surface,
            modifier = Modifier.fillMaxSize(),
        ) {
            Column {
                Row(
                    modifier = Modifier.fillMaxWidth().padding(start = 16.dp, top = 6.dp, end = 4.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Column(Modifier.weight(1f)) {
                        Text(
                            state.item.title,
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.SemiBold,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                        )
                        state.item.appliedLutName?.let { name ->
                            Text(
                                "$name ${((state.item.appliedLutStrength ?: 1f) * 100).toInt()}%",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    }
                    IconButton(
                        onClick = onUndo,
                        enabled = state.canUndo && !state.isProcessing &&
                            pendingCrop == state.geometry.crop,
                    ) {
                        Icon(Icons.AutoMirrored.Filled.Undo, contentDescription = "Undo edit")
                    }
                    IconButton(
                        onClick = onRedo,
                        enabled = state.canRedo && !state.isProcessing &&
                            pendingCrop == state.geometry.crop,
                    ) {
                        Icon(Icons.AutoMirrored.Filled.Redo, contentDescription = "Redo edit")
                    }
                    IconButton(
                        onClick = { compareMode = !compareMode },
                        enabled = state.originalPreview != null && state.preview != null,
                    ) {
                        Icon(
                            Icons.Default.Compare,
                            contentDescription = if (compareMode) {
                                "Close comparison"
                            } else {
                                "Compare original and edited"
                            },
                            tint = if (compareMode) {
                                MaterialTheme.colorScheme.primary
                            } else {
                                MaterialTheme.colorScheme.onSurface
                            },
                        )
                    }
                    IconButton(onClick = onSave, enabled = canSave) {
                        Icon(Icons.Default.Check, contentDescription = "Save edited copy")
                    }
                    IconButton(onClick = requestDismiss) {
                        Icon(Icons.Default.Close, contentDescription = "Close LUT editor")
                    }
                }
                Box(
                    modifier = Modifier.fillMaxWidth().weight(1f).background(Color.Black)
                        .clipToBounds()
                        .onSizeChanged { previewSize = it },
                    contentAlignment = Alignment.Center,
                ) {
                    if (
                        compareMode && state.preview != null &&
                        state.originalPreview != null
                    ) {
                        Image(
                            bitmap = state.originalPreview.asImageBitmap(),
                            contentDescription = "Original preview",
                            modifier = Modifier.fillMaxSize(),
                            contentScale = ContentScale.Fit,
                        )
                        Image(
                            bitmap = state.preview.asImageBitmap(),
                            contentDescription = "Edited preview",
                            modifier = Modifier.fillMaxSize().drawWithContent {
                                clipRect(left = size.width * comparisonFraction) {
                                    this@drawWithContent.drawContent()
                                }
                            },
                            contentScale = ContentScale.Fit,
                        )
                        Canvas(
                            Modifier.fillMaxSize().draggable(
                                state = rememberDraggableState { delta ->
                                    comparisonFraction = updateComparisonFraction(
                                        comparisonFraction,
                                        delta,
                                        previewSize.width,
                                    )
                                },
                                orientation = Orientation.Horizontal,
                            ),
                        ) {
                            val dividerX = size.width * comparisonFraction
                            drawLine(
                                color = Color.White,
                                start = Offset(dividerX, 0f),
                                end = Offset(dividerX, size.height),
                                strokeWidth = 2.dp.toPx(),
                            )
                            drawCircle(
                                color = Color.White,
                                radius = 16.dp.toPx(),
                                center = Offset(dividerX, size.height / 2f),
                            )
                            drawCircle(
                                color = Color.Black,
                                radius = 11.dp.toPx(),
                                center = Offset(dividerX, size.height / 2f),
                            )
                        }
                        Text(
                            "Original",
                            color = Color.White,
                            style = MaterialTheme.typography.labelMedium,
                            modifier = Modifier.align(Alignment.TopStart)
                                .padding(12.dp)
                                .background(Color.Black.copy(alpha = 0.62f))
                                .padding(horizontal = 8.dp, vertical = 4.dp),
                        )
                        Text(
                            "Edited",
                            color = Color.White,
                            style = MaterialTheme.typography.labelMedium,
                            modifier = Modifier.align(Alignment.TopEnd)
                                .padding(12.dp)
                                .background(Color.Black.copy(alpha = 0.62f))
                                .padding(horizontal = 8.dp, vertical = 4.dp),
                        )
                    } else if (state.preview != null) {
                        Image(
                            bitmap = state.preview.asImageBitmap(),
                            contentDescription = "Edited preview",
                            modifier = Modifier.fillMaxSize()
                                .graphicsLayer {
                                    scaleX = scale
                                    scaleY = scale
                                    translationX = imageOffset.x
                                    translationY = imageOffset.y
                                }
                                .transformable(
                                    state = transformState,
                                    enabled = activeTool !in setOf(
                                        EditorTool.Crop,
                                        EditorTool.Perspective,
                                    ),
                                    canPan = { scale > 1.01f },
                                )
                                .pointerInput(state.item.id, scale) {
                                    detectTapGestures(
                                        onDoubleTap = {
                                            scale = if (scale > 1.01f) 1f else 2.5f
                                            imageOffset = Offset.Zero
                                        },
                                    )
                                },
                            contentScale = ContentScale.Fit,
                        )
                    }
                    if (
                        !compareMode && state.preview != null &&
                        activeTool in setOf(EditorTool.Crop, EditorTool.Perspective)
                    ) {
                        EditorGeometryOverlay(
                            bitmap = state.preview,
                            geometry = if (activeTool == EditorTool.Crop) {
                                state.geometry.copy(crop = pendingCrop)
                            } else {
                                state.geometry
                            },
                            tool = activeTool,
                            onSetCrop = { pendingCrop = it },
                            onCropHandleMoved = { cropRatioLabel = "Free" },
                            onSetPerspective = onSetPerspective,
                            onPerspectiveGestureStart = onBeginGesture,
                            onPerspectiveGestureEnd = onEndGesture,
                        )
                    }
                    if (state.isProcessing || state.preview == null) {
                        CircularProgressIndicator(color = Color.White)
                    }
                    ainrProgress?.let { progress ->
                        Column(
                            modifier = Modifier.align(Alignment.BottomCenter)
                                .fillMaxWidth()
                                .background(Color.Black.copy(alpha = 0.72f))
                                .padding(12.dp),
                        ) {
                            Text(
                                "${progress.detail} ${if (progress.total > 0) "${progress.completed}/${progress.total}" else ""}",
                                color = Color.White,
                                style = MaterialTheme.typography.labelMedium,
                            )
                            progress.fraction?.let { fraction ->
                                LinearProgressIndicator(
                                    progress = { fraction },
                                    modifier = Modifier.fillMaxWidth().padding(top = 6.dp),
                                )
                            }
                        }
                    }
                }
                Box(
                    modifier = Modifier.fillMaxWidth().height(30.dp)
                        .background(MaterialTheme.colorScheme.surface)
                        .draggable(
                            state = rememberDraggableState { delta -> panelDrag += delta },
                            orientation = Orientation.Vertical,
                            onDragStopped = {
                                if (panelDrag > 20f) controlsExpanded = false
                                if (panelDrag < -20f) controlsExpanded = true
                                panelDrag = 0f
                            },
                        )
                        .clickable { controlsExpanded = !controlsExpanded },
                    contentAlignment = Alignment.Center,
                ) {
                    Box(
                        modifier = Modifier.width(42.dp).height(5.dp)
                            .background(
                                MaterialTheme.colorScheme.onSurfaceVariant,
                                CircleShape,
                            ),
                    )
                }
                if (controlsExpanded) {
                    Column(modifier = Modifier.heightIn(max = 390.dp)) {
                        SingleChoiceSegmentedButtonRow(
                            modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp),
                        ) {
                            EditorImageSource.entries.forEachIndexed { index, source ->
                                SegmentedButton(
                                    selected = state.source == source,
                                    onClick = { onSetSource(source) },
                                    enabled = !state.isProcessing && (
                                        source == EditorImageSource.Processed ||
                                            state.hasRetainedOriginal
                                        ),
                                    shape = SegmentedButtonDefaults.itemShape(
                                        index,
                                        EditorImageSource.entries.size,
                                    ),
                                    label = { Text(source.label) },
                                )
                            }
                        }
                        if (state.source == EditorImageSource.Processed) {
                            Row(
                                modifier = Modifier.fillMaxWidth()
                                    .padding(horizontal = 16.dp, vertical = 4.dp),
                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                            ) {
                                state.item.appliedLutName?.let { name ->
                                    BakedSettingBadge(
                                        "$name ${((state.item.appliedLutStrength ?: 1f) * 100).toInt()}%",
                                    )
                                }
                                state.item.appliedDenoiseModel?.let {
                                    BakedSettingBadge(
                                        "Denoised ${((state.item.appliedDenoiseStrength ?: 1f) * 100).toInt()}%",
                                    )
                                }
                                if (state.item.appliedGeometry?.hasChanges == true) {
                                    BakedSettingBadge("Geometry")
                                }
                            }
                        }
                        LazyRow(
                            contentPadding = PaddingValues(horizontal = 12.dp),
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                        ) {
                            items(EditorTool.entries, key = EditorTool::name) { tool ->
                                FilterChip(
                                    selected = activeTool == tool,
                                    onClick = {
                                        if (tool == EditorTool.Crop) {
                                            pendingCrop = state.geometry.crop
                                            cropRatioLabel = "Free"
                                        }
                                        if (tool == EditorTool.Lut) {
                                            onRequestLutThumbnails()
                                        }
                                        activeTool = tool
                                        compareMode = false
                                        scale = 1f
                                        imageOffset = Offset.Zero
                                    },
                                    label = { Text(tool.label) },
                                    leadingIcon = {
                                        Icon(
                                            when (tool) {
                                                EditorTool.Adjust -> Icons.Default.Tune
                                                EditorTool.Lut -> Icons.Default.Palette
                                                EditorTool.Denoise -> Icons.Default.AutoFixHigh
                                                EditorTool.Crop -> Icons.Default.Crop
                                                EditorTool.Rotate -> Icons.Default.RotateRight
                                                EditorTool.Perspective -> Icons.Default.Transform
                                            },
                                            contentDescription = null,
                                            modifier = Modifier.size(18.dp),
                                        )
                                    },
                                )
                            }
                        }
                        when (activeTool) {
                            EditorTool.Adjust -> {
                                BasicEditSlider(
                                    "Exposure", state.exposure, onBeginGesture, onEndGesture,
                                ) {
                                    onSetBasicEdits(it, state.contrast, state.saturation)
                                }
                                BasicEditSlider(
                                    "Contrast", state.contrast, onBeginGesture, onEndGesture,
                                ) {
                                    onSetBasicEdits(state.exposure, it, state.saturation)
                                }
                                BasicEditSlider(
                                    "Saturation", state.saturation, onBeginGesture, onEndGesture,
                                ) {
                                    onSetBasicEdits(state.exposure, state.contrast, it)
                                }
                            }
                            EditorTool.Lut -> {
                                val activeLut = state.selectedImportedLabel != null ||
                                    state.preset != LutPreset.Neutral
                                val showingBakedLut = state.source ==
                                    EditorImageSource.Processed &&
                                    !activeLut && state.item.appliedLutName != null
                                EffectStrengthSlider(
                                    label = "Strength",
                                    value = if (showingBakedLut) {
                                        state.item.appliedLutStrength ?: 1f
                                    } else {
                                        state.intensity
                                    },
                                    onValueChange = onSetIntensity,
                                    enabled = !showingBakedLut,
                                    onGestureStart = onBeginGesture,
                                    onGestureEnd = onEndGesture,
                                )
                                LutEditorFilmstrip(state, onSelectPreset, onSelectImported)
                            }
                            EditorTool.Denoise -> {
                                Row(
                                    modifier = Modifier.fillMaxWidth()
                                        .padding(horizontal = 16.dp, vertical = 2.dp),
                                    verticalAlignment = Alignment.CenterVertically,
                                ) {
                                    FilterChip(
                                        selected = state.denoiseEnabled,
                                        onClick = {
                                            onSetDenoiseEnabled(!state.denoiseEnabled)
                                        },
                                        enabled = !state.isProcessing,
                                        label = { Text("Denoise") },
                                    )
                                }
                                if (state.denoiseEnabled) {
                                    SingleChoiceSegmentedButtonRow(
                                        modifier = Modifier.fillMaxWidth()
                                            .padding(horizontal = 16.dp),
                                    ) {
                                        AinrDenoiseModel.entries.forEachIndexed { index, model ->
                                            SegmentedButton(
                                                selected = state.denoiseModel == model,
                                                onClick = { onSetDenoiseModel(model) },
                                                enabled = !state.isProcessing,
                                                shape = SegmentedButtonDefaults.itemShape(
                                                    index,
                                                    AinrDenoiseModel.entries.size,
                                                ),
                                                label = { Text(model.label) },
                                            )
                                        }
                                    }
                                    EffectStrengthSlider(
                                        label = "Amount",
                                        value = state.denoiseStrength,
                                        onValueChange = onSetDenoiseStrength,
                                        onGestureStart = onBeginGesture,
                                        onGestureEnd = onEndGesture,
                                    )
                                }
                            }
                            EditorTool.Crop -> {
                                val imageAspect = activeBitmap.width.toFloat() /
                                    activeBitmap.height.coerceAtLeast(1)
                                LazyRow(
                                    contentPadding = PaddingValues(horizontal = 12.dp),
                                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                                ) {
                                    val ratios = listOf(
                                        "Free" to null,
                                        "Original" to imageAspect,
                                        "1:1" to 1f,
                                        "4:3" to 4f / 3f,
                                        "3:2" to 3f / 2f,
                                        "16:9" to 16f / 9f,
                                    )
                                    items(ratios, key = { it.first }) { (label, ratio) ->
                                        FilterChip(
                                            selected = cropRatioLabel == label,
                                            onClick = {
                                                cropRatioLabel = label
                                                if (ratio != null) {
                                                    pendingCrop = cropPresetForAspect(
                                                        ratio,
                                                        imageAspect,
                                                    )
                                                }
                                            },
                                            label = { Text(label) },
                                        )
                                    }
                                }
                                Row(
                                    modifier = Modifier.fillMaxWidth()
                                        .padding(horizontal = 16.dp),
                                    horizontalArrangement = Arrangement.End,
                                    verticalAlignment = Alignment.CenterVertically,
                                ) {
                                    TextButton(onClick = {
                                        pendingCrop = NormalizedCrop()
                                        cropRatioLabel = "Free"
                                    }) { Text("Reset") }
                                    TextButton(onClick = {
                                        pendingCrop = state.geometry.crop
                                        cropRatioLabel = "Free"
                                        activeTool = EditorTool.Adjust
                                    }) { Text("Cancel") }
                                    Button(
                                        onClick = {
                                            onSetCrop(pendingCrop)
                                            activeTool = EditorTool.Adjust
                                        },
                                        enabled = pendingCrop != state.geometry.crop &&
                                            !state.isProcessing,
                                    ) { Text("Apply") }
                                }
                            }
                            EditorTool.Rotate -> {
                                var straightenGestureActive by remember {
                                    mutableStateOf(false)
                                }
                                Row(
                                    modifier = Modifier.fillMaxWidth()
                                        .padding(horizontal = 12.dp),
                                    horizontalArrangement = Arrangement.SpaceEvenly,
                                ) {
                                    IconButton(onClick = {
                                        onRotateQuarterTurn(false)
                                    }) {
                                        Icon(Icons.Default.RotateLeft, "Rotate left")
                                    }
                                    IconButton(onClick = {
                                        onRotateQuarterTurn(true)
                                    }) {
                                        Icon(Icons.Default.RotateRight, "Rotate right")
                                    }
                                    TextButton(onClick = { onAutoGeometry(false) }) {
                                        Icon(Icons.Default.AutoFixHigh, null)
                                        Text("Auto")
                                    }
                                    TextButton(onClick = {
                                        onResetGeometry(EditorTool.Rotate)
                                    }) { Text("Reset") }
                                }
                                Row(
                                    modifier = Modifier.fillMaxWidth()
                                        .padding(horizontal = 16.dp),
                                    verticalAlignment = Alignment.CenterVertically,
                                ) {
                                    Text("Straighten", style = MaterialTheme.typography.labelMedium)
                                    Slider(
                                        value = state.geometry.straightenDegrees,
                                        onValueChange = {
                                            if (!straightenGestureActive) {
                                                straightenGestureActive = true
                                                onBeginGesture()
                                            }
                                            onSetStraighten(it)
                                        },
                                        onValueChangeFinished = {
                                            if (straightenGestureActive) {
                                                straightenGestureActive = false
                                                onEndGesture()
                                            }
                                        },
                                        valueRange = -45f..45f,
                                        modifier = Modifier.weight(1f).padding(horizontal = 12.dp),
                                    )
                                    Text(
                                        "${state.geometry.straightenDegrees.roundToInt()}°",
                                        style = MaterialTheme.typography.labelSmall,
                                    )
                                }
                            }
                            EditorTool.Perspective -> {
                                Row(
                                    modifier = Modifier.fillMaxWidth()
                                        .padding(horizontal = 16.dp),
                                    horizontalArrangement = Arrangement.SpaceEvenly,
                                ) {
                                    TextButton(onClick = { onAutoGeometry(true) }) {
                                        Icon(Icons.Default.AutoFixHigh, null)
                                        Text("Auto")
                                    }
                                    TextButton(onClick = {
                                        onResetGeometry(EditorTool.Perspective)
                                    }) { Text("Reset") }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if (confirmCancel) {
        AlertDialog(
            onDismissRequest = { confirmCancel = false },
            title = { Text("Cancel denoising?") },
            text = { Text("The current image processing will stop and no edited copy will be saved.") },
            confirmButton = {
                TextButton(onClick = {
                    confirmCancel = false
                    onDismiss()
                }) { Text("Cancel processing") }
            },
            dismissButton = {
                TextButton(onClick = { confirmCancel = false }) { Text("Keep processing") }
            },
        )
    }
}

@Composable
private fun BakedSettingBadge(label: String) {
    Surface(
        color = MaterialTheme.colorScheme.secondaryContainer,
        shape = MaterialTheme.shapes.small,
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 8.dp, vertical = 5.dp),
            horizontalArrangement = Arrangement.spacedBy(4.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Icon(
                Icons.Default.Lock,
                contentDescription = null,
                modifier = Modifier.size(14.dp),
            )
            Text(label, style = MaterialTheme.typography.labelSmall)
        }
    }
}

@Composable
private fun EditorGeometryOverlay(
    bitmap: Bitmap,
    geometry: EditorGeometry,
    tool: EditorTool,
    onSetCrop: (NormalizedCrop) -> Unit,
    onCropHandleMoved: () -> Unit,
    onSetPerspective: (List<NormalizedPoint>) -> Unit,
    onPerspectiveGestureStart: () -> Unit,
    onPerspectiveGestureEnd: () -> Unit,
) {
    val density = LocalDensity.current
    val handleRadius = with(density) { 13.dp.toPx() }
    val handleColor = MaterialTheme.colorScheme.primary
    var activeHandle by remember(tool) { mutableStateOf<Int?>(null) }
    Canvas(
        modifier = Modifier.fillMaxSize().pointerInput(
            bitmap.width,
            bitmap.height,
            geometry,
            tool,
        ) {
            fun displayRect(): Rect = fittedImageRect(
                viewportWidth = size.width.toFloat(),
                viewportHeight = size.height.toFloat(),
                imageWidth = bitmap.width,
                imageHeight = bitmap.height,
            )

            fun displayedPoints(rect: Rect): List<Offset> {
                val normalized = if (tool == EditorTool.Crop) {
                    val crop = geometry.crop
                    listOf(
                        NormalizedPoint(crop.left, crop.top),
                        NormalizedPoint(crop.right, crop.top),
                        NormalizedPoint(crop.right, crop.bottom),
                        NormalizedPoint(crop.left, crop.bottom),
                    )
                } else {
                    geometry.perspective
                }
                return normalized.map {
                    Offset(
                        rect.left + it.x * rect.width,
                        rect.top + it.y * rect.height,
                    )
                }
            }

            detectDragGestures(
                onDragStart = { position ->
                    activeHandle = displayedPoints(displayRect())
                        .mapIndexed { index, point -> index to (point - position).getDistance() }
                        .minByOrNull { it.second }
                        ?.takeIf { it.second <= handleRadius * 2.5f }
                        ?.first
                    if (tool == EditorTool.Perspective && activeHandle != null) {
                        onPerspectiveGestureStart()
                    }
                },
                onDragCancel = {
                    if (tool == EditorTool.Perspective && activeHandle != null) {
                        onPerspectiveGestureEnd()
                    }
                    activeHandle = null
                },
                onDragEnd = {
                    if (tool == EditorTool.Perspective && activeHandle != null) {
                        onPerspectiveGestureEnd()
                    }
                    activeHandle = null
                },
            ) { change, _ ->
                val index = activeHandle ?: return@detectDragGestures
                change.consume()
                val rect = displayRect()
                val point = NormalizedPoint(
                    ((change.position.x - rect.left) / rect.width).coerceIn(0f, 1f),
                    ((change.position.y - rect.top) / rect.height).coerceIn(0f, 1f),
                )
                if (tool == EditorTool.Crop) {
                    onCropHandleMoved()
                    val crop = geometry.crop
                    onSetCrop(
                        when (index) {
                            0 -> crop.copy(left = point.x, top = point.y)
                            1 -> crop.copy(right = point.x, top = point.y)
                            2 -> crop.copy(right = point.x, bottom = point.y)
                            else -> crop.copy(left = point.x, bottom = point.y)
                        }.normalized(),
                    )
                } else {
                    onSetPerspective(
                        geometry.perspective.toMutableList().apply {
                            if (index in indices) this[index] = point
                        },
                    )
                }
            }
        },
    ) {
        val rect = fittedImageRect(
            viewportWidth = size.width,
            viewportHeight = size.height,
            imageWidth = bitmap.width,
            imageHeight = bitmap.height,
        )
        val normalized = if (tool == EditorTool.Crop) {
            val crop = geometry.crop
            listOf(
                NormalizedPoint(crop.left, crop.top),
                NormalizedPoint(crop.right, crop.top),
                NormalizedPoint(crop.right, crop.bottom),
                NormalizedPoint(crop.left, crop.bottom),
            )
        } else {
            geometry.perspective
        }
        val points = normalized.map {
            Offset(
                rect.left + it.x * rect.width,
                rect.top + it.y * rect.height,
            )
        }
        if (tool == EditorTool.Crop && points.size == 4) {
            val shade = Color.Black.copy(alpha = 0.52f)
            drawRect(shade, topLeft = rect.topLeft, size = androidx.compose.ui.geometry.Size(rect.width, points[0].y - rect.top))
            drawRect(shade, topLeft = Offset(rect.left, points[3].y), size = androidx.compose.ui.geometry.Size(rect.width, rect.bottom - points[3].y))
            drawRect(shade, topLeft = Offset(rect.left, points[0].y), size = androidx.compose.ui.geometry.Size(points[0].x - rect.left, points[3].y - points[0].y))
            drawRect(shade, topLeft = Offset(points[1].x, points[1].y), size = androidx.compose.ui.geometry.Size(rect.right - points[1].x, points[2].y - points[1].y))
        }
        if (points.size == 4) {
            points.indices.forEach { index ->
                drawLine(
                    Color.White,
                    points[index],
                    points[(index + 1) % points.size],
                    strokeWidth = 2.dp.toPx(),
                )
                drawCircle(
                    color = Color.White,
                    radius = handleRadius,
                    center = points[index],
                )
                drawCircle(
                    color = handleColor,
                    radius = handleRadius * 0.62f,
                    center = points[index],
                )
            }
        }
    }
}

internal fun fittedImageRect(
    viewportWidth: Float,
    viewportHeight: Float,
    imageWidth: Int,
    imageHeight: Int,
): Rect {
    if (
        viewportWidth <= 0f || viewportHeight <= 0f ||
        imageWidth <= 0 || imageHeight <= 0
    ) return Rect.Zero
    val scale = minOf(viewportWidth / imageWidth, viewportHeight / imageHeight)
    val width = imageWidth * scale
    val height = imageHeight * scale
    val left = (viewportWidth - width) / 2f
    val top = (viewportHeight - height) / 2f
    return Rect(left, top, left + width, top + height)
}

@Composable
private fun EffectStrengthSlider(
    label: String,
    value: Float,
    onValueChange: (Float) -> Unit,
    enabled: Boolean = true,
    onGestureStart: () -> Unit = {},
    onGestureEnd: () -> Unit = {},
) {
    var gestureActive by remember { mutableStateOf(false) }
    Row(
        modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(label, style = MaterialTheme.typography.labelMedium, modifier = Modifier.width(76.dp))
        Slider(
            value = value,
            onValueChange = {
                if (!gestureActive) {
                    gestureActive = true
                    onGestureStart()
                }
                onValueChange(it)
            },
            onValueChangeFinished = {
                if (gestureActive) {
                    gestureActive = false
                    onGestureEnd()
                }
            },
            enabled = enabled,
            modifier = Modifier.weight(1f),
        )
        Text("${(value * 100).toInt()}%", style = MaterialTheme.typography.labelSmall, modifier = Modifier.width(42.dp))
    }
}

@Composable
private fun LutEditorFilmstrip(
    state: LutEditorUiState,
    onSelectPreset: (LutPreset) -> Unit,
    onSelectImported: (String) -> Unit,
) {
    val hasActiveLut = state.selectedImportedLabel != null ||
        state.preset != LutPreset.Neutral
    val bakedLutName = state.item.appliedLutName.takeIf {
        state.source == EditorImageSource.Processed && !hasActiveLut
    }
    LazyRow(
        contentPadding = androidx.compose.foundation.layout.PaddingValues(12.dp),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        items(LutPreset.entries, key = { it.name }) { preset ->
            val bakedSelected = bakedLutName == preset.label
            val activeSelected = state.selectedImportedLabel == null &&
                state.preset == preset && (
                    preset != LutPreset.Neutral || bakedLutName == null
                )
            Surface(
                modifier = Modifier.width(82.dp).height(64.dp)
                    .border(
                        if (activeSelected || bakedSelected) 2.dp else 0.dp,
                        MaterialTheme.colorScheme.primary,
                        MaterialTheme.shapes.small,
                    )
                    .clickable(
                        enabled = !state.isProcessing && !bakedSelected,
                    ) { onSelectPreset(preset) },
                shape = MaterialTheme.shapes.small,
            ) {
                Box {
                    state.lutThumbnails[preset]?.let {
                        Image(it.asImageBitmap(), preset.label, Modifier.fillMaxSize(), contentScale = ContentScale.Crop)
                    }
                    if (
                        state.isLutThumbnailsLoading &&
                        state.lutThumbnails[preset] == null
                    ) {
                        CircularProgressIndicator(
                            modifier = Modifier.align(Alignment.Center).size(18.dp),
                            strokeWidth = 2.dp,
                        )
                    }
                    if (bakedSelected) {
                        Icon(
                            Icons.Default.Lock,
                            contentDescription = "Baked LUT",
                            tint = Color.White,
                            modifier = Modifier.align(Alignment.TopEnd)
                                .padding(4.dp)
                                .size(16.dp),
                        )
                    }
                    Text(
                        preset.label,
                        color = Color.White,
                        style = MaterialTheme.typography.labelSmall,
                        modifier = Modifier.align(Alignment.BottomCenter).fillMaxWidth()
                            .background(Color.Black.copy(alpha = 0.65f)).padding(3.dp),
                    )
                }
            }
        }
        items(state.importedLuts, key = { "editor:${it.label}" }) { imported ->
            val bakedSelected = bakedLutName == imported.label
            val activeSelected = state.selectedImportedLabel == imported.label
            Surface(
                modifier = Modifier.width(82.dp).height(64.dp)
                    .border(
                        if (activeSelected || bakedSelected) 2.dp else 0.dp,
                        MaterialTheme.colorScheme.primary,
                        MaterialTheme.shapes.small,
                    )
                    .clickable(
                        enabled = !state.isProcessing && !bakedSelected,
                    ) { onSelectImported(imported.label) },
                shape = MaterialTheme.shapes.small,
            ) {
                Box {
                    state.importedLutThumbnails[imported.label]?.let {
                        Image(it.asImageBitmap(), imported.label, Modifier.fillMaxSize(), contentScale = ContentScale.Crop)
                    }
                    if (
                        state.isLutThumbnailsLoading &&
                        state.importedLutThumbnails[imported.label] == null
                    ) {
                        CircularProgressIndicator(
                            modifier = Modifier.align(Alignment.Center).size(18.dp),
                            strokeWidth = 2.dp,
                        )
                    }
                    if (bakedSelected) {
                        Icon(
                            Icons.Default.Lock,
                            contentDescription = "Baked LUT",
                            tint = Color.White,
                            modifier = Modifier.align(Alignment.TopEnd)
                                .padding(4.dp)
                                .size(16.dp),
                        )
                    }
                    Text(
                        imported.label,
                        color = Color.White,
                        style = MaterialTheme.typography.labelSmall,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                        modifier = Modifier.align(Alignment.BottomCenter).fillMaxWidth()
                            .background(Color.Black.copy(alpha = 0.65f)).padding(3.dp),
                    )
                }
            }
        }
    }
}

@Composable
private fun BasicEditSlider(
    label: String,
    value: Float,
    onGestureStart: () -> Unit,
    onGestureEnd: () -> Unit,
    onValueChange: (Float) -> Unit,
) {
    var gestureActive by remember { mutableStateOf(false) }
    Row(
        modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(label, style = MaterialTheme.typography.labelMedium, modifier = Modifier.width(76.dp))
        Slider(
            value = value,
            onValueChange = {
                if (!gestureActive) {
                    gestureActive = true
                    onGestureStart()
                }
                onValueChange(it)
            },
            onValueChangeFinished = {
                if (gestureActive) {
                    gestureActive = false
                    onGestureEnd()
                }
            },
            valueRange = -1f..1f,
            modifier = Modifier.weight(1f),
        )
        Text("${(value * 100).toInt()}", style = MaterialTheme.typography.labelSmall, modifier = Modifier.width(36.dp))
    }
}

@Composable
private fun StatusBand(
    text: String,
    isError: Boolean,
    modifier: Modifier = Modifier,
) {
    val container = if (isError) MaterialTheme.colorScheme.errorContainer else MaterialTheme.colorScheme.secondaryContainer
    val content = if (isError) MaterialTheme.colorScheme.onErrorContainer else MaterialTheme.colorScheme.onSecondaryContainer
    Surface(modifier = modifier.fillMaxWidth(), color = container, shape = MaterialTheme.shapes.small) {
        Text(text, modifier = Modifier.padding(12.dp), color = content, style = MaterialTheme.typography.bodySmall)
    }
}

@Composable
private fun DiagnosticsDialog(
    device: SonyCameraDevice,
    capabilities: CameraCapabilities,
    onExport: () -> Unit,
    onDismiss: () -> Unit,
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Camera diagnostics") },
        text = {
            Column(
                modifier = Modifier.heightIn(max = 420.dp).verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                DiagnosticValue("Device", device.friendlyName)
                DiagnosticValue("Model", device.modelName)
                DiagnosticValue("Camera app", capabilities.applicationName ?: "Not reported")
                DiagnosticValue("App version", capabilities.applicationVersion ?: "Not reported")
                DiagnosticValue(
                    "Services",
                    device.endpoints.entries.sortedBy { it.key }
                        .joinToString("\n") { (name, endpoint) -> "$name  $endpoint" },
                )
                DiagnosticValue("Available commands", capabilities.availableApis.size.toString())
                Text(
                    capabilities.availableApis.sorted().joinToString("\n"),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                HorizontalDivider()
                Text(
                    "Legacy Camera Remote API prototype. Public distribution requires Sony licensing review.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        },
        confirmButton = {
            TextButton(onClick = onDismiss) { Text("Close") }
        },
        dismissButton = {
            TextButton(onClick = onExport) { Text("Export log") }
        },
    )
}

@Composable
private fun DiagnosticValue(label: String, value: String) {
    Column {
        Text(label, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text(value, style = MaterialTheme.typography.bodyMedium)
    }
}
