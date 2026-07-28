package com.ryu.sonyremote

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext
import androidx.core.content.ContextCompat
import androidx.core.content.IntentCompat
import androidx.core.net.toUri
import android.net.Uri
import android.provider.OpenableColumns
import android.content.ContentResolver
import android.util.Log
import java.io.ByteArrayOutputStream
import java.io.InputStream
import java.util.zip.ZipInputStream
import androidx.lifecycle.viewmodel.compose.viewModel
import com.ryu.sonyremote.ui.AlphaCaptureLabApp
import com.ryu.sonyremote.ui.CameraViewModel
import com.ryu.sonyremote.ui.theme.AlphaCaptureLabTheme

class MainActivity : ComponentActivity() {
    private var incomingLutBatches by mutableStateOf<List<List<Uri>>>(emptyList())

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enqueueSharedLuts(intent)
        enableEdgeToEdge()
        setContent {
            AlphaCaptureLabTheme {
                AlphaCaptureLabRoot(
                    incomingLutUris = incomingLutBatches.firstOrNull().orEmpty(),
                    onIncomingLutsConsumed = {
                        incomingLutBatches = incomingLutBatches.drop(1)
                    },
                )
            }
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        enqueueSharedLuts(intent)
    }

    private fun enqueueSharedLuts(intent: Intent) {
        val uris = intent.sharedLutUris()
        Log.i(LUT_IMPORT_TAG, "Received ${uris.size} shared LUT URI(s) for ${intent.action}")
        if (uris.isNotEmpty()) incomingLutBatches = incomingLutBatches + listOf(uris)
    }
}

@Composable
private fun AlphaCaptureLabRoot(
    incomingLutUris: List<Uri>,
    onIncomingLutsConsumed: () -> Unit,
    cameraViewModel: CameraViewModel = viewModel(),
) {
    val context = LocalContext.current
    var pendingPair by remember { mutableStateOf<Triple<String, String, Boolean>?>(null) }
    val wifiPanel = rememberLauncherForActivityResult(ActivityResultContracts.StartActivityForResult()) {}
    val nearbyPermission = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted ->
        val pair = pendingPair
        pendingPair = null
        if (granted && pair != null) {
            cameraViewModel.pairAndConnectCamera(pair.first, pair.second, pair.third)
        } else {
            cameraViewModel.onNearbyPermissionResult(granted)
        }
    }

    LaunchedEffect(incomingLutUris) {
        if (incomingLutUris.isEmpty()) return@LaunchedEffect
        try {
            incomingLutUris.forEach { uri ->
                val name = context.contentResolver.query(
                    uri,
                    arrayOf(OpenableColumns.DISPLAY_NAME),
                    null,
                    null,
                    null,
                )?.use { cursor -> if (cursor.moveToFirst()) cursor.getString(0) else null }
                    ?: uri.lastPathSegment.orEmpty()
                context.contentResolver.readSharedLuts(uri, name).forEach { (lutName, text) ->
                    Log.i(LUT_IMPORT_TAG, "Importing shared LUT $lutName")
                    cameraViewModel.importCubeLut(lutName, text)
                }
            }
        } catch (error: Throwable) {
            Log.e(LUT_IMPORT_TAG, "Shared LUT import failed", error)
            cameraViewModel.reportLutImportFailure(error)
        } finally {
            onIncomingLutsConsumed()
        }
    }

    AlphaCaptureLabApp(
        viewModel = cameraViewModel,
        onOpenWifi = {
            val panelIntent = Intent(Settings.Panel.ACTION_WIFI)
            runCatching { wifiPanel.launch(panelIntent) }
                .onFailure { wifiPanel.launch(Intent(Settings.ACTION_WIFI_SETTINGS)) }
        },
        onFindCamera = {
            val needsAndroid16Permission =
                Build.VERSION.SDK_INT >= 36 &&
                    ContextCompat.checkSelfPermission(
                        context,
                        Manifest.permission.NEARBY_WIFI_DEVICES,
                    ) != PackageManager.PERMISSION_GRANTED
            if (needsAndroid16Permission) {
                nearbyPermission.launch(Manifest.permission.NEARBY_WIFI_DEVICES)
            } else {
                cameraViewModel.onNearbyPermissionResult(true)
            }
        },
        onPairCamera = { ssid, password, autoConnect ->
            val needsPermission = Build.VERSION.SDK_INT >= 33 &&
                ContextCompat.checkSelfPermission(
                    context,
                    Manifest.permission.NEARBY_WIFI_DEVICES,
                ) != PackageManager.PERMISSION_GRANTED
            if (needsPermission) {
                pendingPair = Triple(ssid, password, autoConnect)
                nearbyPermission.launch(Manifest.permission.NEARBY_WIFI_DEVICES)
            } else {
                cameraViewModel.pairAndConnectCamera(ssid, password, autoConnect)
            }
        },
        onOpenAppSettings = {
            context.startActivity(
                Intent(
                    Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
                    "package:${context.packageName}".toUri(),
                ),
            )
        },
    )
}

internal fun Intent.sharedLutUris(): List<Uri> = when (action) {
    Intent.ACTION_SEND,
    Intent.ACTION_SEND_MULTIPLE,
    -> buildList {
        if (action == Intent.ACTION_SEND) {
            IntentCompat.getParcelableExtra(this@sharedLutUris, Intent.EXTRA_STREAM, Uri::class.java)?.let(::add)
        } else {
            IntentCompat.getParcelableArrayListExtra(this@sharedLutUris, Intent.EXTRA_STREAM, Uri::class.java)
                ?.let(::addAll)
        }
        clipData?.let { clips ->
            repeat(clips.itemCount) { clips.getItemAt(it).uri?.let(::add) }
        }
        data?.let(::add)
    }.distinct()
    Intent.ACTION_VIEW -> listOfNotNull(data)
    else -> emptyList()
}

private fun ContentResolver.readSharedLuts(uri: Uri, displayName: String): List<Pair<String, String>> {
    val isZip = displayName.endsWith(".zip", ignoreCase = true) ||
        getType(uri) in ZIP_MIME_TYPES
    if (!isZip) {
        val text = openInputStream(uri)?.use { it.readLimitedUtf8() }
            ?: error("Could not open shared LUT")
        return listOf(displayName to text)
    }

    openInputStream(uri)?.use { input ->
        ZipInputStream(input.buffered()).use { zip ->
            val luts = mutableListOf<Pair<String, String>>()
            repeat(MAX_ZIP_ENTRIES) {
                val entry = zip.nextEntry ?: return luts.ifEmpty {
                    error("The shared ZIP does not contain a .cube LUT")
                }
                if (!entry.isDirectory && entry.name.endsWith(".cube", ignoreCase = true)) {
                    val safeName = entry.name.replace('\\', '/').trimStart('/')
                    luts += safeName to zip.readLimitedUtf8()
                }
                zip.closeEntry()
            }
        }
    } ?: error("Could not open shared LUT archive")
    error("The shared ZIP contains too many files")
}

private fun InputStream.readLimitedUtf8(): String {
    val output = ByteArrayOutputStream()
    val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
    var total = 0
    while (true) {
        val count = read(buffer)
        if (count < 0) break
        total += count
        require(total <= MAX_LUT_BYTES) { "The LUT is larger than 8 MB" }
        output.write(buffer, 0, count)
    }
    return output.toString(Charsets.UTF_8.name())
}

private val ZIP_MIME_TYPES = setOf("application/zip", "application/x-zip-compressed")
private const val MAX_ZIP_ENTRIES = 100
private const val MAX_LUT_BYTES = 8 * 1024 * 1024
private const val LUT_IMPORT_TAG = "LutImport"
