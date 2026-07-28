package com.ryu.sonyremote.data

import android.content.ContentResolver
import android.content.ContentValues
import android.net.Uri
import android.location.Location
import android.os.Environment
import android.os.Build
import android.provider.MediaStore
import android.content.ContentUris
import android.util.Size
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import java.io.ByteArrayOutputStream
import androidx.exifinterface.media.ExifInterface
import java.io.ByteArrayInputStream
import java.time.Instant
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import com.ryu.sonyremote.model.OutputImageFormat

class JpegMediaStore(private val contentResolver: ContentResolver) {
    suspend fun delete(uri: Uri): Boolean = withContext(Dispatchers.IO) {
        contentResolver.delete(uri, null, null) > 0
    }

    suspend fun listSaved(): List<SavedMediaItem> = withContext(Dispatchers.IO) {
        val collection = MediaStore.Images.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
        val projection = arrayOf(
            MediaStore.Images.Media._ID,
            MediaStore.Images.Media.DISPLAY_NAME,
            MediaStore.Images.Media.DATE_ADDED,
        )
        contentResolver.query(
            collection,
            projection,
            "(${MediaStore.Images.Media.RELATIVE_PATH}=? OR " +
                "${MediaStore.Images.Media.RELATIVE_PATH}=?)",
            arrayOf("$CURRENT_RELATIVE_PATH/", "$LEGACY_RELATIVE_PATH/"),
            "${MediaStore.Images.Media.DATE_ADDED} ASC",
        )?.use { cursor ->
            val idColumn = cursor.getColumnIndexOrThrow(MediaStore.Images.Media._ID)
            val nameColumn = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DISPLAY_NAME)
            val dateColumn = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATE_ADDED)
            buildList {
                while (cursor.moveToNext()) {
                    val uri = ContentUris.withAppendedId(collection, cursor.getLong(idColumn))
                    val thumbnail = runCatching {
                        contentResolver.loadThumbnail(uri, Size(320, 320), null)
                    }.getOrNull() ?: continue
                    val attribution = readLutAttribution(uri)
                    add(SavedMediaItem(
                        uri, cursor.getString(nameColumn), cursor.getLong(dateColumn), thumbnail,
                        attribution?.first, attribution?.second,
                    ))
                }
            }
        }.orEmpty()
    }
    suspend fun save(
        image: ByteArray,
        prefix: String = "SONY",
        format: OutputImageFormat = OutputImageFormat.Jpeg,
    ): Uri = withContext(Dispatchers.IO) {
        val normalizedJpeg = JpegValidator.normalize(image)
        val normalizedImage = if (format == OutputImageFormat.Jpeg) {
            normalizedJpeg
        } else {
            val bitmap = requireNotNull(BitmapFactory.decodeByteArray(normalizedJpeg, 0, normalizedJpeg.size))
            try {
                ByteArrayOutputStream().use { output ->
                    val webpFormat = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                        Bitmap.CompressFormat.WEBP_LOSSY
                    } else {
                        @Suppress("DEPRECATION")
                        Bitmap.CompressFormat.WEBP
                    }
                    check(bitmap.compress(webpFormat, 95, output))
                    output.toByteArray()
                }
            } finally {
                bitmap.recycle()
            }
        }
        require(prefix.matches(FILE_PREFIX)) { "Invalid image filename prefix" }
        val extension = if (format == OutputImageFormat.Webp) "webp" else "jpg"
        val displayName = "${prefix}_${FILE_TIME_FORMAT.format(Instant.now())}.$extension"
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, displayName)
            put(MediaStore.Images.Media.MIME_TYPE, if (format == OutputImageFormat.Webp) "image/webp" else "image/jpeg")
            put(
                MediaStore.Images.Media.RELATIVE_PATH,
                CURRENT_RELATIVE_PATH,
            )
            put(MediaStore.Images.Media.IS_PENDING, 1)
        }
        val collection = MediaStore.Images.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
        val item = requireNotNull(contentResolver.insert(collection, values)) {
            "Android could not create an image in MediaStore"
        }
        try {
            contentResolver.openOutputStream(item, "w").use { output ->
                requireNotNull(output) { "Android could not open the destination image" }
                output.write(normalizedImage)
                output.flush()
            }
            contentResolver.update(
                item,
                ContentValues().apply { put(MediaStore.Images.Media.IS_PENDING, 0) },
                null,
                null,
            )
            item
        } catch (error: Throwable) {
            contentResolver.delete(item, null, null)
            throw error
        }
    }

    suspend fun copyExif(sourceJpeg: ByteArray, destination: Uri) = withContext(Dispatchers.IO) {
        val source = ExifInterface(ByteArrayInputStream(sourceJpeg))
        contentResolver.openFileDescriptor(destination, "rw").use { descriptor ->
            val target = ExifInterface(
                requireNotNull(descriptor) { "Android could not open the saved image metadata" }.fileDescriptor,
            )
            COPIED_EXIF_TAGS.forEach { tag ->
                source.getAttribute(tag)?.let { target.setAttribute(tag, it) }
            }
            source.latLong?.let { target.setLatLong(it[0], it[1]) }
            source.getAltitude(Double.NaN).takeIf { !it.isNaN() }?.let(target::setAltitude)
            target.setAttribute(
                ExifInterface.TAG_ORIENTATION,
                ExifInterface.ORIENTATION_NORMAL.toString(),
            )
            target.saveAttributes()
        }
    }

    suspend fun setLocation(destination: Uri, location: Location) = withContext(Dispatchers.IO) {
        contentResolver.openFileDescriptor(destination, "rw").use { descriptor ->
            val exif = ExifInterface(requireNotNull(descriptor).fileDescriptor)
            exif.setLatLong(location.latitude, location.longitude)
            if (location.hasAltitude()) exif.setAltitude(location.altitude)
            exif.setAttribute(ExifInterface.TAG_GPS_PROCESSING_METHOD, location.provider ?: "PHONE")
            exif.saveAttributes()
        }
    }

    suspend fun setLutAttribution(destination: Uri, name: String, strength: Float) =
        withContext(Dispatchers.IO) {
            contentResolver.openFileDescriptor(destination, "rw").use { descriptor ->
                val exif = ExifInterface(requireNotNull(descriptor).fileDescriptor)
                val encodedName = android.util.Base64.encodeToString(
                    name.toByteArray(), android.util.Base64.NO_WRAP or android.util.Base64.URL_SAFE,
                )
                exif.setAttribute(
                    ExifInterface.TAG_USER_COMMENT,
                    "$LUT_COMMENT_PREFIX$encodedName:${strength.coerceIn(0f, 1f)}",
                )
                exif.saveAttributes()
            }
        }

    private fun readLutAttribution(uri: Uri): Pair<String, Float>? = runCatching {
        contentResolver.openFileDescriptor(uri, "r").use { descriptor ->
            val value = ExifInterface(requireNotNull(descriptor).fileDescriptor)
                .getAttribute(ExifInterface.TAG_USER_COMMENT)
                ?.takeIf { it.startsWith(LUT_COMMENT_PREFIX) }
                ?.removePrefix(LUT_COMMENT_PREFIX) ?: return@use null
            val encodedName = value.substringBeforeLast(':')
            val strength = value.substringAfterLast(':').toFloat()
            String(android.util.Base64.decode(encodedName, android.util.Base64.URL_SAFE)) to strength
        }
    }.getOrNull()

    private companion object {
        val CURRENT_RELATIVE_PATH = "${Environment.DIRECTORY_PICTURES}/Alpha Capture Lab"
        val LEGACY_RELATIVE_PATH = "${Environment.DIRECTORY_PICTURES}/Sony Remote"
        val FILE_PREFIX = Regex("[A-Z0-9_]{1,32}")
        const val LUT_COMMENT_PREFIX = "REMOTE_CAPTURE_LUT:"
        val COPIED_EXIF_TAGS = listOf(
            ExifInterface.TAG_MAKE,
            ExifInterface.TAG_MODEL,
            ExifInterface.TAG_SOFTWARE,
            ExifInterface.TAG_ARTIST,
            ExifInterface.TAG_COPYRIGHT,
            ExifInterface.TAG_DATETIME,
            ExifInterface.TAG_DATETIME_ORIGINAL,
            ExifInterface.TAG_DATETIME_DIGITIZED,
            ExifInterface.TAG_SUBSEC_TIME,
            ExifInterface.TAG_SUBSEC_TIME_ORIGINAL,
            ExifInterface.TAG_SUBSEC_TIME_DIGITIZED,
            ExifInterface.TAG_EXPOSURE_TIME,
            ExifInterface.TAG_F_NUMBER,
            ExifInterface.TAG_PHOTOGRAPHIC_SENSITIVITY,
            ExifInterface.TAG_EXPOSURE_BIAS_VALUE,
            ExifInterface.TAG_EXPOSURE_PROGRAM,
            ExifInterface.TAG_METERING_MODE,
            ExifInterface.TAG_FLASH,
            ExifInterface.TAG_FOCAL_LENGTH,
            ExifInterface.TAG_FOCAL_LENGTH_IN_35MM_FILM,
            ExifInterface.TAG_LENS_MAKE,
            ExifInterface.TAG_LENS_MODEL,
            ExifInterface.TAG_WHITE_BALANCE,
            ExifInterface.TAG_COLOR_SPACE,
            ExifInterface.TAG_GPS_TIMESTAMP,
            ExifInterface.TAG_GPS_DATESTAMP,
            ExifInterface.TAG_GPS_PROCESSING_METHOD,
        )
        val FILE_TIME_FORMAT: DateTimeFormatter =
            DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss_SSS").withZone(ZoneOffset.UTC)
    }
}

data class SavedMediaItem(
    val uri: Uri,
    val name: String,
    val dateAdded: Long,
    val thumbnail: Bitmap,
    val lutName: String? = null,
    val lutStrength: Float? = null,
)
