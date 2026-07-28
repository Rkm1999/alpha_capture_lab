package com.ryu.scunetdenoiser;

import android.content.ContentResolver;
import android.content.ContentUris;
import android.content.ContentValues;
import android.graphics.Bitmap;
import android.graphics.ColorSpace;
import android.graphics.ImageDecoder;
import android.net.Uri;
import android.os.Environment;
import android.provider.MediaStore;
import android.provider.OpenableColumns;

import androidx.exifinterface.media.ExifInterface;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.UnsupportedEncodingException;
import java.net.URLDecoder;
import java.util.Locale;
import java.util.concurrent.CancellationException;
import java.util.concurrent.atomic.AtomicBoolean;

final class ImageStore {
    private static final String[] COPIED_EXIF_TAGS = {
        ExifInterface.TAG_APERTURE_VALUE,
        ExifInterface.TAG_ARTIST,
        ExifInterface.TAG_BRIGHTNESS_VALUE,
        ExifInterface.TAG_COPYRIGHT,
        ExifInterface.TAG_DATETIME,
        ExifInterface.TAG_DATETIME_DIGITIZED,
        ExifInterface.TAG_DATETIME_ORIGINAL,
        ExifInterface.TAG_EXPOSURE_BIAS_VALUE,
        ExifInterface.TAG_EXPOSURE_MODE,
        ExifInterface.TAG_EXPOSURE_PROGRAM,
        ExifInterface.TAG_EXPOSURE_TIME,
        ExifInterface.TAG_FLASH,
        ExifInterface.TAG_F_NUMBER,
        ExifInterface.TAG_FOCAL_LENGTH,
        ExifInterface.TAG_FOCAL_LENGTH_IN_35MM_FILM,
        ExifInterface.TAG_GPS_ALTITUDE,
        ExifInterface.TAG_GPS_ALTITUDE_REF,
        ExifInterface.TAG_GPS_DATESTAMP,
        ExifInterface.TAG_GPS_LATITUDE,
        ExifInterface.TAG_GPS_LATITUDE_REF,
        ExifInterface.TAG_GPS_LONGITUDE,
        ExifInterface.TAG_GPS_LONGITUDE_REF,
        ExifInterface.TAG_GPS_PROCESSING_METHOD,
        ExifInterface.TAG_GPS_TIMESTAMP,
        ExifInterface.TAG_ISO_SPEED_RATINGS,
        ExifInterface.TAG_LENS_MAKE,
        ExifInterface.TAG_LENS_MODEL,
        ExifInterface.TAG_MAKE,
        ExifInterface.TAG_METERING_MODE,
        ExifInterface.TAG_MODEL,
        ExifInterface.TAG_SCENE_CAPTURE_TYPE,
        ExifInterface.TAG_SOFTWARE,
        ExifInterface.TAG_SUBSEC_TIME,
        ExifInterface.TAG_SUBSEC_TIME_DIGITIZED,
        ExifInterface.TAG_SUBSEC_TIME_ORIGINAL,
        ExifInterface.TAG_USER_COMMENT,
        ExifInterface.TAG_WHITE_BALANCE
    };

    static final class Preview {
        final Bitmap bitmap;
        final int width;
        final int height;
        final String displayName;

        Preview(Bitmap bitmap, int width, int height, String displayName) {
            this.bitmap = bitmap;
            this.width = width;
            this.height = height;
            this.displayName = displayName;
        }
    }

    static final class RgbImage {
        final byte[] pixels;
        final int width;
        final int height;

        RgbImage(byte[] pixels, int width, int height) {
            this.pixels = pixels;
            this.width = width;
            this.height = height;
        }
    }

    Preview decodePreview(ContentResolver resolver, Uri uri, int maximumDimension)
        throws IOException {
        int[] decodedSize = new int[2];
        ImageDecoder.Source source = ImageDecoder.createSource(resolver, uri);
        Bitmap bitmap = ImageDecoder.decodeBitmap(source, (decoder, info, sourceInfo) -> {
            configureDecoder(decoder);
            decodedSize[0] = info.getSize().getWidth();
            decodedSize[1] = info.getSize().getHeight();
            setTargetSize(decoder, decodedSize[0], decodedSize[1], maximumDimension);
        });
        boolean sourceLandscape = decodedSize[0] >= decodedSize[1];
        boolean previewLandscape = bitmap.getWidth() >= bitmap.getHeight();
        if (sourceLandscape != previewLandscape) {
            int swap = decodedSize[0];
            decodedSize[0] = decodedSize[1];
            decodedSize[1] = swap;
        }
        return new Preview(
            bitmap,
            decodedSize[0],
            decodedSize[1],
            displayName(resolver, uri));
    }

    Bitmap decodeFilePreview(File file, int maximumDimension) throws IOException {
        ImageDecoder.Source source = ImageDecoder.createSource(file);
        return ImageDecoder.decodeBitmap(source, (decoder, info, sourceInfo) -> {
            configureDecoder(decoder);
            setTargetSize(decoder, info.getSize().getWidth(), info.getSize().getHeight(),
                maximumDimension);
        });
    }

    RgbImage decodeFull(ContentResolver resolver, Uri uri, AtomicBoolean canceled)
        throws IOException {
        Bitmap bitmap = decode(resolver, uri, 0);
        try {
            int width = bitmap.getWidth();
            int height = bitmap.getHeight();
            long byteCount = (long) width * height * 3;
            if (byteCount > Integer.MAX_VALUE) {
                throw new IOException("Image dimensions are too large");
            }
            byte[] rgb = new byte[(int) byteCount];
            int[] row = new int[width];
            for (int y = 0; y < height; y++) {
                checkCanceled(canceled);
                bitmap.getPixels(row, 0, width, 0, y, width, 1);
                int destination = y * width * 3;
                for (int x = 0; x < width; x++) {
                    int color = row[x];
                    rgb[destination++] = (byte) ((color >>> 16) & 0xff);
                    rgb[destination++] = (byte) ((color >>> 8) & 0xff);
                    rgb[destination++] = (byte) (color & 0xff);
                }
            }
            return new RgbImage(rgb, width, height);
        } finally {
            bitmap.recycle();
        }
    }

    void encodeJpeg(
        byte[] rgb,
        int width,
        int height,
        File destination,
        int quality,
        AtomicBoolean canceled
    ) throws IOException {
        long expected = (long) width * height * 3;
        if (expected != rgb.length) throw new IOException("Invalid RGB output size");
        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        try {
            int[] row = new int[width];
            for (int y = 0; y < height; y++) {
                checkCanceled(canceled);
                int source = y * width * 3;
                for (int x = 0; x < width; x++) {
                    int red = rgb[source++] & 0xff;
                    int green = rgb[source++] & 0xff;
                    int blue = rgb[source++] & 0xff;
                    row[x] = 0xff000000 | red << 16 | green << 8 | blue;
                }
                bitmap.setPixels(row, 0, width, 0, y, width, 1);
            }
            File parent = destination.getParentFile();
            if (parent != null && !parent.isDirectory() && !parent.mkdirs()) {
                throw new IOException("Could not create output directory");
            }
            try (OutputStream output = new BufferedOutputStream(
                new FileOutputStream(destination))) {
                if (!bitmap.compress(Bitmap.CompressFormat.JPEG, quality, output)) {
                    throw new IOException("JPEG encoder failed");
                }
            }
        } finally {
            bitmap.recycle();
        }
    }

    void copyExif(
        ContentResolver resolver,
        Uri source,
        File destination,
        int outputWidth,
        int outputHeight
    ) throws IOException {
        try (InputStream input = new BufferedInputStream(resolver.openInputStream(source))) {
            if (input == null) throw new IOException("Could not reopen source metadata");
            ExifInterface sourceExif = new ExifInterface(input);
            ExifInterface destinationExif = new ExifInterface(destination.getAbsolutePath());
            for (String tag : COPIED_EXIF_TAGS) {
                String value = sourceExif.getAttribute(tag);
                if (value != null) destinationExif.setAttribute(tag, value);
            }
            // ImageDecoder applies source rotation before inference and JPEG encoding.
            destinationExif.setAttribute(
                ExifInterface.TAG_ORIENTATION,
                Integer.toString(ExifInterface.ORIENTATION_NORMAL));
            destinationExif.setAttribute(
                ExifInterface.TAG_PIXEL_X_DIMENSION,
                Integer.toString(outputWidth));
            destinationExif.setAttribute(
                ExifInterface.TAG_PIXEL_Y_DIMENSION,
                Integer.toString(outputHeight));
            destinationExif.setAttribute(
                ExifInterface.TAG_IMAGE_WIDTH,
                Integer.toString(outputWidth));
            destinationExif.setAttribute(
                ExifInterface.TAG_IMAGE_LENGTH,
                Integer.toString(outputHeight));
            destinationExif.saveAttributes();
        }
    }

    Uri saveToGallery(
        ContentResolver resolver,
        File source,
        String displayName,
        int width,
        int height
    ) throws IOException {
        ContentValues values = new ContentValues();
        values.put(MediaStore.Images.Media.DISPLAY_NAME, displayName);
        values.put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg");
        values.put(MediaStore.Images.Media.WIDTH, width);
        values.put(MediaStore.Images.Media.HEIGHT, height);
        values.put(
            MediaStore.Images.Media.RELATIVE_PATH,
            Environment.DIRECTORY_PICTURES + "/SCUNet Denoiser");
        values.put(MediaStore.Images.Media.IS_PENDING, 1);
        Uri uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
        if (uri == null) throw new IOException("Could not create gallery image");
        try {
            try (
                InputStream input = new BufferedInputStream(new FileInputStream(source));
                OutputStream output = resolver.openOutputStream(uri, "w")
            ) {
                if (output == null) throw new IOException("Could not open gallery destination");
                byte[] buffer = new byte[1024 * 1024];
                int count;
                while ((count = input.read(buffer)) >= 0) output.write(buffer, 0, count);
            }
            ContentValues complete = new ContentValues();
            complete.put(MediaStore.Images.Media.IS_PENDING, 0);
            resolver.update(uri, complete, null, null);
            return uri;
        } catch (Throwable error) {
            resolver.delete(uri, null, null);
            if (error instanceof IOException) throw (IOException) error;
            throw new IOException(error);
        }
    }

    static String outputName(String inputName) {
        String source = inputName == null || inputName.isBlank() ? "image" : inputName.trim();
        int extension = source.lastIndexOf('.');
        if (extension > 0) source = source.substring(0, extension);
        source = source.replaceAll("[^A-Za-z0-9._ -]", "_");
        if (source.isBlank()) source = "image";
        return String.format(Locale.US, "%s_denoised.jpg", source);
    }

    private static Bitmap decode(ContentResolver resolver, Uri uri, int maximumDimension)
        throws IOException {
        ImageDecoder.Source source = ImageDecoder.createSource(resolver, uri);
        return ImageDecoder.decodeBitmap(source, (decoder, info, sourceInfo) -> {
            configureDecoder(decoder);
            if (maximumDimension > 0) {
                setTargetSize(decoder, info.getSize().getWidth(), info.getSize().getHeight(),
                    maximumDimension);
            }
        });
    }

    private static void configureDecoder(ImageDecoder decoder) {
        decoder.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE);
        decoder.setTargetColorSpace(ColorSpace.get(ColorSpace.Named.SRGB));
        decoder.setOnPartialImageListener(error -> false);
    }

    private static void setTargetSize(
        ImageDecoder decoder,
        int width,
        int height,
        int maximumDimension
    ) {
        int largest = Math.max(width, height);
        if (largest <= maximumDimension) return;
        double scale = maximumDimension / (double) largest;
        decoder.setTargetSize(
            Math.max(1, (int) Math.round(width * scale)),
            Math.max(1, (int) Math.round(height * scale)));
    }

    static String displayName(ContentResolver resolver, Uri uri) {
        try (android.database.Cursor cursor = resolver.query(
            uri,
            new String[] {
                MediaStore.PickerMediaColumns.DATA,
                OpenableColumns.DISPLAY_NAME
            },
            null,
            null,
            null)) {
            if (cursor != null && cursor.moveToFirst()) {
                int dataColumn = cursor.getColumnIndex(MediaStore.PickerMediaColumns.DATA);
                if (dataColumn >= 0) {
                    String value = fileNameFromPath(cursor.getString(dataColumn));
                    if (value != null) return value;
                }
                String value = cursorString(cursor, OpenableColumns.DISPLAY_NAME);
                if (value != null) {
                    String mediaStoreName = mediaStoreDisplayName(resolver, value);
                    return mediaStoreName == null ? value : mediaStoreName;
                }
            }
        } catch (Throwable ignored) {
            // Older and third-party providers may reject picker-only columns.
        }
        try (android.database.Cursor cursor = resolver.query(
            uri, new String[] {OpenableColumns.DISPLAY_NAME}, null, null, null)) {
            if (cursor != null && cursor.moveToFirst()) {
                String value = cursorString(cursor, OpenableColumns.DISPLAY_NAME);
                if (value != null) {
                    String mediaStoreName = mediaStoreDisplayName(resolver, value);
                    return mediaStoreName == null ? value : mediaStoreName;
                }
            }
        } catch (Throwable ignored) {
            // Some share providers do not implement metadata queries.
        }
        String segment = uri.getLastPathSegment();
        return segment == null || segment.isBlank() ? "image" : segment;
    }

    private static String cursorString(android.database.Cursor cursor, String columnName) {
        int column = cursor.getColumnIndex(columnName);
        if (column < 0) return null;
        String value = cursor.getString(column);
        return value == null || value.isBlank() ? null : value;
    }

    private static String mediaStoreDisplayName(
        ContentResolver resolver,
        String pickerName
    ) {
        int extension = pickerName.lastIndexOf('.');
        if (extension <= 0) return null;
        String idText = pickerName.substring(0, extension);
        for (int index = 0; index < idText.length(); index++) {
            if (!Character.isDigit(idText.charAt(index))) return null;
        }
        try {
            long id = Long.parseLong(idText);
            Uri mediaUri = ContentUris.withAppendedId(
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI, id);
            try (android.database.Cursor cursor = resolver.query(
                mediaUri,
                new String[] {MediaStore.Images.Media.DISPLAY_NAME},
                null,
                null,
                null)) {
                if (cursor == null || !cursor.moveToFirst()) return null;
                String value = cursorString(cursor, MediaStore.Images.Media.DISPLAY_NAME);
                return pickerName.equals(value) ? null : value;
            }
        } catch (RuntimeException ignored) {
            return null;
        }
    }

    static String fileNameFromPath(String path) {
        if (path == null || path.isBlank()) return null;
        String normalized = path.replace('\\', '/');
        int separator = normalized.lastIndexOf('/');
        String value = separator >= 0 ? normalized.substring(separator + 1) : normalized;
        if (value.indexOf('%') >= 0) {
            try {
                value = URLDecoder.decode(value.replace("+", "%2B"), "UTF-8");
            } catch (UnsupportedEncodingException impossible) {
                throw new AssertionError(impossible);
            }
        }
        return value.isBlank() ? null : value;
    }

    private static void checkCanceled(AtomicBoolean canceled) {
        if (canceled.get() || Thread.currentThread().isInterrupted()) {
            throw new CancellationException("Operation canceled");
        }
    }
}
