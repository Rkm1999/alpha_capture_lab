package com.ryu.scunetdenoiser;

import android.content.Context;
import android.util.Log;

import com.google.ai.edge.litert.Environment;

import java.io.File;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Map;

/** Bridges LiteRT 2.1.6's native compiler cache until its Java overload is public. */
final class CachedEnvironmentFactory {
    private static final String TAG = "SCUNetDenoiser";
    private static final int COMPILER_CACHE_OPTION = 18;

    private CachedEnvironmentFactory() {}

    static Environment create(
        Context context,
        Map<Environment.Option, String> options,
        String cacheKey
    ) throws Exception {
        File directory = new File(context.getFilesDir(), "litert_npu_cache/" + cacheKey);
        if (!directory.isDirectory() && !directory.mkdirs()) {
            throw new IllegalStateException("Could not create NPU compiler cache");
        }

        int[] keys = new int[options.size() + 1];
        String[] values = new String[options.size() + 1];
        int index = 0;
        for (Map.Entry<Environment.Option, String> option : options.entrySet()) {
            keys[index] = option.getKey().getValue();
            values[index] = option.getValue();
            index++;
        }
        keys[index] = COMPILER_CACHE_OPTION;
        values[index] = directory.getAbsolutePath();

        try {
            Method create = Environment.class.getDeclaredMethod(
                "nativeCreate", int[].class, String[].class);
            create.setAccessible(true);
            long handle = (long) create.invoke(null, keys, values);
            Constructor<Environment> constructor =
                Environment.class.getDeclaredConstructor(long.class);
            constructor.setAccessible(true);
            Environment environment = constructor.newInstance(handle);
            Log.i(TAG, "NPU_CACHE_ENABLED path=" + directory.getAbsolutePath());
            return environment;
        } catch (InvocationTargetException error) {
            Throwable cause = error.getCause();
            if (cause instanceof Exception) throw (Exception) cause;
            if (cause instanceof Error) throw (Error) cause;
            throw new IllegalStateException(cause);
        } catch (ReflectiveOperationException | RuntimeException error) {
            Log.w(TAG, "NPU compiler cache bridge unavailable; using Real JIT", error);
            return Environment.create(options);
        }
    }
}
