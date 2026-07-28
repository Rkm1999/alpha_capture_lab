package com.ryu.scunetdenoiser;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

public final class AcceleratorEngineTest {
    private static final int PLANE = DenoiseProcessor.TILE * DenoiseProcessor.TILE;

    @Test
    public void convertsNchwFloatToNhwcInt8() {
        ModelStore.TensorSpec spec =
            ModelStore.TensorSpec.int8Nhwc(7, 0.01f, -3, 0.02f, 4);
        float[] input = new float[7 * PLANE];
        for (int channel = 0; channel < 7; channel++) {
            input[channel * PLANE] = channel * 0.1f;
            input[channel * PLANE + 1] = channel * 0.1f + 0.01f;
        }

        byte[] output = AcceleratorEngine.quantizeNchwToNhwc(input, spec);

        for (int channel = 0; channel < 7; channel++) {
            assertEquals(channel * 10 - 3, output[channel]);
            assertEquals(channel * 10 - 2, output[7 + channel]);
        }
    }

    @Test
    public void convertsNhwcInt8ToNchwFloat() {
        ModelStore.TensorSpec spec =
            ModelStore.TensorSpec.int8Nhwc(7, 0.01f, -3, 0.02f, 4);
        byte[] input = new byte[3 * PLANE];
        input[0] = 4;
        input[1] = 9;
        input[2] = -1;
        input[3] = 14;
        input[4] = 19;
        input[5] = 24;

        float[] output = AcceleratorEngine.dequantizeNhwcToNchw(input, spec);

        assertEquals(0.0f, output[0], 0.000001f);
        assertEquals(0.1f, output[PLANE], 0.000001f);
        assertEquals(-0.1f, output[2 * PLANE], 0.000001f);
        assertEquals(0.2f, output[1], 0.000001f);
        assertEquals(0.3f, output[PLANE + 1], 0.000001f);
        assertEquals(0.4f, output[2 * PLANE + 1], 0.000001f);
    }
}
