package com.ryu.scunetdenoiser;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

public final class NoiseStrengthEstimatorTest {
    private static final int PLANE = DenoiseProcessor.TILE * DenoiseProcessor.TILE;

    @Test
    public void flatTileProducesZeroConditionPlane() {
        float[] tile = new float[7 * PLANE];
        for (int index = 0; index < 3 * PLANE; index++) {
            tile[index] = 0.2f;
        }
        float strength = NoiseStrengthEstimator.estimateAndFill(
            tile, new NoiseStrengthEstimator.Workspace());
        assertEquals(0.0f, strength, 0.0f);
        for (int index = 3 * PLANE; index < 4 * PLANE; index++) {
            assertEquals(strength, tile[index], 0.0f);
        }
        for (int index = 4 * PLANE; index < 7 * PLANE; index++) {
            float expected = index < 5 * PLANE ? 0.6f : 0.0f;
            assertEquals(expected, tile[index], 0.000001f);
        }
    }

    @Test
    public void noisyShadowProducesBoundedNonzeroStrength() {
        float[] tile = new float[7 * PLANE];
        long state = 0x1234abcdL;
        for (int index = 0; index < PLANE; index++) {
            state = (1664525L * state + 1013904223L) & 0xffffffffL;
            float noise = (((state >>> 8) & 0xffff) / 65535.0f - 0.5f) * 0.08f;
            float value = Math.max(0.0f, Math.min(1.0f, 0.15f + noise));
            tile[index] = value;
            tile[PLANE + index] = value;
            tile[2 * PLANE + index] = value;
        }
        float strength = NoiseStrengthEstimator.estimateAndFill(
            tile, new NoiseStrengthEstimator.Workspace());
        assertTrue(strength > 0.25f);
        assertTrue(strength <= 1.0f);
        float gate = tile[6 * PLANE];
        assertTrue(gate >= 0.0f);
        assertTrue(gate <= 1.0f);
        for (int index = 6 * PLANE; index < 7 * PLANE; index++) {
            assertEquals(gate, tile[index], 0.0f);
        }
    }

    @Test
    public void fivePlaneModelReceivesStrengthAndSmoothGate() {
        float[] tile = new float[5 * PLANE];
        long state = 0x18f5a12bL;
        for (int index = 0; index < PLANE; index++) {
            state = (1664525L * state + 1013904223L) & 0xffffffffL;
            float noise = (((state >>> 8) & 0xffff) / 65535.0f - 0.5f) * 0.08f;
            float value = Math.max(0.0f, Math.min(1.0f, 0.15f + noise));
            tile[index] = value;
            tile[PLANE + index] = value;
            tile[2 * PLANE + index] = value;
        }

        float strength = NoiseStrengthEstimator.estimateAndFill(
            tile, new NoiseStrengthEstimator.Workspace());
        float gate = tile[4 * PLANE];

        assertTrue(strength > 0.25f);
        assertTrue(strength <= 1.0f);
        assertTrue(gate >= 0.0f);
        assertTrue(gate <= 1.0f);
        for (int index = 3 * PLANE; index < 4 * PLANE; index++) {
            assertEquals(strength, tile[index], 0.0f);
        }
        for (int index = 4 * PLANE; index < 5 * PLANE; index++) {
            assertEquals(gate, tile[index], 0.0f);
        }
    }

    @Test
    public void chromaMapMatchesReplicatedNineByNineMean() {
        float[] tile = new float[7 * PLANE];
        for (int y = 0; y < DenoiseProcessor.TILE; y++) {
            for (int x = 0; x < DenoiseProcessor.TILE; x++) {
                int index = y * DenoiseProcessor.TILE + x;
                tile[index] = ((x * 7 + y * 3) % 101) / 100.0f;
                tile[PLANE + index] = ((x * 2 + y * 5) % 89) / 88.0f;
                tile[2 * PLANE + index] = ((x * 11 + y) % 97) / 96.0f;
            }
        }
        NoiseStrengthEstimator.estimateAndFill(
            tile, new NoiseStrengthEstimator.Workspace());

        int[][] points = {{0, 0}, {4, 4}, {53, 97}, {191, 191}};
        for (int[] point : points) {
            int x = point[0];
            int y = point[1];
            float redMean = 0.0f;
            float blueMean = 0.0f;
            for (int dy = -4; dy <= 4; dy++) {
                int sourceY = Math.max(0, Math.min(DenoiseProcessor.TILE - 1, y + dy));
                for (int dx = -4; dx <= 4; dx++) {
                    int sourceX = Math.max(0, Math.min(DenoiseProcessor.TILE - 1, x + dx));
                    int source = sourceY * DenoiseProcessor.TILE + sourceX;
                    redMean += tile[source] - tile[PLANE + source];
                    blueMean += tile[2 * PLANE + source] - tile[PLANE + source];
                }
            }
            int index = y * DenoiseProcessor.TILE + x;
            float red = tile[index] - tile[PLANE + index];
            float blue = tile[2 * PLANE + index] - tile[PLANE + index];
            float expected = Math.max(0.0f, Math.min(1.0f,
                (Math.abs(red - redMean / 81.0f)
                    + Math.abs(blue - blueMean / 81.0f)) * 0.5f / 0.04f));
            assertEquals(expected, tile[5 * PLANE + index], 0.00002f);
        }
    }
}
