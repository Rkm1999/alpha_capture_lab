package com.ryu.scunetdenoiser;

import java.util.Arrays;

final class NoiseStrengthEstimator {
    private static final int TILE = DenoiseProcessor.TILE;
    private static final int PLANE = TILE * TILE;
    private static final float SIGMA_MIN = 0.0015f;
    private static final float SIGMA_MAX = 0.07f;
    private static final float SHADOW_LIMIT = 0.5f;
    private static final float FLAT_FRACTION = 0.4f;
    private static final float MAD_NORMALIZER = 0.67448975f;
    private static final float GATE_START = 0.35f;
    private static final float GATE_END = 0.75f;

    static final class Workspace {
        final float[] luma = new float[PLANE];
        final float[] residual = new float[PLANE];
        final float[] gradient = new float[PLANE];
        final float[] selection = new float[PLANE];
        final float[] opponentRed = new float[PLANE];
        final float[] opponentBlue = new float[PLANE];
        final float[] horizontalRed = new float[PLANE];
        final float[] horizontalBlue = new float[PLANE];
    }

    private NoiseStrengthEstimator() {
    }

    static float estimateAndFill(float[] nchw, Workspace workspace) {
        if (nchw.length != 5 * PLANE && nchw.length != 7 * PLANE) {
            throw new IllegalArgumentException("Conditioned tile must contain five or seven planes");
        }
        float[] luma = workspace.luma;
        float[] residual = workspace.residual;
        float[] gradient = workspace.gradient;
        for (int index = 0; index < PLANE; index++) {
            luma[index] = 0.2126f * nchw[index]
                + 0.7152f * nchw[PLANE + index]
                + 0.0722f * nchw[2 * PLANE + index];
        }
        for (int y = 0; y < TILE; y++) {
            for (int x = 0; x < TILE; x++) {
                int index = y * TILE + x;
                float sum = 0.0f;
                for (int dy = -1; dy <= 1; dy++) {
                    int sourceY = y + dy;
                    if (sourceY < 0 || sourceY >= TILE) continue;
                    for (int dx = -1; dx <= 1; dx++) {
                        int sourceX = x + dx;
                        if (sourceX >= 0 && sourceX < TILE) {
                            sum += luma[sourceY * TILE + sourceX];
                        }
                    }
                }
                residual[index] = Math.abs(luma[index] - sum / 9.0f);
                int horizontalX = x < TILE - 1 ? x : TILE - 2;
                int verticalY = y < TILE - 1 ? y : TILE - 2;
                float horizontal = Math.abs(
                    luma[y * TILE + horizontalX + 1]
                        - luma[y * TILE + horizontalX]);
                float vertical = Math.abs(
                    luma[(verticalY + 1) * TILE + x]
                        - luma[verticalY * TILE + x]);
                gradient[index] = 0.5f * (horizontal + vertical);
            }
        }

        float[] selection = workspace.selection;
        int shadowCount = 0;
        for (int index = 0; index < PLANE; index++) {
            if (luma[index] < SHADOW_LIMIT) {
                selection[shadowCount++] = gradient[index];
            }
        }
        boolean useAll = shadowCount < 64;
        int gradientCount = useAll ? PLANE : shadowCount;
        if (useAll) {
            System.arraycopy(gradient, 0, selection, 0, PLANE);
        }
        float gradientLimit = quantile(selection, gradientCount, FLAT_FRACTION);

        int candidateCount = 0;
        for (int index = 0; index < PLANE; index++) {
            boolean shadow = useAll || luma[index] < SHADOW_LIMIT;
            if (shadow && gradient[index] <= gradientLimit) {
                selection[candidateCount++] = residual[index];
            }
        }
        if (candidateCount < 64) {
            System.arraycopy(residual, 0, selection, 0, PLANE);
            candidateCount = PLANE;
        }
        float sigma = quantile(selection, candidateCount, 0.5f) / MAD_NORMALIZER;
        float clampedSigma = Math.max(SIGMA_MIN, sigma);
        float strength = (float) (
            (Math.log(clampedSigma) - Math.log(SIGMA_MIN))
                / (Math.log(SIGMA_MAX) - Math.log(SIGMA_MIN)));
        strength = Math.max(0.0f, Math.min(1.0f, strength));
        Arrays.fill(nchw, 3 * PLANE, 4 * PLANE, strength);

        if (nchw.length == 5 * PLANE) {
            Arrays.fill(nchw, 4 * PLANE, 5 * PLANE, noiseGate(strength));
            return strength;
        }

        for (int index = 0; index < PLANE; index++) {
            float shadow = (SHADOW_LIMIT - luma[index]) / SHADOW_LIMIT;
            nchw[4 * PLANE + index] = clampUnit(shadow);
            workspace.opponentRed[index] = nchw[index] - nchw[PLANE + index];
            workspace.opponentBlue[index] = nchw[2 * PLANE + index]
                - nchw[PLANE + index];
        }
        horizontalBoxSum(
            workspace.opponentRed, workspace.opponentBlue,
            workspace.horizontalRed, workspace.horizontalBlue);
        verticalChromaMap(
            workspace, nchw);

        float gate = noiseGate(strength);
        Arrays.fill(nchw, 6 * PLANE, 7 * PLANE, gate);
        return strength;
    }

    private static float noiseGate(float strength) {
        float position = Math.max(
            0.0f,
            Math.min(1.0f, (strength - GATE_START) / (GATE_END - GATE_START)));
        return position * position * (3.0f - 2.0f * position);
    }

    private static int clampIndex(int value) {
        return Math.max(0, Math.min(TILE - 1, value));
    }

    private static void horizontalBoxSum(
        float[] red,
        float[] blue,
        float[] redSum,
        float[] blueSum
    ) {
        for (int y = 0; y < TILE; y++) {
            int row = y * TILE;
            float runningRed = 5.0f * red[row];
            float runningBlue = 5.0f * blue[row];
            for (int x = 1; x <= 4; x++) {
                runningRed += red[row + x];
                runningBlue += blue[row + x];
            }
            for (int x = 0; x < TILE; x++) {
                int index = row + x;
                redSum[index] = runningRed;
                blueSum[index] = runningBlue;
                runningRed += red[row + clampIndex(x + 5)]
                    - red[row + clampIndex(x - 4)];
                runningBlue += blue[row + clampIndex(x + 5)]
                    - blue[row + clampIndex(x - 4)];
            }
        }
    }

    private static void verticalChromaMap(Workspace workspace, float[] nchw) {
        for (int x = 0; x < TILE; x++) {
            float runningRed = 5.0f * workspace.horizontalRed[x];
            float runningBlue = 5.0f * workspace.horizontalBlue[x];
            for (int y = 1; y <= 4; y++) {
                runningRed += workspace.horizontalRed[y * TILE + x];
                runningBlue += workspace.horizontalBlue[y * TILE + x];
            }
            for (int y = 0; y < TILE; y++) {
                int index = y * TILE + x;
                float chroma = (
                    Math.abs(workspace.opponentRed[index] - runningRed / 81.0f)
                        + Math.abs(workspace.opponentBlue[index] - runningBlue / 81.0f)
                ) * 0.5f / 0.04f;
                nchw[5 * PLANE + index] = clampUnit(chroma);
                runningRed += workspace.horizontalRed[clampIndex(y + 5) * TILE + x]
                    - workspace.horizontalRed[clampIndex(y - 4) * TILE + x];
                runningBlue += workspace.horizontalBlue[clampIndex(y + 5) * TILE + x]
                    - workspace.horizontalBlue[clampIndex(y - 4) * TILE + x];
            }
        }
    }

    private static float clampUnit(float value) {
        return Math.max(0.0f, Math.min(1.0f, value));
    }

    private static float quantile(float[] values, int count, float fraction) {
        Arrays.sort(values, 0, count);
        float position = (count - 1) * fraction;
        int lower = (int) Math.floor(position);
        int upper = Math.min(count - 1, lower + 1);
        float weight = position - lower;
        return values[lower] + weight * (values[upper] - values[lower]);
    }
}
