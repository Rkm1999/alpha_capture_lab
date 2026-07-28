# Baseline vs NIND ISO Comparison

This directory compares already-rendered outputs. No student or SCUNet inference is
performed by the comparison script. Every image panel is an unscaled 768 x 768 crop,
and every comparison sheet is lossless PNG.

The source Sony JPEGs have no paired clean targets. PSNR and SSIM below measure
agreement with SCUNet only; they are not clean-reference quality scores.

The alpha 0.7 report predates the ablation identity fields. Its baseline identity is
validated from alpha=0.7 plus the report and actual checkpoint SHA-256 values.

## Full-frame PSNR to SCUNet

| ISO | Baseline | Teacher-only | Full-reference | T - B | F - B | F - T |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1600 | 43.971 | 44.150 | 44.299 | +0.179 | +0.328 | +0.149 |
| 3200 | 42.500 | 42.039 | 42.436 | -0.460 | -0.064 | +0.396 |
| 6400 | 39.213 | 38.307 | 39.039 | -0.906 | -0.174 | +0.732 |
| 12800 | 37.474 | 37.044 | 37.356 | -0.430 | -0.117 | +0.312 |
| 25600 | 32.723 | 32.619 | 32.728 | -0.104 | +0.004 | +0.109 |
| 51200 | 29.881 | 29.925 | 29.911 | +0.044 | +0.030 | -0.014 |
| **Mean** | **37.627** | **37.347** | **37.628** | **-0.280** | **+0.001** | **+0.281** |

## Full-frame preview SSIM to SCUNet

| ISO | Baseline | Teacher-only | Full-reference | T - B | F - B | F - T |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1600 | 0.983967 | 0.982231 | 0.983199 | -0.001736 | -0.000768 | +0.000968 |
| 3200 | 0.970385 | 0.966290 | 0.968768 | -0.004096 | -0.001617 | +0.002479 |
| 6400 | 0.937951 | 0.931349 | 0.936960 | -0.006602 | -0.000991 | +0.005611 |
| 12800 | 0.921835 | 0.915501 | 0.919462 | -0.006334 | -0.002373 | +0.003961 |
| 25600 | 0.828426 | 0.825815 | 0.828460 | -0.002611 | +0.000033 | +0.002645 |
| 51200 | 0.711839 | 0.715112 | 0.715030 | +0.003273 | +0.003191 | -0.000081 |
| **Mean** | **0.892401** | **0.889383** | **0.891980** | **-0.003018** | **-0.000421** | **+0.002597** |

`B` is baseline, `T` is teacher-only, and `F` is full-reference. See
[metrics.json](metrics.json) for all full-frame, luma-band, native-crop, and
pairwise delta values, or [metrics.csv](metrics.csv) for a flat export.

## Per-ISO native sheets

- [ISO 1600](per_iso/ISO1600_DSC00001.png)
- [ISO 3200](per_iso/ISO3200_DSC00002.png)
- [ISO 6400](per_iso/ISO6400_DSC00003.png)
- [ISO 12800](per_iso/ISO12800_DSC00004.png)
- [ISO 25600](per_iso/ISO25600_DSC00005.png)
- [ISO 51200](per_iso/ISO51200_DSC00006.png)

## Cross-ISO native region sheets

- [01 Dark shelf](by_region/01_dark_shelf.png)
- [02 Smooth blue background](by_region/02_blue_background.png)
- [03 Bright screen and text](by_region/03_screen.png)
- [04 Red cap and colored edge](by_region/04_red_cap.png)
- [05 White figure and face detail](by_region/05_white_figure.png)
- [06 Specular metal](by_region/06_metal_bit.png)
- [07 Color blocks and edges](by_region/07_color_blocks.png)
- [08 Dark tool body](by_region/08_dark_tool.png)
- [09 Fine dot pattern](by_region/09_dot_pattern.png)
- [10 Black table and edge](by_region/10_black_table.png)
- [11 Fine printed label](by_region/11_fine_label.png)
- [12 Transparent bottle and desk](by_region/12_bottle_desk.png)

## Reproduce

From `paper_192`:

```bash
$PYTHON scripts/compare_iso_arms.py
```

Input and output hashes, dimensions, and validation assertions are recorded in
[manifest.json](manifest.json).
