# ISO quality comparison

This is a no-reference comparison on the six 24 MP Sony JPEGs from ISO 1600
through ISO 51200. Full SCUNet is used as the teacher reference. The scores
measure agreement with SCUNet, not objective restoration accuracy.

The W32 FP16 column was rendered from the pre-QAT float checkpoint that is the
source of the FP16 mobile model. W32 W8A8 QAT was rendered with the exact
fully-quantized LiteRT file deployed in the Android app.

## Per-ISO agreement

| ISO | W24 to SCUNet PSNR | W32 FP16 to SCUNet PSNR | W32 QAT to SCUNet PSNR | QAT vs FP16 PSNR | QAT vs FP16 MAE |
|---:|---:|---:|---:|---:|---:|
| 1600 | 48.24 | 48.25 | 40.80 | 41.32 | 1.72/255 |
| 3200 | 47.70 | 47.35 | 40.71 | 41.28 | 1.73/255 |
| 6400 | 44.98 | 44.00 | 39.50 | 41.18 | 1.75/255 |
| 12800 | 42.22 | 41.55 | 38.41 | 41.15 | 1.77/255 |
| 25600 | 36.06 | 34.72 | 33.79 | 41.10 | 1.77/255 |
| 51200 | 32.77 | 30.32 | 30.04 | 40.72 | 1.86/255 |

## Shadow denoising

The following is each student's shadow correction magnitude divided by the
teacher's correction magnitude. Values near 100% mean a similar amount of
change, but do not prove that the corrected pixels are identical.

| ISO | W24 | W32 FP16 | W32 QAT |
|---:|---:|---:|---:|
| 1600 | 95.4% | 94.4% | 122.0% |
| 3200 | 94.2% | 93.2% | 111.9% |
| 6400 | 93.8% | 89.7% | 99.9% |
| 12800 | 87.4% | 84.8% | 96.0% |
| 25600 | 70.6% | 62.2% | 70.4% |
| 51200 | 67.8% | 55.3% | 60.8% |

## Visual findings

- ISO 1600-6400: all three mobile models preserve useful edges and remove the
  visible fine grain. QAT is visually close to W32 FP16.
- ISO 12800: all mobile outputs are usable. Full SCUNet is smoother; the
  students retain slightly more texture and residual chroma noise.
- ISO 25600-51200: full SCUNet is clearly strongest on chroma grain and dark
  blotching. W24 removes more of the teacher-targeted shadow noise than W32.
- QAT does not introduce obvious tile seams or checkerboard artifacts in these
  six images. Its mean absolute difference from W32 FP16 is 1.73/255.
- The QAT checkpoint is not only a quantized copy: it also includes its QAT
  fine-tuning. The QAT-versus-FP16 difference therefore includes both training
  changes and INT8 rounding.

## Files

Each ISO folder retains its compact `detail_comparison.jpg`. The `by_region`
folder compares the same crop across all ISO levels. Exact metrics, source
hashes, and model hashes are in `report.json`. The regenerable lossless
full-resolution outputs were removed during training-workspace cleanup.
