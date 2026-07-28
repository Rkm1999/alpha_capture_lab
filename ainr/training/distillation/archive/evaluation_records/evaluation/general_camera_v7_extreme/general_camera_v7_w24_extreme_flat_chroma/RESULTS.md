# Extreme Flat-Chroma Fine-Tune Evaluation

## Checkpoints

The experimental v7 epoch-43 `target-score-best.pt` was compared directly with
the guarded v6 epoch-124 checkpoint. Both use the same width-24 architecture and
were evaluated on the same validation manifest and target definition.

## Paired Validation

| Metric | v6 epoch 124 | v7 epoch 43 | Delta |
| --- | ---: | ---: | ---: |
| PSNR | 37.365210 | 37.064350 | -0.300860 dB |
| SSIM | 0.930583 | 0.922360 | -0.008222 |
| Target score | 0.680983 | 0.718149 | +0.037166 |
| Weakest component | 0.606111 | 0.654205 | +0.048094 |
| Shadow correction | 0.805929 | 0.830278 | +0.024349 |
| Shadow chroma correction | 0.809146 | 0.836100 | +0.026954 |
| Medium/coarse correction | 0.827234 | 0.838728 | +0.011494 |
| Medium/coarse chroma correction | 0.847996 | 0.858107 | +0.010111 |
| Very-coarse chroma correction | 0.638715 | 0.654205 | +0.015490 |
| Row/column chroma correction | 0.606111 | 0.675143 | +0.069033 |

V7 improves every teacher-correction metric but fails both clean-reference
quality guardrails.

## Held-Out ISO Test

The ISO set has no clean references. These ratios measure SCUNet correction
agreement rather than restoration accuracy.

### Shadow Correction

| ISO | v6 | v7 | Delta |
| ---: | ---: | ---: | ---: |
| 1600 | 0.954007 | 0.950484 | -0.003523 |
| 3200 | 0.942542 | 0.945529 | +0.002987 |
| 6400 | 0.937114 | 0.944552 | +0.007438 |
| 12800 | 0.873274 | 0.879174 | +0.005900 |
| 25600 | 0.705784 | 0.713163 | +0.007378 |
| 51200 | 0.672641 | 0.702745 | +0.030103 |

Average shadow correction rises from 0.847560 to 0.855941.

### Midtone Correction

| ISO | v6 | v7 | Delta |
| ---: | ---: | ---: | ---: |
| 1600 | 0.963743 | 0.965888 | +0.002145 |
| 3200 | 0.928589 | 0.944181 | +0.015592 |
| 6400 | 0.900348 | 0.923091 | +0.022743 |
| 12800 | 0.889978 | 0.899457 | +0.009479 |
| 25600 | 0.840149 | 0.837178 | -0.002971 |
| 51200 | 0.779867 | 0.782329 | +0.002462 |

Average midtone correction rises from 0.883779 to 0.892021.

### Full-Image Teacher Agreement

| ISO | v6 PSNR | v7 PSNR | Delta |
| ---: | ---: | ---: | ---: |
| 1600 | 48.2365 | 48.3069 | +0.0704 dB |
| 3200 | 47.6821 | 47.9928 | +0.3107 dB |
| 6400 | 44.9652 | 45.5811 | +0.6159 dB |
| 12800 | 42.2163 | 42.3179 | +0.1015 dB |
| 25600 | 36.0617 | 36.0566 | -0.0051 dB |
| 51200 | 32.7168 | 32.9005 | +0.1837 dB |

## Visual Review

ISO 51200 shows a small reduction in shadow chroma texture, but coarse blotches
remain clearly visible. ISO 25600 is visually very close to v6, consistent with
its negligible full-image agreement change. The paired validation sheet shows
that the stronger teacher matching comes with a broader clean-reference
quality loss; the improvement is not confined to only extreme-noise inputs.

## Decision

Keep v6 epoch 124 as the default checkpoint. The v7 tradeoff is not favorable
enough to replace it: ISO 25600 does not materially improve, and the modest ISO
51200 gain costs 0.301 dB PSNR and 0.00822 SSIM on paired validation.

Retain v7 epoch 43 as an experimental strong-denoise checkpoint. A future
extreme-noise path should use explicit noise-strength conditioning or a
separately selected mode rather than applying this behavior to every image.

## Artifacts

- Paired comparison: `paired_validation.json`
- Validation sheet: `validation_contact_sheet/validation_contact_sheet.jpg`
- ISO report: `iso_target/report.json`
- ISO images and detailed sheets: `iso_target/`
- Checkpoint SHA-256:
  `1cd86c5afd3155d0da5731d90e873f3c0f40a5f64ccdd872c5da5da18e1e47cd`
