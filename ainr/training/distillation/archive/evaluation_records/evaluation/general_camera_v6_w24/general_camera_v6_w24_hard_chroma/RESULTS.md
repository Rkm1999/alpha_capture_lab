# Width-24 Hard-Chroma Evaluation

## Checkpoint

The evaluated candidate is the epoch-124 width-24 `target-best.pt`. It was
compared directly with the v5 width-16 target checkpoint on the same 2,724-row
validation manifest using the v6 target datasets and weakest-component-aware
score.

## Paired Validation

| Metric | v5 width 16 | v6 width 24 | Delta |
| --- | ---: | ---: | ---: |
| PSNR | 36.934770 | 37.365210 | +0.430440 dB |
| SSIM | 0.924388 | 0.930583 | +0.006195 |
| Target score | 0.602870 | 0.680983 | +0.078113 |
| Weakest component | 0.522050 | 0.606111 | +0.084061 |
| Shadow correction | 0.738446 | 0.805929 | +0.067483 |
| Shadow chroma correction | 0.730801 | 0.809146 | +0.078345 |
| Medium/coarse correction | 0.781494 | 0.827234 | +0.045739 |
| Medium/coarse chroma correction | 0.801747 | 0.847996 | +0.046250 |
| Very-coarse chroma correction | 0.522050 | 0.638715 | +0.116665 |
| Row/column chroma correction | 0.527598 | 0.606111 | +0.078513 |

The width-24 checkpoint improves every paired-reference and teacher-correction
metric. The largest improvement is in very-coarse chroma correction.

## Held-Out ISO Test

The ISO photographs have no clean references and were not used for training or
checkpoint selection. Correction ratios measure how much of SCUNet's change the
student applies, rather than restoration accuracy.

### Shadow Correction

| ISO | v5 width 16 | v6 width 24 | Delta |
| ---: | ---: | ---: | ---: |
| 1600 | 0.917396 | 0.954007 | +0.036611 |
| 3200 | 0.896271 | 0.942542 | +0.046271 |
| 6400 | 0.834460 | 0.937114 | +0.102654 |
| 12800 | 0.748832 | 0.873274 | +0.124442 |
| 25600 | 0.557587 | 0.705784 | +0.148198 |
| 51200 | 0.535853 | 0.672641 | +0.136788 |

Average shadow correction rises from 0.748400 to 0.847560.

### Midtone Correction

| ISO | v5 width 16 | v6 width 24 | Delta |
| ---: | ---: | ---: | ---: |
| 1600 | 0.929709 | 0.963743 | +0.034035 |
| 3200 | 0.883194 | 0.928589 | +0.045395 |
| 6400 | 0.817450 | 0.900348 | +0.082897 |
| 12800 | 0.776415 | 0.889978 | +0.113563 |
| 25600 | 0.724012 | 0.840149 | +0.116136 |
| 51200 | 0.702840 | 0.779867 | +0.077027 |

Average midtone correction rises from 0.805603 to 0.883779.

### Full-Image Teacher Agreement

| ISO | v5 PSNR | v6 PSNR | Delta |
| ---: | ---: | ---: | ---: |
| 1600 | 45.7966 | 48.2365 | +2.4399 dB |
| 3200 | 44.7815 | 47.6821 | +2.9006 dB |
| 6400 | 40.9582 | 44.9652 | +4.0070 dB |
| 12800 | 38.9499 | 42.2163 | +3.2665 dB |
| 25600 | 33.8290 | 36.0617 | +2.2327 dB |
| 51200 | 30.7828 | 32.7168 | +1.9340 dB |

## Visual Review

The paired-validation sheet does not show an obvious general detail collapse.
The ISO 25600 sheet shows a clear reduction in colored grain and coarse texture
on flat fields, dark surfaces, and colored edges. ISO 51200 also improves
substantially, but the student still retains coarse chromatic blotches and
structured texture that SCUNet removes. Fine printed detail remains less clean
than SCUNet but is not erased.

## Decision

Promote epoch 124 over v5 for the next device test. The gain transfers to every
held-out ISO and both luminance regions, while paired PSNR and SSIM also improve.
Width 24 is not yet a complete SCUNet replacement at ISO 51200, and its
workstation tiled runtime is about 3.3 seconds per 24 MP image versus about 2.1
seconds for width 16.

## Artifacts

- Paired comparison: `paired_validation.json`
- Validation sheet: `validation_contact_sheet/validation_contact_sheet.jpg`
- ISO report: `iso_target/report.json`
- ISO images and detailed sheets: `iso_target/`
- Checkpoint SHA-256:
  `9637c88a747e216595a57822d1c4bb805ca2c5dbeb593f1fe607b8c5e77666a0`
