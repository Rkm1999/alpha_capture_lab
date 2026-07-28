# Chroma and Blotch Fine-Tune Evaluation

## Checkpoint

The evaluated candidate is epoch 30 `target-best.pt` from
`general_camera_v5_chroma_blotch_tune`. It was compared directly with the
previously selected v4 synthetic-shadow `target-best.pt`.

## Paired Validation

Both checkpoints were evaluated on the same full paired-validation manifest.
The target score below combines the six explicit correction metrics introduced
for this experiment.

| Metric | v4 | v5 | Delta |
| --- | ---: | ---: | ---: |
| PSNR | 36.923671 | 36.934770 | +0.011099 dB |
| SSIM | 0.924073 | 0.924388 | +0.000315 |
| Six-metric target | 0.704944 | 0.714038 | +0.009094 |
| Shadow correction | 0.762575 | 0.766928 | +0.004353 |
| Shadow chroma correction | 0.758470 | 0.765449 | +0.006980 |
| Medium/coarse correction | 0.809934 | 0.810909 | +0.000975 |
| Medium/coarse chroma correction | 0.838478 | 0.839524 | +0.001046 |
| Very-coarse chroma correction | 0.530744 | 0.541115 | +0.010371 |
| Row/column chroma correction | 0.529461 | 0.560302 | +0.030841 |

The candidate improves every paired metric. The largest gain is in row/column
chroma correction, which was the main target of this fine-tune.

## Held-Out ISO Test

The ISO 1600-51200 photographs were not used for training or checkpoint
selection. They have no clean references. The ratios therefore measure how much
of SCUNet's correction the student applies, not restoration accuracy.

### Shadow Correction

| ISO | v4 | v5 | Delta |
| ---: | ---: | ---: | ---: |
| 1600 | 0.915789 | 0.917396 | +0.001607 |
| 3200 | 0.894936 | 0.896271 | +0.001335 |
| 6400 | 0.834125 | 0.834460 | +0.000335 |
| 12800 | 0.748536 | 0.748832 | +0.000296 |
| 25600 | 0.557536 | 0.557587 | +0.000050 |
| 51200 | 0.535668 | 0.535853 | +0.000185 |

Average shadow correction rises from 0.747765 to 0.748400. The gain is
consistent but small, including at ISO 25600-51200.

### Midtone Correction

| ISO | v4 | v5 | Delta |
| ---: | ---: | ---: | ---: |
| 1600 | 0.932923 | 0.929709 | -0.003215 |
| 3200 | 0.886326 | 0.883194 | -0.003132 |
| 6400 | 0.824135 | 0.817450 | -0.006685 |
| 12800 | 0.781588 | 0.776415 | -0.005172 |
| 25600 | 0.731785 | 0.724012 | -0.007772 |
| 51200 | 0.710356 | 0.702840 | -0.007516 |

Average midtone correction falls from 0.811185 to 0.805603. This is a small
but consistent regression in SCUNet agreement.

## Visual Review

The paired-validation sheet is consistent with the metric gains and does not
show an obvious loss of general detail. On the held-out ISO sheets, v5 remains
visually close to v4. ISO 25600 and 51200 still retain substantial chromatic
grain and coarse blotching that SCUNet removes, so this fine-tune does not close
the high-ISO gap.

## Decision

Promote v5 over v4 for the next on-device test: it improves PSNR, SSIM, every
paired correction metric, and shadow correction at every held-out ISO. Treat it
as an incremental checkpoint rather than a solved model. The paired chroma gain
is real, but the held-out visual improvement is marginal and comes with reduced
midtone teacher agreement.

## Artifacts

- Paired metrics: `paired_validation.json`
- Validation sheet: `validation_contact_sheet/validation_contact_sheet.jpg`
- Validation metadata:
  `validation_contact_sheet/validation_contact_sheet.json`
- ISO report: `iso_target/report.json`
- ISO full images and detailed sheets: `iso_target/`
- Checkpoint SHA-256:
  `e1decc705505a7b9995d8055ed7f27d305c01922bfdd8e0970f4167cccf596f0`
