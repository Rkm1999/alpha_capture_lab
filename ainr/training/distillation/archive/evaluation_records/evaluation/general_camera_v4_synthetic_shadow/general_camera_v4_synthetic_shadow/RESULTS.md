# Synthetic Shadow Fine-Tune Evaluation

## Checkpoint

The evaluated candidate is epoch 40 `target-best.pt` from
`general_camera_v4_synthetic_shadow`. It was compared with the previously
selected `general_camera_v1_stage2_hard_shadow/target-best.pt`.

## Paired Validation

The full 2,724-row validation manifest was evaluated for both checkpoints.
Checkpoint selection metrics use only the six real paired domains; synthetic
samples do not contribute to selection.

| Metric | Previous | Epoch 40 | Delta |
| --- | ---: | ---: | ---: |
| PSNR | 36.43166 | 36.92367 | +0.49201 dB |
| SSIM | 0.910934 | 0.924073 | +0.013139 |
| Shadow correction capture | 0.727344 | 0.762574 | +0.035230 |
| Medium/coarse correction capture | 0.782646 | 0.809934 | +0.027288 |
| Combined target | 0.754995 | 0.786254 | +0.031259 |

The candidate improves every paired-validation selection metric.

## Held-Out ISO Test

The ISO photographs were not used for training or checkpoint selection. They
have no paired clean targets, so these ratios measure how much of SCUNet's
correction the student applies, not restoration accuracy.

### Shadow Correction

| ISO | Previous | Epoch 40 | Delta |
| ---: | ---: | ---: | ---: |
| 1600 | 0.88950 | 0.91579 | +0.02629 |
| 3200 | 0.81481 | 0.89494 | +0.08013 |
| 6400 | 0.69455 | 0.83413 | +0.13958 |
| 12800 | 0.64426 | 0.74854 | +0.10427 |
| 25600 | 0.47588 | 0.55754 | +0.08165 |
| 51200 | 0.53950 | 0.53567 | -0.00383 |

Average shadow capture increased from 0.67642 to 0.74777. The improvement is
strong from ISO 1600 through 25600, but it does not transfer to ISO 51200.

### Midtone Correction

| ISO | Previous | Epoch 40 | Delta |
| ---: | ---: | ---: | ---: |
| 1600 | 0.91433 | 0.93292 | +0.01859 |
| 3200 | 0.87089 | 0.88633 | +0.01544 |
| 6400 | 0.84186 | 0.82414 | -0.01772 |
| 12800 | 0.81724 | 0.78159 | -0.03565 |
| 25600 | 0.76677 | 0.73178 | -0.03499 |
| 51200 | 0.78022 | 0.71036 | -0.06987 |

The synthetic fine-tune moved correction toward shadows, but reduced
SCUNet-like correction in high-ISO midtones. This table is luminance-region
behavior and is distinct from the paired validation's spatial-frequency
medium/coarse metric.

## Visual Review

The real paired validation sheet shows improved denoising without an obvious
general detail collapse. The ISO crops confirm stronger shadow processing,
especially at ISO 6400-25600. At ISO 25600 and 51200, the student still retains
visible chromatic grain and blotchy texture that SCUNet removes.

## Decision

Epoch 40 is a better candidate than the previous model for the next on-device
test because it improves all paired metrics and substantially improves
held-out shadow behavior through ISO 25600. It should not yet be treated as a
complete SCUNet replacement at ISO 51200.

## Artifacts

- Paired metrics: `paired_validation.json`
- Validation sheet: `validation_contact_sheet/validation_contact_sheet.jpg`
- Validation sheet metadata:
  `validation_contact_sheet/validation_contact_sheet.json`
- ISO report: `iso_target/report.json`
- ISO full images and detailed sheets: `iso_target/`
