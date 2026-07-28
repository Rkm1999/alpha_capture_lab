# Balanced Shadow/Multiscale Run Results

The training run completed all 200 epochs. All paired metrics below use the
same 1,982-row validation manifest.

## Paired validation

| Checkpoint | Epoch | Balanced PSNR | Balanced SSIM | Target score | Shadow capture | Medium/coarse capture |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Previous model | 85 | 38.8732 | 0.90799 | 0.68148 | 65.60% | 70.69% |
| 1:1:1 patch run | 110 | 38.5605 | 0.90022 | 0.66628 | 64.35% | 68.91% |
| New general (`best.pt`) | 92 | 38.8957 | 0.90780 | 0.73260 | 69.93% | 76.59% |
| New target (`target-best.pt`) | 175 | 38.7045 | 0.90093 | 0.75293 | 72.34% | 78.24% |

The general checkpoint is the best choice when paired-reference PSNR is the
priority. The target checkpoint is the recommended denoising checkpoint: it
retains the configured 38.70 dB guardrail while capturing more of SCUNet's
shadow and medium/coarse correction.

## ISO teacher agreement

These camera JPEGs have no clean reference. PSNR and SSIM therefore measure
agreement with SCUNet, not absolute reconstruction quality.

| ISO | General PSNR | Target PSNR | Delta | General SSIM | Target SSIM |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,600 | 44.9835 | 45.3095 | +0.3260 | 0.98655 | 0.98810 |
| 3,200 | 43.2715 | 43.5110 | +0.2395 | 0.97426 | 0.97644 |
| 6,400 | 38.9785 | 39.4624 | +0.4839 | 0.94337 | 0.94874 |
| 12,800 | 37.8810 | 38.2329 | +0.3520 | 0.93397 | 0.93815 |
| 25,600 | 33.1428 | 33.5192 | +0.3763 | 0.85260 | 0.86395 |
| 51,200 | 30.4349 | 31.1188 | +0.6838 | 0.75774 | 0.77851 |

Average agreement improves from 38.1154 to 38.5256 dB and from 0.90808 to
0.91565 SSIM with the target checkpoint. Average shadow correction capture
increases from 65.77% to 67.46%.

## Artifacts

- `paired_validation.json`: general versus target paired validation.
- `paired_validation_with_baselines.json`: same-manifest four-model comparison.
- `iso_comparison.json`: per-ISO general versus target metrics.
- `iso_general/`: full images and detail contact sheets for `best.pt`.
- `iso_target/`: full images and detail contact sheets for `target-best.pt`.
