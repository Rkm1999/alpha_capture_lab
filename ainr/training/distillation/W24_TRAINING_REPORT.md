# LiteDenoise Width-24 Training Report

> Historical report for the earlier W24-v4 branch. The final selected
> production model is `general_camera_v9_w24_residual_adapter`; see
> [`EXPERIMENT_RETROSPECTIVE.md`](EXPERIMENT_RETROSPECTIVE.md).

## Selected checkpoint

- Run: `runs/litedenoise_w24_v4`
- Checkpoint: `best_distilled.pt`, epoch 208
- Parameters: 4,415,643
- Compute: 4.449 GMAC per 192 x 192 tile
- Validation teacher PSNR: 40.638 dB
- Checkpoint SHA-256: `338b972b279b3681c613bc96c2d7d0275d2f8db587e0c54577c8470545fa18e3`

The Sony ISO 1600-51200 set was excluded from all training and validation. It
was used only for full-resolution audits after each experiment converged.

## Experiment results

| Experiment | Result |
| --- | --- |
| Width-24 correction-weighted distillation | Selected; 70.82% RGB-only held-out projection |
| Residual projection loss | High-ISO projection peaked near 54%; did not transfer sufficiently |
| Gaussian/shot-noise JPEG augmentation | High-ISO projection stayed near 53% |
| Amplified real-noise augmentation | High-ISO projection reached about 55%, then saturated |
| PolyU real JPEG pairs, including Sony A7 II | High-ISO projection regressed near 51% |
| Extra refinement depth | Validation improved, but high-ISO projection stayed near 53% |

These trials show that ISO 25600-51200 processed Sony JPEGs are outside the
available training distribution. Additional RGB-only fitting reduced
clean-reference quality without closing that gap.

## ISO-conditioned export

The deployable model accepts RGB plus one constant ISO-strength plane. The
trained RGB residual is multiplied by a bounded gain:

```text
gain = clamp((ISO / 6400)^(1/3), 1.0, 1.6)
condition = (gain - 1.0) / 0.6
```

The gain is applied inside the exported graph. The held-out correction
projection is:

| ISO | Projection |
| ---: | ---: |
| 1600 | 85.10% |
| 3200 | 84.69% |
| 6400 | 81.40% |
| 12800 | 86.21% |
| 25600 | 78.81% |
| 51200 | 80.24% |
| **Mean** | **82.74%** |

Export: `export/litedenoise_w24_v4_iso.tflite`

- Size: 17,687,820 bytes
- SHA-256: `21d20ee04352c37011cb878cd578ac3f8fef20bdd03938134d0101b1d8d92085`
- PyTorch/LiteRT maximum numerical error: 1.37e-6

Audit outputs and metrics are in
`verification/litedenoise-w24-v4-iso-calibrated`.

## Remaining deployment work

Android and iOS must read ISO from EXIF, fill the fourth input plane with the
condition value, and fall back to zero when ISO is unavailable. The conditioned
graph must be compiled and timed on each target accelerator before replacing
the current performance model in a release build.
