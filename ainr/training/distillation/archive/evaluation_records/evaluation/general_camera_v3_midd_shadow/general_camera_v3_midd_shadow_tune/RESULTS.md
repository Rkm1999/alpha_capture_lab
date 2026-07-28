# MIDD Shadow-Focused Fine-Tune

## Goal

Test whether dark, high-noise MIDD samples improve the student's weak shadow
denoising on the held-out ISO 1600-51200 camera JPEG set.

MIDD's prepared metadata and source images do not contain usable ISO values.
The selection is therefore a **high-noise/dark proxy**, not an ISO-labelled
subset.

## Dataset Selection

`scripts/build_midd_shadow_manifest.py` evaluates the existing paired MIDD
patches and selects the darkest high-noise crop per scene. Candidates must have
at least 25% shadow pixels. They are ranked independently for each camera using:

- 55% shadow high-frequency noisy-to-clean residual
- 35% SCUNet shadow correction magnitude
- 10% shadow coverage

The top 25% for each of 20 camera sensors were added as teacher-only
`midd_shadow` records:

| Split | Selected patches |
| --- | ---: |
| Train | 2,470 |
| Validation | 280 |

The fine-tune sampled `midd_shadow` at 30%; PolyU, RENOIR, and ordinary MIDD
each received 23.33%. It ran for all 60 configured epochs from the previous
selected checkpoint.

## Paired Validation

| Checkpoint | PSNR | SSIM | Four-domain target | Shadow capture | Medium/coarse capture |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous selected | 38.7149 | 0.901805 | 0.78005 | 0.75784 | 0.80225 |
| MIDD shadow epoch 60 | 38.6460 | 0.901741 | 0.78579 | 0.76600 | 0.80559 |

The adapted model improved the four-domain target by 0.00574, including MIDD
shadow capture from 0.85374 to 0.85905. It lost 0.069 dB PSNR, so no adapted
epoch passed the configured 38.70 dB selection guardrail. `target-best.pt`
therefore remains the unmodified starting checkpoint.

## Held-Out Camera JPEGs

Shadow correction captured relative to SCUNet:

| ISO | Previous selected | Epoch 60 | Delta |
| ---: | ---: | ---: | ---: |
| 1600 | 0.89067 | 0.88950 | -0.00118 |
| 3200 | 0.82371 | 0.81481 | -0.00890 |
| 6400 | 0.71812 | 0.69455 | -0.02357 |
| 12800 | 0.66480 | 0.64426 | -0.02053 |
| 25600 | 0.49283 | 0.47588 | -0.01695 |
| 51200 | 0.54764 | 0.53950 | -0.00814 |

Medium/coarse correction captured relative to SCUNet:

| ISO | Previous selected | Epoch 60 | Delta |
| ---: | ---: | ---: | ---: |
| 1600 | 0.91939 | 0.91433 | -0.00506 |
| 3200 | 0.87779 | 0.87089 | -0.00690 |
| 6400 | 0.83999 | 0.84186 | +0.00187 |
| 12800 | 0.81642 | 0.81724 | +0.00082 |
| 25600 | 0.74926 | 0.76678 | +0.01752 |
| 51200 | 0.74658 | 0.78022 | +0.03364 |

At ISO 25600-51200, medium/coarse capture improved by 0.02558 on average, but
shadow capture fell by 0.01255. The MIDD proxy is useful for coarse-noise
training, but it does not match the processed camera-JPEG shadow distribution
well enough to replace the previous model.

## Decision

Keep
`runs/general_camera_v1_no_namcc/general_camera_v1_stage2_hard_shadow/target-best.pt`
as the selected model.

The next shadow-focused dataset should contain real high-ISO processed JPEG
inputs with clean references, or synthetic samples calibrated to those JPEG
shadow residuals. Increasing the weight of this MIDD proxy is not supported by
the held-out results.

## Artifacts

- Config: `configs/general_camera_v3_midd_shadow_tune.yaml`
- Selection manifest:
  `data/general_camera_v2_double_patches/cache/manifest_midd_shadow.json`
- Candidate checkpoint: `runs/general_camera_v3_midd_shadow/general_camera_v3_midd_shadow_tune/last.pt`
- Candidate SHA-256:
  `13871510a74f1c3bd193da7033fe9df6e3cb02e305c1243acf1b9d755c01199b`
- Paired metrics: `paired_validation.json`
- ISO metrics and images: `iso_last/`
