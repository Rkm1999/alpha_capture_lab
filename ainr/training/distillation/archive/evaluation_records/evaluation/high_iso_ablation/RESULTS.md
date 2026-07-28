# NIND High-ISO Ablation Results

## Decision

Carry `nind_full_reference/best.pt` forward as the NIND research candidate.
It is stronger than the teacher-only variant on clean/reference validation and
recovers the teacher-only variant's loss on the external Sony ISO set.

This is not yet an unconditional replacement for the original `alpha_0p7`
checkpoint. Full-reference and alpha 0.7 are effectively tied on the six
targetless Sony JPEGs, and every student still leaves substantially more
high-ISO chroma noise and grain than SCUNet.

Selected checkpoint:

- Path: `runs/high_iso_ablation/nind_full_reference/best.pt`
- Epoch: 199 of 200
- SHA-256: `f6db8904fb47d545841d02171e8599f8f08e82ac69420d270158f86de575cf03`
- Student: width-16 LiteDenoiseNet, 1,963,411 parameters, fixed 192 x 192 input

Do not use `last.pt`: the selection PSNR fell from 41.09483 at epoch 199 to
41.01760 at epoch 200.

## Experiment

The two new runs start from the same initialization and use the same seed,
manifest, data order, teacher, alpha, optimizer, schedule, and selection
datasets. They differ only in how NIND is supervised:

| Run | NIND ground-truth loss | NIND teacher loss | Best epoch |
| --- | ---: | ---: | ---: |
| Teacher-only | disabled | 700 x KD MSE | 174 |
| Full-reference | 300 x GT MSE + 50 x GT L1 | 700 x KD MSE | 199 |

The original alpha 0.7 checkpoint is included as a historical baseline, not a
causal ablation. It was trained only on the legacy MIDD+SIDD cache, whereas the
new runs add PolyU Sony and NIND.

## Reference Validation

All three checkpoints were evaluated on the identical 850-image combined
validation set. Values are RGB PSNR in dB / SSIM against the dataset reference.

| Dataset | Alpha 0.7 | Teacher-only | Full-reference |
| --- | ---: | ---: | ---: |
| MIDD | 41.3060 / .963058 | 41.2881 / .963403 | **41.3094** / .963268 |
| NIND | 31.9329 / .824542 | 31.4910 / .804652 | **32.4317 / .848923** |
| PolyU Sony | 34.7138 / **.969456** | 34.4764 / .965484 | **34.7920** / .968887 |
| SIDD | 41.4444 / .949888 | 41.4087 / .949724 | **41.4837 / .950174** |
| All 850 | 39.3067 / .931119 | 39.1913 / .927187 | **39.4234 / .935995** |

Relative to alpha 0.7, full-reference gains 0.1167 dB / .004876 SSIM overall
and 0.4988 dB / .024381 SSIM on NIND. NIND uses exposure-corrected lower-ISO
references, so those values must not be interpreted as strict clean-target
quality.

Full-reference also wins reference PSNR and SSIM at every NIND noise level:

| NIND level | Alpha 0.7 | Teacher-only | Full-reference |
| --- | ---: | ---: | ---: |
| 6400 | 34.5000 / .870005 | 34.1075 / .857413 | **34.8774 / .882012** |
| H1 | 31.9491 / .837623 | 31.1954 / .752231 | **32.8019 / .865103** |
| H2 | 29.3781 / .781179 | 28.9821 / .805340 | **29.7083 / .818426** |
| H3 | 25.7227 / .686324 | 25.6517 / .708039 | **26.2107 / .733521** |

Machine-readable results:

- [Controlled pair summary](comparison/summary.json)
- [Alpha 0.7 combined-validation summary](alpha_0p7_combined_validation/summary.json)
- [Controlled validation contact sheet](comparison/comparison_contact_sheet.png)

## Sony ISO Test

The six 24 MP Sony JPEGs have no clean targets. The following metrics therefore
measure agreement with SCUNet, not denoising quality against ground truth. All
models used the same 192 px tile, 8 px pad, 176 px core, and whole-tile weighted
overlap configuration.

| ISO | Alpha 0.7 | Teacher-only | Full-reference |
| ---: | ---: | ---: | ---: |
| 1600 | 43.971 | 44.150 | **44.299** |
| 3200 | **42.500** | 42.039 | 42.436 |
| 6400 | **39.213** | 38.307 | 39.039 |
| 12800 | **37.474** | 37.044 | 37.356 |
| 25600 | 32.723 | 32.619 | **32.728** |
| 51200 | 29.881 | **29.925** | 29.911 |
| Mean | 37.627 | 37.347 | **37.628** |

Full-reference versus alpha 0.7 is a tie: +0.0011 dB mean PSNR and -0.000421
mean preview SSIM. Full-reference versus teacher-only improves by 0.2807 dB
and .002597 SSIM. Native-crop inspection found no new seams or structural
artifacts, but all three students retain visible coarse color noise and fine
grain at ISO 25600 and 51200 relative to SCUNet.

Visual and machine-readable comparisons:

- [Three-model ISO comparison](iso_arm_comparison/README.md)
- [ISO 51200 native comparison](iso_arm_comparison/per_iso/ISO51200_DSC00006.png)
- [Dark-region comparison across ISO levels](iso_arm_comparison/by_region/01_dark_shelf.png)
- [ISO comparison metrics](iso_arm_comparison/metrics.json)

## Reproduce

From `paper_192`, after defining `PYTHON` as described in the project README:

```bash
$PYTHON scripts/evaluate_ablation.py
$PYTHON scripts/evaluate_alpha_combined.py
$PYTHON scripts/validate_iso_set.py \
  --config configs/high_iso_ablation_teacher_only.yaml \
  --checkpoint runs/high_iso_ablation/nind_teacher_only/best.pt \
  --output-dir evaluation/high_iso_ablation/nind_teacher_only_iso_test
$PYTHON scripts/validate_iso_set.py \
  --config configs/high_iso_ablation_full_reference.yaml \
  --checkpoint runs/high_iso_ablation/nind_full_reference/best.pt \
  --output-dir evaluation/high_iso_ablation/nind_full_reference_iso_test
$PYTHON scripts/compare_iso_arms.py
```

The original alpha ISO render was reused only after verifying the checkpoint,
teacher, all six input images, tiling configuration, and byte-identical teacher
outputs. The comparison manifest records hashes for every input and artifact.

## Limits

- Both new runs use one seed; there are no confidence intervals.
- NIND contains 19 scenes and PolyU Sony contains 2 scenes in this cache.
- NIND references can retain noise and are not strict clean targets.
- The Sony ISO set contains six unpaired JPEGs, so it cannot measure true
  reconstruction quality.
- A final deployment decision still requires clean-paired or averaged images
  from the target camera pipeline, especially at ISO 25600 and 51200.
