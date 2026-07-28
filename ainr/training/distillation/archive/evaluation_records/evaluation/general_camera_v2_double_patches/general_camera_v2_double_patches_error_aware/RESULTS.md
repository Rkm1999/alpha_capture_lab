# Doubled-patch error-aware fine-tune

This run doubled training crops for all datasets while retaining the original
validation partition byte-for-byte. It initialized model weights from the
previous stage-two target checkpoint and reset optimizer/scheduler/scaler state.

## Data

| Dataset | Previous training patches | Doubled training patches |
| --- | ---: | ---: |
| MIDD | 15,838 | 31,676 |
| PolyU | 544 | 1,088 |
| Renoir | 392 | 784 |

The difficulty index included current-student shadow and medium/coarse errors
against SCUNet. Dataset draw probability remained one third per dataset.

## Frozen paired validation

| Checkpoint | PSNR | SSIM | Target | Shadow | Medium/coarse |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous target | 38.71493 | 0.901805 | 0.754470 | 0.725878 | 0.783061 |
| Doubled target | 38.77892 | 0.906422 | 0.757215 | 0.728758 | 0.785673 |
| Doubled general | 38.85906 | 0.909012 | 0.753819 | 0.722986 | 0.784653 |

## Held-out ISO shadow capture

| ISO | Previous target | Doubled target | Change |
| ---: | ---: | ---: | ---: |
| 1600 | 0.8907 | 0.8776 | -0.0130 |
| 3200 | 0.8237 | 0.8076 | -0.0161 |
| 6400 | 0.7181 | 0.6867 | -0.0314 |
| 12800 | 0.6648 | 0.6375 | -0.0273 |
| 25600 | 0.4928 | 0.4650 | -0.0278 |
| 51200 | 0.5476 | 0.5125 | -0.0352 |

## Held-out ISO medium/coarse capture

| ISO | Previous target | Doubled target | Change |
| ---: | ---: | ---: | ---: |
| 1600 | 0.9194 | 0.9088 | -0.0106 |
| 3200 | 0.8778 | 0.8685 | -0.0093 |
| 6400 | 0.8400 | 0.8405 | +0.0006 |
| 12800 | 0.8164 | 0.8155 | -0.0009 |
| 25600 | 0.7493 | 0.7644 | +0.0152 |
| 51200 | 0.7466 | 0.7752 | +0.0286 |

## Decision

Do not promote the doubled-patch checkpoint as the default model. It improves
paired-reference fidelity and high-ISO medium/coarse agreement, but consistently
reduces held-out shadow correction. Keep the previous stage-two target checkpoint
as the selected general model until a training change recovers shadow transfer.

Doubled target SHA-256:
`36b4bd68de11e1531046443ec3bfd9e500dc161aa3bfca5f2d086bc79e1e5122`
