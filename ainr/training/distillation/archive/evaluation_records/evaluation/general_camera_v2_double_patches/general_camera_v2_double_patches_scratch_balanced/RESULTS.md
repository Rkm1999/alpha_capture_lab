# Doubled-patch scratch comparison

This model was trained from random initialization for 200 epochs on the doubled
training cache. It used equal dataset sampling without the stage-two
current-student difficulty index. Validation remained byte-for-byte identical to
the previous runs.

## Frozen paired validation

| Checkpoint | PSNR | SSIM | Target | Shadow | Medium/coarse |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous target | 38.71493 | 0.901805 | 0.754470 | 0.725878 | 0.783061 |
| Doubled fine-tune | 38.77892 | 0.906422 | 0.757215 | 0.728758 | 0.785673 |
| Scratch target | 38.76200 | 0.904586 | 0.765093 | 0.738967 | 0.791219 |
| Scratch general | 38.90984 | 0.906848 | 0.756275 | 0.727449 | 0.785102 |

The scratch target passes both quality guardrails and exceeds the planned 0.765
paired target threshold.

## Held-out ISO shadow capture

| ISO | Previous target | Doubled fine-tune | Scratch target |
| ---: | ---: | ---: | ---: |
| 1600 | 0.8907 | 0.8776 | 0.8758 |
| 3200 | 0.8237 | 0.8076 | 0.7899 |
| 6400 | 0.7181 | 0.6867 | 0.6456 |
| 12800 | 0.6648 | 0.6375 | 0.6181 |
| 25600 | 0.4928 | 0.4650 | 0.4208 |
| 51200 | 0.5476 | 0.5125 | 0.4595 |

Average ISO 25600/51200 shadow capture:

- Previous target: 0.5202
- Doubled fine-tune: 0.4888
- Scratch target: 0.4402

## Held-out ISO medium/coarse capture

| ISO | Previous target | Doubled fine-tune | Scratch target |
| ---: | ---: | ---: | ---: |
| 1600 | 0.9194 | 0.9088 | 0.9375 |
| 3200 | 0.8778 | 0.8685 | 0.8984 |
| 6400 | 0.8400 | 0.8405 | 0.8426 |
| 12800 | 0.8164 | 0.8155 | 0.8258 |
| 25600 | 0.7493 | 0.7644 | 0.7797 |
| 51200 | 0.7466 | 0.7752 | 0.7543 |

## Decision

Do not promote the scratch checkpoint as the default model. It is the strongest
model on frozen paired validation and generally improves medium/coarse ISO
agreement, but it has the weakest held-out shadow correction. The doubled
general-camera data is shifting the model away from the target camera JPEG
shadow distribution.

Keep the previous stage-two target checkpoint as the selected model for now.
Further improvement requires representative high-ISO shadow supervision or a
camera-domain validation term; additional optimization on the present general
datasets is unlikely to fix this transfer gap.

Scratch target SHA-256:
`ae4675d8c26988d28c7e1748ae7e6e004ede3f9781a09c0ccc21c57f1d89ef10`
