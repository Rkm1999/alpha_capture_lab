# Model Selection Record

Evaluation ID: `21cae7fc078cd4b77afb55cfacdfbb0ef99cf773d98f9330eaa0c4a268923489`

This record covers the guide's controlled alpha matrix on 654 held-out public
MIDD/SIDD patches. Every run used an independent random initialization, 200
epochs, the same scene split, and the same exact 192 x 192 input pipeline.

## Clean-Reference Metrics

| Model | PSNR | SSIM | Minimum PSNR |
| --- | ---: | ---: | ---: |
| Noisy input | 32.9129 | 0.717783 | 15.4114 |
| SCUNet teacher | 39.6446 | 0.943655 | 20.1308 |
| Alpha 0.0 | **41.5514** | **0.956773** | **29.4692** |
| Alpha 0.7 | 41.3805 | 0.955970 | 26.7537 |
| Alpha 0.9 | 41.2340 | 0.955063 | 24.9869 |

The paired references retain visible noise in some severe SIDD patches, so
clean-reference PSNR alone favors copying that residual noise. Teacher agreement
and visual review are therefore included in the decision.

## Teacher Agreement

| Scope | Model | Teacher PSNR | Teacher SSIM |
| --- | --- | ---: | ---: |
| All 654 | Alpha 0.0 | 42.6962 | 0.976503 |
| All 654 | Alpha 0.7 | 43.7475 | **0.981954** |
| All 654 | Alpha 0.9 | **43.8156** | 0.980821 |
| 64 noisiest | Alpha 0.7 | 34.3845 | **0.947663** |
| 64 noisiest | Alpha 0.9 | **34.8201** | 0.944449 |
| 64 darkest | Alpha 0.7 | **43.9701** | **0.952688** |
| 64 darkest | Alpha 0.9 | 43.8790 | 0.949824 |

## Visual Gate

The original-pixel sheets in [`visual_review`](visual_review) cover severe
noise, darkest inputs, strongest texture, and each student's largest teacher
disagreements. No student shows tiling, color shifts, ringing, hallucinated
edges, or material texture erasure. Alpha 0.9 is marginally smoother on severe
flat regions, but gives up more clean-reference detail. None of the students
fully reproduces SCUNet's cleanup on the hardest SIDD samples.

## Decision

Select **alpha 0.7, epoch 181** as the balanced distilled research checkpoint.
It gains 1.05 dB teacher agreement over alpha 0.0 across the full validation set
for only a 0.17 dB clean-reference PSNR tradeoff, and it is the strongest model
on the darkest teacher-agreement category. Alpha 0.9 does not provide enough
visual benefit to justify its larger clean-reference loss.

This is not a product-release gate. There is no clean-paired target-camera
validation set, and the local MIDD data is CC BY-NC-SA 4.0. Target-camera
validation and a distribution/license decision remain required.

## Export Artifacts

| Artifact | Size | SHA-256 | Maximum parity error |
| --- | ---: | --- | ---: |
| `scunet_student_alpha_0p7_192_fp32.tflite` | 7,891,096 bytes | `bbf93f9396bfa20b1cb9889e81369bbfa8b0a266bf02d04b1cb38da6007a0c79` | 0.0000008345 |
| `scunet_student_alpha_0p7_192_fp16.tflite` | 3,975,104 bytes | `11a1bafef3f3518786d2559161b8fbe58e3c7ca7e9ff3dfb27510255371be510` | 0.0005657077 |

Both models have static float32 NHWC input and output tensors shaped
`[1, 192, 192, 3]`. FP16 refers to weight storage; computation and I/O remain
float32. FP32 uses a `1e-4` parity gate and FP16 uses a `1e-3` gate.
