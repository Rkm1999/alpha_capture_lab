# Stage-two hard-shadow fine-tune

The run fine-tuned the previous epoch-175 target checkpoint for 50 epochs at
5e-6 to 1e-6. Training used equal dataset sampling, within-dataset hard-patch
quartiles, normalized shadow correction loss, and paired-reference gradient
loss. The held-out ISO set was not used for training or checkpoint selection.

## Paired validation

| Checkpoint | Epoch | PSNR | SSIM | Target score | Shadow capture | Medium/coarse capture |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Source target | 175 | 38.70450 | 0.900934 | 0.752930 | 0.723431 | 0.782429 |
| Stage-two target | 19 | 38.71493 | 0.901805 | 0.754470 | 0.725878 | 0.783061 |
| Stage-two general | 26 | 38.76382 | 0.902771 | 0.752785 | 0.722761 | 0.782809 |

The stage-two target checkpoint passes the 38.70 PSNR and 0.900 SSIM
guardrails. It improves all paired metrics over the source target checkpoint.

## Held-out ISO shadow capture

| ISO | Source target | Stage-two target | Change |
| ---: | ---: | ---: | ---: |
| 1600 | 0.8821 | 0.8907 | +0.0086 |
| 3200 | 0.8152 | 0.8237 | +0.0085 |
| 6400 | 0.7078 | 0.7181 | +0.0104 |
| 12800 | 0.6488 | 0.6648 | +0.0160 |
| 25600 | 0.4666 | 0.4928 | +0.0262 |
| 51200 | 0.5269 | 0.5476 | +0.0207 |

The average ISO 25600/51200 shadow capture rises from 0.4968 to 0.5202. The
planned 0.55 stretch target was not reached, but the improvement is consistent
at every tested ISO and paired-reference quality also increased.

## Selection

Use `target-best.pt` for denoising behavior. Use `best.pt` only when the
highest paired PSNR/SSIM is more important than matching SCUNet's shadow
correction.

Stage-two target SHA-256:
`15b8e17768273e9fb90a92a289a28b5299a69c8d203817c976777217c8cc9802`
