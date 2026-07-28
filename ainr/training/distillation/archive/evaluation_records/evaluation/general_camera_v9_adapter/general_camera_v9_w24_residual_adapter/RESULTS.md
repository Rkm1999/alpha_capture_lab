# Width-24 Noise-Gated Residual Adapter Evaluation

## Checkpoint

The evaluated candidate is the epoch-9 `target-best.pt` from the v9
noise-gated residual-adapter run. The width-24 v6 epoch-124 checkpoint is the
baseline. Both models were evaluated on the same 2,724-row validation manifest
and the same six held-out full-resolution ISO photographs.

## Paired Validation

| Metric | v6 backbone | v9 adapter | Delta |
| --- | ---: | ---: | ---: |
| PSNR | 37.365213 | 37.269629 | -0.095585 dB |
| SSIM | 0.930583 | 0.929075 | -0.001508 |
| Target score | 0.680983 | 0.700080 | +0.019097 |

The candidate remains inside the configured `-0.1 dB` PSNR and `-0.002` SSIM
limits. Its frozen v6 backbone is unchanged; the target gain comes from the
3,039-parameter residual adapter.

## Held-Out ISO Test

The ISO photographs have no clean reference. Correction ratios measure how
much of SCUNet's change the student applies, while teacher-agreement PSNR
measures output similarity to SCUNet.

| ISO | Shadow ratio | Delta | Midtone ratio | Delta | Teacher PSNR | Delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1600 | 0.953887 | -0.000120 | 0.964171 | +0.000428 | 48.2438 | +0.0074 dB |
| 3200 | 0.942492 | -0.000050 | 0.929868 | +0.001279 | 47.6971 | +0.0150 dB |
| 6400 | 0.937812 | +0.000698 | 0.901879 | +0.001531 | 44.9799 | +0.0146 dB |
| 12800 | 0.873696 | +0.000421 | 0.891294 | +0.001316 | 42.2169 | +0.0005 dB |
| 25600 | 0.705947 | +0.000163 | 0.841648 | +0.001500 | 36.0553 | -0.0064 dB |
| 51200 | 0.677708 | +0.005066 | 0.784948 | +0.005081 | 32.7658 | +0.0489 dB |

Average shadow correction rises from `0.847560` to `0.848590`; average midtone
correction rises from `0.883779` to `0.885635`. Average teacher-agreement PSNR
rises from `41.9798` to `41.9931 dB`.

## Runtime And Visual Review

Average workstation runtime is `3.429 s` per 24 MP image versus `3.316 s` for
v6, a 3.4% increase. Composition seam-gradient ratios remain close to `1.0`.

The paired sheet shows no broad detail collapse. The held-out ISO images show
the intended behavior: negligible change at low and moderate ISO, with the
largest measurable gain at ISO 51200. The adapter remains far from SCUNet's
removal of coarse chromatic grain at ISO 25600 and 51200, so this is an
incremental improvement rather than a complete solution.

## Artifacts

- `paired_validation.json`
- `validation_contact_sheet/validation_contact_sheet.jpg`
- `iso_target/report.json`
- `iso_target/ISO*/detail_contact_sheet.jpg`
- Checkpoint SHA-256:
  `ca6fc701080d7d1ef4dc8f988b7d2a1cebda957f9550f7e032d831204ae9dbeb`
