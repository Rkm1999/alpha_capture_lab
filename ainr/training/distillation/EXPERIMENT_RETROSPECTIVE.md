# SCUNet mobile distillation retrospective

## Decision

Keep `general_camera_v9_w24_residual_adapter` as the mobile Performance model.
On the held-out Sony ISO 1600-51200 JPEG audit it gave the best visual balance
of denoising, detail, color stability, and mobile compute among the students.

The retained artifacts are in `archive/final_w24/`. The exact experiment is
defined by `paper_192/configs/general_camera_v9_w24_residual_adapter.yaml`.

## What worked

1. **Exact 192 px teacher/student inputs.** Matching the deployed tile size and
   preprocessing avoided the context mismatch seen in earlier experiments.
2. **Response distillation plus paired targets.** Alpha 0.7 was a better
   compromise than clean-only or teacher-dominant alpha 0.9 training.
3. **Real-noise dataset diversity.** MIDD, SIDD, PolyU, NIND, UHD-LL, and SNIC
   improved general behavior more reliably than fitting one camera's ISO set.
4. **Balanced hard-example sampling.** Dataset/scene balancing and explicit
   high-ISO sampling improved shadow and chroma correction without letting a
   large dataset dominate.
5. **Correction-aware validation.** Measuring shadow, medium/coarse,
   chromatic, and row/column correction exposed failures hidden by global
   PSNR and SSIM.
6. **Width 24 with the residual noise adapter.** The estimated strength and
   smooth gate let the model change behavior with observed noise while keeping
   the 4.42 M-parameter backbone practical.
7. **Full-resolution visual audits.** The reserved ISO set caught residual
   grain, blotching, softness, and tile artifacts that patch scores missed.

## What did not justify deployment

1. **Synthetic Gaussian/shot noise alone.** It did not reproduce processed
   high-ISO JPEG chroma blotching or camera-ISP texture.
2. **Teacher-only domain data.** It increased teacher agreement but generally
   underperformed properly aligned noisy/clean supervision.
3. **Aggressive shadow/chroma losses.** These improved target metrics but could
   soften detail, shift color, or optimize the score without producing the
   preferred full-image result.
4. **Repeated narrow fine-tunes.** Chroma heads, profile heads, residual
   scaling, checkpoint soups, and weak-band targeting produced small metric
   gains that did not beat W24 visually.
5. **W32/W40/W48/W64 expansion.** Larger students improved selected target
   scores, but compute and latency grew rapidly. The W32 result retained more
   high-ISO grain than W24 in the final audit.
6. **W32 W8A8 QAT.** Quantization preserved the W32 appearance reasonably well
   (`1.73/255` mean absolute difference from the pre-QAT W32 output), but it
   did not improve S25 NPU latency and did not match W24 image quality.
7. **Full SCUNet on mobile.** It remains the strongest high-ISO denoiser, but
   its approximately 18 M parameters, attention graph, and mobile latency make
   it unsuitable as the Performance model.

## Final ISO audit

Agreement to SCUNet is a diagnostic, not clean-reference quality:

| ISO | W24 PSNR | W32 FP16 PSNR | W32 INT8 QAT PSNR |
|---:|---:|---:|---:|
| 1600 | 48.24 | 48.25 | 40.80 |
| 3200 | 47.70 | 47.35 | 40.71 |
| 6400 | 44.98 | 44.00 | 39.50 |
| 12800 | 42.22 | 41.55 | 38.41 |
| 25600 | 36.06 | 34.72 | 33.79 |
| 51200 | 32.77 | 30.32 | 30.04 |

W24 captured 70.6% and 67.8% of SCUNet's shadow correction magnitude at ISO
25600 and 51200. W32 FP16 captured 62.2% and 55.3%. Full SCUNet still removes
substantially more extreme-ISO chroma grain and blotching.

## Retained reproduction path

1. Download the licensed datasets and SCUNet teacher separately.
2. Rebuild prepared data with `paper_192/scripts/prepare_general_dataset.py`
   and the selected config.
3. Train with `paper_192/scripts/train_mixed.py`.
4. Evaluate clean-reference and ISO behavior with
   `paper_192/scripts/evaluate.py` and `validate_iso_set.py`.
5. Export Android with `export_litert.py --precomputed-noise-gate`.
6. Export Apple with `export_ios_coreml.py --precomputed-noise-gate`.

All alternative configurations remain available for reference, and compact
run/evaluation evidence is under `archive/`.
