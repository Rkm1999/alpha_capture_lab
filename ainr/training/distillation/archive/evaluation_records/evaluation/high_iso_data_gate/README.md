# Sony-Adjacent High-ISO Data Gate

Status: **pending visual review**. No model has been trained from this candidate
cache.

## Review Artifacts

- [`data_gate_contact_sheet.png`](data_gate_contact_sheet.png): all eight PolyU
  Sony scenes plus representative NIND ISO 6400/H1-H4 native-detail crops.
- [`nind_progression_contact_sheet.png`](nind_progression_contact_sheet.png):
  identical scene-stable crops across ISO 6400, H1, H2, and H3.
- [`alignment_review_contact_sheet.png`](alignment_review_contact_sheet.png):
  the 22 crops flagged by low-pass phase correlation.
- [`data_gate_report.json`](data_gate_report.json): complete sample-level
  metrics, provenance hashes, licenses, selections, and alignment exceptions.

Every row is native `192x192` RGB and uses the exact deployment preprocessing.
The columns are noisy input, full-precision SCUNet output, dataset reference,
and an amplified visualization of `SCUNet - noisy` centered at gray.

## Composition

| Source | Source pairs | Cached crops | Supervision |
| --- | ---: | ---: | --- |
| PolyU Sony A7 II | 8 | 128 | Paired noisy/mean |
| NIND ISO 6400/H1-H4 | 170 | 680 | SCUNet teacher only |

The cache contains 808 samples and 86 scene groups. Splits are scene-level with
no leakage. PolyU holds out complete ISO 3200 and ISO 6400 scenes, producing 96
training and 32 validation crops. NIND produces 516 training and 164 validation
crops.

## Automated Result

| Dataset | Mean teacher gain | Teacher better than reference baseline |
| --- | ---: | ---: |
| PolyU Sony | +4.39 dB | 93.0% of crops |
| NIND | +8.37 dB | 99.6% of crops |

The metric compares noisy and SCUNet outputs to each dataset's reference. It
does not prove that the reference is perfectly clean or that SCUNet preserves
the preferred texture.

All 22 phase-correlation exceptions are NIND teacher-only samples, mostly flat
or nearly black regions where registration is weakly observable. PolyU has no
flagged crop. NIND references are not used in the proposed training loss, but
the exception sheet remains part of the visual audit.

## Visual Interpretation

- PolyU transfers subtle denoising behavior from camera-processed Sony JPEGs.
- NIND adds the missing medium/coarse chroma-noise and deep-shadow stress cases.
- SCUNet becomes increasingly aggressive at H2-H4 and can remove fine texture.
  These levels should be teacher-only, sampled conservatively, and evaluated by
  severity instead of dominating batches.
- NIND H labels represent increasing underexposure correction, not literal ISO
  values. The existing Sony ISO 1600-51200 set remains final evaluation only.

Training should begin only after the main sheet and progression sheet are
accepted. The first controlled run should retain the public-data baseline,
mix target-domain samples at a bounded ratio, and report shadow/detail metrics
separately from whole-image PSNR.
