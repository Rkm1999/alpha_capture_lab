# SCUNet Mobile Distillation, Exact 192 px

> **Archived study:** the production decision is the width-24 residual-adapter
> model. Large datasets, repeated checkpoints, logs, and rendered evaluations
> have been removed. Reusable code/configs remain here; compact run evidence,
> the selected checkpoint/export, and conclusions are in
> [`../archive`](../archive) and
> [`../EXPERIMENT_RETROSPECTIVE.md`](../EXPERIMENT_RETROSPECTIVE.md).

This directory is an isolated implementation of
`SCUNet_Mobile_Distillation_192x192_Guide.pdf`. It does not import prior
width-24 checkpoints, custom residual/frequency losses, EMA weights, shadow
weighting, correction samplers, or unpaired camera labels.

## Controlled Experiment

- Frozen teacher: `scunet_color_real_psnr`, exact 192 x 192 input
- Student: fixed width-16 LiteDenoiseNet, 1,963,411 parameters
- Input: NCHW RGB float in `[0, 1]`; deployment: fixed NHWC
- Optimizer: Adam, `1e-4`, cosine to `1e-5`, 200 epochs
- Batch size: 16; AMP on CUDA; global gradient clip: 0.1
- Validation: per-image RGB PSNR and Gaussian SSIM after a one-pixel crop
- Runs: independent random initialization at alpha `0.0`, `0.7`, and `0.9`

The local public-data study uses MIDD and SIDD paired images. It deliberately
excludes the existing A6300/Sony records because those records use SCUNet as
both `clean` and `teacher` and are not clean-paired samples. A final product
decision still requires clean-paired target-camera scenes as specified by the
guide.

The 15% deterministic scene validation split and the 8 SIDD / 2 MIDD native
patches per source pair are local experiment choices; the guide does not
prescribe a split ratio or patch count. Every source filename and crop
coordinate is retained in the generated manifest.

## Reproduce

Use the existing CUDA training environment on this workstation:

```bash
PYTHON=/home/ryu/.cache/scunet-int8-venv/bin/python

$PYTHON scripts/model_report.py
$PYTHON scripts/prepare_dataset.py --replace
$PYTHON scripts/validate_dataset.py
$PYTHON scripts/test_smoke.py

$PYTHON scripts/train.py --alpha 0.0
$PYTHON scripts/train.py --alpha 0.7
$PYTHON scripts/train.py --alpha 0.9

$PYTHON scripts/evaluate.py --require-paper-matrix \
  --checkpoint alpha_0p0=runs/alpha_0p0/best.pt \
  --checkpoint alpha_0p7=runs/alpha_0p7/best.pt \
  --checkpoint alpha_0p9=runs/alpha_0p9/best.pt

$PYTHON scripts/visual_review.py \
  --checkpoint alpha_0p0=runs/alpha_0p0/best.pt \
  --checkpoint alpha_0p7=runs/alpha_0p7/best.pt \
  --checkpoint alpha_0p9=runs/alpha_0p9/best.pt \
  --summary evaluation/matrix/summary.json \
  --per-image evaluation/matrix/per_image.jsonl \
  --output-dir evaluation/matrix/visual_review
```

Run the unpaired camera ISO set as a separate visual and teacher-agreement
study. This produces full-resolution outputs, twelve native-detail crop sheets
per ISO, and twelve cross-ISO region sheets:

```bash
$PYTHON scripts/validate_iso_set.py \
  --input /home/ryu/projects/iso_test_image \
  --checkpoint runs/alpha_0p7/best.pt \
  --output-dir evaluation/iso_test_alpha_0p7
```

The current rendered index is
[`evaluation/iso_test_alpha_0p7/README.md`](evaluation/iso_test_alpha_0p7/README.md).
It intentionally reports student/teacher agreement rather than clean-reference
quality because these JPEGs have no paired clean targets.

## Sony-Adjacent High-ISO Data Gate

The candidate target-domain gate is isolated from the completed public-data
experiment. It combines all eight full-resolution PolyU Sony A7 II JPEG pairs
with the local NIND ISO 6400 and H-series sequences. NIND H1-H4 are
exposure-corrected underexposures captured at the camera's maximum ISO; they
are not relabeled as literal ISO 12800-51200 samples.

PolyU uses paired noisy/burst-mean supervision. NIND's base-ISO image is kept
for alignment and visual diagnostics, but NIND is marked teacher-only because
its reference can retain noise and differs from an in-camera Sony JPEG target.
The only local CC BY-SA NIND scene is excluded; retained NIND inputs are CC BY
or public domain. PolyU remains subject to the separate permission reported by
the project owner, whose written terms should be archived outside this repo.

Build and inspect the candidate gate without changing `data/cache` or the
completed baseline runs:

```bash
PYTHON=/home/ryu/.cache/scunet-int8-venv/bin/python

$PYTHON scripts/build_high_iso_manifest.py
$PYTHON scripts/prepare_dataset.py \
  --config configs/high_iso_data_gate.yaml --replace
$PYTHON scripts/validate_high_iso_gate.py \
  --config configs/high_iso_data_gate.yaml
```

The visual decision artifacts and current measurements are in
[`evaluation/high_iso_data_gate/README.md`](evaluation/high_iso_data_gate/README.md).
This is a pending data gate, not permission to start a new training run.

Export is intentionally separate because LiteRT Torch has different dependency
constraints:

```bash
/home/ryu/.cache/ainr-export-venv/bin/python scripts/export_litert.py \
  --checkpoint runs/alpha_0p7/best.pt \
  --precision fp32 \
  --output export/scunet_student_alpha_0p7_192_fp32.tflite

/home/ryu/.cache/ainr-export-venv/bin/python scripts/export_litert.py \
  --checkpoint runs/alpha_0p7/best.pt \
  --precision fp16 \
  --output export/scunet_student_alpha_0p7_192_fp16.tflite
```

The existing Android compositor uses planar NCHW tensors and clips model output
while converting it back to image bytes. Export its Performance model with that
contract and without the redundant terminal clamp so the full graph can remain
in one GPU delegate partition:

```bash
/home/ryu/.cache/ainr-export-venv/bin/python scripts/export_litert.py \
  --checkpoint evaluation/domain_expansion/uhd_snic_alpha_0p7_final/best_snapshot.pt \
  --config configs/uhd_snic_alpha_0p7.yaml \
  --precision fp32 \
  --layout nchw \
  --omit-output-clamp \
  --output export/litedenoise_uhd_snic_alpha_0p7_e195_192_nchw_fp32.tflite
```

## W32 W8A8 QAT

The Android and Apple integer models branch independently from the completed
epoch-200 W32 float checkpoint. Both fine-tune all weights while simulating
8-bit weights and activations in the forward pass. They do not share observer
ranges because LiteRT and Core ML use different quantization backends.

Android uses LiteRT PT2E QAT with signed per-tensor INT8 activations and signed
per-channel INT8 convolution weights. Observers freeze at epoch 26. The export
step recalibrates deployment activation ranges from the QAT-adapted weights,
then rejects the artifact unless its I/O and every convolution are integer and
the graph contains no float tensor or `DEQUANTIZE` operation:

```bash
FLOAT=runs/general_camera_v15_chroma/general_camera_v84_w32_int8_scratch/epoch_200.pt
ANDROID_QAT=runs/general_camera_v15_chroma/general_camera_v85_w32_int8_qat

/home/ryu/.cache/ainr-export-venv/bin/python scripts/train_mixed.py \
  --config configs/general_camera_v85_w32_int8_qat.yaml \
  --init-checkpoint "$FLOAT" \
  --disable-early-stopping

/home/ryu/.cache/ainr-export-venv/bin/python scripts/export_litert_int8.py \
  --checkpoint "$ANDROID_QAT/epoch_030.pt" \
  --output export/litedenoise_w32_w8a8_qat_192_nhwc.tflite
```

Apple uses Core ML Tools `LinearQuantizer` QAT with per-channel signed INT8
weights and per-tensor unsigned INT8 activations. Its fixed 192 px graph uses
static scale-factor resizes so conversion cannot introduce dynamic shape
fallbacks. The exporter finalizes the learned fake-quantization graph and
rejects the ML Program unless it contains activation quantize/dequantize
operations and compressed INT8 weights. Activation quantization requires iOS
17 or newer; INT8-INT8 Neural Engine acceleration targets A17 Pro or newer:

```bash
APPLE_QAT=runs/general_camera_v15_chroma/general_camera_v86_w32_coreml_int8_qat

/home/ryu/.cache/ainr-coreml-venv/bin/python scripts/train_mixed.py \
  --config configs/general_camera_v86_w32_coreml_int8_qat.yaml \
  --init-checkpoint "$FLOAT" \
  --disable-early-stopping

/home/ryu/.cache/ainr-coreml-venv/bin/python scripts/export_coreml_int8_qat.py \
  --checkpoint "$APPLE_QAT/epoch_030.pt" \
  --output export/litedenoise_w32_w8a8_qat_192.mlpackage
```

QAT checkpoints retain both the deployable float parameter state in `model`
and the complete fake-quantized graph state in `qat_model`. The latter is
required to resume training or finalize the platform-specific artifact.

Each training run writes `status.json`, `history.jsonl`, `best.pt`, `last.pt`,
and a checkpoint every ten epochs. `best.pt` is selected only by validation
clean-target PSNR.

## Current Result

All three controlled 200-epoch runs are complete. Alpha `0.7` is the selected
research checkpoint: it gives the best balance of SCUNet agreement and
clean-reference fidelity, including the strongest teacher agreement on the 64
darkest validation patches. The full metrics, category review, decision rule,
and artifact hashes are recorded in
[`evaluation/matrix/SELECTION.md`](evaluation/matrix/SELECTION.md).

The subsequent NIND high-ISO ablation is also complete. The NIND
full-reference run is the stronger new research candidate on the combined
reference validation set, but it is effectively tied with the original alpha
`0.7` checkpoint on the six full-resolution Sony ISO images. See the complete
three-way validation, ISO comparison, visual artifacts, provenance, and limits
in
[`evaluation/high_iso_ablation/RESULTS.md`](evaluation/high_iso_ablation/RESULTS.md).

## UHD-LL And SNIC Domain Expansion

The next fresh run adds all 2,150 UHD-LL image pairs and the Sony A7R III
portion of SNIC. SNIC uses every indoor/outdoor archive, ISO 1600, 3200, 6400,
and 12800, and exact native-resolution crops. SNIC keeps its native paired
reference behavior.

UHD-LL uses a denoising-specific hybrid target. A 480 px thumbnail estimates a
per-channel clean-to-noisy illumination field in linear RGB using a Gaussian
equivalent to sigma 128 at source resolution. Each exact crop is mapped to the
input exposure, then combined with the frozen SCUNet output as:

```text
target_linear = Gaussian8(teacher_linear)
              + mapped_clean_linear
              - Gaussian8(mapped_clean_linear)
```

This keeps SCUNet's brightness, color, and coarse denoising while replacing
its medium/fine bands with clean captured detail. The detail multiplier is
fixed at `1.0`. The hybrid is built from the exact float16 teacher tensor kept
in the cache, so the release validator can reconstruct the target without a
second inference. In the 160-pair hard-case prototype this reduced median
low-frequency drift by 8.8x and structured correction by 21.2x versus the old
global affine target, while retaining `0.9998x` clean-detail energy.

Clean supervision is decided per crop, not per source image. The gate compares
signed gradients over shifts from `-3` through `+3` pixels after a sigma `0.7`
prefilter. It requires texture `>= 0.012`, zero-shift correlation `>= 0.50`,
and rejects any nonzero shift improving correlation by more than `0.018`.
Passing rows use `GT=1.0, KD=0.7`; shifted, flat, or unverifiable rows remain
full-strength teacher-only samples with `GT=0.0, KD=1.0`. The deliberately hard
prototype accepted 113/160 crops (70.6%); that is a conservative diagnostic,
not an assumed full-dataset pass rate.

Build, audit, combine, train, and monitor the run with:

```bash
PYTHON=/home/ryu/.cache/scunet-int8-venv/bin/python

uv run --with 'remotezip>=0.12,<1' --with 'tqdm>=4.66,<5' \
  python scripts/download_snic_subset.py
$PYTHON scripts/build_domain_manifest.py
$PYTHON scripts/prepare_domain_dataset.py --gate-only
$PYTHON scripts/prepare_domain_dataset.py --replace
$PYTHON scripts/validate_domain_gate.py
$PYTHON scripts/build_mixed_cache.py --replace
$PYTHON scripts/test_mixed_smoke.py
$PYTHON scripts/train_mixed.py --config configs/uhd_snic_alpha_0p7.yaml
$PYTHON scripts/watch_mixed.py --interval 5
```

The full gate reconstructs every UHD crop, local gain field, alignment
decision, cached input, and hybrid target from the immutable source pair.
Smoke mode reconstructs only its stratified sample and is explicitly not a
release gate. The mixed manifest also records and validates the legacy and
domain preprocessing versions separately before training can start.

The mixed sampler gives equal configured influence to dataset domains rather
than allowing UHD-LL's image count to dominate. Checkpoint selection is the
equal mean of validation PSNR for MIDD, SIDD, PolyU Sony, UHD-LL, and SNIC;
teacher-only UHD rows are excluded from clean-reference metrics.

## Targeted High-ISO Continuation

The targeted experiment continues from the selected epoch-195 model with a
fresh optimizer, scheduler, scaler, and deterministic RNG stream. The control
keeps the normalized real-domain sampling mix. The synthetic arm assigns an
exact configured probability of `0.12` to `synthetic_camera_jpeg` and scales
every real-domain probability by `0.88`. Within SNIC training data, ISO 12800
has the strongest record multiplier; validation remains uniformly weighted.

Synthetic data cannot enter a mixed training cache until the full generator
and calibration are non-smoke, the release gate is clean, and the rendered
contact sheet has a hash-bound visual acceptance. The acceptance command does
not rerun validation or redraw the sheets: it requires the SHA-256 of the exact
pending report that was inspected and signs that report and artifact set. The
analysis also records the active validator code SHA and semantic contract, so a
validator change invalidates an older approval. The builder independently
hashes every input, clean target, and teacher tensor and rejects stale,
provisional, smoke, or unsigned caches. The synthetic arm requires
`synthetic_camera_jpeg_linear_post_isp_covariance_v3`; this includes calibrated
post-ISP band covariance and per-array integrity metadata, so older caches fail the
preprocessing contract instead of being reused silently. Generation and gate
validation also require the profile's recorded fitter SHA-256 to match the active
fitter code, preventing a stale same-version calibration profile from being reused.
The release analysis likewise binds the manifest's generator path and SHA-256 to
the active generator implementation, so an older generator cannot pass under the
same preprocessing label.
Before training, the
consumer hashes every synthetic mixed-cache tensor again and reconstructs the
accepted cache identity. It then keeps a read-only descriptor for every verified
synthetic array. DataLoader workers read through those pinned inodes and hash the
exact bytes passed to NumPy, so an atomic path replacement still consumes the
accepted payload while an in-place mutation stops training. This low-memory
strategy requires Linux procfs and enough file descriptors for three arrays per
synthetic record; it buffers only the array currently being decoded rather than
the full cache. Training also opens and snapshots the mixed manifest exactly once;
array preflight, both dataset splits, the training contract, the run fingerprint,
and environment provenance all consume those same bytes. Replacing the published
cache during startup therefore cannot bind pinned arrays to different manifest
metadata. Legacy and real-domain arrays predate per-record hashes and remain
outside this additional consumer-side integrity guarantee.

```bash
PYTHON=/home/ryu/.cache/scunet-int8-venv/bin/python
E195=evaluation/domain_expansion/uhd_snic_alpha_0p7_final/best_snapshot.pt

$PYTHON scripts/fit_synthetic_noise_profiles.py \
  --config configs/synthetic_camera_jpeg_gate.yaml --replace
$PYTHON scripts/prepare_synthetic_camera_jpeg.py \
  --config configs/synthetic_camera_jpeg_gate.yaml --replace
$PYTHON scripts/validate_synthetic_camera_jpeg.py \
  --config configs/synthetic_camera_jpeg_gate.yaml
REPORT=data/synthetic_camera_jpeg/synthetic_camera_jpeg_gate_report.json
REPORT_SHA=$(sha256sum "$REPORT" | awk '{print $1}')

# Inspect both generated contact sheets before signing this exact artifact set:
# data/synthetic_camera_jpeg/synthetic_camera_jpeg_contact_sheet.png
# data/synthetic_camera_jpeg/synthetic_camera_jpeg_contact_sheet.jpg
$PYTHON scripts/validate_synthetic_camera_jpeg.py \
  --config configs/synthetic_camera_jpeg_gate.yaml \
  --accept-visual --reviewer "$USER" \
  --accept-report-sha256 "$REPORT_SHA"

$PYTHON scripts/build_mixed_cache.py \
  --config configs/high_iso_targeted_control.yaml \
  --output-root data/high_iso_targeted_control/cache --replace
$PYTHON scripts/build_mixed_cache.py \
  --config configs/high_iso_targeted_synthetic.yaml \
  --synthetic-root data/synthetic_camera_jpeg/cache \
  --output-root data/high_iso_targeted_synthetic/cache --replace

$PYTHON scripts/train_mixed.py \
  --config configs/high_iso_targeted_control.yaml \
  --init-checkpoint "$E195"
$PYTHON scripts/train_mixed.py \
  --config configs/high_iso_targeted_synthetic.yaml \
  --init-checkpoint "$E195"
```

Both runs report the unchanged general clean-reference PSNR and a real-only
high-ISO target score. The target score equally combines NIND and SNIC ISO
12800, measuring teacher-correction capture in shadow pixels and in Gaussian
`G1-G4` medium and `G4-G12` coarse bands. Samples without shadow pixels are
excluded from the shadow component and reported through explicit contributor
counts. Synthetic validation records are excluded. A model-only continuation
validates epoch 0 before training, so the source checkpoint remains eligible
for both rankings and seeds early stopping without spending patience.
`general-best.pt` tracks the highest general PSNR, while
`target-best.pt` tracks the highest target score only while general PSNR stays
at or above `38.30` dB. `best.pt` remains a compatibility alias for the
general checkpoint. Early stopping follows the target score; `status.json`
and `history.jsonl` retain both metrics and the general guardrail result.

## Data And Release Limits

- Fresh input/clean cache tensors are float32; teacher tensors are generated by
  full-precision SCUNet inference and stored as float16. UHD hybrid targets are
  constructed from that stored-precision teacher tensor.
- Splits are made by dataset/scene before crop generation.
- MIDD's local license is CC BY-NC-SA 4.0. Treat the resulting checkpoints as
  research artifacts unless separate distribution and commercial rights are
  confirmed.
- There is currently no clean-paired target-camera validation set. Public-data
  PSNR/SSIM cannot substitute for that final guide gate.
