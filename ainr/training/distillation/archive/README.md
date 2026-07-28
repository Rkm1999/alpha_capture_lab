# Distillation archive

This directory is the compact record retained after the July 2026 SCUNet
mobile-distillation study.

## Selected production model

`final_w24/` contains the selected width-24 residual-adapter checkpoint, its
five-plane Android LiteRT export, exact configuration, run metadata, training
history, export parity report, and SHA-256 manifest.

The five input planes are RGB, the app-estimated scalar noise-strength plane,
and its precomputed smooth gate. The model processes fixed 192 x 192 tiles.

## Experiment evidence

- `run_records/` preserves every available `run.json` and `status.json` from
  the paper-guided experiment series. Per-epoch history is retained only for
  the selected W24 run under `final_w24/`.
- `evaluation_records/` preserves evaluation Markdown and JSON results without
  multi-gigabyte rendered images.
- The executable methods remain under `paper_192/scripts/` and
  `paper_192/src/`.
- Every experiment configuration remains under `paper_192/configs/`.

Raw datasets, prepared patch caches, repeated checkpoints, logs, smoke
outputs, and full-resolution generated images were removed because they are
regenerable and consumed approximately 771 GB.

See `../EXPERIMENT_RETROSPECTIVE.md` for the conclusions and navigation guide.
