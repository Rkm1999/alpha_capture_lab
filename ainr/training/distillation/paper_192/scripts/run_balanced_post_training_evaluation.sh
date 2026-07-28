#!/usr/bin/env bash
set -euo pipefail

root="/home/ryu/projects/alpha-capture-lab/ainr/training/distillation/paper_192"
python_bin="/home/ryu/.cache/scunet-int8-venv/bin/python"
run="$root/runs/general_camera_v1_no_namcc/run_balanced_shadow_multiscale_v1"
evaluation="$root/evaluation/general_camera_v1_no_namcc/run_balanced_shadow_multiscale_v1"
status="$run/status.json"

while [[ "$(jq -r .state "$status")" == "running" ]]; do
  jq -c '{state,epoch,epochs,best_selection_psnr,best_target_score,updated_at}' "$status"
  sleep 60
done

if [[ "$(jq -r .state "$status")" != "complete" ]]; then
  jq . "$status"
  exit 1
fi

mkdir -p "$evaluation"
cd "$root"
"$python_bin" scripts/evaluate_mixed_checkpoints.py \
  --config configs/general_camera_v1_no_namcc.yaml \
  --checkpoint "general=$run/best.pt" \
  --checkpoint "target=$run/target-best.pt" \
  --output "$evaluation/paired_validation.json"

"$python_bin" scripts/validate_iso_set.py \
  --input /home/ryu/projects/iso_test_image \
  --checkpoint "$run/best.pt" \
  --output-dir "$evaluation/iso_general"

"$python_bin" scripts/validate_iso_set.py \
  --input /home/ryu/projects/iso_test_image \
  --checkpoint "$run/target-best.pt" \
  --output-dir "$evaluation/iso_target"

touch "$evaluation/complete"
