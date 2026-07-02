#!/usr/bin/env bash

# Evaluate a trained HGT station model and optionally score an unseen lon/lat area.
# Override CUDA_VISIBLE_DEVICES, HGT_CONDA_ENV, HGT_GRAPH_DIR, HGT_TRAIN_DIR, or HGT_EVAL_OUTPUT_DIR as needed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HGT_CONDA_ENV="${HGT_CONDA_ENV:-HGT}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_VISIBLE_DEVICES

extra_args=()
[[ -n "${HGT_GRAPH_DIR:-}" ]] && extra_args+=(--graph-dir "${HGT_GRAPH_DIR}")
[[ -n "${HGT_TRAIN_DIR:-}" ]] && extra_args+=(--train-dir "${HGT_TRAIN_DIR}")
[[ -n "${HGT_EVAL_OUTPUT_DIR:-}" ]] && extra_args+=(--output-dir "${HGT_EVAL_OUTPUT_DIR}")

echo "[eval_hgt] env=${HGT_CONDA_ENV}"
echo "[eval_hgt] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
[[ -n "${HGT_GRAPH_DIR:-}" ]] && echo "[eval_hgt] graph-dir=${HGT_GRAPH_DIR}"
[[ -n "${HGT_TRAIN_DIR:-}" ]] && echo "[eval_hgt] train-dir=${HGT_TRAIN_DIR}"
[[ -n "${HGT_EVAL_OUTPUT_DIR:-}" ]] && echo "[eval_hgt] output-dir=${HGT_EVAL_OUTPUT_DIR}"

conda run --no-capture-output -n "${HGT_CONDA_ENV}" python "${SCRIPT_DIR}/evaluate_hgt_station_area.py" \
  "${extra_args[@]}" \
  "$@"
