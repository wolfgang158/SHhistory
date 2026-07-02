#!/usr/bin/env bash

# Train station-level HGT conflict classifier on the latest preprocessed graph.
# Override CUDA_VISIBLE_DEVICES, HGT_CONDA_ENV, HGT_GRAPH_DIR, or HGT_TRAIN_OUTPUT_DIR as needed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

HGT_CONDA_ENV="${HGT_CONDA_ENV:-HGT}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_VISIBLE_DEVICES

extra_args=()
[[ -n "${HGT_GRAPH_DIR:-}" ]] && extra_args+=(--graph-dir "${HGT_GRAPH_DIR}")
[[ -n "${HGT_TRAIN_OUTPUT_DIR:-}" ]] && extra_args+=(--output-dir "${HGT_TRAIN_OUTPUT_DIR}")

echo "[train_hgt] env=${HGT_CONDA_ENV}"
echo "[train_hgt] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
[[ -n "${HGT_GRAPH_DIR:-}" ]] && echo "[train_hgt] graph-dir=${HGT_GRAPH_DIR}"
[[ -n "${HGT_TRAIN_OUTPUT_DIR:-}" ]] && echo "[train_hgt] output-dir=${HGT_TRAIN_OUTPUT_DIR}"

conda run --no-capture-output -n "${HGT_CONDA_ENV}" python "${SCRIPT_DIR}/train_hgt_station_conflict.py" \
  "${extra_args[@]}" \
  "$@"
