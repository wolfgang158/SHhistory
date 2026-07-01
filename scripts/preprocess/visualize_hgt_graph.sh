#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_ID="$(date +%Y%m%d-%H%M%S)"

HGT_CONDA_ENV="${HGT_CONDA_ENV:-HGT}"
HGT_GRAPH_DIR="${HGT_GRAPH_DIR:-${ROOT}/data/hgt_graph}"
HGT_VIZ_OUTPUT_DIR="${HGT_VIZ_OUTPUT_DIR:-${ROOT}/outputs/preprocess/hgt_graph_viz/runs/${RUN_ID}}"

echo "[visualize_hgt_graph] env=${HGT_CONDA_ENV}"
echo "[visualize_hgt_graph] graph-dir=${HGT_GRAPH_DIR}"
echo "[visualize_hgt_graph] output-dir=${HGT_VIZ_OUTPUT_DIR}"

conda run -n "${HGT_CONDA_ENV}" python "${SCRIPT_DIR}/visualize_hgt_graph.py" \
  --graph-dir "${HGT_GRAPH_DIR}" \
  --output-dir "${HGT_VIZ_OUTPUT_DIR}" \
  "$@"

echo "[visualize_hgt_graph] completed: ${HGT_VIZ_OUTPUT_DIR}"
