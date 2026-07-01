#!/usr/bin/env bash

#scripts/preprocess/build_hgt_graph.sh --radius-m 2560 --max-poi-per-station-group 120

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_ID="$(date +%Y%m%d-%H%M%S)"

HGT_CONDA_ENV="${HGT_CONDA_ENV:-HGT}"
HGT_DATA_DIR="${HGT_DATA_DIR:-${ROOT}/data/raw}"
HGT_OUTPUT_DIR="${HGT_OUTPUT_DIR:-${ROOT}/outputs/preprocess/hgt_graph/runs/${RUN_ID}}"

extra_args=()
[[ -n "${HGT_STATIONS_CSV:-}" ]] && extra_args+=(--stations-csv "${HGT_STATIONS_CSV}")
[[ -n "${HGT_BUILDINGS_CSV:-}" ]] && extra_args+=(--buildings-csv "${HGT_BUILDINGS_CSV}")
[[ -n "${HGT_CONSERVATION_GEOJSON:-}" ]] && extra_args+=(--conservation-geojson "${HGT_CONSERVATION_GEOJSON}")
[[ -n "${HGT_ADMIN_GEOJSON:-}" ]] && extra_args+=(--admin-geojson "${HGT_ADMIN_GEOJSON}")
[[ -n "${HGT_ROADS_CSV:-}" ]] && extra_args+=(--roads-csv "${HGT_ROADS_CSV}")
[[ -n "${HGT_POI_CSV:-}" ]] && extra_args+=(--poi-csv "${HGT_POI_CSV}")

echo "[build_hgt_graph] env=${HGT_CONDA_ENV}"
echo "[build_hgt_graph] data-dir=${HGT_DATA_DIR}"
echo "[build_hgt_graph] output-dir=${HGT_OUTPUT_DIR}"

conda run -n "${HGT_CONDA_ENV}" python "${SCRIPT_DIR}/build_hgt_graph.py" \
  --data-dir "${HGT_DATA_DIR}" \
  --output-dir "${HGT_OUTPUT_DIR}" \
  "${extra_args[@]}" \
  "$@"

echo "[build_hgt_graph] completed: ${HGT_OUTPUT_DIR}"
