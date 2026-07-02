# DF26 HGT API Package

This package contains the HGT real-time inference API delivery assets for the SHhistory project.

## Source Runs
- Graph timestamp: 20260702-015536
- Model timestamp: 20260702-092440
- Outputs timestamp: 20260702-092440
- best_model.pt comes from 20260702-092440
- hgt_tensors.pt / hetero_data.pt / pyhgt_graph.pkl / edges.csv / nodes come from 20260702-015536

## Git LFS Note
The model and graph artifacts in this package are stored as Git LFS pointers in the repository. To obtain the real binary content, use Git LFS support (for example `git lfs pull`) in an environment that has Git LFS installed.

## Inference Stub
- inference_stub.py is a minimal API integration scaffold for the packaged assets.
- The current package provides model/data assets, JSON I/O documentation, and an integration stub.
- A fully runnable service still needs the original HGT model class and training code from the upstream project.
- Do not treat this stub as a final production inference service.

## Contents
- docs/HGT_API_JSON_IO_Bilingual_v2.md: API input/output specification
- model/: trained HGT model artifacts
- graph/: preprocessing graph artifacts
- examples/: example requests and responses
- outputs/: sample outputs

## Notes
- best_model.pt cannot be used alone for inference. It must be used together with the graph artifacts in graph/.
- Inference requires: graph/hgt_tensors.pt, graph/hetero_data.pt, graph/pyhgt_graph.pkl, graph/edges.csv, and graph/nodes/*.csv.
- The API accepts station_id / station_name / lon-lat JSON and returns conflict_score, confidence, conflict_level, color, radius, and geometry.
