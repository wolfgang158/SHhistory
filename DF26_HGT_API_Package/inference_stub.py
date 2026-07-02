"""This is an inference integration stub, not a fully runnable production API.

This module provides a minimal, clearly marked scaffold for how an HGT-based
real-time inference service could be wired around the packaged assets.

Important notes:
- best_model.pt is an HGT checkpoint file.
- hgt_tensors.pt / hetero_data.pt / pyhgt_graph.pkl are graph artifacts.
- nodes/station.csv is used for station_id / station_name / coordinate mapping.
- Real forward inference requires the original HGT model class and training code
  from the upstream project; this stub intentionally does not fake a real model
  prediction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


class InferenceStubError(RuntimeError):
    """Raised when the stub is used before real model assets are wired in."""


def load_package(package_dir: str | Path) -> Dict[str, Any]:
    """Load package-level paths and metadata.

    This function only establishes file locations. It does not perform any real
    model inference.
    """
    package_path = Path(package_dir).resolve()
    return {
        "package_dir": str(package_path),
        "model_dir": str(package_path / "model"),
        "graph_dir": str(package_path / "graph"),
        "examples_dir": str(package_path / "examples"),
        "outputs_dir": str(package_path / "outputs"),
        "docs_dir": str(package_path / "docs"),
        "checkpoint_path": str(package_path / "model" / "best_model.pt"),
        "graph_artifact_paths": {
            "hgt_tensors": str(package_path / "graph" / "hgt_tensors.pt"),
            "hetero_data": str(package_path / "graph" / "hetero_data.pt"),
            "pyhgt_graph": str(package_path / "graph" / "pyhgt_graph.pkl"),
            "edges": str(package_path / "graph" / "edges.csv"),
            "nodes_dir": str(package_path / "graph" / "nodes"),
        },
    }


def load_station_mapping(package_dir: str | Path) -> Dict[str, Dict[str, Any]]:
    """Load station mapping from nodes/station.csv.

    In a production implementation, this mapping would be used to resolve:
    - station_id
    - station_name
    - lon / lat
    - optional geometry fields
    """
    package_path = Path(package_dir).resolve()
    station_csv = package_path / "graph" / "nodes" / "station.csv"
    if not station_csv.exists():
        raise FileNotFoundError(f"station mapping file not found: {station_csv}")

    # Placeholder implementation: parse CSV lazily and return a mapping.
    # Replace this with the project's own station metadata loader.
    raise NotImplementedError(
        "Station mapping loading is not fully implemented in this stub. "
        "Wire in the project's existing station metadata loader here."
    )


def predict_by_station_id(station_id: str, package_dir: Optional[str | Path] = None) -> Dict[str, Any]:
    """Return a placeholder prediction for a given station ID.

    This should eventually call the real HGT model using the checkpoint and graph
    artifacts from the package. The stub intentionally raises a clear placeholder
    error rather than fabricating fake results.
    """
    if package_dir is None:
        raise InferenceStubError("package_dir is required for this stub")

    raise NotImplementedError(
        "Real inference is not implemented in this stub. "
        "Connect the original HGT model class and training code here."
    )


def predict_by_station_name(station_name: str, package_dir: Optional[str | Path] = None) -> Dict[str, Any]:
    """Return a placeholder prediction for a given station name."""
    if package_dir is None:
        raise InferenceStubError("package_dir is required for this stub")

    raise NotImplementedError(
        "Real inference is not implemented in this stub. "
        "Connect the original HGT model class and training code here."
    )


def predict_nearest_station(lon: float, lat: float, package_dir: Optional[str | Path] = None) -> Dict[str, Any]:
    """Return a placeholder prediction for the nearest station to lon/lat."""
    if package_dir is None:
        raise InferenceStubError("package_dir is required for this stub")

    raise NotImplementedError(
        "Real nearest-station inference is not implemented in this stub. "
        "Connect the original HGT model class and training code here."
    )


def format_api_response(station: Dict[str, Any], prediction: Dict[str, Any]) -> Dict[str, Any]:
    """Format a minimal JSON response payload for the API contract.

    This intentionally keeps the structure aligned with the API documentation but
    does not fabricate a real prediction result.
    """
    return {
        "status": "success",
        "data": [
            {
                "station": station,
                "prediction": prediction,
                "visualization": {
                    "color": "#7F8C8D",
                    "radius": 6,
                    "opacity": 0.8,
                },
            }
        ],
        "model": {
            "model_type": "HGT",
            "checkpoint": "model/best_model.pt",
        },
        "meta": {
            "request_type": "stub",
            "result_count": 1,
            "crs": "EPSG:4326",
        },
    }


if __name__ == "__main__":
    package_dir = Path(__file__).resolve().parent
    print(json.dumps(load_package(package_dir), indent=2))
