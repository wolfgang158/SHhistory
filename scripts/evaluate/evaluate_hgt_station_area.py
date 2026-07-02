#!/usr/bin/env python3
"""Evaluate trained HGT station predictions and score an unseen area."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error


ROOT = Path(__file__).resolve().parents[2]
HGT_DIR = ROOT / "HGT"
if str(HGT_DIR) not in sys.path:
    sys.path.insert(0, str(HGT_DIR))

from pyHGT.model import Classifier, GNN  # noqa: E402


def log(message: str) -> None:
    print(f"[eval_hgt {time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def safe_torch_load(path: Path, map_location: str | torch.device) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def latest_run(base_dir: Path, marker: str) -> Path:
    candidates = [p for p in base_dir.glob("runs/*") if (p / marker).exists()]
    if not candidates:
        raise FileNotFoundError(f"No run with {marker} found under {base_dir / 'runs'}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def lonlat_to_webmerc(lon: float, lat: float) -> tuple[float, float]:
    lat = min(max(float(lat), -85.05112878), 85.05112878)
    x = float(lon) * 20037508.34 / 180.0
    y = math.log(math.tan((90.0 + lat) * math.pi / 360.0)) * 20037508.34 / math.pi
    return x, y


def mask_station_target_features(x: torch.Tensor, station_global_idx: torch.Tensor) -> torch.Tensor:
    x = x.clone()
    if x.shape[1] > 12:
        x[station_global_idx, 11:13] = 0.0
    return x


def stratified_split(labels: torch.Tensor, train_ratio: float, val_ratio: float, seed: int) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    labels_np = labels.cpu().numpy()
    split = {"train": [], "val": [], "test": []}
    for cls in sorted(np.unique(labels_np).tolist()):
        cls_idx = np.where(labels_np == cls)[0]
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio))) if n >= 3 else 0
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1 if n - n_train > 1 else 0
        split["train"].extend(cls_idx[:n_train].tolist())
        split["val"].extend(cls_idx[n_train : n_train + n_val].tolist())
        split["test"].extend(cls_idx[n_train + n_val :].tolist())
    for key, values in split.items():
        rng.shuffle(values)
        split[key] = torch.tensor(values, dtype=torch.long)
    return split


def build_model(checkpoint: dict[str, Any], tensors: dict[str, Any], hidden_dim: int | None, device: torch.device):
    ckpt_args = checkpoint.get("args", {})
    hidden = int(hidden_dim or ckpt_args.get("hidden_dim", 64))
    num_classes = int(tensors["station_y_conflict"].max().item()) + 1
    encoder = GNN(
        in_dim=tensors["x"].shape[1],
        n_hid=hidden,
        num_types=len(tensors["node_type_map"]),
        num_relations=int(tensors["edge_type"].max().item()) + 1,
        n_heads=int(ckpt_args.get("num_heads", 4)),
        n_layers=int(ckpt_args.get("num_layers", 2)),
        dropout=float(ckpt_args.get("dropout", 0.2)),
        conv_name=ckpt_args.get("conv_name", "hgt"),
        prev_norm=True,
        last_norm=True,
        use_RTE=bool(ckpt_args.get("use_rte", False)),
    ).to(device)
    classifier = Classifier(hidden, num_classes).to(device)
    historic_head = nn.Linear(hidden, 1).to(device)
    daily_head = nn.Linear(hidden, 1).to(device)

    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    classifier.load_state_dict(checkpoint["classifier_state_dict"])
    historic_head.load_state_dict(checkpoint["historic_head_state_dict"])
    daily_head.load_state_dict(checkpoint["daily_head_state_dict"])
    encoder.eval()
    classifier.eval()
    historic_head.eval()
    daily_head.eval()
    return encoder, classifier, historic_head, daily_head


def metrics(df: pd.DataFrame, mask: np.ndarray) -> dict[str, float]:
    if mask.sum() == 0:
        return {}
    y_true = df.loc[mask, "true_conflict_label"].to_numpy()
    y_pred = df.loc[mask, "pred_conflict_label"].to_numpy()
    hist_true = df.loc[mask, "true_historic_probability"].to_numpy()
    hist_pred = df.loc[mask, "pred_historic_probability"].to_numpy()
    daily_true = df.loc[mask, "true_daily_probability"].to_numpy()
    daily_pred = df.loc[mask, "pred_daily_probability"].to_numpy()
    return {
        "n": int(mask.sum()),
        "conflict_accuracy": float(accuracy_score(y_true, y_pred)),
        "conflict_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "historic_mae": float(mean_absolute_error(hist_true, hist_pred)),
        "historic_rmse": float(np.sqrt(np.mean((hist_true - hist_pred) ** 2))),
        "daily_mae": float(mean_absolute_error(daily_true, daily_pred)),
        "daily_rmse": float(np.sqrt(np.mean((daily_true - daily_pred) ** 2))),
    }


def grade_probability(value: float, low: float = 0.33, high: float = 0.66) -> str:
    if value >= high:
        return "high"
    if value >= low:
        return "medium"
    return "low"


def conflict_label_name(label: int) -> str:
    names = {0: "low", 1: "medium", 2: "high"}
    return names.get(int(label), f"class_{int(label)}")


def score_area(df: pd.DataFrame, lon: float, lat: float, radius_m: float) -> dict[str, Any]:
    if not {"x", "y"}.issubset(df.columns):
        if not {"lon", "lat"}.issubset(df.columns):
            raise KeyError("Station metadata must contain either x/y or lon/lat columns for area scoring")
        xy = np.array([lonlat_to_webmerc(float(a), float(b)) for a, b in zip(df["lon"], df["lat"])])
        sx, sy = xy[:, 0], xy[:, 1]
    else:
        sx, sy = df["x"].to_numpy(dtype=float), df["y"].to_numpy(dtype=float)
    qx, qy = lonlat_to_webmerc(lon, lat)
    dist = np.sqrt((sx - qx) ** 2 + (sy - qy) ** 2)
    local = df.loc[dist <= radius_m].copy()
    local["distance_m"] = dist[dist <= radius_m]
    if len(local) == 0:
        nearest = int(np.argmin(dist))
        local = df.iloc[[nearest]].copy()
        local["distance_m"] = [float(dist[nearest])]

    weights = 1.0 / np.maximum(local["distance_m"].to_numpy(dtype=float), 1.0)
    weights = weights / weights.sum()
    conflict_score = float(np.average(local["pred_conflict_expected"], weights=weights))
    historic_score = float(np.average(local["pred_historic_probability"], weights=weights))
    daily_score = float(np.average(local["pred_daily_probability"], weights=weights))
    return {
        "query_lon": lon,
        "query_lat": lat,
        "radius_m": radius_m,
        "matched_station_count": int(len(local)),
        "conflict_score": conflict_score,
        "conflict_grade": grade_probability(conflict_score / max(float(df["pred_conflict_label"].max()), 1.0)),
        "historic_degree": historic_score,
        "historic_grade": grade_probability(historic_score),
        "daily_degree": daily_score,
        "daily_grade": grade_probability(daily_score),
        "nearest_stations": local.sort_values("distance_m")
        .head(10)[
            [
                c
                for c in [
                    "station_local_idx",
                    "name",
                    "lon",
                    "lat",
                    "distance_m",
                    "pred_conflict_label",
                    "pred_conflict_level",
                    "pred_conflict_expected",
                    "pred_historic_probability",
                    "pred_daily_probability",
                ]
                if c in local.columns
            ]
        ]
        .to_dict(orient="records"),
    }


def score_station(
    df: pd.DataFrame,
    station_name: str | None,
    station_local_idx: int | None,
    required_split: str | None,
) -> dict[str, Any]:
    if station_name is None and station_local_idx is None:
        raise ValueError("Provide --station-name or --station-local-idx for station scoring")
    if station_local_idx is not None:
        matches = df.loc[df["station_local_idx"] == station_local_idx].copy()
    else:
        if "name" not in df.columns:
            raise KeyError("Station metadata does not contain a name column")
        needle = str(station_name).strip()
        matches = df.loc[df["name"].astype(str).str.contains(needle, case=False, regex=False)].copy()
    if required_split:
        matches = matches.loc[matches["split"] == required_split].copy()
    if len(matches) == 0:
        split_note = f" in split={required_split}" if required_split else ""
        query = f"idx={station_local_idx}" if station_local_idx is not None else f"name contains {station_name!r}"
        raise ValueError(f"No station matched {query}{split_note}")
    if len(matches) > 1:
        cols = [c for c in ["station_local_idx", "name", "split", "lon", "lat"] if c in matches.columns]
        raise ValueError(f"Multiple stations matched; use --station-local-idx. Matches: {matches[cols].to_dict(orient='records')}")

    row = matches.iloc[0]
    result_cols = [
        "station_local_idx",
        "id",
        "raw_id",
        "name",
        "split",
        "lon",
        "lat",
        "true_conflict_label",
        "pred_conflict_label",
        "pred_conflict_expected",
        "pred_conflict_confidence",
        "true_historic_probability",
        "pred_historic_probability",
        "pred_historic_grade",
        "true_daily_probability",
        "pred_daily_probability",
        "pred_daily_grade",
    ]
    result = {col: row[col].item() if hasattr(row[col], "item") else row[col] for col in result_cols if col in row.index}
    prob_cols = [col for col in df.columns if col.startswith("prob_conflict_class_")]
    result["conflict_probabilities"] = {col: float(row[col]) for col in prob_cols}
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained HGT station model and optionally score an unseen area.")
    parser.add_argument("--graph-dir", type=Path, default=None, help="Graph artifact directory containing hgt_tensors.pt.")
    parser.add_argument("--train-dir", type=Path, default=None, help="Training output directory containing best_model.pt.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Override checkpoint path. Defaults to train-dir/best_model.pt.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Evaluation output directory.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden-dim", type=int, default=None, help="Override hidden dim if checkpoint lacks args.")
    parser.add_argument("--train-ratio", type=float, default=None, help="Override split ratio used for metrics.")
    parser.add_argument("--val-ratio", type=float, default=None, help="Override split ratio used for metrics.")
    parser.add_argument("--seed", type=int, default=None, help="Override split seed used for metrics.")
    parser.add_argument("--no-mask-station-target-features", action="store_true")
    parser.add_argument("--area-lon", type=float, default=None, help="Longitude for unseen-area scoring.")
    parser.add_argument("--area-lat", type=float, default=None, help="Latitude for unseen-area scoring.")
    parser.add_argument("--area-radius-m", type=float, default=800.0)
    parser.add_argument("--station-name", default=None, help="Score one station by substring match, for example '人民广场'.")
    parser.add_argument("--station-local-idx", type=int, default=None, help="Score one station by station_local_idx.")
    parser.add_argument("--station-split", choices=["train", "val", "test"], default=None, help="Require the queried station to be in this split.")
    parser.add_argument("--random-station", action="store_true", help="Randomly choose one station, optionally constrained by --station-split.")
    parser.add_argument("--random-seed", type=int, default=None, help="Seed for --random-station.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    graph_dir = args.graph_dir or latest_run(ROOT / "outputs" / "preprocess" / "hgt_graph", "hgt_tensors.pt")
    train_dir = args.train_dir or latest_run(ROOT / "outputs" / "train" / "hgt_station_conflict", "best_model.pt")
    checkpoint_path = args.checkpoint or (train_dir / "best_model.pt")
    run_id = time.strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir or (ROOT / "outputs" / "evaluate" / "hgt_station_area" / "runs" / run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    log(f"graph-dir={graph_dir}")
    log(f"train-dir={train_dir}")
    log(f"checkpoint={checkpoint_path}")
    log(f"output-dir={output_dir}")
    log(f"device={device}")

    tensors = safe_torch_load(graph_dir / "hgt_tensors.pt", map_location="cpu")
    checkpoint = safe_torch_load(checkpoint_path, map_location=device)
    ckpt_args = checkpoint.get("args", {})
    train_ratio = float(args.train_ratio if args.train_ratio is not None else ckpt_args.get("train_ratio", 0.7))
    val_ratio = float(args.val_ratio if args.val_ratio is not None else ckpt_args.get("val_ratio", 0.15))
    seed = int(args.seed if args.seed is not None else ckpt_args.get("seed", 42))

    labels = tensors["station_y_conflict"].long()
    historic_target = tensors["station_historic_probability"].float()
    daily_target = tensors["station_daily_probability"].float()
    station_offset = int(tensors["node_offsets"]["station"])
    station_global_idx = torch.arange(station_offset, station_offset + len(labels), dtype=torch.long)
    if not args.no_mask_station_target_features:
        tensors["x"] = mask_station_target_features(tensors["x"], station_global_idx)

    encoder, classifier, historic_head, daily_head = build_model(checkpoint, tensors, args.hidden_dim, device)
    for key in ["x", "node_type", "edge_index", "edge_type", "edge_time"]:
        tensors[key] = tensors[key].to(device)
    station_global_idx = station_global_idx.to(device)

    with torch.no_grad():
        emb = encoder(tensors["x"], tensors["node_type"], tensors["edge_time"], tensors["edge_index"], tensors["edge_type"])
        station_emb = emb[station_global_idx]
        logits = classifier(station_emb)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        historic_pred = torch.sigmoid(historic_head(station_emb).squeeze(-1)).cpu().numpy()
        daily_pred = torch.sigmoid(daily_head(station_emb).squeeze(-1)).cpu().numpy()

    df = pd.DataFrame(
        {
            "station_local_idx": np.arange(len(labels)),
            "true_conflict_label": labels.numpy(),
            "pred_conflict_label": probs.argmax(axis=1),
            "pred_conflict_level": [conflict_label_name(v) for v in probs.argmax(axis=1)],
            "pred_conflict_expected": probs @ np.arange(probs.shape[1]),
            "pred_conflict_confidence": probs.max(axis=1),
            "true_historic_probability": historic_target.numpy(),
            "pred_historic_probability": historic_pred,
            "true_daily_probability": daily_target.numpy(),
            "pred_daily_probability": daily_pred,
            "pred_historic_grade": [grade_probability(v) for v in historic_pred],
            "pred_daily_grade": [grade_probability(v) for v in daily_pred],
        }
    )
    for cls in range(probs.shape[1]):
        df[f"prob_conflict_class_{cls}"] = probs[:, cls]

    station_csv = graph_dir / "nodes" / "station.csv"
    if station_csv.exists():
        station_df = pd.read_csv(station_csv)
        keep_cols = [c for c in ["id", "raw_id", "name", "lon", "lat", "x", "y", "conflict_level_500m"] if c in station_df]
        df = pd.concat([station_df[keep_cols].reset_index(drop=True), df], axis=1)

    split = stratified_split(labels, train_ratio, val_ratio, seed)
    split_name = np.full(len(df), "unused", dtype=object)
    for name, idx in split.items():
        split_name[idx.numpy()] = name
    df["split"] = split_name
    df.to_csv(output_dir / "station_area_predictions.csv", index=False, encoding="utf-8-sig")

    metric_summary = {name: metrics(df, df["split"].to_numpy() == name) for name in ["train", "val", "test"]}
    station_summary = None
    if args.random_station:
        candidates = df
        if args.station_split:
            candidates = candidates.loc[candidates["split"] == args.station_split]
        if len(candidates) == 0:
            raise ValueError(f"No station candidates found for split={args.station_split}")
        rng = np.random.default_rng(args.random_seed)
        chosen = candidates.sample(n=1, random_state=int(rng.integers(0, np.iinfo(np.int32).max))).iloc[0]
        args.station_local_idx = int(chosen["station_local_idx"])

    if args.station_name is not None or args.station_local_idx is not None:
        station_summary = score_station(df, args.station_name, args.station_local_idx, args.station_split)
        (output_dir / "station_score.json").write_text(json.dumps(station_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    area_summary = None
    if args.area_lon is not None or args.area_lat is not None:
        if args.area_lon is None or args.area_lat is None:
            raise ValueError("--area-lon and --area-lat must be provided together")
        area_summary = score_area(df, args.area_lon, args.area_lat, args.area_radius_m)
        (output_dir / "area_score.json").write_text(json.dumps(area_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "graph_dir": str(graph_dir),
        "train_dir": str(train_dir),
        "checkpoint": str(checkpoint_path),
        "output_dir": str(output_dir),
        "metrics": metric_summary,
        "station_score": station_summary,
        "area_score": area_summary,
        "split": {"train_ratio": train_ratio, "val_ratio": val_ratio, "seed": seed},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"complete: wrote {output_dir / 'station_area_predictions.csv'}")
    if area_summary:
        log(
            "area score: "
            f"conflict={area_summary['conflict_score']:.4f} ({area_summary['conflict_grade']}), "
            f"historic={area_summary['historic_degree']:.4f} ({area_summary['historic_grade']}), "
            f"daily={area_summary['daily_degree']:.4f} ({area_summary['daily_grade']})"
        )
    if station_summary:
        log(
            "station score: "
            f"{station_summary.get('name', station_summary.get('station_local_idx'))}, "
            f"split={station_summary.get('split')}, "
            f"conflict={station_summary.get('pred_conflict_label')}, "
            f"historic={station_summary.get('pred_historic_probability'):.4f} ({station_summary.get('pred_historic_grade')}), "
            f"daily={station_summary.get('pred_daily_probability'):.4f} ({station_summary.get('pred_daily_grade')})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
