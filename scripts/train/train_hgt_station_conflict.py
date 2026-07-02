#!/usr/bin/env python3
"""Train HGT on station conflict, historicity, and dailiness targets."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[2]
HGT_DIR = ROOT / "HGT"
if str(HGT_DIR) not in sys.path:
    sys.path.insert(0, str(HGT_DIR))

from pyHGT.model import Classifier, GNN  # noqa: E402


def log(message: str) -> None:
    print(f"[train_hgt {time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def latest_graph_run(base_dir: Path) -> Path:
    candidates = [p for p in base_dir.glob("runs/*") if (p / "hgt_tensors.pt").exists()]
    if not candidates:
        raise FileNotFoundError(f"No graph run with hgt_tensors.pt found under {base_dir / 'runs'}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stratified_split(labels: torch.Tensor, train_ratio: float, val_ratio: float, seed: int) -> dict[str, torch.Tensor]:
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("--train-ratio and --val-ratio must leave a positive test split")

    rng = np.random.default_rng(seed)
    labels_np = labels.cpu().numpy()
    train_idx, val_idx, test_idx = [], [], []
    for cls in sorted(np.unique(labels_np).tolist()):
        cls_idx = np.where(labels_np == cls)[0]
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio))) if n >= 3 else 0
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1 if n - n_train > 1 else 0
        train_idx.extend(cls_idx[:n_train].tolist())
        val_idx.extend(cls_idx[n_train : n_train + n_val].tolist())
        test_idx.extend(cls_idx[n_train + n_val :].tolist())

    for values in (train_idx, val_idx, test_idx):
        rng.shuffle(values)
    return {
        "train": torch.tensor(train_idx, dtype=torch.long),
        "val": torch.tensor(val_idx, dtype=torch.long),
        "test": torch.tensor(test_idx, dtype=torch.long),
    }


def safe_torch_load(path: Path, map_location: str | torch.device) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def metric_dict(y_true: torch.Tensor, logits: torch.Tensor) -> dict[str, float]:
    if y_true.numel() == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0}
    pred = logits.argmax(dim=-1).detach().cpu().numpy()
    true = y_true.detach().cpu().numpy()
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
    }


def regression_metric_dict(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict[str, float]:
    if y_true.numel() == 0:
        return {"mae": 0.0, "rmse": 0.0}
    err = y_pred.detach() - y_true.detach()
    return {
        "mae": float(err.abs().mean().cpu()),
        "rmse": float(torch.sqrt((err * err).mean()).cpu()),
    }


def evaluate(
    encoder: GNN,
    classifier: Classifier,
    historic_head: nn.Module,
    daily_head: nn.Module,
    tensors: dict[str, Any],
    station_global_idx: torch.Tensor,
    split_idx: torch.Tensor,
    labels: torch.Tensor,
    historic_target: torch.Tensor,
    daily_target: torch.Tensor,
) -> dict[str, float]:
    encoder.eval()
    classifier.eval()
    historic_head.eval()
    daily_head.eval()
    with torch.no_grad():
        node_emb = encoder(
            tensors["x"],
            tensors["node_type"],
            tensors["edge_time"],
            tensors["edge_index"],
            tensors["edge_type"],
        )
        station_emb = node_emb[station_global_idx]
        logits = classifier(station_emb)[split_idx]
        historic_pred = torch.sigmoid(historic_head(station_emb).squeeze(-1))[split_idx]
        daily_pred = torch.sigmoid(daily_head(station_emb).squeeze(-1))[split_idx]
    cls_metrics = metric_dict(labels[split_idx], logits)
    historic_metrics = regression_metric_dict(historic_target[split_idx], historic_pred)
    daily_metrics = regression_metric_dict(daily_target[split_idx], daily_pred)
    return {
        "conflict_accuracy": cls_metrics["accuracy"],
        "conflict_macro_f1": cls_metrics["macro_f1"],
        "historic_mae": historic_metrics["mae"],
        "historic_rmse": historic_metrics["rmse"],
        "daily_mae": daily_metrics["mae"],
        "daily_rmse": daily_metrics["rmse"],
    }


def mask_station_target_features(x: torch.Tensor, station_global_idx: torch.Tensor) -> torch.Tensor:
    """Avoid leaking target probabilities into station-level regression heads.

    Station feature layout from build_hgt_graph.py:
    columns 11 and 12 are historic_probability_500m and daily_probability_500m.
    """
    x = x.clone()
    if x.shape[1] > 12:
        x[station_global_idx, 11:13] = 0.0
    return x


def save_predictions(
    output_dir: Path,
    graph_dir: Path,
    station_logits: torch.Tensor,
    historic_pred: torch.Tensor,
    daily_pred: torch.Tensor,
    labels: torch.Tensor,
    historic_target: torch.Tensor,
    daily_target: torch.Tensor,
    split: dict[str, torch.Tensor],
) -> None:
    probs = torch.softmax(station_logits, dim=-1).detach().cpu().numpy()
    pred = probs.argmax(axis=1)
    df = pd.DataFrame(
        {
            "station_local_idx": np.arange(len(pred)),
            "true_label": labels.detach().cpu().numpy(),
            "pred_label": pred,
            "pred_confidence": probs.max(axis=1),
            "true_historic_probability": historic_target.detach().cpu().numpy(),
            "pred_historic_probability": historic_pred.detach().cpu().numpy(),
            "true_daily_probability": daily_target.detach().cpu().numpy(),
            "pred_daily_probability": daily_pred.detach().cpu().numpy(),
        }
    )
    for cls in range(probs.shape[1]):
        df[f"prob_class_{cls}"] = probs[:, cls]

    station_csv = graph_dir / "nodes" / "station.csv"
    if station_csv.exists():
        station_df = pd.read_csv(station_csv)
        keep_cols = [c for c in ["id", "raw_id", "name", "lon", "lat", "conflict_index_500m", "conflict_level_500m"] if c in station_df]
        df = pd.concat([station_df[keep_cols].reset_index(drop=True), df], axis=1)

    split_name = np.full(len(df), "unused", dtype=object)
    for name, idx in split.items():
        split_name[idx.detach().cpu().numpy()] = name
    df["split"] = split_name
    df.to_csv(output_dir / "station_predictions.csv", index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HGT for station-level conflict, historicity, and dailiness prediction.")
    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=None,
        help="Preprocess output directory containing hgt_tensors.pt. Defaults to latest outputs/preprocess/hgt_graph/runs/*.",
    )
    parser.add_argument(
        "--graph-base-dir",
        type=Path,
        default=ROOT / "outputs" / "preprocess" / "hgt_graph",
        help="Base graph output directory used when --graph-dir is omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Training output directory. Defaults to outputs/train/hgt_station_conflict/runs/<timestamp>.",
    )
    parser.add_argument("--device", default="cuda:0", help="Torch device. With CUDA_VISIBLE_DEVICES=1, cuda:0 is physical GPU 1.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--conflict-loss-weight", type=float, default=1.0)
    parser.add_argument("--historic-loss-weight", type=float, default=0.5)
    parser.add_argument("--daily-loss-weight", type=float, default=0.5)
    parser.add_argument("--regression-loss", choices=["mse", "smooth_l1"], default="mse")
    parser.add_argument("--selection-objective", choices=["multitask", "conflict", "degrees"], default="multitask")
    parser.add_argument("--degree-score-weight", type=float, default=0.5)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--conv-name", default="hgt", choices=["hgt", "dense_hgt", "gcn", "gat"])
    parser.add_argument("--use-rte", action="store_true", help="Enable relative temporal encoding. Static graph training leaves it off.")
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision.")
    parser.add_argument(
        "--no-mask-station-target-features",
        action="store_true",
        help="Do not mask station feature columns 11/12, which contain historic/daily target probabilities.",
    )
    parser.add_argument("--log-every", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    graph_dir = args.graph_dir or latest_graph_run(args.graph_base_dir)
    tensors_path = graph_dir / "hgt_tensors.pt"
    if not tensors_path.exists():
        raise FileNotFoundError(f"Missing tensor artifact: {tensors_path}")

    run_id = time.strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir or (ROOT / "outputs" / "train" / "hgt_station_conflict" / "runs" / run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    log(f"graph-dir={graph_dir}")
    log(f"output-dir={output_dir}")
    log(f"device={device}")

    log("loading tensors")
    tensors = safe_torch_load(tensors_path, map_location="cpu")
    required = [
        "x",
        "node_type",
        "edge_index",
        "edge_type",
        "edge_time",
        "station_y_conflict",
        "station_historic_probability",
        "station_daily_probability",
        "node_type_map",
        "node_offsets",
    ]
    missing = [key for key in required if key not in tensors]
    if missing:
        raise KeyError(f"Missing keys in {tensors_path}: {missing}")

    labels = tensors["station_y_conflict"].long()
    historic_target = tensors["station_historic_probability"].float()
    daily_target = tensors["station_daily_probability"].float()
    split = stratified_split(labels, args.train_ratio, args.val_ratio, args.seed)
    station_offset = int(tensors["node_offsets"]["station"])
    station_global_idx = torch.arange(station_offset, station_offset + len(labels), dtype=torch.long)
    num_classes = int(labels.max().item()) + 1

    log(
        "data summary: "
        f"nodes={tuple(tensors['x'].shape)}, edges={tensors['edge_index'].shape[1]}, "
        f"node_types={len(tensors['node_type_map'])}, edge_types={int(tensors['edge_type'].max().item()) + 1}, "
        f"stations={len(labels)}, classes={num_classes}, "
        f"split={ {k: int(len(v)) for k, v in split.items()} }"
    )

    if not args.no_mask_station_target_features:
        tensors["x"] = mask_station_target_features(tensors["x"], station_global_idx)
        log("masked station feature columns 11/12 to avoid historic/daily target leakage")

    for key in ["x", "node_type", "edge_index", "edge_type", "edge_time"]:
        tensors[key] = tensors[key].to(device)
    labels = labels.to(device)
    historic_target = historic_target.to(device)
    daily_target = daily_target.to(device)
    station_global_idx = station_global_idx.to(device)
    split = {key: value.to(device) for key, value in split.items()}

    encoder = GNN(
        in_dim=tensors["x"].shape[1],
        n_hid=args.hidden_dim,
        num_types=len(tensors["node_type_map"]),
        num_relations=int(tensors["edge_type"].max().item()) + 1,
        n_heads=args.num_heads,
        n_layers=args.num_layers,
        dropout=args.dropout,
        conv_name=args.conv_name,
        prev_norm=True,
        last_norm=True,
        use_RTE=args.use_rte,
    ).to(device)
    classifier = Classifier(args.hidden_dim, num_classes).to(device)
    historic_head = nn.Linear(args.hidden_dim, 1).to(device)
    daily_head = nn.Linear(args.hidden_dim, 1).to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(classifier.parameters()) + list(historic_head.parameters()) + list(daily_head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    loss_fn = nn.NLLLoss()
    reg_loss_fn = nn.SmoothL1Loss(beta=0.05) if args.regression_loss == "smooth_l1" else nn.MSELoss()

    best_val = -1.0
    best_epoch = 0
    history = []
    log("starting training")
    for epoch in range(1, args.epochs + 1):
        encoder.train()
        classifier.train()
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
            node_emb = encoder(tensors["x"], tensors["node_type"], tensors["edge_time"], tensors["edge_index"], tensors["edge_type"])
            station_emb = node_emb[station_global_idx]
            station_logits = classifier(station_emb)
            historic_pred = torch.sigmoid(historic_head(station_emb).squeeze(-1))
            daily_pred = torch.sigmoid(daily_head(station_emb).squeeze(-1))
            conflict_loss = loss_fn(station_logits[split["train"]], labels[split["train"]])
            historic_loss = reg_loss_fn(historic_pred[split["train"]], historic_target[split["train"]])
            daily_loss = reg_loss_fn(daily_pred[split["train"]], daily_target[split["train"]])
            loss = (
                args.conflict_loss_weight * conflict_loss
                + args.historic_loss_weight * historic_loss
                + args.daily_loss_weight * daily_loss
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            train_cls_metrics = metric_dict(labels[split["train"]], station_logits[split["train"]])
            train_historic_metrics = regression_metric_dict(historic_target[split["train"]], historic_pred[split["train"]])
            train_daily_metrics = regression_metric_dict(daily_target[split["train"]], daily_pred[split["train"]])
            val_metrics = evaluate(
                encoder,
                classifier,
                historic_head,
                daily_head,
                tensors,
                station_global_idx,
                split["val"],
                labels,
                historic_target,
                daily_target,
            )
            row = {
                "epoch": epoch,
                "loss": float(loss.detach().cpu()),
                "conflict_loss": float(conflict_loss.detach().cpu()),
                "historic_loss": float(historic_loss.detach().cpu()),
                "daily_loss": float(daily_loss.detach().cpu()),
                "train_conflict_accuracy": train_cls_metrics["accuracy"],
                "train_conflict_macro_f1": train_cls_metrics["macro_f1"],
                "train_historic_mae": train_historic_metrics["mae"],
                "train_daily_mae": train_daily_metrics["mae"],
                "val_conflict_accuracy": val_metrics["conflict_accuracy"],
                "val_conflict_macro_f1": val_metrics["conflict_macro_f1"],
                "val_historic_mae": val_metrics["historic_mae"],
                "val_daily_mae": val_metrics["daily_mae"],
            }
            history.append(row)
            log(
                f"epoch={epoch:04d} loss={row['loss']:.4f} "
                f"train_f1={row['train_conflict_macro_f1']:.4f} "
                f"val_f1={row['val_conflict_macro_f1']:.4f} "
                f"val_hist_mae={row['val_historic_mae']:.4f} val_daily_mae={row['val_daily_mae']:.4f}"
            )
            degree_penalty = row["val_historic_mae"] + row["val_daily_mae"]
            if args.selection_objective == "conflict":
                score = row["val_conflict_macro_f1"]
            elif args.selection_objective == "degrees":
                score = -degree_penalty
            else:
                score = row["val_conflict_macro_f1"] - args.degree_score_weight * degree_penalty
            if score > best_val:
                best_val = score
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "encoder_state_dict": encoder.state_dict(),
                        "classifier_state_dict": classifier.state_dict(),
                        "historic_head_state_dict": historic_head.state_dict(),
                        "daily_head_state_dict": daily_head.state_dict(),
                        "args": json_safe(vars(args)),
                        "node_type_map": tensors["node_type_map"],
                        "node_offsets": tensors["node_offsets"],
                    },
                    output_dir / "best_model.pt",
                )

        if best_epoch and epoch - best_epoch >= args.patience:
            log(f"early stopping at epoch={epoch}, best_epoch={best_epoch}, best_validation_score={best_val:.4f}")
            break

    checkpoint = safe_torch_load(output_dir / "best_model.pt", map_location=device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    classifier.load_state_dict(checkpoint["classifier_state_dict"])
    historic_head.load_state_dict(checkpoint["historic_head_state_dict"])
    daily_head.load_state_dict(checkpoint["daily_head_state_dict"])
    test_metrics = evaluate(
        encoder,
        classifier,
        historic_head,
        daily_head,
        tensors,
        station_global_idx,
        split["test"],
        labels,
        historic_target,
        daily_target,
    )

    encoder.eval()
    classifier.eval()
    with torch.no_grad():
        final_emb = encoder(tensors["x"], tensors["node_type"], tensors["edge_time"], tensors["edge_index"], tensors["edge_type"])
        final_station_emb = final_emb[station_global_idx]
        final_logits = classifier(final_station_emb)
        final_historic_pred = torch.sigmoid(historic_head(final_station_emb).squeeze(-1))
        final_daily_pred = torch.sigmoid(daily_head(final_station_emb).squeeze(-1))

    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "classifier_state_dict": classifier.state_dict(),
            "historic_head_state_dict": historic_head.state_dict(),
            "daily_head_state_dict": daily_head.state_dict(),
            "args": json_safe(vars(args)),
            "test_metrics": test_metrics,
        },
        output_dir / "last_model.pt",
    )
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    save_predictions(
        output_dir,
        graph_dir,
        final_logits,
        final_historic_pred,
        final_daily_pred,
        labels,
        historic_target,
        daily_target,
        split,
    )

    manifest = {
        "graph_dir": str(graph_dir),
        "tensors_path": str(tensors_path),
        "output_dir": str(output_dir),
        "best_epoch": best_epoch,
        "best_validation_score": best_val,
        "test_metrics": test_metrics,
        "args": json_safe(vars(args)),
        "split_sizes": {k: int(len(v)) for k, v in split.items()},
        "num_classes": num_classes,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log(
        "complete: "
        f"test_conflict_acc={test_metrics['conflict_accuracy']:.4f}, "
        f"test_conflict_f1={test_metrics['conflict_macro_f1']:.4f}, "
        f"test_historic_mae={test_metrics['historic_mae']:.4f}, "
        f"test_daily_mae={test_metrics['daily_mae']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
