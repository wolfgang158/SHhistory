#!/usr/bin/env python3
"""Visualize the generated HGT heterogeneous graph.

Outputs:
- meta_graph.png: type-level schema graph.
- station_subgraph_<id>.png: sampled local station subgraph.
- station_subgraph_<id>.html: lightweight interactive local subgraph.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


ROOT = Path(__file__).resolve().parents[2]
GRAPH_DIR = ROOT / "data" / "hgt_graph"
OUT_DIR = ROOT / "data" / "hgt_graph" / "viz"

TYPE_COLORS = {
    "station": "#d33f49",
    "building": "#9471a5",
    "road": "#50514f",
    "poi": "#2f80ed",
    "conservation_area": "#0b6e4f",
    "admin_area": "#f2c14e",
}

REL_COLORS = {
    "near_building": "#9471a5",
    "near_road": "#50514f",
    "near_poi_daily": "#2f80ed",
    "near_poi_tour": "#00a896",
    "near_poi_transport": "#f77f00",
    "near_poi_other": "#6c757d",
    "inside_conservation": "#0b6e4f",
    "inside_admin": "#c49a00",
}


def safe_name(value: Any) -> str:
    text = str(value) if value is not None else ""
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = re.sub(r"[^\w\-]+", "_", text, flags=re.UNICODE).strip("_")
    return text[:80] or "station"


def read_nodes(graph_dir: Path) -> dict[str, pd.DataFrame]:
    nodes_dir = graph_dir / "nodes"
    return {path.stem: pd.read_csv(path) for path in sorted(nodes_dir.glob("*.csv"))}


def load_edges(graph_dir: Path) -> pd.DataFrame:
    return pd.read_csv(graph_dir / "edges.csv")


def draw_meta_graph(graph_dir: Path, out_dir: Path) -> Path:
    manifest = json.loads((graph_dir / "manifest.json").read_text(encoding="utf-8"))
    graph = nx.MultiDiGraph()
    for node_type, count in manifest["node_counts"].items():
        graph.add_node(node_type, count=count)
    for key, count in manifest["edge_counts"].items():
        src, rel, dst = key.split("|")
        if rel.startswith("rev_"):
            continue
        graph.add_edge(src, dst, relation=rel, count=count)

    pos = {
        "station": (0.0, 0.0),
        "building": (-1.8, 1.0),
        "road": (0.0, 1.55),
        "poi": (1.8, 1.0),
        "conservation_area": (-1.4, -1.15),
        "admin_area": (1.4, -1.15),
    }
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor("#f7f4ef")
    sizes = [1200 + math.log1p(graph.nodes[n]["count"]) * 450 for n in graph.nodes]
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=[TYPE_COLORS.get(n, "#888888") for n in graph.nodes],
        node_size=sizes,
        edgecolors="#252525",
        linewidths=1.4,
        ax=ax,
    )
    nx.draw_networkx_labels(
        graph,
        pos,
        labels={n: f"{n}\n{graph.nodes[n]['count']:,}" for n in graph.nodes},
        font_size=10,
        font_color="#ffffff",
        ax=ax,
    )
    for src, dst, data in graph.edges(data=True):
        rel = data["relation"]
        rad = 0.12 if rel.startswith("near_poi") else 0.04
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=[(src, dst)],
            width=1.5 + math.log1p(data["count"]) / 3,
            edge_color=REL_COLORS.get(rel, "#333333"),
            arrows=True,
            arrowsize=18,
            connectionstyle=f"arc3,rad={rad}",
            ax=ax,
        )
        sx, sy = pos[src]
        tx, ty = pos[dst]
        lx, ly = (sx + tx) / 2, (sy + ty) / 2
        ax.text(lx, ly, f"{rel}\n{data['count']:,}", fontsize=8, ha="center", va="center", color="#202020")
    ax.set_title("HGT Heterogeneous Graph Schema", fontsize=16, pad=18)
    ax.axis("off")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "meta_graph.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def choose_station(stations: pd.DataFrame, query: str | None) -> int:
    if not query:
        cx, cy = stations["x"].median(), stations["y"].median()
        dist = (stations["x"] - cx) ** 2 + (stations["y"] - cy) ** 2
        return int(dist.idxmin())
    if query.isdigit() and int(query) in stations.index:
        return int(query)
    q = query.lower()
    hits = stations[stations["id"].astype(str).str.lower().str.contains(q, regex=False)]
    if hits.empty:
        hits = stations[stations["name"].astype(str).str.lower().str.contains(q, regex=False)]
    if hits.empty:
        raise ValueError(f"No station matched query: {query}")
    return int(hits.index[0])


def sample_station_edges(edges: pd.DataFrame, station_id: int, max_per_relation: int) -> pd.DataFrame:
    direct = edges[(edges["source_type"].eq("station")) & (edges["source"].eq(station_id))]
    direct = direct[~direct["relation"].str.startswith("rev_")]
    groups = []
    for _, group in direct.groupby("relation", sort=True):
        if "weight" in group:
            group = group.sort_values("weight", ascending=False)
        groups.append(group.head(max_per_relation))
    return pd.concat(groups, ignore_index=True) if groups else direct.head(0)


def build_local_graph(nodes: dict[str, pd.DataFrame], sampled_edges: pd.DataFrame, station_id: int) -> nx.Graph:
    graph = nx.Graph()
    station = nodes["station"].iloc[station_id]
    graph.add_node(("station", station_id), type="station", label=str(station["name"]), x=station["x"], y=station["y"])
    for _, edge in sampled_edges.iterrows():
        src = (edge["source_type"], int(edge["source"]))
        dst = (edge["target_type"], int(edge["target"]))
        for node_type, node_id in [src, dst]:
            if graph.has_node((node_type, node_id)):
                continue
            row = nodes[node_type].iloc[node_id]
            label = row.get("name", row.get("id", f"{node_type}:{node_id}"))
            graph.add_node((node_type, node_id), type=node_type, label=str(label), x=float(row["x"]), y=float(row["y"]))
        graph.add_edge(src, dst, relation=edge["relation"], weight=float(edge.get("weight", 1.0)))
    return graph


def draw_station_subgraph(graph: nx.Graph, station_label: str, out_path: Path) -> None:
    xs = np.array([graph.nodes[n]["x"] for n in graph.nodes], dtype=float)
    ys = np.array([graph.nodes[n]["y"] for n in graph.nodes], dtype=float)
    cx, cy = graph.nodes[next(n for n in graph.nodes if n[0] == "station")]["x"], graph.nodes[next(n for n in graph.nodes if n[0] == "station")]["y"]
    pos = {n: ((graph.nodes[n]["x"] - cx) / 1000.0, (graph.nodes[n]["y"] - cy) / 1000.0) for n in graph.nodes}

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_facecolor("#f8f7f2")
    for rel, edge_list in group_edges(graph).items():
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=edge_list,
            width=0.8,
            alpha=0.28,
            edge_color=REL_COLORS.get(rel, "#777777"),
            ax=ax,
        )
    for node_type in TYPE_COLORS:
        nodelist = [n for n in graph.nodes if graph.nodes[n]["type"] == node_type]
        if not nodelist:
            continue
        size = 430 if node_type == "station" else 75 if node_type in {"conservation_area", "admin_area"} else 26
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=nodelist,
            node_color=TYPE_COLORS[node_type],
            node_size=size,
            alpha=0.9,
            linewidths=0.6,
            edgecolors="#222222",
            label=f"{node_type} ({len(nodelist)})",
            ax=ax,
        )
    labels = {n: graph.nodes[n]["label"] for n in graph.nodes if n[0] in {"station", "conservation_area", "admin_area"}}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8, ax=ax)
    ax.set_title(f"Station Local Heterogeneous Subgraph: {station_label}", fontsize=15, pad=14)
    ax.set_xlabel("km from station center")
    ax.set_ylabel("km from station center")
    ax.grid(True, color="#ddd8cf", linewidth=0.6)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def group_edges(graph: nx.Graph) -> dict[str, list[tuple[Any, Any]]]:
    grouped: dict[str, list[tuple[Any, Any]]] = {}
    for src, dst, data in graph.edges(data=True):
        grouped.setdefault(data.get("relation", "edge"), []).append((src, dst))
    return grouped


def write_interactive_html(graph: nx.Graph, station_label: str, out_path: Path) -> None:
    station_node = next(n for n in graph.nodes if n[0] == "station")
    cx, cy = graph.nodes[station_node]["x"], graph.nodes[station_node]["y"]
    node_payload = []
    for node in graph.nodes:
        data = graph.nodes[node]
        node_payload.append(
            {
                "id": f"{node[0]}:{node[1]}",
                "type": data["type"],
                "label": data["label"][:60],
                "x": (data["x"] - cx) / 1000.0,
                "y": -(data["y"] - cy) / 1000.0,
                "color": TYPE_COLORS.get(data["type"], "#999999"),
            }
        )
    edge_payload = [
        {
            "source": f"{src[0]}:{src[1]}",
            "target": f"{dst[0]}:{dst[1]}",
            "relation": data.get("relation", "edge"),
            "color": REL_COLORS.get(data.get("relation", ""), "#999999"),
        }
        for src, dst, data in graph.edges(data=True)
    ]
    payload = json.dumps({"nodes": node_payload, "edges": edge_payload}, ensure_ascii=False)
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Station HGT Subgraph</title>
  <style>
    body {{ margin: 0; font-family: Segoe UI, sans-serif; background: #f8f7f2; color: #222; }}
    header {{ padding: 14px 18px; border-bottom: 1px solid #d8d2c8; display: flex; gap: 20px; align-items: baseline; }}
    h1 {{ margin: 0; font-size: 18px; font-weight: 650; }}
    #meta {{ font-size: 13px; color: #555; }}
    svg {{ display: block; width: 100vw; height: calc(100vh - 52px); cursor: grab; }}
    .edge {{ stroke-opacity: 0.32; stroke-width: 1.1; }}
    .node {{ stroke: #222; stroke-width: 0.7; }}
    .label {{ font-size: 11px; paint-order: stroke; stroke: #f8f7f2; stroke-width: 3px; }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(station_label)}</h1>
    <div id="meta"></div>
  </header>
  <svg id="viz" viewBox="-3 -3 6 6" preserveAspectRatio="xMidYMid meet"></svg>
  <script>
    const data = {payload};
    const svg = document.getElementById('viz');
    document.getElementById('meta').textContent = `${{data.nodes.length}} nodes / ${{data.edges.length}} edges`;
    const ns = 'http://www.w3.org/2000/svg';
    const x = v => v;
    const y = v => v;
    const byId = Object.fromEntries(data.nodes.map(n => [n.id, n]));
    for (const e of data.edges) {{
      const s = byId[e.source], t = byId[e.target];
      const line = document.createElementNS(ns, 'line');
      line.setAttribute('x1', x(s.x));
      line.setAttribute('y1', y(s.y));
      line.setAttribute('x2', x(t.x));
      line.setAttribute('y2', y(t.y));
      line.setAttribute('stroke', e.color);
      line.setAttribute('class', 'edge');
      const title = document.createElementNS(ns, 'title');
      title.textContent = e.relation;
      line.appendChild(title);
      svg.appendChild(line);
    }}
    for (const n of data.nodes) {{
      const r = n.type === 'station' ? 0.07 : (n.type.endsWith('_area') ? 0.045 : 0.026);
      const c = document.createElementNS(ns, 'circle');
      c.setAttribute('cx', x(n.x));
      c.setAttribute('cy', y(n.y));
      c.setAttribute('r', r);
      c.setAttribute('fill', n.color);
      c.setAttribute('class', 'node');
      const title = document.createElementNS(ns, 'title');
      title.textContent = `${{n.type}}: ${{n.label}}`;
      c.appendChild(title);
      svg.appendChild(c);
      if (n.type === 'station' || n.type.endsWith('_area')) {{
        const text = document.createElementNS(ns, 'text');
        text.setAttribute('x', x(n.x) + 0.06);
        text.setAttribute('y', y(n.y) - 0.04);
        text.setAttribute('class', 'label');
        text.textContent = n.label;
        svg.appendChild(text);
      }}
    }}
  </script>
</body>
</html>
"""
    out_path.write_text(html_text, encoding="utf-8")


def visualize_station(graph_dir: Path, out_dir: Path, query: str | None, max_per_relation: int) -> tuple[Path, Path]:
    nodes = read_nodes(graph_dir)
    edges = load_edges(graph_dir)
    station_id = choose_station(nodes["station"], query)
    station_label = str(nodes["station"].iloc[station_id]["name"])
    sampled = sample_station_edges(edges, station_id, max_per_relation)
    local_graph = build_local_graph(nodes, sampled, station_id)
    stem = f"station_subgraph_{station_id:04d}_{safe_name(station_label)}"
    png_path = out_dir / f"{stem}.png"
    html_path = out_dir / f"{stem}.html"
    draw_station_subgraph(local_graph, station_label, png_path)
    write_interactive_html(local_graph, station_label, html_path)
    return png_path, html_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize generated HGT graph.")
    parser.add_argument("--graph-dir", type=Path, default=GRAPH_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--station", default=None, help="Station index, id fragment, or name fragment.")
    parser.add_argument("--max-per-relation", type=int, default=80, help="Max local edges shown per relation type.")
    parser.add_argument("--skip-meta", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    if not args.skip_meta:
        outputs.append(draw_meta_graph(args.graph_dir, args.output_dir))
    outputs.extend(visualize_station(args.graph_dir, args.output_dir, args.station, args.max_per_relation))
    for path in outputs:
        print(path.relative_to(ROOT).as_posix().encode("ascii", errors="ignore").decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
