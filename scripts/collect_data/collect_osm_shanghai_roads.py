#!/usr/bin/env python3
"""Download Shanghai road network from OpenStreetMap and export edge segments."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
import requests
from osmnx._errors import InsufficientResponseError


PLACE_NAME = "Shanghai, China"
SHANGHAI_BBOX = (120.85, 30.67, 122.12, 31.88)  # west, south, east, north
OUTPUT_DIR = Path("data/raw/road_segments")
TILE_DIR = OUTPUT_DIR / "tiles"
FAILED_TILES_PATH = OUTPUT_DIR / "failed_tiles.jsonl"
EMPTY_TILES_PATH = OUTPUT_DIR / "empty_tiles.jsonl"
GPKG_PATH = OUTPUT_DIR / "shanghai_road_segments.gpkg"
GEOJSON_PATH = OUTPUT_DIR / "shanghai_road_segments.geojson"
CSV_PATH = OUTPUT_DIR / "shanghai_road_segments.csv"
SUMMARY_PATH = OUTPUT_DIR / "shanghai_road_segments_summary.json"


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def stringify_osm_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(item) for item in value)
    return str(value)


def get_text_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return df[column].apply(stringify_osm_value)
    return pd.Series([""] * len(df), index=df.index)


def build_segments(edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    segments = edges.reset_index().copy()
    segments["segment_id"] = [
        f"seg_{idx:09d}" for idx in range(1, len(segments) + 1)
    ]
    segments["from_node"] = segments["u"]
    segments["to_node"] = segments["v"]
    segments["road_name"] = get_text_column(segments, "name")
    segments["osmid"] = get_text_column(segments, "osmid")
    segments["road_type"] = get_text_column(segments, "highway")
    segments["highway"] = segments["road_type"]
    if "length" not in segments.columns:
        segments["length"] = segments.geometry.length

    keep_cols = [
        "segment_id",
        "osmid",
        "road_name",
        "highway",
        "road_type",
        "length",
        "from_node",
        "to_node",
        "geometry",
    ]
    return segments[keep_cols].set_geometry("geometry")


def build_bbox_grid(
    bbox: tuple[float, float, float, float],
    rows: int,
    cols: int,
) -> list[tuple[float, float, float, float]]:
    west, south, east, north = bbox
    lon_step = (east - west) / cols
    lat_step = (north - south) / rows
    tiles = []
    for row in range(rows):
        for col in range(cols):
            tile_west = west + col * lon_step
            tile_east = west + (col + 1) * lon_step
            tile_south = south + row * lat_step
            tile_north = south + (row + 1) * lat_step
            # OSMnx 2.x bbox order is left, bottom, right, top.
            tiles.append((tile_west, tile_south, tile_east, tile_north))
    return tiles


def save_outputs(segments: gpd.GeoDataFrame) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    segments.to_file(GPKG_PATH, layer="road_segments", driver="GPKG")
    segments.to_file(GEOJSON_PATH, driver="GeoJSON")

    csv_df = segments.copy()
    csv_df["geometry"] = csv_df.geometry.to_wkt()
    csv_df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

    road_type_counts = (
        segments["road_type"]
        .replace("", "unknown")
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
    )
    missing_road_name_count = int(segments["road_name"].eq("").sum())
    summary = {
        "place": PLACE_NAME,
        "network_type": "drive",
        "segment_count": int(len(segments)),
        "road_type_counts": road_type_counts,
        "missing_road_name_count": missing_road_name_count,
        "outputs": {
            "gpkg": str(GPKG_PATH),
            "geojson": str(GEOJSON_PATH),
            "csv": str(CSV_PATH),
        },
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def collect_roads(network_type: str = "drive") -> gpd.GeoDataFrame:
    logging.info("Downloading OSM road network: place=%s network_type=%s", PLACE_NAME, network_type)
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(OUTPUT_DIR / "osmnx_cache")
    ox.settings.log_console = False
    ox.settings.timeout = 240

    graph = ox.graph_from_place(
        PLACE_NAME,
        network_type=network_type,
        simplify=True,
        retain_all=True,
        truncate_by_edge=True,
    )
    logging.info(
        "Downloaded graph: nodes=%s edges=%s",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )

    if not isinstance(graph, nx.MultiDiGraph):
        graph = nx.MultiDiGraph(graph)

    _, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True, fill_edge_geometry=True)
    logging.info("Converted graph edges to GeoDataFrame: rows=%s crs=%s", len(edges), edges.crs)
    return build_segments(edges)


def collect_tile(
    tile_id: int,
    bbox: tuple[float, float, float, float],
    network_type: str,
    force: bool,
    retries: int,
    retry_sleep: float,
) -> Path | None:
    TILE_DIR.mkdir(parents=True, exist_ok=True)
    tile_path = TILE_DIR / f"tile_{tile_id:03d}.gpkg"
    if tile_path.exists() and not force:
        logging.info("Skip existing tile %03d: %s", tile_id, tile_path)
        return tile_path
    if is_recorded_tile(EMPTY_TILES_PATH, tile_id) and not force:
        logging.info("Skip known empty tile %03d", tile_id)
        return None
    if is_recorded_tile(FAILED_TILES_PATH, tile_id) and not force:
        logging.info("Skip known failed tile %03d", tile_id)
        return None

    graph = None
    for attempt in range(1, retries + 1):
        logging.info("Downloading tile %03d bbox=%s attempt=%s/%s", tile_id, bbox, attempt, retries)
        try:
            graph = ox.graph_from_bbox(
                bbox,
                network_type=network_type,
                simplify=True,
                retain_all=True,
                truncate_by_edge=True,
            )
            break
        except InsufficientResponseError as exc:
            logging.warning("Tile %03d returned no data; skip. error=%s", tile_id, exc)
            record_empty_tile(tile_id, bbox, network_type, exc)
            return None
        except ValueError as exc:
            if "Found no graph nodes within the requested polygon" in str(exc):
                logging.warning("Tile %03d has no graph nodes; skip. error=%s", tile_id, exc)
                record_empty_tile(tile_id, bbox, network_type, exc)
                return None
            raise
        except requests.exceptions.RequestException as exc:
            logging.warning("Tile %03d request failed attempt=%s/%s error=%s", tile_id, attempt, retries, exc)
            if attempt == retries:
                record_failed_tile(tile_id, bbox, network_type, exc)
                return None
            time.sleep(retry_sleep * attempt)

    if graph is None:
        raise RuntimeError(f"Tile {tile_id:03d} did not produce a graph.")
    logging.info(
        "Tile %03d graph: nodes=%s edges=%s",
        tile_id,
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    if graph.number_of_edges() == 0:
        logging.info("Tile %03d has no edges; skip tile file.", tile_id)
        return None

    _, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True, fill_edge_geometry=True)
    segments = build_segments(edges)
    segments["tile_id"] = tile_id
    segments.to_file(tile_path, layer="road_segments", driver="GPKG")
    logging.info("Saved tile %03d segments=%s path=%s", tile_id, len(segments), tile_path)
    return tile_path


def record_failed_tile(
    tile_id: int,
    bbox: tuple[float, float, float, float],
    network_type: str,
    exc: BaseException,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "tile_id": tile_id,
        "bbox": bbox,
        "network_type": network_type,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with FAILED_TILES_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    logging.error("Tile %03d failed after retries; recorded in %s", tile_id, FAILED_TILES_PATH)


def record_empty_tile(
    tile_id: int,
    bbox: tuple[float, float, float, float],
    network_type: str,
    exc: BaseException,
) -> None:
    if is_recorded_tile(EMPTY_TILES_PATH, tile_id):
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "tile_id": tile_id,
        "bbox": bbox,
        "network_type": network_type,
        "reason": type(exc).__name__,
        "message": str(exc),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with EMPTY_TILES_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def is_recorded_tile(path: Path, tile_id: int) -> bool:
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("tile_id") == tile_id:
                return True
    return False


def read_recorded_tile_ids(path: Path) -> set[int]:
    tile_ids: set[int] = set()
    if not path.exists():
        return tile_ids
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            tile_id = payload.get("tile_id")
            if isinstance(tile_id, int):
                tile_ids.add(tile_id)
    return tile_ids


def parse_tile_ids(value: str | None) -> set[int] | None:
    if not value:
        return None
    tile_ids: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start > end:
                raise ValueError(f"Invalid tile range: {part}")
            tile_ids.update(range(start, end + 1))
        else:
            tile_ids.add(int(part))
    return tile_ids


def collect_roads_by_bbox_grid(
    network_type: str,
    rows: int,
    cols: int,
    sleep_seconds: float,
    force_tiles: bool,
    retries: int,
    retry_sleep: float,
    overpass_url: str | None,
    only_failed: bool,
    only_tiles: set[int] | None,
) -> gpd.GeoDataFrame:
    logging.info(
        "Downloading OSM road network by bbox grid: bbox=%s rows=%s cols=%s network_type=%s",
        SHANGHAI_BBOX,
        rows,
        cols,
        network_type,
    )
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(OUTPUT_DIR / "osmnx_cache")
    ox.settings.log_console = False
    ox.settings.timeout = 240
    if overpass_url:
        ox.settings.overpass_url = overpass_url
        logging.info("Using custom Overpass URL: %s", overpass_url)

    selected_tile_ids = only_tiles
    if only_failed:
        failed_tile_ids = read_recorded_tile_ids(FAILED_TILES_PATH)
        selected_tile_ids = failed_tile_ids if selected_tile_ids is None else selected_tile_ids & failed_tile_ids
        logging.info("Only failed tile mode enabled: tile_ids=%s", sorted(selected_tile_ids))

    tile_paths: list[Path] = []
    for tile_id, bbox in enumerate(build_bbox_grid(SHANGHAI_BBOX, rows, cols), start=1):
        if selected_tile_ids is not None and tile_id not in selected_tile_ids:
            continue
        tile_path = collect_tile(
            tile_id,
            bbox,
            network_type,
            force=force_tiles,
            retries=retries,
            retry_sleep=retry_sleep,
        )
        if tile_path:
            tile_paths.append(tile_path)
        time.sleep(sleep_seconds)

    all_tile_paths = sorted(TILE_DIR.glob("tile_*.gpkg"))
    if all_tile_paths:
        tile_paths = all_tile_paths

    if not tile_paths:
        raise RuntimeError("No tile files were generated.")

    frames = [gpd.read_file(path, layer="road_segments") for path in tile_paths]
    merged = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    merged["dedupe_key"] = (
        merged["osmid"].astype(str)
        + "|"
        + merged["from_node"].astype(str)
        + "|"
        + merged["to_node"].astype(str)
        + "|"
        + merged.geometry.to_wkb().map(lambda value: value.hex())
    )
    merged = merged.drop_duplicates("dedupe_key").drop(columns=["dedupe_key", "tile_id"], errors="ignore")
    merged = merged.reset_index(drop=True)
    merged["segment_id"] = [f"seg_{idx:09d}" for idx in range(1, len(merged) + 1)]
    logging.info("Merged tile segments: unique_rows=%s", len(merged))
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Shanghai OSM road network and export edge segments."
    )
    parser.add_argument(
        "--network-type",
        choices=["drive", "all", "all_private", "walk"],
        default="drive",
        help="OSMnx network type. Default prefers drive roads.",
    )
    parser.add_argument(
        "--method",
        choices=["bbox-grid", "place"],
        default="bbox-grid",
        help="Use bbox-grid by default to avoid one huge Overpass request.",
    )
    parser.add_argument("--grid-rows", type=int, default=8)
    parser.add_argument("--grid-cols", type=int, default=8)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--force-tiles", action="store_true")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=10.0)
    parser.add_argument("--overpass-url", default=None)
    parser.add_argument(
        "--only-failed",
        action="store_true",
        help="Only retry tile ids recorded in data/raw/road_segments/failed_tiles.jsonl, then merge all existing tile files.",
    )
    parser.add_argument(
        "--only-tiles",
        default=None,
        help="Comma-separated tile ids or ranges to process, e.g. 95,98-101. Final output still merges existing tile files.",
    )
    return parser.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()
    if args.method == "place":
        segments = collect_roads(network_type=args.network_type)
    else:
        segments = collect_roads_by_bbox_grid(
            network_type=args.network_type,
            rows=args.grid_rows,
            cols=args.grid_cols,
            sleep_seconds=args.sleep,
            force_tiles=args.force_tiles,
            retries=args.retries,
            retry_sleep=args.retry_sleep,
            overpass_url=args.overpass_url,
            only_failed=args.only_failed,
            only_tiles=parse_tile_ids(args.only_tiles),
        )
    summary = save_outputs(segments)

    print("街段总数:", summary["segment_count"])
    print("道路类型统计:")
    for road_type, count in summary["road_type_counts"].items():
        print(f"  {road_type}: {count}")
    print("缺失道路名数量:", summary["missing_road_name_count"])
    print("文件保存路径:")
    for label, path in summary["outputs"].items():
        print(f"  {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
