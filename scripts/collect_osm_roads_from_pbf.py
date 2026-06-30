#!/usr/bin/env python3
"""Extract Shanghai driving road segments from a local OSM PBF file."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from pyrosm import OSM


DEFAULT_PBF_PATH = Path("data/raw/shanghai-260629.osm.pbf")
OUTPUT_DIR = Path("data/osm")
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


def first_existing_column(gdf: gpd.GeoDataFrame, candidates: list[str]) -> pd.Series:
    for column in candidates:
        if column in gdf.columns:
            return gdf[column].apply(stringify_osm_value)
    return pd.Series([""] * len(gdf), index=gdf.index)


def calculate_lengths_meters(roads: gpd.GeoDataFrame) -> pd.Series:
    if roads.empty:
        return pd.Series([], dtype="float64", index=roads.index)
    metric = roads.to_crs("EPSG:32651")
    return metric.geometry.length


def build_segments(roads: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    roads = roads.copy()
    roads = roads[roads.geometry.notna()].copy()
    roads = roads[~roads.geometry.is_empty].copy()
    roads = roads.reset_index(drop=True)

    segments = gpd.GeoDataFrame(
        {
            "segment_id": [f"seg_{idx:09d}" for idx in range(1, len(roads) + 1)],
            "osmid": first_existing_column(roads, ["id", "osm_id", "osmid"]),
            "road_name": first_existing_column(roads, ["name"]),
            "highway": first_existing_column(roads, ["highway"]),
            "length": calculate_lengths_meters(roads),
        },
        geometry=roads.geometry,
        crs=roads.crs,
    )
    return segments


def save_outputs(segments: gpd.GeoDataFrame, pbf_path: Path) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    segments.to_file(GPKG_PATH, layer="road_segments", driver="GPKG")
    segments.to_file(GEOJSON_PATH, driver="GeoJSON")

    csv_df = pd.DataFrame(segments.drop(columns="geometry"))
    csv_df["geometry"] = segments.geometry.to_wkt()
    csv_df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

    highway_counts = (
        segments["highway"]
        .replace("", "unknown")
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
    )
    summary = {
        "source": str(pbf_path),
        "method": "pyrosm.get_network(network_type='driving')",
        "segment_count": int(len(segments)),
        "highway_counts": highway_counts,
        "missing_road_name_count": int(segments["road_name"].eq("").sum()),
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


def collect_from_pbf(pbf_path: Path) -> gpd.GeoDataFrame:
    if not pbf_path.exists():
        raise FileNotFoundError(f"PBF file not found: {pbf_path}")

    logging.info("Reading local PBF with pyrosm: %s", pbf_path)
    osm = OSM(str(pbf_path))
    roads = osm.get_network(network_type="driving", nodes=False)
    logging.info("Extracted driving road rows: %s", len(roads))
    logging.info("Input columns: %s", ", ".join(roads.columns))

    segments = build_segments(roads)
    logging.info("Built road segments: %s", len(segments))
    return segments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Shanghai driving road segments from a local OSM PBF file."
    )
    parser.add_argument(
        "--pbf",
        type=Path,
        default=DEFAULT_PBF_PATH,
        help=f"Local OSM PBF path. Default: {DEFAULT_PBF_PATH}",
    )
    return parser.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()
    segments = collect_from_pbf(args.pbf)
    summary = save_outputs(segments, args.pbf)

    print("街段总数:", summary["segment_count"])
    print("道路类型统计:")
    for highway, count in summary["highway_counts"].items():
        print(f"  {highway}: {count}")
    print("缺失道路名数量:", summary["missing_road_name_count"])
    print("文件保存路径:")
    for label, path in summary["outputs"].items():
        print(f"  {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
