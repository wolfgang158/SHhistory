#!/usr/bin/env python3
"""Extract Shanghai administrative boundaries from a local OSM PBF file."""

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
ADMIN_GPKG_PATH = OUTPUT_DIR / "shanghai_admin_boundary.gpkg"
ADMIN_GEOJSON_PATH = OUTPUT_DIR / "shanghai_admin_boundary.geojson"
ADMIN_CSV_PATH = OUTPUT_DIR / "shanghai_admin_boundary.csv"
STREET_GPKG_PATH = OUTPUT_DIR / "shanghai_street_boundary.gpkg"
SUMMARY_PATH = OUTPUT_DIR / "shanghai_admin_boundary_summary.json"


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


def normalize_admin_level(value: Any) -> str:
    text = stringify_osm_value(value)
    if text.endswith(".0"):
        return text[:-2]
    return text


def prepare_admin_boundaries(boundaries: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    boundaries = boundaries.copy()
    boundaries = boundaries[boundaries.geometry.notna()].copy()
    boundaries = boundaries[~boundaries.geometry.is_empty].copy()

    if "boundary" not in boundaries.columns:
        boundaries["boundary"] = ""
    if "admin_level" not in boundaries.columns:
        boundaries["admin_level"] = ""
    if "name" not in boundaries.columns:
        boundaries["name"] = ""

    boundaries["boundary"] = boundaries["boundary"].apply(stringify_osm_value)
    boundaries["admin_level"] = boundaries["admin_level"].apply(normalize_admin_level)
    boundaries["name"] = boundaries["name"].apply(stringify_osm_value)
    admin = boundaries[boundaries["boundary"].eq("administrative")].copy()
    admin = admin[["name", "admin_level", "boundary", "geometry"]].reset_index(drop=True)
    return admin


def numeric_admin_levels(admin: gpd.GeoDataFrame) -> pd.Series:
    return pd.to_numeric(admin["admin_level"], errors="coerce")


def classify_street_level(admin: gpd.GeoDataFrame) -> str | None:
    levels = sorted(
        int(level)
        for level in numeric_admin_levels(admin).dropna().unique()
        if int(level) > 6
    )
    if 10 in levels:
        return "10"
    if levels:
        return str(levels[-1])
    return None


def build_summary(admin: gpd.GeoDataFrame, street_level: str | None) -> dict[str, Any]:
    level_counts = (
        admin["admin_level"]
        .replace("", "unknown")
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
    )
    shanghai_city_count = int(
        admin["name"].str.contains("上海", case=False, na=False).sum()
    )
    county_count = int(admin["admin_level"].eq("6").sum())
    street_count = (
        int(admin["admin_level"].eq(street_level).sum())
        if street_level is not None
        else 0
    )
    return {
        "source": str(DEFAULT_PBF_PATH),
        "method": "pyrosm.get_boundaries(boundary_type='all') then boundary == administrative",
        "admin_boundary_count": int(len(admin)),
        "shanghai_city_boundary_count": shanghai_city_count,
        "county_boundary_count_admin_level_6": county_count,
        "street_boundary_level_used": street_level,
        "street_boundary_count": street_count,
        "admin_level_counts": level_counts,
        "outputs": {
            "admin_gpkg": str(ADMIN_GPKG_PATH),
            "admin_geojson": str(ADMIN_GEOJSON_PATH),
            "admin_csv": str(ADMIN_CSV_PATH),
            "street_gpkg": str(STREET_GPKG_PATH) if street_level is not None else "",
        },
    }


def save_outputs(admin: gpd.GeoDataFrame, summary: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    admin.to_file(ADMIN_GPKG_PATH, layer="admin_boundary", driver="GPKG")
    admin.to_file(ADMIN_GEOJSON_PATH, driver="GeoJSON")

    csv_df = pd.DataFrame(admin.drop(columns="geometry"))
    csv_df["geometry"] = admin.geometry.to_wkt()
    csv_df.to_csv(ADMIN_CSV_PATH, index=False, encoding="utf-8-sig")

    street_level = summary["street_boundary_level_used"]
    if street_level is not None:
        street = admin[admin["admin_level"].eq(street_level)].copy()
        street.to_file(STREET_GPKG_PATH, layer="street_boundary", driver="GPKG")

    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def collect_admin_boundaries(pbf_path: Path) -> gpd.GeoDataFrame:
    if not pbf_path.exists():
        raise FileNotFoundError(f"PBF file not found: {pbf_path}")

    logging.info("Reading local PBF with pyrosm: %s", pbf_path)
    osm = OSM(str(pbf_path))
    boundaries = osm.get_boundaries(
        boundary_type="all",
        tags_to_keep=["name", "admin_level", "boundary"],
    )
    logging.info("Extracted boundary rows: %s", len(boundaries))
    logging.info("Input columns: %s", ", ".join(boundaries.columns))
    admin = prepare_admin_boundaries(boundaries)
    logging.info("Administrative boundary rows: %s", len(admin))
    return admin


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Shanghai administrative boundaries from a local OSM PBF file."
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
    admin = collect_admin_boundaries(args.pbf)
    street_level = classify_street_level(admin)
    summary = build_summary(admin, street_level)
    save_outputs(admin, summary)

    print("上海市边界数量:", summary["shanghai_city_boundary_count"])
    print("区县数量:", summary["county_boundary_count_admin_level_6"])
    print("街道数量:", summary["street_boundary_count"])
    print("街道级 admin_level 使用:", summary["street_boundary_level_used"])
    print("各 admin_level 数量:")
    for admin_level, count in summary["admin_level_counts"].items():
        print(f"  {admin_level}: {count}")
    print("文件保存路径:")
    for label, path in summary["outputs"].items():
        if path:
            print(f"  {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
