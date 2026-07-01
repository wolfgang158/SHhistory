#!/usr/bin/env python3
"""Build a station-centered heterogeneous graph for HGT experiments.

The script follows docs/preprocess.md at graph level: metro stations are sample
anchors, and nearby historic buildings, roads, POIs, conservation areas, and
administrative areas become typed nodes connected by typed spatial relations.
It writes both a pyHGT Graph pickle and PyG HeteroData/tensor artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import dill
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from shapely.geometry import Point, shape
from shapely.ops import transform
from shapely import wkt
from torch_geometric.data import HeteroData


ROOT = Path(__file__).resolve().parents[2]
HGT_DIR = ROOT / "HGT"
if str(HGT_DIR) not in sys.path:
    sys.path.insert(0, str(HGT_DIR))

from pyHGT.data import Graph  # noqa: E402


FEATURE_DIM = 16
YEAR = 2026
DAILY_KEYWORDS = ("餐饮", "购物", "生活", "医疗", "金融", "公司", "商务", "住宅", "政府")
TOUR_KEYWORDS = ("风景", "住宿", "体育", "休闲", "科教", "文化", "事件")
TRANSPORT_KEYWORDS = ("交通", "道路", "通行")
MAJOR_HIGHWAYS = {"motorway", "trunk", "primary", "secondary"}
LOCAL_HIGHWAYS = {"tertiary", "residential", "service", "unclassified", "living_street", "road"}
DAILY_KEYWORDS_UTF8 = (
    "餐饮",
    "购物",
    "生活",
    "医疗",
    "金融",
    "公司",
    "商务",
    "住宅",
    "政府",
    "学校",
    "市场",
    "超市",
    "便利店",
    "公厕",
    "社区",
    "居委",
)
TOUR_KEYWORDS_UTF8 = ("风景", "景点", "酒店", "住宿", "体育", "休闲", "科教", "文化", "公园", "旅游", "展览")
TRANSPORT_KEYWORDS_UTF8 = ("交通", "道路", "通行", "公交", "地铁", "停车")
HISTORIC_KEYWORDS = (
    "历史",
    "文物",
    "遗址",
    "旧址",
    "故居",
    "古镇",
    "古街",
    "老街",
    "博物馆",
    "纪念",
    "名人",
    "文化",
    "寺",
    "庙",
    "教堂",
    "祠",
    "牌坊",
    "会馆",
    "公馆",
    "石库门",
)
HISTORIC_FLAG_COLUMNS = ("is_historic", "historic", "history", "历史", "历史标注", "是否历史", "历史节点")


def out_text(value: Any) -> str:
    return str(value).encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")


def gcj02_to_wgs84(lon: float, lat: float) -> tuple[float, float]:
    if not (72.004 <= lon <= 137.8347 and 0.8293 <= lat <= 55.8271):
        return lon, lat

    def transform_lat(x: float, y: float) -> float:
        ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y
        ret += 0.2 * math.sqrt(abs(x))
        ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(y * math.pi) + 40.0 * math.sin(y / 3.0 * math.pi)) * 2.0 / 3.0
        ret += (160.0 * math.sin(y / 12.0 * math.pi) + 320 * math.sin(y * math.pi / 30.0)) * 2.0 / 3.0
        return ret

    def transform_lon(x: float, y: float) -> float:
        ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y
        ret += 0.1 * math.sqrt(abs(x))
        ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(x * math.pi) + 40.0 * math.sin(x / 3.0 * math.pi)) * 2.0 / 3.0
        ret += (150.0 * math.sin(x / 12.0 * math.pi) + 300.0 * math.sin(x / 30.0 * math.pi)) * 2.0 / 3.0
        return ret

    a = 6378245.0
    ee = 0.00669342162296594323
    dlat = transform_lat(lon - 105.0, lat - 35.0)
    dlon = transform_lon(lon - 105.0, lat - 35.0)
    radlat = lat / 180.0 * math.pi
    magic = 1 - ee * math.sin(radlat) ** 2
    sqrt_magic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrt_magic) * math.pi)
    dlon = (dlon * 180.0) / (a / sqrt_magic * math.cos(radlat) * math.pi)
    return lon * 2 - (lon + dlon), lat * 2 - (lat + dlat)


def lonlat_to_webmerc(lon: float, lat: float) -> tuple[float, float]:
    lon = float(lon)
    lat = min(max(float(lat), -85.05112878), 85.05112878)
    x = lon * 20037508.34 / 180.0
    y = math.log(math.tan((90.0 + lat) * math.pi / 360.0)) * 20037508.34 / math.pi
    return x, y


def project_geom(geom):
    return transform(lambda x, y, z=None: lonlat_to_webmerc(x, y), geom)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def log_norm(value: float, scale: float) -> float:
    return float(np.log1p(max(value, 0.0)) / np.log1p(scale))


def point_feature(x: float, y: float, bounds: tuple[float, float, float, float]) -> list[float]:
    minx, miny, maxx, maxy = bounds
    return [
        (x - minx) / max(maxx - minx, 1.0),
        (y - miny) / max(maxy - miny, 1.0),
    ]


def pad_feature(values: list[float]) -> list[float]:
    values = [float(v) for v in values[:FEATURE_DIM]]
    return values + [0.0] * (FEATURE_DIM - len(values))


def poi_group(text: str) -> str:
    if any(k in text for k in DAILY_KEYWORDS) or any(k in text for k in DAILY_KEYWORDS_UTF8):
        return "daily"
    if any(k in text for k in TOUR_KEYWORDS) or any(k in text for k in TOUR_KEYWORDS_UTF8):
        return "tour"
    if any(k in text for k in TRANSPORT_KEYWORDS) or any(k in text for k in TRANSPORT_KEYWORDS_UTF8):
        return "transport"
    return "other"


def truthy_flag(value: Any) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "是", "历史", "historic", "history"}


def first_existing_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {str(col).strip().lower(): col for col in columns}
    for candidate in candidates:
        hit = lowered.get(candidate.lower())
        if hit is not None:
            return hit
    return None


def is_historic_poi(row: pd.Series, text: str, historic_col: str | None) -> bool:
    if historic_col is not None:
        return truthy_flag(row.get(historic_col))
    return any(keyword in text for keyword in HISTORIC_KEYWORDS)


def load_stations(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rows = []
    for i, row in df.iterrows():
        lon, lat = gcj02_to_wgs84(safe_float(row["longitude"]), safe_float(row["latitude"]))
        x, y = lonlat_to_webmerc(lon, lat)
        rows.append(
            {
                "id": f"station:{row['id']}",
                "raw_id": row["id"],
                "name": row.get("name", ""),
                "lon": lon,
                "lat": lat,
                "x": x,
                "y": y,
                "adname": row.get("adname", ""),
            }
        )
    return pd.DataFrame(rows)


def load_buildings(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rows = []
    quality_map = {"high": 1.0, "medium": 0.5, "low": 0.0}
    for _, row in df.iterrows():
        lon, lat = safe_float(row["longitude"]), safe_float(row["latitude"])
        x, y = lonlat_to_webmerc(lon, lat)
        built_year = safe_float(row.get("built_year"), 0.0)
        rows.append(
            {
                "id": f"building:{row['uid']}",
                "raw_id": row["uid"],
                "name": row.get("display_name", ""),
                "lon": lon,
                "lat": lat,
                "x": x,
                "y": y,
                "batch": safe_float(row.get("batch"), 0.0),
                "built_year": built_year,
                "quality": quality_map.get(str(row.get("coordinate_quality", "")).lower(), 0.0),
            }
        )
    return pd.DataFrame(rows)


def load_areas(path: Path, node_type: str) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for i, feat in enumerate(data["features"]):
        props = feat.get("properties") or {}
        geom = shape(feat["geometry"])
        projected = project_geom(geom)
        centroid = projected.centroid
        rows.append(
            {
                "id": f"{node_type}:{i}",
                "raw_id": str(i),
                "name": props.get("name", props.get("NAME", str(i))),
                "x": centroid.x,
                "y": centroid.y,
                "area": projected.area,
                "geometry": projected,
            }
        )
    return pd.DataFrame(rows)


def load_roads(path: Path, station_xy: np.ndarray, radius_m: float) -> pd.DataFrame:
    tree = cKDTree(station_xy)
    rows = []
    for chunk in pd.read_csv(path, chunksize=20000):
        for _, row in chunk.iterrows():
            try:
                geom = wkt.loads(row["geometry"])
            except Exception:
                continue
            projected = project_geom(geom)
            centroid = projected.centroid
            length = safe_float(row.get("length"), projected.length)
            # Coarse filter by centroid, followed by exact distance later.
            if not tree.query_ball_point([centroid.x, centroid.y], radius_m + min(length / 2.0, radius_m)):
                continue
            rows.append(
                {
                    "id": f"road:{row['segment_id']}",
                    "raw_id": row["segment_id"],
                    "name": row.get("road_name", ""),
                    "x": centroid.x,
                    "y": centroid.y,
                    "highway": str(row.get("highway", "")),
                    "length": length,
                    "geometry": projected,
                }
            )
    return pd.DataFrame(rows)


def load_pois(path: Path, station_xy: np.ndarray, radius_m: float, max_per_station_group: int) -> pd.DataFrame:
    tree = cKDTree(station_xy)
    selected: dict[tuple[int, str], list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    for chunk in pd.read_csv(path, chunksize=50000):
        cols = list(chunk.columns)
        if len(cols) < 5:
            continue
        name_col, big_col, mid_col, lon_col, lat_col = cols[:5]
        historic_col = first_existing_column(cols, HISTORIC_FLAG_COLUMNS)
        for i, row in chunk.iterrows():
            lon_raw, lat_raw = safe_float(row[lon_col], np.nan), safe_float(row[lat_col], np.nan)
            if not np.isfinite(lon_raw) or not np.isfinite(lat_raw):
                continue
            lon, lat = gcj02_to_wgs84(lon_raw, lat_raw)
            x, y = lonlat_to_webmerc(lon, lat)
            station_ids = tree.query_ball_point([x, y], radius_m)
            if not station_ids:
                continue
            text = f"{row.get(big_col, '')};{row.get(mid_col, '')}"
            group = poi_group(text)
            historic = is_historic_poi(row, f"{row.get(name_col, '')};{text}", historic_col)
            rec = {
                "id": f"poi:{i}",
                "raw_id": str(i),
                "name": row.get(name_col, ""),
                "category": text,
                "group": group,
                "is_historic": int(historic),
                "is_daily": int(group == "daily"),
                "lon": lon,
                "lat": lat,
                "x": x,
                "y": y,
            }
            for sid in station_ids:
                dist = float(np.linalg.norm(station_xy[sid] - np.array([x, y])))
                bucket = selected[(sid, group)]
                bucket.append((dist, rec))
                if len(bucket) > max_per_station_group * 3:
                    bucket.sort(key=lambda item: item[0])
                    del bucket[max_per_station_group:]

    unique: dict[str, dict[str, Any]] = {}
    for bucket in selected.values():
        bucket.sort(key=lambda item: item[0])
        for _, rec in bucket[:max_per_station_group]:
            unique[rec["id"]] = rec
    return pd.DataFrame(unique.values())


def add_bidirectional(edges: list[dict[str, Any]], src_type: str, src: int, rel: str, dst_type: str, dst: int, weight: float = 1.0):
    edges.append({"source_type": src_type, "source": src, "relation": rel, "target_type": dst_type, "target": dst, "weight": weight})
    edges.append({"source_type": dst_type, "source": dst, "relation": f"rev_{rel}", "target_type": src_type, "target": src, "weight": weight})


def compute_station_weak_labels(
    stations: pd.DataFrame,
    buildings: pd.DataFrame,
    pois: pd.DataFrame,
    label_radius_m: float,
    conflict_link_m: float,
    conflict_threshold: float,
    min_conflict_nodes: int,
) -> pd.DataFrame:
    """Create station-level weak labels from 500m historic/daily node balance.

    Historic nodes are historic building nodes plus POIs with explicit or
    keyword-derived historic flags. Non-historic daily nodes are daily POIs
    without a historic flag. Connectivity is a bipartite density proxy between
    historic and daily nodes inside the station label radius.
    """

    rows = []
    station_xy = stations[["x", "y"]].to_numpy()
    building_xy = buildings[["x", "y"]].to_numpy() if len(buildings) else np.empty((0, 2))
    building_tree = cKDTree(building_xy) if len(building_xy) else None

    if len(pois):
        poi_xy = pois[["x", "y"]].to_numpy()
        historic_poi_mask = pois["is_historic"].astype(bool).to_numpy() if "is_historic" in pois else np.zeros(len(pois), dtype=bool)
        daily_poi_mask = pois["is_daily"].astype(bool).to_numpy() if "is_daily" in pois else pois["group"].eq("daily").to_numpy()
        poi_tree = cKDTree(poi_xy)
    else:
        poi_xy = np.empty((0, 2))
        historic_poi_mask = np.zeros(0, dtype=bool)
        daily_poi_mask = np.zeros(0, dtype=bool)
        poi_tree = None

    for sid, sxy in enumerate(station_xy):
        building_ids = building_tree.query_ball_point(sxy, label_radius_m) if building_tree is not None else []
        poi_ids = poi_tree.query_ball_point(sxy, label_radius_m) if poi_tree is not None else []

        historic_poi_ids = [pid for pid in poi_ids if historic_poi_mask[pid]]
        daily_poi_ids = [pid for pid in poi_ids if daily_poi_mask[pid] and not historic_poi_mask[pid]]

        historic_points = []
        if building_ids:
            historic_points.append(building_xy[np.asarray(building_ids, dtype=int)])
        if historic_poi_ids:
            historic_points.append(poi_xy[np.asarray(historic_poi_ids, dtype=int)])
        historic_points_arr = np.vstack(historic_points) if historic_points else np.empty((0, 2))
        daily_points_arr = poi_xy[np.asarray(daily_poi_ids, dtype=int)] if daily_poi_ids else np.empty((0, 2))

        historic_count = int(len(historic_points_arr))
        daily_count = int(len(daily_points_arr))
        total = historic_count + daily_count
        historic_prob = historic_count / total if total else 0.0
        daily_prob = daily_count / total if total else 0.0
        balance = 1.0 - abs(historic_count - daily_count) / total if total else 0.0

        if historic_count and daily_count:
            daily_tree = cKDTree(daily_points_arr)
            cross_links = int(sum(len(ids) for ids in daily_tree.query_ball_point(historic_points_arr, conflict_link_m)))
            full_links = historic_count * daily_count
            connectivity = cross_links / full_links if full_links else 0.0
        else:
            cross_links = 0
            full_links = 0
            connectivity = 0.0

        conflict_index = balance * connectivity
        conflict_label = int(
            historic_count >= min_conflict_nodes
            and daily_count >= min_conflict_nodes
            and conflict_index >= conflict_threshold
        )
        rows.append(
            {
                "station_id": sid,
                "label_radius_m": label_radius_m,
                "historic_node_count_500m": historic_count,
                "daily_node_count_500m": daily_count,
                "historic_probability_500m": historic_prob,
                "daily_probability_500m": daily_prob,
                "historic_daily_balance_500m": balance,
                "historic_daily_cross_links_500m": cross_links,
                "historic_daily_full_links_500m": full_links,
                "historic_daily_connectivity_500m": connectivity,
                "conflict_index_500m": conflict_index,
                "conflict_label_500m": conflict_label,
            }
        )
    return pd.DataFrame(rows)


def first_geojson(path: Path) -> Path:
    matches = sorted(path.glob("*.geojson"))
    if not matches:
        raise FileNotFoundError(f"No GeoJSON file found in {path}")
    return matches[0]


def resolve_sources(args: argparse.Namespace) -> dict[str, Path]:
    data_dir = args.data_dir
    return {
        "stations": args.stations_csv or data_dir / "metro_stations" / "shanghai_metro_stations_amap.csv",
        "buildings": args.buildings_csv or data_dir / "historic_buildings" / "shanghai_excellent_historic_buildings_points.csv",
        "conservation": args.conservation_geojson or first_geojson(data_dir / "historic_conservation_areas"),
        "admin": args.admin_geojson or data_dir / "admin_boundary" / "shanghai_admin_boundary.geojson",
        "roads": args.roads_csv or data_dir / "road_segments" / "shanghai_road_segments.csv",
        "poi": args.poi_csv or data_dir / "poi" / "2026_poi_Shanghai.csv",
    }


def build_graph(args: argparse.Namespace) -> dict[str, Any]:
    sources = resolve_sources(args)
    for label, path in sources.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} source not found: {path}")

    stations = load_stations(sources["stations"])
    station_xy = stations[["x", "y"]].to_numpy()
    station_tree = cKDTree(station_xy)
    buildings = load_buildings(sources["buildings"])
    conservation = load_areas(sources["conservation"], "conservation_area")
    admin = load_areas(sources["admin"], "admin_area")
    roads = load_roads(sources["roads"], station_xy, args.radius_m)
    pois = load_pois(sources["poi"], station_xy, args.radius_m, args.max_poi_per_station_group)
    station_labels = compute_station_weak_labels(
        stations,
        buildings,
        pois,
        args.label_radius_m,
        args.conflict_link_m,
        args.conflict_threshold,
        args.min_conflict_nodes,
    )
    stations = stations.join(station_labels.drop(columns=["station_id"]))

    all_x = np.concatenate(
        [
            stations[["x"]].to_numpy().ravel(),
            buildings[["x"]].to_numpy().ravel(),
            roads[["x"]].to_numpy().ravel() if len(roads) else np.array([]),
            pois[["x"]].to_numpy().ravel() if len(pois) else np.array([]),
            conservation[["x"]].to_numpy().ravel(),
            admin[["x"]].to_numpy().ravel(),
        ]
    )
    all_y = np.concatenate(
        [
            stations[["y"]].to_numpy().ravel(),
            buildings[["y"]].to_numpy().ravel(),
            roads[["y"]].to_numpy().ravel() if len(roads) else np.array([]),
            pois[["y"]].to_numpy().ravel() if len(pois) else np.array([]),
            conservation[["y"]].to_numpy().ravel(),
            admin[["y"]].to_numpy().ravel(),
        ]
    )
    bounds = (float(all_x.min()), float(all_y.min()), float(all_x.max()), float(all_y.max()))

    edges: list[dict[str, Any]] = []
    station_metrics = defaultdict(lambda: defaultdict(float))

    building_xy = buildings[["x", "y"]].to_numpy()
    building_tree = cKDTree(building_xy)
    for sid, sxy in enumerate(station_xy):
        for bid in building_tree.query_ball_point(sxy, args.radius_m):
            dist = float(np.linalg.norm(sxy - building_xy[bid]))
            station_metrics[sid]["historic_buildings"] += 1
            add_bidirectional(edges, "station", sid, "near_building", "building", int(bid), 1.0 / (1.0 + dist))

    if len(pois):
        poi_xy = pois[["x", "y"]].to_numpy()
        poi_tree = cKDTree(poi_xy)
        for sid, sxy in enumerate(station_xy):
            for pid in poi_tree.query_ball_point(sxy, args.radius_m):
                dist = float(np.linalg.norm(sxy - poi_xy[pid]))
                group = pois.iloc[pid]["group"]
                station_metrics[sid][f"poi_{group}"] += 1
                add_bidirectional(edges, "station", sid, f"near_poi_{group}", "poi", int(pid), 1.0 / (1.0 + dist))

    station_points = [Point(xy) for xy in station_xy]
    for rid, row in roads.iterrows():
        geom = row["geometry"]
        candidate_sids = station_tree.query_ball_point([row["x"], row["y"]], args.radius_m + min(row["length"] / 2.0, args.radius_m))
        road_idx = int(rid)
        for sid in candidate_sids:
            dist = geom.distance(station_points[sid])
            if dist <= args.radius_m:
                highway = row["highway"]
                station_metrics[sid]["road_length_m"] += row["length"]
                if highway in MAJOR_HIGHWAYS:
                    station_metrics[sid]["major_road_length_m"] += row["length"]
                add_bidirectional(edges, "station", sid, "near_road", "road", road_idx, 1.0 / (1.0 + dist))

    for aid, row in conservation.iterrows():
        geom = row["geometry"]
        for sid, pt in enumerate(station_points):
            if geom.contains(pt):
                station_metrics[sid]["in_conservation"] = 1.0
                add_bidirectional(edges, "station", sid, "inside_conservation", "conservation_area", int(aid), 1.0)
        for bid, brow in buildings.iterrows():
            if geom.contains(Point(brow["x"], brow["y"])):
                add_bidirectional(edges, "building", int(bid), "inside_conservation", "conservation_area", int(aid), 1.0)

    for aid, row in admin.iterrows():
        geom = row["geometry"]
        for sid, pt in enumerate(station_points):
            if geom.contains(pt):
                add_bidirectional(edges, "station", sid, "inside_admin", "admin_area", int(aid), 1.0)

    node_tables: dict[str, pd.DataFrame] = {}
    feature_tables: dict[str, np.ndarray] = {}

    station_features = []
    for sid, row in stations.iterrows():
        m = station_metrics[sid]
        daily = m["poi_daily"]
        tour = m["poi_tour"]
        mix = min(daily, tour) / max(daily + tour, 1.0)
        station_features.append(
            pad_feature(
                point_feature(row["x"], row["y"], bounds)
                + [
                    log_norm(m["historic_buildings"], 100),
                    log_norm(daily, 200),
                    log_norm(tour, 200),
                    log_norm(m["poi_transport"], 100),
                    log_norm(m["poi_other"], 300),
                    log_norm(m["road_length_m"] / 1000.0, 300),
                    log_norm(m["major_road_length_m"] / 1000.0, 100),
                    m["in_conservation"],
                    mix,
                    row["historic_probability_500m"],
                    row["daily_probability_500m"],
                    log_norm(row["historic_node_count_500m"], 100),
                    log_norm(row["daily_node_count_500m"], 200),
                ]
            )
        )
    node_tables["station"] = stations.drop(columns=[], errors="ignore")
    feature_tables["station"] = np.asarray(station_features, dtype=np.float32)

    building_features = []
    for _, row in buildings.iterrows():
        building_features.append(
            pad_feature(
                point_feature(row["x"], row["y"], bounds)
                + [
                    row["quality"],
                    row["batch"] / 10.0,
                    (row["built_year"] - 1800.0) / 250.0 if row["built_year"] else 0.0,
                ]
            )
        )
    node_tables["building"] = buildings
    feature_tables["building"] = np.asarray(building_features, dtype=np.float32)

    road_features = []
    for _, row in roads.iterrows():
        h = row["highway"]
        road_features.append(
            pad_feature(
                point_feature(row["x"], row["y"], bounds)
                + [
                    log_norm(row["length"], 5000),
                    1.0 if h in MAJOR_HIGHWAYS else 0.0,
                    1.0 if h in LOCAL_HIGHWAYS else 0.0,
                ]
            )
        )
    node_tables["road"] = roads.drop(columns=["geometry"], errors="ignore")
    feature_tables["road"] = np.asarray(road_features, dtype=np.float32)

    poi_features = []
    group_to_idx = {"daily": 0, "tour": 1, "transport": 2, "other": 3}
    for _, row in pois.iterrows():
        one_hot = [0.0, 0.0, 0.0, 0.0]
        one_hot[group_to_idx.get(row["group"], 3)] = 1.0
        poi_features.append(
            pad_feature(
                point_feature(row["x"], row["y"], bounds)
                + one_hot
                + [
                    row.get("is_historic", 0),
                    row.get("is_daily", 0),
                ]
            )
        )
    node_tables["poi"] = pois
    feature_tables["poi"] = np.asarray(poi_features, dtype=np.float32)

    for node_type, df in [("conservation_area", conservation), ("admin_area", admin)]:
        feats = []
        for _, row in df.iterrows():
            feats.append(pad_feature(point_feature(row["x"], row["y"], bounds) + [log_norm(row["area"] / 1_000_000.0, 500)]))
        node_tables[node_type] = df.drop(columns=["geometry"], errors="ignore")
        feature_tables[node_type] = np.asarray(feats, dtype=np.float32)

    hetero = HeteroData()
    for node_type, feats in feature_tables.items():
        hetero[node_type].x = torch.tensor(feats, dtype=torch.float32)
        hetero[node_type].node_id = torch.arange(len(feats), dtype=torch.long)
    hetero["station"].y_conflict = torch.tensor(stations["conflict_label_500m"].to_numpy(), dtype=torch.long)
    hetero["station"].conflict_index = torch.tensor(stations["conflict_index_500m"].to_numpy(), dtype=torch.float32)
    hetero["station"].historic_probability = torch.tensor(stations["historic_probability_500m"].to_numpy(), dtype=torch.float32)
    hetero["station"].daily_probability = torch.tensor(stations["daily_probability_500m"].to_numpy(), dtype=torch.float32)

    edge_df = pd.DataFrame(edges)
    for (src_type, rel, dst_type), group in edge_df.groupby(["source_type", "relation", "target_type"], sort=True):
        edge_index = torch.tensor(group[["source", "target"]].to_numpy().T, dtype=torch.long)
        edge_weight = torch.tensor(group["weight"].to_numpy(), dtype=torch.float32)
        hetero[(src_type, rel, dst_type)].edge_index = edge_index
        hetero[(src_type, rel, dst_type)].edge_weight = edge_weight

    graph = Graph()
    for node_type, df in node_tables.items():
        for i, row in df.reset_index(drop=True).iterrows():
            graph.add_node({"type": node_type, "id": int(i), "name": out_text(row.get("name", row.get("id", i)))})
        graph.node_feature[node_type] = pd.DataFrame(
            {
                "node_id": np.arange(len(df)),
                "source_id": list(df["id"]) if "id" in df else [str(i) for i in range(len(df))],
                "emb": list(feature_tables[node_type]),
                "citation": np.zeros(len(df), dtype=np.float32),
            }
        )
    for _, row in edge_df.iterrows():
        graph.add_edge(
            {"type": row["source_type"], "id": int(row["source"])},
            {"type": row["target_type"], "id": int(row["target"])},
            time=YEAR,
            relation_type=row["relation"],
            directed=False,
        )

    node_type_order = list(feature_tables.keys())
    node_offsets = {}
    xs, node_types = [], []
    offset = 0
    for tid, node_type in enumerate(node_type_order):
        feats = feature_tables[node_type]
        node_offsets[node_type] = offset
        xs.append(torch.tensor(feats, dtype=torch.float32))
        node_types.extend([tid] * len(feats))
        offset += len(feats)
    edge_type_keys = sorted(edge_df[["source_type", "relation", "target_type"]].drop_duplicates().itertuples(index=False, name=None))
    edge_type_map = {key: i for i, key in enumerate(edge_type_keys)}
    edge_index, edge_type, edge_time = [], [], []
    for _, row in edge_df.iterrows():
        src = int(row["source"]) + node_offsets[row["source_type"]]
        dst = int(row["target"]) + node_offsets[row["target_type"]]
        key = (row["source_type"], row["relation"], row["target_type"])
        edge_index.append([src, dst])
        edge_type.append(edge_type_map[key])
        edge_time.append(0)
    tensors = {
        "x": torch.cat(xs, dim=0),
        "node_type": torch.tensor(node_types, dtype=torch.long),
        "edge_index": torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        "edge_type": torch.tensor(edge_type, dtype=torch.long),
        "edge_time": torch.tensor(edge_time, dtype=torch.long),
        "station_y_conflict": torch.tensor(stations["conflict_label_500m"].to_numpy(), dtype=torch.long),
        "station_conflict_index": torch.tensor(stations["conflict_index_500m"].to_numpy(), dtype=torch.float32),
        "station_historic_probability": torch.tensor(stations["historic_probability_500m"].to_numpy(), dtype=torch.float32),
        "station_daily_probability": torch.tensor(stations["daily_probability_500m"].to_numpy(), dtype=torch.float32),
        "node_type_map": {t: i for i, t in enumerate(node_type_order)},
        "edge_type_map": {str(k): v for k, v in edge_type_map.items()},
        "node_offsets": node_offsets,
    }

    return {
        "graph": graph,
        "hetero": hetero,
        "tensors": tensors,
        "nodes": node_tables,
        "edges": edge_df,
        "summary": {
            "radius_m": args.radius_m,
            "label_radius_m": args.label_radius_m,
            "conflict_link_m": args.conflict_link_m,
            "conflict_threshold": args.conflict_threshold,
            "min_conflict_nodes": args.min_conflict_nodes,
            "feature_dim": FEATURE_DIM,
            "node_counts": {k: int(len(v)) for k, v in node_tables.items()},
            "edge_counts": edge_df.groupby(["source_type", "relation", "target_type"]).size().to_dict(),
            "total_edges": int(len(edge_df)),
            "station_conflict_label_counts": {
                str(k): int(v) for k, v in stations["conflict_label_500m"].value_counts().sort_index().items()
            },
            "station_conflict_index": {
                "min": float(stations["conflict_index_500m"].min()),
                "mean": float(stations["conflict_index_500m"].mean()),
                "max": float(stations["conflict_index_500m"].max()),
            },
            "weak_label_definition": (
                "500m station circle; historic nodes = historic buildings + explicit/keyword historic POIs; "
                "daily nodes = non-historic daily POIs; conflict_index = count_balance * bipartite_connectivity"
            ),
            "crs": "EPSG:3857 features from WGS84/GCJ02-normalized source coordinates",
            "sources": {k: str(v) for k, v in sources.items()},
        },
    }


def save_outputs(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "nodes").mkdir(exist_ok=True)
    with (output_dir / "pyhgt_graph.pkl").open("wb") as f:
        dill.dump(payload["graph"], f)
    torch.save(payload["hetero"], output_dir / "hetero_data.pt")
    torch.save(payload["tensors"], output_dir / "hgt_tensors.pt")
    for node_type, df in payload["nodes"].items():
        df.drop(columns=["geometry"], errors="ignore").to_csv(output_dir / "nodes" / f"{node_type}.csv", index=False, encoding="utf-8-sig")
    payload["edges"].to_csv(output_dir / "edges.csv", index=False, encoding="utf-8-sig")
    summary = payload["summary"].copy()
    summary["edge_counts"] = {"|".join(k): int(v) for k, v in summary["edge_counts"].items()}
    (output_dir / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build station-centered heterogeneous graph for HGT.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw", help="Base raw data directory for default source paths.")
    parser.add_argument("--stations-csv", type=Path, default=None, help="Override metro stations CSV source.")
    parser.add_argument("--buildings-csv", type=Path, default=None, help="Override historic buildings CSV source.")
    parser.add_argument("--conservation-geojson", type=Path, default=None, help="Override conservation areas GeoJSON source.")
    parser.add_argument("--admin-geojson", type=Path, default=None, help="Override administrative areas GeoJSON source.")
    parser.add_argument("--roads-csv", type=Path, default=None, help="Override road segments CSV source.")
    parser.add_argument("--poi-csv", type=Path, default=None, help="Override POI CSV source.")
    parser.add_argument("--radius-m", type=float, default=2560.0, help="Station context radius; 512px * 10m / 2 by default.")
    parser.add_argument("--label-radius-m", type=float, default=500.0, help="Radius for weak station conflict labels.")
    parser.add_argument("--conflict-link-m", type=float, default=150.0, help="Cross-group link distance for historic/daily connectivity.")
    parser.add_argument("--conflict-threshold", type=float, default=0.15, help="Weak conflict label threshold on balance * connectivity.")
    parser.add_argument("--min-conflict-nodes", type=int, default=2, help="Minimum nodes required in each group for a positive conflict label.")
    parser.add_argument("--max-poi-per-station-group", type=int, default=120, help="Nearest POI cap per station and POI group.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "preprocess" / "hgt_graph")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_graph(args)
    save_outputs(payload, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
