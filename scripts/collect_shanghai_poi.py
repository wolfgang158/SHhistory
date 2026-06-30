#!/usr/bin/env python3
"""Collect Shanghai POI data from Amap Web Service API.

The collector uses polygon search over a recursive grid. Dense cells are split
again so the API page cap is less likely to hide records. Results are
deduplicated and saved as CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


API_URL = "https://restapi.amap.com/v3/place/polygon"
DEFAULT_KEY = "d309562e163657076f36476891e46798"

# Covers Shanghai Municipality with a small margin. Amap returns GCJ-02 coords.
SHANGHAI_BBOX = (120.85, 30.67, 122.12, 31.88)

# Approximate district bounding boxes with intentional overlap. They reduce
# requests outside Shanghai compared with the full municipal bounding box.
SHANGHAI_DISTRICT_BBOXES = {
    "huangpu": (121.45, 31.19, 121.52, 31.26),
    "xuhui": (121.36, 31.13, 121.49, 31.23),
    "changning": (121.32, 31.17, 121.45, 31.25),
    "jingan": (121.40, 31.21, 121.49, 31.32),
    "putuo": (121.33, 31.22, 121.47, 31.32),
    "hongkou": (121.45, 31.24, 121.53, 31.31),
    "yangpu": (121.47, 31.24, 121.58, 31.35),
    "minhang": (121.23, 30.97, 121.58, 31.25),
    "baoshan": (121.30, 31.25, 121.56, 31.53),
    "jiading": (121.12, 31.20, 121.42, 31.52),
    "pudong": (121.44, 30.93, 122.02, 31.36),
    "jinshan": (120.83, 30.66, 121.42, 30.98),
    "songjiang": (120.98, 30.86, 121.40, 31.24),
    "qingpu": (120.82, 30.91, 121.28, 31.28),
    "fengxian": (121.33, 30.75, 121.82, 31.08),
    "chongming": (121.05, 31.25, 122.12, 31.90),
}

# Broad Amap POI categories. Recursive spatial splitting handles dense areas.
AMAP_TOP_TYPES = [
    "010000",  # 汽车服务
    "020000",  # 汽车销售
    "030000",  # 汽车维修
    "040000",  # 摩托车服务
    "050000",  # 餐饮服务
    "060000",  # 购物服务
    "070000",  # 生活服务
    "080000",  # 体育休闲服务
    "090000",  # 医疗保健服务
    "100000",  # 住宿服务
    "110000",  # 风景名胜
    "120000",  # 商务住宅
    "130000",  # 政府机构及社会团体
    "140000",  # 科教文化服务
    "150000",  # 交通设施服务
    "160000",  # 金融保险服务
    "170000",  # 公司企业
    "180000",  # 道路附属设施
    "190000",  # 地名地址信息
    "200000",  # 公共设施
    "220000",  # 事件活动
    "970000",  # 室内设施
    "990000",  # 通行设施
]

CSV_FIELDS = [
    "id",
    "name",
    "type",
    "typecode",
    "biz_type",
    "address",
    "location",
    "longitude",
    "latitude",
    "pname",
    "cityname",
    "adname",
    "adcode",
    "tel",
    "postcode",
    "website",
    "email",
    "entr_location",
    "exit_location",
    "alias",
    "tag",
    "business_area",
    "grid_level",
    "grid_bbox",
    "source_type",
]


@dataclass(frozen=True)
class Cell:
    min_lng: float
    min_lat: float
    max_lng: float
    max_lat: float
    level: int = 0

    def polygon(self) -> str:
        return f"{self.min_lng:.6f},{self.min_lat:.6f}|{self.max_lng:.6f},{self.max_lat:.6f}"

    def label(self) -> str:
        return (
            f"{self.min_lng:.6f},{self.min_lat:.6f},"
            f"{self.max_lng:.6f},{self.max_lat:.6f}"
        )

    def split(self) -> List["Cell"]:
        mid_lng = (self.min_lng + self.max_lng) / 2
        mid_lat = (self.min_lat + self.max_lat) / 2
        next_level = self.level + 1
        return [
            Cell(self.min_lng, self.min_lat, mid_lng, mid_lat, next_level),
            Cell(mid_lng, self.min_lat, self.max_lng, mid_lat, next_level),
            Cell(self.min_lng, mid_lat, mid_lng, self.max_lat, next_level),
            Cell(mid_lng, mid_lat, self.max_lng, self.max_lat, next_level),
        ]


class AmapCollector:
    def __init__(
        self,
        key: str,
        output_dir: Path,
        offset: int = 25,
        sleep_seconds: float = 0.25,
        max_pages: int = 100,
        split_threshold: int = 850,
        max_level: int = 8,
        retries: int = 3,
        timeout: int = 20,
        log_file: Optional[Path] = None,
        csv_every_tasks: int = 50,
        area_mode: str = "districts",
    ) -> None:
        self.key = key
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_path = output_dir / "shanghai_poi_raw.jsonl"
        self.csv_path = output_dir / "shanghai_poi.csv"
        self.state_path = output_dir / "collector_state.json"
        self.offset = min(max(offset, 1), 25)
        self.sleep_seconds = sleep_seconds
        self.max_pages = max_pages
        self.split_threshold = split_threshold
        self.max_level = max_level
        self.retries = retries
        self.timeout = timeout
        self.log_file = log_file
        self.csv_every_tasks = csv_every_tasks
        self.area_mode = area_mode
        self.ssl_context = self._build_ssl_context()
        self.completed_tasks = self._load_completed_tasks()
        self.seen: Dict[str, Dict[str, Any]] = {}
        self.total_requests = 0
        self.total_raw_records = 0

    def log(self, message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        if self.log_file:
            with self.log_file.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def _load_completed_tasks(self) -> set[str]:
        if not self.state_path.exists():
            return set()
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            return set(data.get("completed_tasks", []))
        except (json.JSONDecodeError, OSError):
            return set()

    @staticmethod
    def _build_ssl_context() -> ssl.SSLContext:
        if os.environ.get("SSL_CERT_FILE") or os.environ.get("SSL_CERT_DIR"):
            return ssl.create_default_context()
        system_ca = Path("/etc/ssl/cert.pem")
        if system_ca.exists():
            return ssl.create_default_context(cafile=str(system_ca))
        return ssl.create_default_context()

    def _save_state(self, task_id: Optional[str] = None) -> None:
        if task_id:
            self.completed_tasks.add(task_id)
        payload = {
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "completed_tasks": sorted(self.completed_tasks),
            "total_completed_tasks": len(self.completed_tasks),
            "total_requests": self.total_requests,
            "total_raw_records": self.total_raw_records,
            "unique_records_loaded": len(self.seen),
        }
        self.state_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if (
            task_id
            and self.csv_every_tasks > 0
            and len(self.completed_tasks) % self.csv_every_tasks == 0
        ):
            self.write_csv()

    def load_existing_raw(self) -> None:
        if not self.raw_path.exists():
            return
        loaded = 0
        with self.raw_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    poi = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = self._dedupe_key(poi)
                self.seen[key] = poi
                loaded += 1
        self.log(f"Loaded {loaded} existing raw records, {len(self.seen)} unique.")

    def request_page(self, cell: Cell, typecode: str, page: int) -> Dict[str, Any]:
        params = {
            "key": self.key,
            "polygon": cell.polygon(),
            "types": typecode,
            "offset": str(self.offset),
            "page": str(page),
            "extensions": "all",
            "output": "json",
            "city": "上海",
            "citylimit": "true",
        }
        url = f"{API_URL}?{urllib.parse.urlencode(params)}"
        last_error: Optional[BaseException] = None
        for attempt in range(1, self.retries + 1):
            try:
                with urllib.request.urlopen(
                    url,
                    timeout=self.timeout,
                    context=self.ssl_context,
                ) as resp:
                    body = resp.read().decode("utf-8")
                self.total_requests += 1
                data = json.loads(body)
                if data.get("status") != "1":
                    info = data.get("info", "UNKNOWN_ERROR")
                    infocode = data.get("infocode", "")
                    raise RuntimeError(f"Amap API error: {info} ({infocode})")
                return data
            except (
                urllib.error.URLError,
                TimeoutError,
                json.JSONDecodeError,
                RuntimeError,
            ) as exc:
                last_error = exc
                self.log(
                    f"Request failed attempt {attempt}/{self.retries}: "
                    f"type={typecode} page={page} bbox={cell.label()} error={exc}"
                )
                time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(f"Request failed after retries: {last_error}")

    def collect_cell(self, cell: Cell, typecode: str, dry_run: bool = False) -> None:
        strategy = (
            f"{self.area_mode}_split{self.split_threshold}_level{self.max_level}"
        )
        task_id = f"{strategy}|{typecode}|{cell.label()}|L{cell.level}"
        if task_id in self.completed_tasks:
            self.log(f"Skip completed task {task_id}")
            return

        first = self.request_page(cell, typecode, 1)
        count = int(first.get("count") or 0)
        pois = first.get("pois") or []
        self.log(
            f"Task {task_id}: count={count}, first_page={len(pois)}, "
            f"unique_so_far={len(self.seen)}"
        )

        if dry_run:
            self.log("Dry run enabled; not saving POIs or descending grid.")
            return

        if count >= self.split_threshold and cell.level < self.max_level:
            self.log(
                f"Task {task_id}: count {count} >= threshold "
                f"{self.split_threshold}; split into 4 child cells."
            )
            for child in cell.split():
                self.collect_cell(child, typecode, dry_run=False)
            self._save_state(task_id)
            return

        self._save_pois(pois, cell, typecode)
        page = 2
        while page <= self.max_pages:
            if len(pois) < self.offset:
                break
            data = self.request_page(cell, typecode, page)
            pois = data.get("pois") or []
            self.log(f"Task {task_id}: page={page}, records={len(pois)}")
            if not pois:
                break
            self._save_pois(pois, cell, typecode)
            page += 1
            time.sleep(self.sleep_seconds)

        if page > self.max_pages:
            self.log(
                f"Task {task_id}: reached max_pages={self.max_pages}; "
                "consider lowering split_threshold or increasing max_level."
            )
        self._save_state(task_id)

    def _save_pois(self, pois: Iterable[Dict[str, Any]], cell: Cell, typecode: str) -> None:
        with self.raw_path.open("a", encoding="utf-8") as f:
            for poi in pois:
                if not self._is_shanghai_poi(poi):
                    continue
                poi = dict(poi)
                poi["_grid_level"] = cell.level
                poi["_grid_bbox"] = cell.label()
                poi["_source_type"] = typecode
                key = self._dedupe_key(poi)
                if key in self.seen:
                    continue
                self.seen[key] = poi
                f.write(json.dumps(poi, ensure_ascii=False) + "\n")
                self.total_raw_records += 1

    @staticmethod
    def _is_shanghai_poi(poi: Dict[str, Any]) -> bool:
        adcode = str(poi.get("adcode", ""))
        cityname = str(poi.get("cityname", ""))
        pname = str(poi.get("pname", ""))
        return adcode.startswith("31") or "上海" in cityname or "上海" in pname

    @staticmethod
    def _dedupe_key(poi: Dict[str, Any]) -> str:
        poi_id = str(poi.get("id") or "").strip()
        if poi_id:
            return f"id:{poi_id}"
        name = str(poi.get("name") or "").strip()
        location = str(poi.get("location") or "").strip()
        address = str(poi.get("address") or "").strip()
        return f"fallback:{name}|{location}|{address}"

    def write_csv(self) -> None:
        with self.csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for poi in sorted(self.seen.values(), key=lambda p: self._dedupe_key(p)):
                writer.writerow(self._flatten_poi(poi))
        self.log(f"CSV saved: {self.csv_path} ({len(self.seen)} unique records)")

    @staticmethod
    def _flatten_poi(poi: Dict[str, Any]) -> Dict[str, str]:
        location = str(poi.get("location") or "")
        lng, lat = "", ""
        if "," in location:
            lng, lat = location.split(",", 1)
        row: Dict[str, str] = {}
        for field in CSV_FIELDS:
            if field == "longitude":
                row[field] = lng
            elif field == "latitude":
                row[field] = lat
            elif field == "grid_level":
                row[field] = str(poi.get("_grid_level", ""))
            elif field == "grid_bbox":
                row[field] = str(poi.get("_grid_bbox", ""))
            elif field == "source_type":
                row[field] = str(poi.get("_source_type", ""))
            else:
                value = poi.get(field, "")
                if isinstance(value, (dict, list)):
                    row[field] = json.dumps(value, ensure_ascii=False)
                else:
                    row[field] = str(value)
        return row

    def run(self, typecodes: List[str], initial_cells: List[Cell], dry_run: bool) -> None:
        self.load_existing_raw()
        for typecode in typecodes:
            self.log(f"Start type {typecode}")
            for cell in initial_cells:
                self.collect_cell(cell, typecode, dry_run=dry_run)
                time.sleep(self.sleep_seconds)
            self.write_csv()
        self.write_csv()
        self._save_state()


def build_initial_cells(bbox: Tuple[float, float, float, float], grid: int) -> List[Cell]:
    min_lng, min_lat, max_lng, max_lat = bbox
    lng_step = (max_lng - min_lng) / grid
    lat_step = (max_lat - min_lat) / grid
    cells: List[Cell] = []
    for y in range(grid):
        for x in range(grid):
            cells.append(
                Cell(
                    min_lng + x * lng_step,
                    min_lat + y * lat_step,
                    min_lng + (x + 1) * lng_step,
                    min_lat + (y + 1) * lat_step,
                    0,
                )
            )
    return cells


def build_district_cells(grid: int) -> List[Cell]:
    cells: List[Cell] = []
    for bbox in SHANGHAI_DISTRICT_BBOXES.values():
        cells.extend(build_initial_cells(bbox, grid))
    return cells


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Shanghai POIs from Amap.")
    parser.add_argument("--key", default=DEFAULT_KEY, help="Amap Web Service key.")
    parser.add_argument("--output-dir", default="data", help="Output directory.")
    parser.add_argument("--grid", type=int, default=2, help="Initial grid per axis.")
    parser.add_argument("--offset", type=int, default=25, help="Page size, max 25.")
    parser.add_argument("--sleep", type=float, default=0.25, help="Delay between requests.")
    parser.add_argument("--max-pages", type=int, default=100, help="Max pages per task.")
    parser.add_argument("--split-threshold", type=int, default=180)
    parser.add_argument("--max-level", type=int, default=8)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument(
        "--types",
        default=",".join(AMAP_TOP_TYPES),
        help="Comma-separated Amap type codes.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only test first pages.")
    parser.add_argument(
        "--runtime-log",
        default="data/runtime.log",
        help="Append machine-readable runtime progress here.",
    )
    parser.add_argument(
        "--area-mode",
        choices=["districts", "bbox"],
        default="districts",
        help="Use district bounding boxes or one municipal bounding box.",
    )
    parser.add_argument(
        "--csv-every-tasks",
        type=int,
        default=50,
        help="Refresh CSV after this many completed tasks; 0 disables.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    runtime_log = Path(args.runtime_log)
    runtime_log.parent.mkdir(parents=True, exist_ok=True)
    typecodes = [item.strip() for item in args.types.split(",") if item.strip()]
    if args.area_mode == "districts":
        cells = build_district_cells(args.grid)
    else:
        cells = build_initial_cells(SHANGHAI_BBOX, args.grid)
    collector = AmapCollector(
        key=args.key,
        output_dir=output_dir,
        offset=args.offset,
        sleep_seconds=args.sleep,
        max_pages=args.max_pages,
        split_threshold=args.split_threshold,
        max_level=args.max_level,
        retries=args.retries,
        timeout=args.timeout,
        log_file=runtime_log,
        csv_every_tasks=args.csv_every_tasks,
        area_mode=args.area_mode,
    )
    collector.log(
        f"Collector configured: types={len(typecodes)}, cells={len(cells)}, "
        f"dry_run={args.dry_run}, output_dir={output_dir}, "
        f"split_threshold={args.split_threshold}, max_level={args.max_level}, "
        f"area_mode={args.area_mode}, grid={args.grid}"
    )
    try:
        collector.run(typecodes, cells, dry_run=args.dry_run)
    except KeyboardInterrupt:
        collector.log("Interrupted by user; refreshing CSV before exit.")
        collector.write_csv()
        collector._save_state()
        return 130
    except Exception as exc:
        collector.log(f"Fatal error: {exc}; refreshing CSV before exit.")
        collector.write_csv()
        collector._save_state()
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
