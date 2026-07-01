# Preprocess wrappers

This directory contains Python preprocess entrypoints plus Windows `.bat` and
Linux/macOS `.sh` wrappers.

## Output layout

Wrappers write timestamped outputs under:

```text
outputs/
  preprocess/
    hgt_graph/
      runs/
        yyyyMMdd-HHmmss/
    hgt_graph_viz/
      runs/
        yyyyMMdd-HHmmss/
```

The Python scripts still accept `--output-dir` if you need a fixed destination.

## Build HGT graph

Default run:

Windows:

```powershell
scripts\preprocess\build_hgt_graph.bat
```

Linux/macOS:

```bash
chmod +x scripts/preprocess/build_hgt_graph.sh
scripts/preprocess/build_hgt_graph.sh
```

Use a different base data directory:

Windows:

```powershell
set HGT_DATA_DIR=E:\path\to\data_raw_variant
scripts\preprocess\build_hgt_graph.bat
```

Linux/macOS:

```bash
export HGT_DATA_DIR=/path/to/data_raw_variant
scripts/preprocess/build_hgt_graph.sh
```

By default, `build_hgt_graph.bat` reads raw source files from:

```text
data/raw/
```

Override individual source files:

Windows:

```powershell
set HGT_STATIONS_CSV=E:\path\to\stations.csv
set HGT_BUILDINGS_CSV=E:\path\to\buildings.csv
set HGT_CONSERVATION_GEOJSON=E:\path\to\conservation.geojson
set HGT_ADMIN_GEOJSON=E:\path\to\admin.geojson
set HGT_ROADS_CSV=E:\path\to\roads.csv
set HGT_POI_CSV=E:\path\to\poi.csv
scripts\preprocess\build_hgt_graph.bat --radius-m 2560 --max-poi-per-station-group 120
```

Linux/macOS:

```bash
export HGT_STATIONS_CSV=/path/to/stations.csv
export HGT_BUILDINGS_CSV=/path/to/buildings.csv
export HGT_CONSERVATION_GEOJSON=/path/to/conservation.geojson
export HGT_ADMIN_GEOJSON=/path/to/admin.geojson
export HGT_ROADS_CSV=/path/to/roads.csv
export HGT_POI_CSV=/path/to/poi.csv
scripts/preprocess/build_hgt_graph.sh --radius-m 2560 --max-poi-per-station-group 120
```

The same options can be passed directly:

Windows:

```powershell
scripts\preprocess\build_hgt_graph.bat --data-dir E:\path\to\data\raw --poi-csv E:\path\to\poi.csv
```

Linux/macOS:

```bash
scripts/preprocess/build_hgt_graph.sh --data-dir /path/to/data/raw --poi-csv /path/to/poi.csv
```

## Visualize HGT graph

Default run uses `data\hgt_graph` as the graph source:

Windows:

```powershell
scripts\preprocess\visualize_hgt_graph.bat
```

Linux/macOS:

```bash
chmod +x scripts/preprocess/visualize_hgt_graph.sh
scripts/preprocess/visualize_hgt_graph.sh
```

Visualize a specific graph run:

Windows:

```powershell
set HGT_GRAPH_DIR=E:\postgrad\others\DF-2026\SHhistory\outputs\preprocess\hgt_graph\runs\yyyyMMdd-HHmmss
scripts\preprocess\visualize_hgt_graph.bat --station 0
```

Linux/macOS:

```bash
export HGT_GRAPH_DIR=/path/to/outputs/preprocess/hgt_graph/runs/yyyyMMdd-HHmmss
scripts/preprocess/visualize_hgt_graph.sh --station 0
```

Or pass the graph source directly:

Windows:

```powershell
scripts\preprocess\visualize_hgt_graph.bat --graph-dir E:\path\to\hgt_graph --station 0
```

Linux/macOS:

```bash
scripts/preprocess/visualize_hgt_graph.sh --graph-dir /path/to/hgt_graph --station 0
```

## Environment knobs

- `HGT_CONDA_ENV`: conda environment name. Default: `HGT`.
- `HGT_OUTPUT_DIR`: explicit output directory for `build_hgt_graph.bat`.
- `HGT_VIZ_OUTPUT_DIR`: explicit output directory for `visualize_hgt_graph.bat`.
- `HGT_DATA_DIR`: base raw-data directory for default graph-building sources. Default: `data/raw`.
- `HGT_GRAPH_DIR`: graph directory to visualize.
