# Preprocess wrappers

This directory contains Python preprocess entrypoints and Windows batch wrappers.
Use the `.bat` files for routine runs on Windows.

## Output layout

Batch wrappers write timestamped outputs under:

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

```powershell
scripts\preprocess\build_hgt_graph.bat
```

Use a different base data directory:

```powershell
set HGT_DATA_DIR=E:\path\to\data_raw_variant
scripts\preprocess\build_hgt_graph.bat
```

By default, `build_hgt_graph.bat` reads raw source files from:

```text
data\raw\
```

Override individual source files:

```powershell
set HGT_STATIONS_CSV=E:\path\to\stations.csv
set HGT_BUILDINGS_CSV=E:\path\to\buildings.csv
set HGT_CONSERVATION_GEOJSON=E:\path\to\conservation.geojson
set HGT_ADMIN_GEOJSON=E:\path\to\admin.geojson
set HGT_ROADS_CSV=E:\path\to\roads.csv
set HGT_POI_CSV=E:\path\to\poi.csv
scripts\preprocess\build_hgt_graph.bat --radius-m 2560 --max-poi-per-station-group 120
```

The same options can be passed directly:

```powershell
scripts\preprocess\build_hgt_graph.bat --data-dir E:\path\to\data\raw --poi-csv E:\path\to\poi.csv
```

## Visualize HGT graph

Default run uses `data\hgt_graph` as the graph source:

```powershell
scripts\preprocess\visualize_hgt_graph.bat
```

Visualize a specific graph run:

```powershell
set HGT_GRAPH_DIR=E:\postgrad\others\DF-2026\SHhistory\outputs\preprocess\hgt_graph\runs\yyyyMMdd-HHmmss
scripts\preprocess\visualize_hgt_graph.bat --station 0
```

Or pass the graph source directly:

```powershell
scripts\preprocess\visualize_hgt_graph.bat --graph-dir E:\path\to\hgt_graph --station 0
```

## Environment knobs

- `HGT_CONDA_ENV`: conda environment name. Default: `HGT`.
- `HGT_OUTPUT_DIR`: explicit output directory for `build_hgt_graph.bat`.
- `HGT_VIZ_OUTPUT_DIR`: explicit output directory for `visualize_hgt_graph.bat`.
- `HGT_DATA_DIR`: base raw-data directory for default graph-building sources. Default: `data\raw`.
- `HGT_GRAPH_DIR`: graph directory to visualize.
