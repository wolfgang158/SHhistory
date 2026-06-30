# HGT 可视化数据预处理 Pipeline 图示说明

本文档根据 `scripts/preprocess/build_hgt_graph.py`、`data/hgt_graph/README.md` 和当前输出文件整理，用于直接指导 `gpt-image2` 生成 CVPR 论文风格的预处理架构示意图。图中应强调：以上海地铁站为中心锚点，将历史建筑、道路、POI、历史文化风貌区和行政边界统一到同一空间坐标系，并构建站点中心的异质图数据，最终导出 HGT / PyG 可用的数据产物。

## 一句话概括

Metro-station-centered heterogeneous spatial graph preprocessing: multi-source urban heritage and mobility context data are normalized into a common metric CRS, queried within a 2560 m station context radius, encoded as typed nodes and typed spatial relations, then exported as `HeteroData`, pyHGT graph, flat HGT tensors, node tables, edge tables, and manifest metadata.

## 建议图名

**Station-Centered Heterogeneous Graph Preprocessing for Shanghai Urban Heritage Context Modeling**

## 图的整体布局

建议采用横向五栏流程图，左到右为：

1. **Raw Multi-source Data**
2. **Spatial Normalization**
3. **Station-centered Context Query**
4. **Heterogeneous Graph Construction**
5. **HGT-ready Outputs**

图形风格建议：

- CVPR 风格、白底、细线框、低饱和配色、少量强调色。
- 每个数据源用一个小卡片表示；每个处理阶段用圆角矩形模块表示；异质图阶段用 typed nodes + colored edges 表示。
- 使用地图底图的抽象线稿，不要放真实地图影像。
- 使用 6 类节点图标：station、building、road、POI、conservation area、admin area。
- 使用箭头标注关键操作：GCJ-02 to WGS84, WGS84 to EPSG:3857, radius query, KD-tree search, containment test, feature encoding, artifact export。

## 输入数据层

图左侧展示 6 类输入数据源：

| 图中标签 | 文件路径 | 几何/表格类型 | 作用 |
| --- | --- | --- | --- |
| Metro stations | `data/metro_stations/shanghai_metro_stations_amap.csv` | point table, GCJ-02 coordinates | 样本中心锚点，共 537 个站点 |
| Historic buildings | `data/historic_buildings/shanghai_excellent_historic_buildings_points.csv` | point table, WGS84-like lon/lat | 历史建筑节点，共 1071 个 |
| Road segments | `data/road_segments/shanghai_road_segments.csv` | WKT LineString | 道路节点，按站点邻域粗筛后保留 |
| POIs 2026 | `data/poi/2026_poi_Shanghai.csv` | point table, GCJ-02 coordinates | POI 节点，按 daily / tour / transport / other 分组 |
| Conservation areas | `data/historic_conservation_areas/*.geojson` | Polygon | 历史文化风貌区节点，共 12 个 |
| Administrative areas | `data/admin_boundary/shanghai_admin_boundary.geojson` | Polygon | 行政区/街镇边界节点 |

图中可将输入层画成 6 个并列小卡片，每张卡片包含一个小图标：

- Station: blue metro symbol
- Historic building: red heritage building
- Road: gray road line
- POI: orange map pin
- Conservation area: purple polygon
- Admin area: green boundary polygon

## Step 1: 空间坐标标准化

对应脚本中的核心函数：

- `gcj02_to_wgs84`
- `lonlat_to_webmerc`
- `project_geom`
- `load_stations`
- `load_buildings`
- `load_areas`
- `load_roads`
- `load_pois`

处理逻辑：

1. AMap 来源的地铁站和 POI 坐标从 GCJ-02 转为 WGS84。
2. OSM 道路、行政边界、历史建筑点和风貌区边界按 WGS84 / GeoJSON / WKT 读取。
3. 所有点、线、面统一投影到 EPSG:3857 Web Mercator 米制坐标。
4. 点对象保存 `lon, lat, x, y`；线和面对象计算投影后的 centroid、length 或 area。

图中建议表达：

```text
GCJ-02 / WGS84 geographic coordinates
        |
        v
Coordinate normalization
        |
        v
EPSG:3857 metric plane for distance query and graph geometry
```

视觉元素：

- 一个中间模块写作 **CRS Normalization**。
- 模块内放两行小字：`GCJ-02 -> WGS84` 和 `WGS84 -> EPSG:3857`。
- 输出箭头标注：`metric x/y coordinates`。

## Step 2: 站点中心上下文窗口

对应脚本参数：

- `--radius-m`, 默认 `2560.0`
- `--max-poi-per-station-group`, 默认 `120`

处理逻辑：

1. 每个地铁站作为一个 context anchor。
2. 默认以站点为中心建立半径 2560 m 的空间上下文。
3. 脚本使用 `scipy.spatial.cKDTree` 加速邻域查询。
4. 道路先用线段 centroid 粗筛，再用线到站点的真实几何距离精筛。
5. POI 按站点和类别分桶，每个站点每类 POI 最多保留最近 120 个候选。
6. 风貌区和行政区使用 polygon contains 判断站点是否位于区域内。

图中建议表达：

- 在地图抽象底图上画一个蓝色 station 点。
- station 周围画半透明圆形 buffer，标注 `r = 2560 m`。
- 圆内分布红色 building 点、橙色 POI 点、灰色 road 线、紫色 conservation polygon、绿色 admin polygon。
- buffer 旁标注 `KD-tree radius query + geometry containment`。

## Step 3: 异质节点构建

当前图包含 6 类节点：

| Node type | 来源 | 主要属性 | 当前数量 |
| --- | --- | --- | ---: |
| `station` | 地铁站 CSV | `id`, `name`, `lon`, `lat`, `x`, `y`, `adname` | 537 |
| `building` | 历史建筑 CSV | `batch`, `built_year`, `coordinate_quality`, position | 1071 |
| `road` | 道路 CSV WKT | `highway`, `length`, centroid | 98402 |
| `poi` | 2026 POI CSV | `category`, `group`, position | 185822 |
| `conservation_area` | 风貌区 GeoJSON | polygon centroid, area | 12 |
| `admin_area` | 行政边界 GeoJSON | polygon centroid, area | 389 |

图中建议将异质图区域画为一个中央 graph panel：

- 六种节点使用不同颜色和形状。
- `station` 节点放在中心或上层，其他节点围绕 station。
- 每个节点类型旁边显示数量，例如 `station x537`, `POI x185,822`。
- 标题写 **Typed Spatial Nodes**。

## Step 4: 异质边与空间关系构建

脚本使用 `add_bidirectional` 为每条空间关系添加正向和反向边。边权为距离衰减或区域包含常数。

主要关系如下：

| Source | Relation | Target | Edge weight |
| --- | --- | --- | --- |
| `station` | `near_building` | `building` | `1 / (1 + distance)` |
| `station` | `near_road` | `road` | `1 / (1 + distance)` |
| `station` | `near_poi_daily` | `poi` | `1 / (1 + distance)` |
| `station` | `near_poi_tour` | `poi` | `1 / (1 + distance)` |
| `station` | `near_poi_transport` | `poi` | `1 / (1 + distance)` |
| `station` | `near_poi_other` | `poi` | `1 / (1 + distance)` |
| `station` | `inside_conservation` | `conservation_area` | `1.0` |
| `building` | `inside_conservation` | `conservation_area` | `1.0` |
| `station` | `inside_admin` | `admin_area` | `1.0` |

每条边均有反向关系：

- `rev_near_building`
- `rev_near_road`
- `rev_near_poi_*`
- `rev_inside_conservation`
- `rev_inside_admin`

图中建议表达：

- 边颜色按 relation family 区分：
  - near-building: red
  - near-road: gray
  - near-POI: orange
  - inside-conservation: purple
  - inside-admin: green
- 边上可标注 `distance decay` 或 `contains`。
- 在 graph panel 角落放一个小 legend：`weight = 1/(1+d)` for near relations, `weight = 1` for containment。

## Step 5: 节点特征编码

统一特征维度：

- `FEATURE_DIM = 16`

所有节点均编码为 16 维 float feature。短向量通过 zero padding 补齐。

### Station feature

站点特征由归一化坐标和上下文统计组成：

1. normalized x
2. normalized y
3. log normalized historic building count
4. log normalized daily POI count
5. log normalized tour POI count
6. log normalized transport POI count
7. log normalized other POI count
8. log normalized total road length
9. log normalized major road length
10. conservation-area membership flag
11. daily-tour mix score
12-16. zero padding

### Building feature

1. normalized x
2. normalized y
3. coordinate quality score: high = 1.0, medium = 0.5, low = 0.0
4. publication batch normalized by 10
5. built year normalized as `(built_year - 1800) / 250`
6-16. zero padding

### Road feature

1. normalized x
2. normalized y
3. log normalized road length
4. major road flag, where motorway / trunk / primary / secondary = 1
5. local road flag, where tertiary / residential / service / unclassified / living_street / road = 1
6-16. zero padding

### POI feature

1. normalized x
2. normalized y
3-6. one-hot POI group: daily, tour, transport, other
7-16. zero padding

### Area feature

For `conservation_area` and `admin_area`:

1. normalized centroid x
2. normalized centroid y
3. log normalized area in square kilometers
4-16. zero padding

图中建议表达：

- 在异质图右侧放一个 **Feature Encoding** 模块。
- 模块内画成一个 16 维向量条：`[x, y, counts, lengths, flags, mix, padding]`。
- 可以放一个小公式：`node feature -> R^16`。

## Step 6: 输出数据产物

脚本输出目录默认为：

```text
data/hgt_graph/
```

输出文件：

| 文件 | 用途 |
| --- | --- |
| `pyhgt_graph.pkl` | legacy `pyHGT.data.Graph` 对象 |
| `hetero_data.pt` | PyTorch Geometric `HeteroData`，包含 typed node stores 和 typed edge stores |
| `hgt_tensors.pt` | flat HGT tensors: `x`, `node_type`, `edge_index`, `edge_type`, `edge_time`, type maps, node offsets |
| `nodes/*.csv` | 每类节点的元数据表，便于检查和可视化 |
| `edges.csv` | 边表：`source_type`, `source`, `relation`, `target_type`, `target`, `weight` |
| `manifest.json` | 参数、节点数量、边数量、CRS 描述 |

当前 manifest 关键统计：

```text
Context radius: 2560 m
Feature dim: 16
Total nodes: 286,233
Total edges: 3,417,482
CRS: EPSG:3857 features from WGS84/GCJ02-normalized source coordinates
```

图中建议输出层画成三个并列 artifact blocks：

1. **PyG HeteroData**: typed node stores + typed edge stores
2. **HGT Tensors**: `x`, `node_type`, `edge_index`, `edge_type`, `edge_time`
3. **Inspection Tables**: `nodes/*.csv`, `edges.csv`, `manifest.json`

## 推荐的架构图文字标注

图中推荐保留这些英文标签，便于 CVPR 风格呈现：

- Raw Multi-source Urban Data
- CRS Normalization
- Station-centered Context Radius
- KD-tree Neighbor Query
- Geometry Containment Test
- Typed Spatial Nodes
- Typed Spatial Relations
- 16-D Node Feature Encoding
- Heterogeneous Graph Assembly
- HGT-ready Artifacts

图中推荐保留这些公式/短注释：

```text
r = 2560 m
w_near = 1 / (1 + distance)
w_inside = 1
x_v in R^16
GCJ-02 -> WGS84 -> EPSG:3857
```

## 可直接给 gpt-image2 的 Prompt

```text
Create a clean CVPR paper-style architecture diagram on a white background, showing a station-centered heterogeneous spatial graph preprocessing pipeline for Shanghai urban heritage context modeling.

Use a horizontal left-to-right layout with five major stages:

1. Raw Multi-source Urban Data:
show six compact input cards with icons and labels:
Metro stations (537, point table, GCJ-02),
Historic buildings (1071, point table),
Road segments (WKT LineString),
2026 POIs (point table),
Conservation areas (12 polygons),
Administrative areas (389 polygons).

2. CRS Normalization:
show arrows from mixed coordinate systems into one module labeled "CRS Normalization".
Inside the module write:
"GCJ-02 -> WGS84"
"WGS84 -> EPSG:3857"
Output label: "metric x/y coordinates".

3. Station-centered Context Query:
draw an abstract map patch with a blue metro station point in the center and a translucent circular buffer labeled "r = 2560 m".
Inside the buffer show red historic building dots, orange POI pins, gray road lines, purple conservation polygon, and green admin boundary.
Add labels "KD-tree radius query", "nearest POIs per group", and "polygon contains test".

4. Heterogeneous Graph Construction:
draw a typed graph panel with six node types:
station, building, road, POI, conservation_area, admin_area.
Use distinct colors and shapes.
Draw colored typed edges:
station--near_building--building,
station--near_road--road,
station--near_poi_daily/tour/transport/other--POI,
station--inside_conservation--conservation_area,
building--inside_conservation--conservation_area,
station--inside_admin--admin_area.
Show reverse edges with subtle dashed arrows.
Add a small legend:
"near edge weight: w = 1/(1+d)"
"containment edge weight: w = 1".

5. HGT-ready Outputs:
draw three output blocks:
"PyG HeteroData: typed node stores + typed edge stores",
"HGT tensors: x, node_type, edge_index, edge_type, edge_time",
"Inspection tables: nodes/*.csv, edges.csv, manifest.json".
Add summary text:
"16-D node features"
"286,233 nodes"
"3,417,482 directed typed edges".

Visual style:
minimal CVPR / NeurIPS diagram style, crisp vector-like graphics, thin gray outlines, muted blue/red/orange/purple/green colors, no photorealistic map, no clutter, no drop shadows, clear typography, concise labels, balanced spacing, suitable for a two-column computer vision paper figure.
```

## 简化版 Caption

**Figure.** Preprocessing pipeline for station-centered heterogeneous graph construction. Multi-source Shanghai urban data are normalized into EPSG:3857, queried within a 2560 m context radius around each metro station, and converted into typed spatial nodes and typed relations. Each node is encoded as a 16-D feature vector, and the final graph is exported as PyG `HeteroData`, pyHGT graph, flat HGT tensors, and inspection tables.

