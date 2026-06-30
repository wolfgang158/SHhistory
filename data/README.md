# data 数据说明

本目录按数据主题整理上海历史建筑、历史文化风貌区边界和上海地铁站点相关数据。PDF 文件为文档材料，暂保留在 `data/` 根目录，不纳入结构化数据整理。

## 目录结构

```text
data/
├── historic_buildings/
│   ├── shanghai_excellent_historic_buildings_points.csv
│   └── shanghai_excellent_historic_buildings_points.geojson
├── historic_conservation_areas/
│   └── 上海中心城区12片历史文化风貌区_保护边界_面积校准拟合.geojson
├── metro_stations/
│   ├── fetch_meta.json
│   ├── shanghai_metro_stations_amap.csv
│   └── shanghai_metro_stations_amap.json
├── DF2026methodology.pdf
└── README.md
```

## 文件夹概览

| 文件夹 | 内容 |
| --- | --- |
| `historic_buildings/` | 上海优秀历史建筑点位数据，包含 CSV 表格版和 GeoJSON 点要素版 |
| `historic_conservation_areas/` | 上海中心城区 12 片历史文化风貌区保护边界面数据 |
| `metro_stations/` | 高德地图抓取的上海地铁站点数据及抓取元数据 |

## 历史建筑点位数据

### `historic_buildings/shanghai_excellent_historic_buildings_points.csv`

上海优秀历史建筑点位的表格版本。共 1071 条记录。

主要字段：

| 字段 | 含义 |
| --- | --- |
| `uid` | 记录唯一标识 |
| `batch` | 历史建筑公布批次 |
| `display_name` | 用于展示的建筑名称 |
| `original_name` | 原名称 |
| `current_name` | 现名称 |
| `district_original` | 原始资料中的区县 |
| `district_current` | 当前整理后的区县 |
| `raw_address` | 原始地址 |
| `geocode_query_address` | 用于地理编码的查询地址 |
| `longitude` / `latitude` | 经度 / 纬度 |
| `geocode_level` | 高德地理编码匹配级别 |
| `geocode_formatted_address` | 高德返回的标准化地址 |
| `protection_category` | 保护类别 |
| `built_year` | 建成年份 |
| `source_url` | 来源链接 |
| `coordinate_quality` | 坐标质量，分为 `high`、`medium`、`low` |

关键统计：

| 维度 | 统计 |
| --- | --- |
| 坐标质量 | `high` 858 条，`medium` 190 条，`low` 23 条 |
| 批次 | 第一批 62 条，第二批 176 条，第三批 162 条，第四批 245 条，第五批 426 条 |
| 主要区县 | 黄浦区 279 条，徐汇区 265 条，静安区 161 条，长宁区 125 条，虹口区 96 条 |

### `historic_buildings/shanghai_excellent_historic_buildings_points.geojson`

历史建筑点位的 GeoJSON 版本。共 1071 个 `Point` 要素。

说明：

- 几何类型为 `Point`。
- 属性字段与 `historic_buildings/shanghai_excellent_historic_buildings_points.csv` 基本一致。
- 适合直接导入 QGIS、ArcGIS、Kepler.gl、Mapbox、Leaflet 等地图工具。
- 如果只是做表格分析，优先使用 CSV；如果要做空间可视化，优先使用 GeoJSON。

## 历史文化风貌区边界数据

### `historic_conservation_areas/上海中心城区12片历史文化风貌区_保护边界_面积校准拟合.geojson`

上海中心城区 12 片历史文化风貌区保护边界数据。共 12 个 `Polygon` 面要素。

包含的风貌区：

| 风貌区 |
| --- |
| 外滩 |
| 人民广场 |
| 老城厢 |
| 衡山路-复兴路 |
| 南京西路 |
| 愚园路 |
| 新华路 |
| 山阴路 |
| 提篮桥 |
| 江湾 |
| 龙华 |
| 虹桥路 |

主要字段：

| 字段 | 含义 |
| --- | --- |
| `name` | 风貌区名称 |
| `source_area_ha` | 来源资料中的面积，单位为公顷 |
| `boundary_text` | 文字描述的边界 |
| `method` | 边界拟合方法 |
| `source` | 数据来源说明 |
| `raw_geom_area_ha` | 原始几何面积，单位为公顷 |
| `raw_area_diff_pct` | 原始几何面积与来源面积的差异比例 |
| `calibrated_area_ha` | 校准后面积，单位为公顷 |
| `area_scale_factor` | 面积校准比例 |
| `precision_note` | 精度说明 |

注意：

- 该数据是根据边界文字描述和 OpenStreetMap 线位拟合得到。
- 文件名中的“面积校准拟合”表示边界经过面积校准，并非官方测绘红线。

## 地铁站点数据

### `metro_stations/shanghai_metro_stations_amap.csv`

高德地图 POI 接口抓取的上海地铁站点表。共 537 条记录。

主要字段：

| 字段 | 含义 |
| --- | --- |
| `id` | 高德 POI ID |
| `name` | 站点名称 |
| `type` | POI 类型 |
| `typecode` | POI 类型编码 |
| `pname` | 省级名称 |
| `cityname` | 城市名称 |
| `adname` | 行政区名称 |
| `address` | 地址，通常为线路信息 |
| `location` | 高德原始坐标字符串 |
| `longitude` / `latitude` | 经度 / 纬度 |
| `citycode` | 城市编码 |
| `adcode` | 行政区编码 |
| `business_area` | 商圈 |
| `query_polygon` | 该记录来源的查询网格 |

主要区县分布：

| 行政区 | 站点数 |
| --- | ---: |
| 浦东新区 | 166 |
| 闵行区 | 65 |
| 宝山区 | 45 |
| 徐汇区 | 38 |
| 普陀区 | 31 |
| 杨浦区 | 28 |
| 嘉定区 | 27 |
| 虹口区 | 25 |
| 青浦区 | 22 |
| 静安区 | 19 |
| 黄浦区 | 17 |
| 长宁区 | 16 |

### `metro_stations/shanghai_metro_stations_amap.json`

高德地图返回的地铁站点原始 POI JSON 数据。共 537 条记录。

说明：

- 该文件保留了比 CSV 更完整的高德 POI 字段。
- `metro_stations/shanghai_metro_stations_amap.csv` 是从该 JSON 中抽取常用字段后形成的轻量版本。
- 如果需要照片、别名、室内地图、出入口位置、业务扩展字段等信息，可以优先查看 JSON。

常见字段：

| 字段 | 含义 |
| --- | --- |
| `id` | 高德 POI ID |
| `name` | 站点名称 |
| `type` / `typecode` | POI 类型及编码 |
| `pname` / `cityname` / `adname` | 省、市、区 |
| `address` | 地址或线路信息 |
| `location` | 坐标 |
| `photos` | 高德返回的图片信息 |
| `business_area` | 商圈 |
| `entr_location` / `exit_location` | 入口 / 出口相关坐标字段 |
| `_query_polygon` | 抓取时对应的查询网格 |

## 抓取元数据

### `metro_stations/fetch_meta.json`

地铁站点数据抓取过程的元信息。

主要字段：

| 字段 | 含义 |
| --- | --- |
| `source` | 数据来源，当前为高德 Web API place/polygon |
| `query` | 查询关键词、类型和扩展参数 |
| `bbox` | 抓取范围和网格步长 |
| `cells_fetched` | 实际抓取的网格数量，当前为 90 |
| `unique_pois` | 去重后的 POI 数量，当前为 537 |
| `by_adname` | 按行政区统计的 POI 数量 |
| `query_pages` | 每个查询网格和分页的抓取记录 |

用途：

- 复核地铁站点数据的抓取范围和查询条件。
- 检查哪些网格返回了多少 POI。
- 追踪 `metro_stations/shanghai_metro_stations_amap.csv` 和 `metro_stations/shanghai_metro_stations_amap.json` 的生成过程。

## 使用建议

- 表格统计、筛选和建模：优先使用 `.csv` 文件。
- 空间分析和地图可视化：优先使用 `.geojson` 文件。
- 需要追溯高德原始字段：使用 `metro_stations/shanghai_metro_stations_amap.json`。
- 需要复核数据抓取过程：使用 `metro_stations/fetch_meta.json`。
- PDF 文件是参考文档或研究材料，不作为结构化数据源整理。
