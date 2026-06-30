  建议的基础版技术实现方案

  1. 先定样本单元

  - 以“地铁站”为基础样本中心，每个站点生成一个空间 patch。
  - 基础版建议两级尺度：
      - local patch：256 x 256，5m/pixel，突出站点周边细节。
      - context patch：512 x 512，10m/pixel，覆盖更大的站城背景。

  - 如果先做一版最小可行方案，直接用 512 x 512 @ 10m/pixel 就够了。

  2. 统一空间坐标系

  - 先把所有数据拉到同一坐标基准，再做栅格化。
  - 处理顺序建议：
      - AMap 地铁站点 GCJ-02 -> WGS84
      - OSM 道路、行政边界通常是 WGS84
      - 历史建筑点和风貌区边界也统一检查到 WGS84
      - 最终 rasterization 时投影到一个米制坐标系，比如 EPSG:3857 或本地等距投影

  - 这一步很关键，不然不同来源会发生空间错位。

  3. 定义异构图像的通道设计

  - 基础版先做“空间结构异构图像”，不要一开始就依赖复杂 POI 语义。
  - 推荐通道如下：
      - C1：地铁站中心点热力图
      - C2：历史建筑点密度图
      - C3：风貌区边界 mask
      - C4：道路总体密度
      - C5：主干路密度
      - C6：次干路/支路密度
      - C7：道路交叉口密度
      - C8：行政边界 mask
      - C9：站点距离变换图
      - C10：历史建筑缓冲区密度

  - 如果后续补上 AMap POI 数据，可以再加：
      - C11：日常生活类 POI 密度
      - C12：旅游休闲类 POI 密度
      - C13：日常-旅游混合度/熵值图

  4. 做矢量到栅格的转换

  - 用 geopandas + shapely + rasterio 做栅格化。
  - 点要素处理：
      - 地铁站、历史建筑点做高斯核扩散，不建议直接打单像素点。
      - 核半径按分辨率设定，比如 20m ~ 50m。

  - 线要素处理：
      - 道路先按等级做不同宽度 buffer，再 rasterize。
      - 例如主干路更宽，支路更窄。

  - 面要素处理：
      - 风貌区、行政边界直接 rasterize 成 mask。

  - 每个通道单独生成，最后按通道堆叠。

  5. 构建候选标签

  - 如果 PDF 基础框架里已经有冲突等级定义，就直接把它映射成监督标签。
  - 如果还没有明确标签，先做两层输出：
      - 候选样本：按站点输出异构图像
      - 候选指标：比如站点周边历史建筑密度、道路强度、风貌区覆盖率、POI 混合度

  - 后续人工标注或规则判别时，就直接基于这些候选图做判定。
  - 建议先支持三类标签格式：
      - 二分类：有/无混合冲突
      - 三分类：低/中/高
      - 多分类：日常主导、旅游主导、混合挤压、过渡平衡等

  6. 定义输出数据结构

  - 建议输出成三类文件：
      - sample_id.npz：多通道图像张量
      - sample_id.json：样本元数据
      - sample_id.png：可视化预览图

  - 元数据建议包含：
      - center_lon, center_lat
      - bbox
      - crs
      - pixel_size
      - channels
      - label
      - source_versions

  - 如果数据量大，推荐再建一个 manifest.csv 或 manifest.parquet 做索引。

  7. 做质量控制

  - 每批数据都跑这几类检查：
      - 坐标是否越界
      - 各层是否对齐
      - 是否存在空 patch
      - 是否有异常密集或异常稀疏的样本
      - 类别是否严重不平衡

  - 视觉检查建议随机抽样 50 张，把多通道结果叠成彩色预览图看一遍。
  - 对于站点中心样本，检查站点是否落在图像中心附近，偏差不能太大。

  8. 工程拆分

  - 建议拆成 5 个脚本/模块：
      - load_sources.py：读取 CSV/GeoJSON/JSON
      - normalize_crs.py：统一坐标系和字段
      - build_samples.py：以站点为中心生成 patch
      - rasterize_layers.py：矢量转多通道图像
      - export_dataset.py：输出 npz/json/png/manifest

  - 这样后续补数据源或改通道时，不用重写全流程。

  结合你现有仓库，最适合先接入的数据

  - 地铁站点：data/metro_stations/shanghai_metro_stations_amap.csv
  - 历史建筑：data/historic_buildings/...
  - 风貌区边界：data/historic_conservation_areas/...
  - 道路：data/osm/shanghai_road_segments.*
  - 行政边界：data/osm/shanghai_admin_boundary.*
  - POI 扩展抓取脚本：scripts/collect_data/collect_shanghai_poi.py

  推荐的第一版落地顺序

  1. 先做站点中心 512 x 512 基础异构图像。
  2. 只接入结构类通道：站点、道路、历史建筑、风貌区、行政边界。
  3. 跑通导出、预览和 QC。
  4. 再补 POI 语义通道，增强“日常-旅游混合”判别能力。
  5. 最后再做多尺度和标签体系升级。
