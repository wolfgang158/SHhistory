# HGT Real-Time Inference API JSON I/O Specification v2

## 中文

### 1. 目标

该 API 用于封装已训练好的 HGT 模型，为外部程序提供实时冲突潜势预测。

后端预加载：

- HGT 模型权重：`20260702-092440/best_model.pt`
- heterogeneous graph
- node features
- edges
- station id / station name / coordinate mapping

客户端只提交查询 JSON，不提交完整图结构。

---

### 2. Endpoint 设计

| Method | Endpoint | 用途 |
|---|---|---|
| `POST` | `/api/v1/predict/station` | 按站点名称或站点 ID 查询 |
| `POST` | `/api/v1/predict/nearest` | 按经纬度匹配最近站点并预测 |
| `GET` | `/api/v1/predict/all` | 返回所有站点预测结果 |
| `GET` | `/api/v1/health` | 检查 API 和模型是否已加载 |

---

### 3. 输入 JSON

#### 3.1 按站点查询

`station_name` 和 `station_id` 至少提供一个。优先使用 `station_id`。

```json
{
  "station_id": "station_000128",
  "station_name": "淮海中路",
  "return_geometry": true,
  "return_visualization": true
}
```

字段说明：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `station_id` | string | 否 | 后端站点节点 ID |
| `station_name` | string | 否 | 站点名称 |
| `return_geometry` | boolean | 否 | 是否返回 GeoJSON geometry，默认 `true` |
| `return_visualization` | boolean | 否 | 是否返回颜色、半径等可视化字段，默认 `true` |

---

#### 3.2 按坐标查询最近站点

```json
{
  "coordinate": {
    "lon": 121.4726,
    "lat": 31.2297
  },
  "search_radius_m": 800,
  "return_geometry": true,
  "return_visualization": true
}
```

字段说明：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `coordinate.lon` | float | 是 | WGS84 经度 |
| `coordinate.lat` | float | 是 | WGS84 纬度 |
| `search_radius_m` | float | 否 | 最近站点搜索半径，单位米，默认 `800` |
| `return_geometry` | boolean | 否 | 是否返回 GeoJSON geometry，默认 `true` |
| `return_visualization` | boolean | 否 | 是否返回可视化字段，默认 `true` |

---

### 4. 统一输出 JSON

无论是单站点还是多站点，核心结构统一为：

```json
{
  "status": "success",
  "data": [
    {
      "station": {},
      "prediction": {},
      "visualization": {}
    }
  ],
  "model": {},
  "meta": {}
}
```

---

### 5. 单站点输出示例

```json
{
  "status": "success",
  "data": [
    {
      "station": {
        "station_id": "station_000128",
        "station_name": "淮海中路",
        "lon": 121.4726,
        "lat": 31.2297,
        "geometry": {
          "type": "Point",
          "coordinates": [121.4726, 31.2297]
        }
      },
      "prediction": {
        "conflict_score": 0.73,
        "conflict_level": "high",
        "confidence": 0.81,
        "daily_activity_score": 0.68,
        "tourism_activity_score": 0.76,
        "heritage_context_score": 0.59,
        "accessibility_score": 0.84
      },
      "visualization": {
        "color": "#E64B35",
        "radius": 9,
        "opacity": 0.9
      }
    }
  ],
  "model": {
    "model_type": "HGT",
    "checkpoint": "20260702-092440/best_model.pt",
    "inference_mode": "real_time_forward",
    "output_type": "continuous_score"
  },
  "meta": {
    "request_type": "station",
    "result_count": 1,
    "crs": "EPSG:4326"
  }
}
```

---

### 6. 全部站点输出示例

```json
{
  "status": "success",
  "data": [
    {
      "station": {
        "station_id": "station_000128",
        "station_name": "淮海中路",
        "lon": 121.4726,
        "lat": 31.2297
      },
      "prediction": {
        "conflict_score": 0.73,
        "conflict_level": "high",
        "confidence": 0.81
      },
      "visualization": {
        "color": "#E64B35",
        "radius": 9
      }
    },
    {
      "station": {
        "station_id": "station_000244",
        "station_name": "人民广场",
        "lon": 121.4752,
        "lat": 31.2329
      },
      "prediction": {
        "conflict_score": 0.61,
        "conflict_level": "medium",
        "confidence": 0.74
      },
      "visualization": {
        "color": "#F39C12",
        "radius": 7
      }
    }
  ],
  "model": {
    "model_type": "HGT",
    "checkpoint": "20260702-092440/best_model.pt",
    "inference_mode": "real_time_forward",
    "output_type": "continuous_score"
  },
  "meta": {
    "request_type": "all",
    "result_count": 537,
    "crs": "EPSG:4326"
  }
}
```

---

### 7. 字段类型约束

| 字段 | 类型 | 范围 / 枚举 | 说明 |
|---|---|---|---|
| `conflict_score` | float | `0.0 - 1.0` | HGT 输出的连续冲突潜势分数 |
| `conflict_level` | string | `low`, `medium`, `high` | 由 `conflict_score` 映射得到的展示等级 |
| `confidence` | float | `0.0 - 1.0` | 模型预测置信度 |
| `daily_activity_score` | float | `0.0 - 1.0` | 日常活动相关分数 |
| `tourism_activity_score` | float | `0.0 - 1.0` | 文旅活动相关分数 |
| `heritage_context_score` | float | `0.0 - 1.0` | 历史文化环境相关分数 |
| `accessibility_score` | float | `0.0 - 1.0` | 可达性相关分数 |
| `color` | string | hex color | 前端可视化颜色 |
| `radius` | float | `> 0` | 前端点半径 |
| `geometry` | object | GeoJSON | 可选返回 |

---

### 8. 错误输出

#### 站点不存在

```json
{
  "status": "error",
  "error_code": "STATION_NOT_FOUND",
  "message": "No station matched the input station_id or station_name.",
  "data": [],
  "meta": {
    "request_type": "station"
  }
}
```

#### 输入格式错误

```json
{
  "status": "error",
  "error_code": "INVALID_INPUT",
  "message": "coordinate.lon and coordinate.lat are required for nearest-station query.",
  "data": [],
  "meta": {
    "request_type": "nearest"
  }
}
```

#### 模型未加载

```json
{
  "status": "error",
  "error_code": "MODEL_NOT_READY",
  "message": "HGT model or graph tensors are not loaded.",
  "data": [],
  "meta": {}
}
```

---

### 9. 实时推理流程

1. API 启动时加载 `best_model.pt`、heterogeneous graph、node features、edges 和 station mapping。
2. 客户端发送 JSON 请求。
3. 后端根据 station ID、station name 或 coordinate 匹配图中的 station node。
4. HGT 模型基于已加载图结构做 forward inference。
5. 后端返回连续 `conflict_score`、展示等级、置信度和可视化字段。

---

### 10. 可视化说明

API 返回的是 JSON 数据，不直接返回图片。

可视化由客户端完成：

- Web map 使用 `lon`、`lat`、`color`、`radius` 绘制站点。
- Grasshopper / Rhino 使用 `conflict_score` 控制点颜色、半径或图层。
- GIS / Python 可将 `data` 转为 GeoJSON 或 GeoDataFrame。

---

## English

### 1. Objective

This API wraps the trained HGT model and provides real-time conflict potential prediction for external clients.

The backend preloads:

- HGT checkpoint: `20260702-092440/best_model.pt`
- heterogeneous graph
- node features
- edges
- station id / station name / coordinate mapping

The client only sends a query JSON. It does not send the full graph structure.

---

### 2. Endpoint Design

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/v1/predict/station` | Query by station name or station ID |
| `POST` | `/api/v1/predict/nearest` | Match nearest station by coordinates and predict |
| `GET` | `/api/v1/predict/all` | Return predictions for all stations |
| `GET` | `/api/v1/health` | Check whether the API and model are loaded |

---

### 3. Input JSON

#### 3.1 Query by station

At least one of `station_name` or `station_id` should be provided. `station_id` has higher priority.

```json
{
  "station_id": "station_000128",
  "station_name": "Huaihai Middle Road",
  "return_geometry": true,
  "return_visualization": true
}
```

Field description:

| Field | Type | Required | Description |
|---|---|---:|---|
| `station_id` | string | No | Backend station node ID |
| `station_name` | string | No | Station name |
| `return_geometry` | boolean | No | Whether to return GeoJSON geometry, default `true` |
| `return_visualization` | boolean | No | Whether to return color/radius visualization fields, default `true` |

---

#### 3.2 Query nearest station by coordinate

```json
{
  "coordinate": {
    "lon": 121.4726,
    "lat": 31.2297
  },
  "search_radius_m": 800,
  "return_geometry": true,
  "return_visualization": true
}
```

Field description:

| Field | Type | Required | Description |
|---|---|---:|---|
| `coordinate.lon` | float | Yes | WGS84 longitude |
| `coordinate.lat` | float | Yes | WGS84 latitude |
| `search_radius_m` | float | No | Nearest-station search radius in meters, default `800` |
| `return_geometry` | boolean | No | Whether to return GeoJSON geometry, default `true` |
| `return_visualization` | boolean | No | Whether to return visualization fields, default `true` |

---

### 4. Unified Output JSON

For both single-station and multi-station queries, the response uses the same structure:

```json
{
  "status": "success",
  "data": [
    {
      "station": {},
      "prediction": {},
      "visualization": {}
    }
  ],
  "model": {},
  "meta": {}
}
```

---

### 5. Single-Station Output Example

```json
{
  "status": "success",
  "data": [
    {
      "station": {
        "station_id": "station_000128",
        "station_name": "Huaihai Middle Road",
        "lon": 121.4726,
        "lat": 31.2297,
        "geometry": {
          "type": "Point",
          "coordinates": [121.4726, 31.2297]
        }
      },
      "prediction": {
        "conflict_score": 0.73,
        "conflict_level": "high",
        "confidence": 0.81,
        "daily_activity_score": 0.68,
        "tourism_activity_score": 0.76,
        "heritage_context_score": 0.59,
        "accessibility_score": 0.84
      },
      "visualization": {
        "color": "#E64B35",
        "radius": 9,
        "opacity": 0.9
      }
    }
  ],
  "model": {
    "model_type": "HGT",
    "checkpoint": "20260702-092440/best_model.pt",
    "inference_mode": "real_time_forward",
    "output_type": "continuous_score"
  },
  "meta": {
    "request_type": "station",
    "result_count": 1,
    "crs": "EPSG:4326"
  }
}
```

---

### 6. All-Station Output Example

```json
{
  "status": "success",
  "data": [
    {
      "station": {
        "station_id": "station_000128",
        "station_name": "Huaihai Middle Road",
        "lon": 121.4726,
        "lat": 31.2297
      },
      "prediction": {
        "conflict_score": 0.73,
        "conflict_level": "high",
        "confidence": 0.81
      },
      "visualization": {
        "color": "#E64B35",
        "radius": 9
      }
    },
    {
      "station": {
        "station_id": "station_000244",
        "station_name": "People's Square",
        "lon": 121.4752,
        "lat": 31.2329
      },
      "prediction": {
        "conflict_score": 0.61,
        "conflict_level": "medium",
        "confidence": 0.74
      },
      "visualization": {
        "color": "#F39C12",
        "radius": 7
      }
    }
  ],
  "model": {
    "model_type": "HGT",
    "checkpoint": "20260702-092440/best_model.pt",
    "inference_mode": "real_time_forward",
    "output_type": "continuous_score"
  },
  "meta": {
    "request_type": "all",
    "result_count": 537,
    "crs": "EPSG:4326"
  }
}
```

---

### 7. Field Constraints

| Field | Type | Range / Enum | Description |
|---|---|---|---|
| `conflict_score` | float | `0.0 - 1.0` | Continuous conflict potential score from HGT |
| `conflict_level` | string | `low`, `medium`, `high` | Display level mapped from `conflict_score` |
| `confidence` | float | `0.0 - 1.0` | Model prediction confidence |
| `daily_activity_score` | float | `0.0 - 1.0` | Daily activity score |
| `tourism_activity_score` | float | `0.0 - 1.0` | Tourism activity score |
| `heritage_context_score` | float | `0.0 - 1.0` | Heritage context score |
| `accessibility_score` | float | `0.0 - 1.0` | Accessibility score |
| `color` | string | hex color | Visualization color |
| `radius` | float | `> 0` | Visualization point radius |
| `geometry` | object | GeoJSON | Optional geometry object |

---

### 8. Error Output

#### Station not found

```json
{
  "status": "error",
  "error_code": "STATION_NOT_FOUND",
  "message": "No station matched the input station_id or station_name.",
  "data": [],
  "meta": {
    "request_type": "station"
  }
}
```

#### Invalid input

```json
{
  "status": "error",
  "error_code": "INVALID_INPUT",
  "message": "coordinate.lon and coordinate.lat are required for nearest-station query.",
  "data": [],
  "meta": {
    "request_type": "nearest"
  }
}
```

#### Model not ready

```json
{
  "status": "error",
  "error_code": "MODEL_NOT_READY",
  "message": "HGT model or graph tensors are not loaded.",
  "data": [],
  "meta": {}
}
```

---

### 9. Real-Time Inference Flow

1. When the API starts, it loads `best_model.pt`, the heterogeneous graph, node features, edges, and station mapping.
2. The client sends a JSON request.
3. The backend matches a station node by station ID, station name, or coordinate.
4. The HGT model runs forward inference on the preloaded graph.
5. The backend returns continuous `conflict_score`, display level, confidence, and visualization fields.

---

### 10. Visualization

The API returns JSON data, not an image.

Visualization is handled by the client:

- Web map uses `lon`, `lat`, `color`, and `radius` to draw station points.
- Grasshopper / Rhino uses `conflict_score` to control point color, radius, or layer.
- GIS / Python can convert `data` into GeoJSON or GeoDataFrame.
