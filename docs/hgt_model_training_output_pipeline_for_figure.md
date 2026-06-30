# HGT 模型训练及产出 Pipeline 图示说明

本文档用于指导 `gpt-image2` 生成一张 CVPR 论文风格的 **HGT 模型训练及产出架构图**。定位是技术实现说明：从 `data/hgt_graph/hgt_tensors.pt` / `hetero_data.pt` 读取预处理后的异质图，经过 HGT 模型编码、监督任务头训练，最终输出站点级“日常-旅游混合冲突”识别结果、模型权重、评估指标和可解释性图层。

参考信息：

- 已实现图数据构建：`scripts/preprocess/build_hgt_graph.py`
- 图数据说明：`data/hgt_graph/README.md`
- HGT 模型实现：`HGT/pyHGT/model.py`, `HGT/pyHGT/conv.py`
- 通用训练范式：`HGT/OAG/train_paper_field.py`
- 研究目标参考：`docs/研究方案简版：轨道站城地区日常—旅游混合冲突识别(4).pdf`

注意：当前仓库已有 HGT 图数据与通用 HGT 模型代码，但没有看到专门面向本项目的训练入口脚本。因此本图应表达“技术实现方案 / training architecture”，不要画成“已经完整跑通的现成训练脚本日志”。

## 一句话概括

HGT training pipeline for station-level daily-tourism mixed-conflict recognition: preprocessed heterogeneous graph tensors are fed into type-specific feature adapters and stacked HGTConv layers; station embeddings are selected for a supervised prediction head; losses and metrics drive optimization; trained weights, station conflict scores, class labels, spatial risk maps, and attention-based explanations are exported.

## 建议图名

**HGT Training and Output Pipeline for Station-level Daily-Tourism Conflict Recognition**

## 图的整体布局

建议采用横向六栏流程图，左到右为：

1. **HGT Graph Artifacts**
2. **Training Data Assembly**
3. **HGT Encoder**
4. **Station-level Task Head**
5. **Optimization and Evaluation**
6. **Model Outputs and Spatial Products**

图形风格建议：

- CVPR / NeurIPS 技术图风格，白底、细线、轻量阴影或无阴影、低饱和色。
- 强调工程实现链路：tensor input -> mini-batch / full-batch training -> HGT encoder -> prediction head -> loss / metrics -> exported products。
- HGT 模型内部用 “type-specific linear adapters + stacked HGTConv blocks” 表示，不要画成普通 CNN。
- 右侧产出层要同时包含机器学习产物和城市空间分析产物。

## Stage 1: HGT Graph Artifacts

左侧输入来自 `data/hgt_graph/`：

| Artifact | 内容 | 图中表达 |
| --- | --- | --- |
| `hgt_tensors.pt` | flat tensors: `x`, `node_type`, `edge_index`, `edge_type`, `edge_time`, `node_type_map`, `edge_type_map`, `node_offsets` | 一个 tensor stack 图标 |
| `hetero_data.pt` | PyG `HeteroData` typed node stores and edge stores | 一个 typed graph 数据块 |
| `nodes/*.csv` | 站点、建筑、道路、POI、风貌区、行政区节点元数据 | 表格图标 |
| `edges.csv` | typed spatial relation edge table | 关系表图标 |
| `manifest.json` | 节点/边数量、特征维度、CRS、半径参数 | metadata 图标 |

关键输入规模：

```text
Node feature dimension: 16
Node types: 6
Directed typed edges: 3,417,482
Total nodes: 286,233
Target prediction unit: station nodes, 537 samples
```

图中建议将输入画成两个并列层次：

- **Tensor Inputs**: `x`, `node_type`, `edge_index`, `edge_type`, `edge_time`
- **Metadata Inputs**: node offsets, type maps, station metadata, optional labels

## Stage 2: Training Data Assembly

训练数据组织需要把图张量和站点级监督标签对齐。标签来自研究方案中的“轨道站城地区日常-旅游混合冲突识别”任务，可按实际实验设计支持二分类、三分类或多分类。

### 目标节点

当前任务建议以 `station` 节点作为 supervised target：

```text
station embeddings -> conflict prediction
```

### 标签形式

可在图中画成三种可选 label schema：

| Label schema | 类别 |
| --- | --- |
| Binary | conflict / non-conflict |
| Ordinal 3-class | low / medium / high |
| Multi-class | daily-dominant, tourism-dominant, mixed-pressure, transitional-balance |

### 数据划分

图中建议表达为：

```text
Station labels + station node offsets
        |
        v
Train / Validation / Test split
```

可视化时把 `station` 节点从全图中高亮出来，并用 `train`, `val`, `test` 三种颜色的小标记表示样本划分。

### Batch 组织方式

当前 `hgt_tensors.pt` 支持 full-graph training；HGT 原始代码也提供 `sample_subgraph` 的 mini-batch 子图采样范式。图中可以并列画两条数据装载路径：

1. **Full-graph mode**: load all tensors on GPU, forward all nodes, compute loss only on labeled station nodes.
2. **Subgraph sampling mode**: sample station-centered typed neighborhoods, train on station mini-batches.

建议图中主路径使用 full-graph mode，因为当前仓库已验证 `hgt_tensors.pt` 可直接通过一层 HGT 前向；旁边用虚线框表示 optional subgraph sampling。

## Stage 3: HGT Encoder

对应代码：

- `HGT/pyHGT/model.py`
  - `GNN`
  - `Classifier`
- `HGT/pyHGT/conv.py`
  - `HGTConv`
  - `GeneralConv`
  - `RelTemporalEncoding`

### 3.1 Type-specific feature adapters

`GNN.forward` 首先按节点类型分别使用线性层把 16 维输入投影到隐藏维度：

```text
x_v in R^16
node_type t
type-specific Linear_t: R^16 -> R^d
tanh + dropout
```

图中画法：

- 输入为 6 类节点的 16-D feature bars。
- 进入 6 个并列的 type-specific adapters：`W_station`, `W_building`, `W_road`, `W_poi`, `W_conservation`, `W_admin`。
- 合并成统一 hidden node representation `h_v in R^d`。

### 3.2 Stacked HGTConv blocks

HGTConv 的核心机制应画成一个可重复堆叠的 transformer-like graph block：

```text
for each meta relation <source_type, relation_type, target_type>:
    target-specific Q projection
    source-specific K/V projection
    relation-specific attention matrix
    relation-specific message matrix
    multi-head attention over incoming typed edges
    target-specific aggregation + skip connection + layer norm
```

图中建议画 2-4 个堆叠模块，并标注：

- **Heterogeneous Mutual Attention**
- **Relation-specific Message Passing**
- **Target-specific Aggregation**
- **Skip Connection + LayerNorm**

关键公式可放在模块旁：

```text
Attention: Q_target · R_att · K_source
Message: R_msg · V_source
Aggregate: softmax over incoming edges
```

实际代码中的参数：

| Component | 代码含义 |
| --- | --- |
| `k_linears[t]` | source-type-specific key projection |
| `q_linears[t]` | target-type-specific query projection |
| `v_linears[t]` | source-type-specific value projection |
| `relation_att[r]` | relation-specific attention transformation |
| `relation_msg[r]` | relation-specific message transformation |
| `relation_pri[r]` | relation-specific attention prior |
| `a_linears[t]` | target-type-specific output transformation |
| `skip[t]` | learnable target-type skip gate |

### 3.3 Optional temporal encoding

代码中有 `RelTemporalEncoding`，但当前图构建脚本把 `edge_time` 设为 0 或 `YEAR=2026` 的静态图语义。因此图中建议将 temporal encoding 画成灰色 optional branch：

```text
edge_time / relation time encoding
optional for dynamic graph extensions
```

不要把它画成当前任务的主要贡献。

## Stage 4: Station-level Task Head

HGT encoder 输出所有节点的 hidden representation：

```text
H = HGT(x, node_type, edge_index, edge_type, edge_time)
H shape: [num_nodes, hidden_dim]
```

当前监督任务只选择 station 节点：

```text
Z_station = H[node_offsets["station"] : node_offsets["station"] + num_stations]
```

然后进入分类或回归任务头：

### 分类头

适合 conflict class prediction：

```text
Linear(hidden_dim -> num_classes)
Softmax / LogSoftmax
CrossEntropyLoss or NLLLoss
```

图中标签：

```text
Station Conflict Classifier
daily-dominant / tourism-dominant / mixed-pressure / transitional-balance
```

### 回归头

适合 continuous conflict intensity score：

```text
MLP(hidden_dim -> 1)
Sigmoid or raw score
MSE / MAE / ranking loss
```

图中标签：

```text
Conflict Intensity Regressor
score in [0, 1]
```

建议主图画分类头，同时在旁边小分支标注 optional regression head。

## Stage 5: Optimization and Evaluation

训练闭环建议画成标准深度学习 training loop：

```text
Prediction logits / scores
        |
        v
Supervised loss on labeled station nodes
        |
        v
Backpropagation
        |
        v
AdamW optimizer + LR scheduler + gradient clipping
        |
        v
Update HGT encoder and task head
```

可参考 `HGT/OAG/train_paper_field.py` 的通用训练范式：

- optimizer: AdamW / Adam / SGD / Adagrad
- scheduler: CosineAnnealingLR
- gradient clipping
- train / validation / test split
- best model selected by validation metric

### 推荐评估指标

根据任务形式选择：

| 任务形式 | 指标 |
| --- | --- |
| Binary classification | Accuracy, Precision, Recall, F1, ROC-AUC |
| 3-class / multi-class classification | Macro-F1, Weighted-F1, Confusion Matrix |
| Ordinal conflict level | Quadratic weighted kappa, MAE |
| Regression score | MAE, RMSE, Spearman correlation |
| Spatial product quality | station-level map inspection, high-risk cluster review |

图中建议把 evaluation block 画成仪表盘：

- `Loss curve`
- `Macro-F1 / AUC`
- `Confusion matrix`
- `Validation-based checkpoint selection`

## Stage 6: 模型产出与空间分析产物

右侧输出层建议分成 4 类：

### 6.1 Model artifacts

```text
best_hgt_model.pt
encoder_state_dict.pt
classifier_head.pt
training_config.json
metrics.json
```

### 6.2 Station-level predictions

```text
station_predictions.csv
station_id, station_name, conflict_score, predicted_class, confidence
```

### 6.3 Spatial visualization products

```text
station_conflict_map.geojson
station_conflict_heatmap.png
high_risk_station_rank.csv
district_summary.csv
```

图中应把这些产物画成城市地图上的彩色站点：

- low conflict: blue / green
- medium conflict: yellow
- high conflict: red
- mixed-pressure class: red-orange

### 6.4 Explanation products

HGTConv 会保存 attention 权重 `self.att`，因此可导出基于 typed relation attention 的解释：

```text
attention_by_relation.csv
top_influential_neighbors.csv
relation_importance_barplot.png
local_station_explanation.html
```

图中建议显示一个 station-centered explanation panel：

- 中央站点
- 周边 POI、道路、历史建筑、风貌区
- 边粗细表示 attention / influence
- 右侧条形图显示 relation importance：`near_poi_tour`, `near_poi_daily`, `near_road`, `near_building`, `inside_conservation`, `inside_admin`

## 推荐图中英文标签

建议保留英文标签，便于生成论文风格图：

- HGT Graph Artifacts
- Tensor Inputs
- Station Labels
- Train / Val / Test Split
- Full-graph Training
- Optional Subgraph Sampling
- Type-specific Feature Adapters
- Stacked HGTConv Layers
- Heterogeneous Mutual Attention
- Relation-specific Message Passing
- Target-specific Aggregation
- Station Embedding Selection
- Conflict Classification Head
- Supervised Loss on Station Nodes
- AdamW Optimization
- Validation Metrics
- Best Checkpoint
- Station Conflict Scores
- Spatial Risk Map
- Attention-based Explanation

## 推荐图中公式与短注释

```text
x_v in R^16
h_v in R^d
z_station = HGT(G)[station nodes]
logits = Linear(z_station)
L = CrossEntropy(logits, y_station)
Q_t, K_s, V_s by node type
R_att, R_msg by relation type
attention over typed incoming edges
```

## 可直接给 gpt-image2 的 Prompt

```text
Create a clean CVPR paper-style technical architecture diagram on a white background. The figure should show an HGT model training and output pipeline for station-level daily-tourism mixed-conflict recognition in Shanghai metro station areas.

Use a horizontal left-to-right layout with six stages:

1. HGT Graph Artifacts:
draw input blocks from data/hgt_graph:
"hgt_tensors.pt: x, node_type, edge_index, edge_type, edge_time",
"hetero_data.pt: PyG HeteroData",
"nodes/*.csv + edges.csv",
"manifest.json".
Add small summary text:
"286,233 nodes, 3,417,482 directed typed edges, 6 node types, 16-D features".

2. Training Data Assembly:
show station nodes being selected as supervised target nodes.
Add "station labels" and "train / validation / test split".
Show the task labels as:
"conflict / non-conflict" and "low / medium / high" and "daily-dominant / tourism-dominant / mixed-pressure / transitional-balance".
Main path: "full-graph training".
Add a small dashed optional branch labeled "subgraph sampling".

3. HGT Encoder:
draw six type-specific feature adapter blocks:
station, building, road, POI, conservation_area, admin_area.
Each adapter maps "x_v in R^16" to "h_v in R^d".
Then draw 3 stacked HGTConv blocks.
Inside each HGTConv block show:
"type-specific Q/K/V projections",
"relation-specific attention R_att",
"relation-specific message R_msg",
"multi-head typed-edge attention",
"target-specific aggregation + skip + layer norm".
Represent typed edges with colored relation arrows: near_poi, near_road, near_building, inside_conservation, inside_admin.
Add a grey optional note: "edge_time / temporal encoding for dynamic extension".

4. Station-level Task Head:
show all node embeddings H from the HGT encoder.
Highlight station embeddings:
"Z_station = H[station nodes]".
Feed them into a "Conflict Classification Head".
Show logits and class probabilities.
Add a small optional side branch "Conflict Intensity Regressor: score in [0,1]".

5. Optimization and Evaluation:
draw a training loop:
"supervised loss on labeled station nodes" -> "backpropagation" -> "AdamW optimizer + LR scheduler + gradient clipping" -> "update HGT encoder and task head".
Add validation dashboard icons:
"loss curve", "Macro-F1 / AUC", "confusion matrix", "best checkpoint".

6. Model Outputs and Spatial Products:
draw four output groups:
"best_hgt_model.pt + training_config.json + metrics.json",
"station_predictions.csv: station_id, conflict_score, predicted_class, confidence",
"station_conflict_map.geojson + heatmap.png + high_risk_station_rank.csv",
"attention-based explanation: relation importance + influential neighbors".
Visualize the spatial output as an abstract Shanghai metro map with stations colored blue/green/yellow/red by predicted conflict level.
Visualize the explanation output as a local station-centered graph where thicker edges indicate higher attention.

Visual style:
minimal CVPR / NeurIPS technical diagram, crisp vector-like graphics, white background, thin gray outlines, muted colors, clear sans-serif typography, no photorealistic map, no decorative background, no clutter, no cartoon style. Use blue for stations, orange for POIs, gray for roads, red for high conflict, purple for heritage/conservation areas, green for administrative areas. Make the diagram suitable for a two-column computer vision paper figure.
```

## 简化版 Caption

**Figure.** HGT training and output pipeline for station-level daily-tourism conflict recognition. Preprocessed heterogeneous graph tensors are assembled with station labels, encoded by type-specific feature adapters and stacked HGTConv layers, and supervised through a station-level classification head. Training optimizes loss on labeled station nodes and selects checkpoints by validation metrics. Final outputs include trained HGT weights, station conflict predictions, spatial risk maps, ranked high-risk stations, and attention-based local explanations.

## 可用于论文方法节的技术摘要

The model consumes the preprocessed heterogeneous graph as flat HGT tensors, including node features, node type IDs, typed edge indices, edge type IDs, and edge-time placeholders. Each node type is first mapped into a shared hidden space through a type-specific linear adapter. Stacked HGTConv layers then perform meta-relation-aware message passing, where query, key, and value projections depend on node types, and attention/message transformations depend on relation types. Station node embeddings are selected from the final node representation matrix and passed to a supervised task head for daily-tourism conflict classification or intensity regression. The training loop computes loss only on labeled station nodes while gradients update the shared HGT encoder and station-level head. Outputs include model checkpoints, station-level conflict scores and classes, spatial visualization layers, and relation-attention explanations.

