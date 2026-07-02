# 项目文件下载与随机地铁站评估说明

这份说明给没有编程背景的成员使用。目标是：

1. 下载 `master` 分支里的全部项目文件，包括大模型文件。
2. 随机选择一个上海地铁站。
3. 用最新的 `best_model.pt` 做评估。
4. 下载评估结果。

## 重要提醒

不要用 GitHub 页面上的 `Download ZIP` 下载本项目。模型和图数据使用 Git LFS 保存，ZIP 可能只包含占位文件，不能正常评估。

必须使用下面的 `git clone` 和 `git lfs pull`。

## 第 1 步：准备 Git LFS

如果服务器上已经有 `git lfs`，可以跳过安装。

检查：

```bash
git lfs version
```

如果提示 `git: 'lfs' is not a git command`，先安装：

```bash
conda install -c conda-forge git-lfs -y
```

安装后执行：

```bash
git lfs install
```

## 第 2 步：下载项目

选择一个你想放项目的目录，然后执行：

```bash
git clone https://github.com/wolfgang158/SHhistory.git
cd SHhistory
git lfs pull
```

`git lfs pull` 会下载训练好的模型、图数据等大文件，时间可能比较久。

## 第 3 步：准备 Python 环境

如果服务器已经有 `HGT` 环境，可以直接进入：

```bash
conda activate HGT
```

如果没有 `HGT` 环境，创建一个：

```bash
conda create -n HGT python=3.10 -y
conda activate HGT
python -m pip install -r HGT/requirements.txt
```

## 第 4 步：随机选择一个测试集地铁站并评估

在项目目录 `SHhistory` 里执行：

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/evaluate/evaluate_hgt_station_area.sh --random-station --station-split test
```

说明：

- `--random-station` 表示随机选一个地铁站。
- `--station-split test` 表示只从测试集里随机选，适合检查模型对未参与训练站点的效果。
- `CUDA_VISIBLE_DEVICES=1` 表示使用第 1 张 GPU。如果机器没有 GPU，请联系维护者，不要随便改命令。

如果想固定随机结果，使用：

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/evaluate/evaluate_hgt_station_area.sh --random-station --station-split test --random-seed 42
```

## 第 5 步：查看结果在哪里

命令运行结束后，终端会显示类似：

```text
[eval_hgt ...] output-dir=/data1/fangxuebin/SHhistory/outputs/evaluate/hgt_station_area/runs/20260702-xxxxxx
[eval_hgt ...] station score: ...
```

结果文件就在这个 `output-dir` 文件夹里。

主要看这两个文件：

```text
station_score.json
station_area_predictions.csv
```

其中：

- `station_score.json`：本次随机站点的单站评估结果。
- `station_area_predictions.csv`：所有站点的预测结果表。

## 结果字段怎么读

`station_score.json` 里常见字段含义：

```text
name
```

被评估的地铁站名称。

```text
split
```

数据划分。这里应该是 `test`。

```text
pred_conflict_label
```

模型预测的冲突等级编号：

```text
0 = low，低潜在冲突
1 = medium，中潜在冲突
2 = high，高潜在冲突
```

```text
pred_conflict_level
```

模型预测的冲突等级文字版：`low`、`medium`、`high`。

```text
pred_historic_probability
```

模型预测的“历史性”程度，范围大致是 `0` 到 `1`。越高表示历史性越强。

```text
pred_historic_grade
```

历史性等级：`low`、`medium`、`high`。

```text
pred_daily_probability
```

模型预测的“日常性”程度，范围大致是 `0` 到 `1`。越高表示日常生活属性越强。

```text
pred_daily_grade
```

日常性等级：`low`、`medium`、`high`。

## 第 6 步：下载结果

如果你使用 VS Code 远程连接服务器：

1. 在左侧文件栏打开 `outputs/evaluate/hgt_station_area/runs/`。
2. 找到最新时间的文件夹。
3. 右键下载整个文件夹，或者下载 `station_score.json` 和 `station_area_predictions.csv`。

如果你使用命令行下载，把下面路径换成终端显示的 `output-dir`：

```bash
scp -r fangxuebin@server-pro6000-1:/data1/fangxuebin/SHhistory/outputs/evaluate/hgt_station_area/runs/具体时间文件夹 ./hgt_eval_result
```

## 常见问题

### 1. 下载后模型文件打不开

很可能没有执行：

```bash
git lfs pull
```

重新进入项目目录执行一次即可。

### 2. 提示找不到 `best_model.pt`

说明训练结果没有下载完整，或者仓库里还没有上传训练输出。先执行：

```bash
git lfs pull
find outputs/train/hgt_station_conflict/runs -name best_model.pt
```

如果还是没有结果，请联系维护者。

### 3. 提示 CUDA 或 GPU 错误

先确认正在使用 `HGT` 环境：

```bash
conda activate HGT
```

然后重新运行评估命令。如果仍然报错，把完整报错截图发给维护者。

### 4. 想评估指定地铁站

例如评估“莲花路”：

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/evaluate/evaluate_hgt_station_area.sh --station-name "莲花路"
```

只在测试集里找“莲花路”：

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/evaluate/evaluate_hgt_station_area.sh --station-name "莲花路" --station-split test
```
