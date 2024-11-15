## 完整的数据处理过程

### 检查文件完整性

检查场景标注文件（包围盒，地图边界）完整性

检查每个样本文件（观测数据，位姿）完整性

### 创建 cad 地图

统计每个场景地图的尺寸边界

创建 cad 地图

### 单视角 bev

### 全景图 bev

## 运行所需的最小操作

### 检查 bev 文件完整性

scripts/bev_pesp/check_file.py

### 数据集目录 csv

metric_learning/v1/pesp/gen_csv.py
metric_learning/v1/pesp/train_test_split.py

### 数据集 dataloader 遍历

metric_learning/v1/test/test_s3d_dataset.py
