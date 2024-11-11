# Structured3D

## Observation Process

## BEV Observation

## CAD Mapping

### v0

所有场景地图使用相同的尺寸

### v1

根据地图大小选择不同的尺寸

#### scene_boundary.py

统计不同场景的坐标边界
1. 读取场景的外墙 outwall 标注和包围盒 bbox 标注
2. 统计坐标的极值、中心和地图尺寸（坐标的跨度）

#### coord_conv.py

将真实世界坐标转换到地图
1. 地图的原点（0，0）为坐标中心
2. 坐标除以分辨率，从 mm 转换为 pixel
3. 根据地图尺寸，平移像素坐标
4. 上下翻转 y 轴

#### cad_mapping.py

## Metric Learning

### neighbor_sample.py
- 采样近邻 near neighbor，作为正样例 positive sample
- 采样远邻 far neighbor，作为负样例 negative sample

### v0 

每个样本独立处理。数据目录中 csv 每一行存储 bev 路径、cad 地图路径、位姿真值


### v1

同场景样本批量处理。数据目录中 csv 每一行存储 bev 路径列表，cad 地图路径，位姿真值列表

## Hypothesis


### Perspective Observation

视野受限情况下的视觉观测数据

在已有全景图和周围 360 度 BEV 的基础上，模拟受限视野下的视觉观测

## Utils

针对样本数据的处理

- 坐标转换

### Virtual Observation

针对场景数据的处理

- 读取地图信息
- 采样位姿假设
- 虚拟观测
- 匹配最相近的位姿
