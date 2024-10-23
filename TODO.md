- [x] __错误样本统计__
    - [x] 统计观测数据样本缺失的场景，记录为表 download/check_scene_obs.py
    - [x] 统计标注数据缺失的场景，记录为表 download/check_scene_annos.py

- [ ] __场景物体地图__
    作者提供了每个场景的语义分割观测和实例地图，但没有提供实例和语义之间的对应关系。
    - [x] 根据实例对应的语义类别，获取真实的语义物体地图 preprocessing/instance_semantic.py
    - [x] 统计场景内物体的坐标范围 preprocessing/boundary_statics.py
    - [ ] 从标注文件创建物体地图 preprocessing/scene_map_img.py

- [ ] __BEV 观测__
    - [ ] 每个场景采集的数据包括深度图、语义分割图和 RGB 图，创建 BEV 占用网格 bev_observation/generate_bev.py

- [ ] __度量学习数据集__
    - [ ] 从地图上随机采样近邻位置和远邻位置
    - [ ] 获取近 / 远邻位置的局部地图

- [ ] __偏移量预测数据集__
    - [ ] 从地图上随机采样近邻，并计算偏移量
    - [ ] 获取近邻位置的局部地图
