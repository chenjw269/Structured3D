- [ ] __下载数据__

- [x] __错误样本统计__
    - [x] 统计观测数据样本缺失的场景，记录为表 download/check_scene_obs.py
    - [x] 统计标注数据缺失的场景，记录为表 download/check_scene_annos.py

- [ ] __CAD 地图__
    作者提供了每个场景的语义分割观测和实例地图，但没有提供实例和语义之间的对应关系。
    - [x] 根据实例对应的语义类别，获取真实的语义物体地图 preprocessing/instance_semantic.py
    - [x] 统计场景内物体的坐标范围 preprocessing/boundary_statics.py
    - [ ] 从标注文件创建物体地图 preprocessing/scene_map_img.py

- [ ] __BEV 观测__
每个场景采集的数据包括深度图、语义分割图和 RGB 图。通过深度图计算 3d 点云，将不同语义类别的点云投影到 bev，创建 2d bev 占用网格
    - [ ] 全景（panorama）图 bev  bev_observation/generate_bev.py
    - [ ] 单视角（perspective）视图 bev

- [ ] __度量学习 / 正负样例__
    - [ ] 从地图上随机采样近邻位置和远邻位置
    - [ ] 获取近 / 远邻位置的局部地图

- [ ] __匹配定位 / 均匀采样__
    - [ ] 

- [ ] __偏移量预测数据集__
    - [ ] 从地图上随机采样近邻，并计算偏移量
    - [ ] 获取近邻位置的局部地图
