# 202410261946
# Description
# 在完整 S3D 数据集上训练时，部分样本不能随机采样近邻和远邻
############################################################
# Logs
# scene_02785 141
# gt_pos 88., 494.
# scene_bound [128, 944], [128, 734]
############################################################
# Conclusion
# 在地图上可视化场景的位置，发现地图包括了两个不同的室内环境
# 原因是建图过程中，有多个 outwall，但只选择了其中一个 outwall 作为建图的边界
#############################################################

import os
import cv2
import numpy as np
import pandas as pd

import sys
sys.path.append(".")

from s3d import *
from scripts.cad_mapping_v1.coord_conv import position_to_pixel


scene_index = "scene_02785"
sample_index = "141"
scene_pth = os.path.join(s3d_data_pth, scene_index)
sample_pth = os.path.join(s3d_data_pth, scene_index, "2D_rendering", sample_index, "panorama")

# 读取场景参数
scene_annos = os.path.join(scene_pth, "boundary.csv")
scene_annos = pd.read_csv(scene_annos)
# 读取样本位置坐标
sample_pos = os.path.join(sample_pth, "camera_xyz.txt")
sample_pos = np.genfromtxt(sample_pos, delimiter=" ")[:2]
# 坐标转换
sample_pos = position_to_pixel(sample_pos, resolution, scene_annos)

# 读取场景地图
scene_map = os.path.join(s3d_map_pth, scene_index, "map.png")
scene_map = cv2.imread(scene_map)
# 在地图上可视化样本位置
scene_map = cv2.circle(scene_map, sample_pos, 3, (0,0,0), -1)
# 可视化地图
cv2.imshow("Sample on map", scene_map)

# 读取样本 BEV
sample_bev = os.path.join(s3d_bev_pth, scene_index, "2D_rendering", sample_index, "panorama/full/bev.png")
sample_bev = cv2.imread(sample_bev)
# 可视化样本 BEV
cv2.imshow("Sample bev", sample_bev)

cv2.waitKey(0)
cv2.destroyAllWindows()
