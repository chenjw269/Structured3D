# 20241025
# 验证从地图上采样的位置假设

# 202410251420
# 202410251510

# 50 min，合格

import os
import cv2
import copy
import platform
import numpy as np
import pandas as pd

import sys
sys.path.append(".")

from scripts.cad_mapping_v1.coord_conv import position_to_pixel
from scripts.hypothesis.v1.pano.position_hypothesis import generate_scene_hypothesis
from scripts.utils.visualize_occ import visualize_occ # 可视化占用网格
from assets.semantic2label import LABEL_TO_COLOR
from scripts.utils.patches_matching import patches_matching_positive # 匹配最近邻地图块


# 获取系统类型
system_type = platform.system()
# 本地路径和服务器路径
if system_type == 'Windows':
    data_pth = "e:/datasets/Structure3D/Structured3D"
    map_pth = "e:/datasets/Structure3D_map/Structured3D"
    scene_annos_loss = "logs/scene_annos.txt"
    scene_line_err = "logs/scene_line_err.txt"
    output_pth = "e:/datasets/Structure3D_map/Structured3D"
else:
    data_pth = "/data1/chenjiwei/S3D/zip/Structured3D"
    map_pth = "/data1/chenjiwei/S3D/Structure3D_map/Structured3D"
    scene_annos_loss = "../../logs/scene_annos.txt"
    scene_line_err = "../../logs/scene_line_err.txt"
    output_pth = "/data1/chenjiwei/S3D/Structure3D_map/Structured3D"


if __name__ == "__main__":

    scene_index = "scene_00000"
    sample_index = "485142"

    resolution = 25  # 2.5 cm, 0.025 m / pixel

    # 读取场景地图
    scene_map = os.path.join(map_pth, scene_index, "map.npy")
    scene_map = np.load(scene_map)
    # 场景边界数据
    scene_info = os.path.join(data_pth, scene_index, "boundary.csv")
    scene_info = pd.read_csv(scene_info)

    #############################
    # 样本位置
    #############################
    # 读取样本位置
    sample_pos = os.path.join(
        data_pth , scene_index, "2D_rendering", sample_index, "panorama/camera_xyz.txt"
    )
    sample_pos = np.genfromtxt(sample_pos, delimiter=" ")[:2]
    # 坐标变换
    sample_pos = position_to_pixel(sample_pos, resolution, scene_info)
    # 在地图上可视化样本位置
    tem_scene_map = copy.copy(scene_map)
    sample_on_map = cv2.circle(tem_scene_map, sample_pos, 2, 1, -1)
    sample_on_map = visualize_occ(sample_on_map, LABEL_TO_COLOR)
    cv2.imshow("Sample on map", sample_on_map)

    #############################
    # 采样位置假设及匹配
    #############################
    # 从场景中采样位置假设
    scene_pth = os.path.join(data_pth, scene_index)
    hypothesis_loc = generate_scene_hypothesis(scene_info)
    # 查找距离最近的位置假设
    nearest_indices = patches_matching_positive(sample_pos, hypothesis_loc, 1)
    nearest_pos = hypothesis_loc[nearest_indices][0]
    # 从地图上获取最近的观测假设
    nearest_obs = scene_map[
        nearest_pos[1] - 128: nearest_pos[1] + 128,
        nearest_pos[0] - 128: nearest_pos[0] + 128
    ]
    # 可视化最近的观测假设
    nearest_obs = visualize_occ(nearest_obs, LABEL_TO_COLOR)
    cv2.imshow("Nearest obs hypothesis", nearest_obs)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
