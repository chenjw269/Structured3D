# 20241025
# 测试：在地图上采样正负样例
import os
import cv2
import math
import random
import platform
import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append(".")

from scripts.cad_mapping_v1.coord_conv import position_to_pixel # 坐标转换
from scripts.metric_learning.v1.neighbor_sample import * # 采样正负样例位置
from scripts.utils.visualize_occ import visualize_occ # 可视化占用网格
from assets.semantic2label import LABEL_TO_COLOR


# 获取系统类型
system_type = platform.system()
# 本地路径和服务器路径
if system_type == 'Windows':
    data_pth = "e:/datasets/Structure3D/Structured3D"
    bev_pth = "e:/datasets/Structure3D_bev/Structured3D"
    map_pth = "e:/datasets/Structure3D_map/Structured3D"
    scene_annos_loss = "logs/scene_annos.txt"
    scene_line_err = "logs/scene_line_err.txt"
    output_pth = "e:/datasets/Structure3D_map/Structured3D"
else:
    data_pth = "/data1/chenjiwei/S3D/zip/Structured3D"
    bev_pth = "/data1/chenjiwei/S3D/Structure3D_bev/Structured3D"
    map_pth = "/data1/chenjiwei/S3D/Structure3D_map/Structured3D"
    scene_annos_loss = "../../logs/scene_annos.txt"
    scene_line_err = "../../logs/scene_line_err.txt"
    output_pth = "/data1/chenjiwei/S3D/Structure3D_map/Structured3D"

resolution = 25 # 2.5 cm, 0.025 m / pixel


if __name__ == "__main__":
    
    # # Structured3D 包括 3500 个场景
    # scene_index_list = [f"scene_{num:05}" for num in range(3500)]

    # TODO:随机场景，随机样本
    # obs_dir = os.path.join(data_pth, scene_index, "2D_rendering")
    # obs_list = os.listdir(obs_dir)
    
    scene_index = "scene_00000"
    obs_item_index = "485142"
    
    bev_dir = os.path.join(bev_pth, scene_index, "2D_rendering")
    obs_dir = os.path.join(data_pth, scene_index, "2D_rendering")
    ###############################
    # 场景数据
    ###############################
    # 当前场景地图
    scene_map = os.path.join(map_pth, scene_index, "map.npy")
    scene_map = np.load(scene_map)
    # 当前场景标注
    scene_annos = os.path.join(data_pth, scene_index, "boundary.csv")
    scene_annos_df = pd.read_csv(scene_annos)
    # 场景坐标范围
    scene_bound = np.array([
        [128, int(scene_annos_df['size_x'].item() / resolution) - 128],
        [128, int(scene_annos_df['size_y'].item() / resolution) - 128]
    ])

    ###############################
    # 样本数据
    ###############################
    # 局部地图
    obs_item_bev = os.path.join(bev_dir, obs_item_index, "panorama/full/bev.png")
    obs_item_bev = cv2.imread(obs_item_bev)
    cv2.imshow("Local map", obs_item_bev)

    # 真实位置
    obs_item_pos = os.path.join(obs_dir, obs_item_index, "panorama/camera_xyz.txt")
    obs_item_pos = np.genfromtxt(obs_item_pos, delimiter=" ")[:2]
    # 坐标转换
    obs_item_pos = position_to_pixel(obs_item_pos, resolution, scene_annos_df)

    ################################
    # 采样正负样例位置
    ################################
    # 随机正样例位置
    positive_radius = 60 # 60 * 0.025 = 1.5
    positive_nums = 5
    positive_sample_pos = generate_neighbor_within(
        obs_item_pos, positive_nums, positive_radius, scene_bound
    )
    
    # 随机负样例位置
    negative_radius_1 = 140 # 140 * 0.025 = 3.5
    negative_radius_2 = 280 # 280 * 0.025 = 7
    negative_nums = 5
    negative_sample_pos = generate_neighbor_between(
        obs_item_pos, negative_nums,
        radius_1=negative_radius_1,
        radius_2=negative_radius_2,
        p_bound=scene_bound
    )

    ################################
    # 可视化正负样例
    ################################
    # 可视化正样例
    for i in range(positive_nums):
        positive_item_pos = positive_sample_pos[i]
        positive_item_lm = scene_map[
            positive_item_pos[1]-128 : positive_item_pos[1]+128,
            positive_item_pos[0]-128 : positive_item_pos[0]+128
        ]
        positive_item_lm_img = visualize_occ(positive_item_lm, LABEL_TO_COLOR)
        cv2.imshow(f"Positive sample {i}", positive_item_lm_img)

    # 可视化负样例
    for i in range(negative_nums):
        negative_item_pos = negative_sample_pos[i]
        negative_item_lm = scene_map[
            negative_item_pos[1]-128 : negative_item_pos[1]+128,
            negative_item_pos[0]-128 : negative_item_pos[0]+128
        ]
        negative_item_lm_img = visualize_occ(negative_item_lm, LABEL_TO_COLOR)
        cv2.imshow(f"Negative sample {i}", negative_item_lm_img)
        
    cv2.waitKey(0)
    cv2.destroyAllWindows()
