# 20241019
# 在地图上采样观测假设
import os
import cv2
import numpy as np
from tqdm import tqdm

import sys
sys.path.append(".")

from scripts.utils.generate_neighbor import *


if __name__ == "__main__":
    
    data_pth = "e:/datasets/Structure3D/Structured3D"
    bev_pth = "e:/datasets/Structure3D_bev/Structured3D"
    map_pth = "e:/datasets/Structure3D_map/Structured3D"
    
    # # Structured3D 包括 3500 个场景
    # scene_index_list = [f"scene_{num:05}" for num in range(3500)]
    # 统计前 100 个场景
    scene_index_list = [f"scene_{num:05}" for num in range(100)]

    # 遍历场景
    for scene_index in tqdm(scene_index_list):

        # 当前场景下的观测数据
        obs_dir = os.path.join(data_pth, scene_index, "2D_rendering")
        obs_list = os.listdir(obs_dir)

        # 当前场景的地图
        scene_map = os.path.join(map_pth, scene_index, "map.npy")
        scene_map = np.load(scene_map)

        # 遍历观测数据
        for obs_item in obs_list:

            # 局部地图路径
            obs_item_bev = os.path.join(bev_pth, scene_index, "2D_rendering", obs_item, "panorama/full/bev.png")
            obs_item_bev = cv2.imread(obs_item_bev)
            
            # 真实观测位置
            
            # 随机正样例位置
            
            # 随机负样例位置
            
            # 可视化正样例
            # 可视化负样例
            
            # 可视化结果