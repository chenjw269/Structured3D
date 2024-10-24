# 20241019
# 在地图上采样观测假设
import os
import cv2
import numpy as np
from tqdm import tqdm

import sys
sys.path.append(".")

from scripts.utils.coord_conv import position_to_pixel # 坐标转换
from scripts.utils.generate_neighbor import *
from scripts.utils.visualize_occ import visualize_occ # 可视化占用网格
from assets.semantic2label import LABEL_TO_COLOR


if __name__ == "__main__":
    
    data_pth = "d:/datasets/Structure3D/Structured3D"
    bev_pth = "d:/datasets/Structure3D_bev/Structured3D"
    map_pth = "d:/datasets/Structure3D_map/Structured3D"
    
    # # Structured3D 包括 3500 个场景
    # scene_index_list = [f"scene_{num:05}" for num in range(3500)]
    # 统计前 100 个场景
    scene_index_list = [f"scene_{num:05}" for num in range(100)]

    resolution = 25 # 2.5 cm, 0.025 m / pixel
    map_occ_size = (1600, 1200) # x 轴范围为 (-20m, 20m) y 轴范围为 (-15m, 15m)
    coord_bound = ((0, 1600), (0, 1200)) # 坐标边界

    # 遍历场景
    for scene_index in tqdm(scene_index_list):

        # 遍历场景
        # 当前场景的地图
        scene_map = os.path.join(map_pth, scene_index, "map.npy")
        scene_map = np.load(scene_map)

        # 遍历观测数据
        for obs_item in obs_list:

            # 局部地图路径
            obs_item_bev = os.path.join(bev_pth, scene_index, "2D_rendering", obs_item, "panorama/full/bev.png")
            obs_item_bev = cv2.imread(obs_item_bev)
            
            # 真实观测位置
            obs_item_pos = os.path.join(data_pth, scene_index, "2D_rendering", obs_item, "panorama/camera_xyz.txt")
            obs_item_pos = np.genfromtxt(obs_item_pos, delimiter=" ")[:2]
            # 坐标转换
            obs_item_pos = position_to_pixel(obs_item_pos, resolution, map_occ_size)

            # 随机正样例位置
            positive_radius = 60 # 60 * 0.025 = 1.5
            positive_nums = 5
            positive_sample_pos = generate_neighbor_within(
                obs_item_pos, positive_nums, positive_radius, coord_bound
            )
            
            # 随机负样例位置
            negative_radius_1 = 100 # 100 * 0.025 = 2.5
            negative_radius_2 = 200 # 200 * 0.025 = 5
            negative_nums = 5
            negative_sample_pos = generate_neighbor_between(
                obs_item_pos, negative_nums,
                radius_1=negative_radius_1,
                radius_2=negative_radius_2,
                p_bound=coord_bound
            )
            
            # 可视化正样例
            for i in range(positive_nums):
                positive_item_pos = positive_sample_pos[i]
                positive_item_lm = scene_map[
                    positive_item_pos[0]-128 : positive_item_pos[0]+128,
                    positive_item_pos[1]-128 : positive_item_pos[1]+128
                ]
                positive_item_lm_img = visualize_occ(positive_item_lm, LABEL_TO_COLOR)
                cv2.imshow(f"Positive sample {i}", positive_item_lm_img)

            # 可视化负样例
            for i in range(negative_nums):
                negative_item_pos = negative_sample_pos[i]
                negative_item_lm = scene_map[
                    negative_item_pos[0]-128 : negative_item_pos[0]+128,
                    negative_item_pos[1]-128 : negative_item_pos[1]+128
                ]
                negative_item_lm_img = visualize_occ(negative_item_lm, LABEL_TO_COLOR)
                cv2.imshow(f"Negative sample {i}", negative_item_lm_img)
                
            cv2.waitKey(0)
            cv2.destroyAllWindows()
