# 20241023
# 测试：从地图上随机采样近邻和远邻

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append(".")

from scripts.utils.coord_conv import position_to_pixel # 坐标转换，从真实世界转换到地图平面
from scripts.utils.generate_neighbor import * # 随机采样，随机近邻和远邻
from scripts.utils.visualize_occ import visualize_occ # 可视化占用网格
from assets.semantic2label import LABEL_TO_COLOR


if __name__ == "__main__":

    data_pth = "e:/datasets/Structure3D/Structured3D"
    bev_pth = "e:/datasets/Structure3D_bev/Structured3D"
    map_pth = "e:/datasets/Structure3D_map/Structured3D"

    # Structured3D 包括 3500 个场景
    scene_index_list = [f"scene_{num:05}" for num in range(1000)]

    resolution = 25 # 2.5 cm, 0.025 m / pixel
    map_occ_size = (1600, 1600) # x 轴范围为 (-20m, 20m) y 轴范围为 (-20m, 20m)
    coord_bound = ((0, 1600), (0, 1600)) # 坐标边界

    # 遍历场景
    for scene_index in tqdm(scene_index_list):
        
        # 读取场景地图
        scene_map = os.path.join(map_pth, scene_index, "map.png")
        scene_map = cv2.imread(scene_map)
        # 读取场景坐标边界
        scene_boundary = os.path.join(data_pth, scene_index, "boundary.csv")
        scene_boundary = pd.read_csv(scene_boundary)
        # 场景坐标中心点
        scene_center_point = np.array([scene_boundary['x_center'].item(), scene_boundary['y_center'].item()])
        
        scene_data_pth = os.path.join(data_pth, scene_index, "2D_rendering")

        # 遍历样本
        scene_sample_list = os.listdir(scene_data_pth)
        for sample_item in scene_sample_list:

            # 读取样本位置
            sample_pos = os.path.join(scene_data_pth, sample_item, "panorama/camera_xyz.txt")
            sample_pos = np.genfromtxt(sample_pos, delimiter=" ")[:2]
            # 坐标转换
            sample_pos_norm = sample_pos - scene_center_point
            sample_pos_pixel = position_to_pixel(sample_pos_norm, resolution, map_occ_size)
            # 随机正样例位置
            positive_radius = 40 # 40 * 0.025 = 1
            positive_nums = 5
            positive_sample_pos = generate_neighbor_within(
                sample_pos_pixel, positive_nums,
                positive_radius,
                coord_bound
            )
            # 随机负样例位置
            negative_radius_1 = 100 # 100 * 0.025 = 2.5
            negative_radius_2 = 200 # 200 * 0.025 = 5
            negative_nums = 5
            negative_sample_pos = generate_neighbor_between(
                sample_pos_pixel, negative_nums,
                radius_1=negative_radius_1,
                radius_2=negative_radius_2,
                p_bound=coord_bound
            )

            # # 在地图上可视化正样例
            # for i in positive_sample_pos:
            #     scene_map = cv2.circle(scene_map, i, 1, 
            #         color=(0,0,255),
            #         thickness=-1)
            # # 在地图上可视化负样例
            # for i in negative_sample_pos:
            #     scene_map = cv2.circle(scene_map, i, 1, 
            #         color=(255,0,0),
            #         thickness=-1)
            # cv2.imshow("Sample on map", scene_map)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # 读取周围 BEV
            sample_bev = os.path.join(bev_pth, scene_index, "2D_rendering", sample_item, "panorama/full/bev.png")
            sample_bev = cv2.imread(sample_bev)
            # cv2.imshow("Local BEV", sample_bev)
            cv2.imwrite("logs/bev.png", sample_bev)

            # 可视化正样例
            for i in range(positive_nums):
                positive_item_pos = positive_sample_pos[i]
                positive_item_lm = scene_map[
                    positive_item_pos[0]-128 : positive_item_pos[0]+128,
                    positive_item_pos[1]-128 : positive_item_pos[1]+128
                ]
                # positive_item_lm_img = visualize_occ(positive_item_lm, LABEL_TO_COLOR)
                # cv2.imshow(f"Positive sample {i}", positive_item_lm)
                cv2.imwrite(f"logs/positive/{i}.png", positive_item_lm)
            # 可视化负样例
            for i in range(negative_nums):
                negative_item_pos = negative_sample_pos[i]
                negative_item_lm = scene_map[
                    negative_item_pos[0]-128 : negative_item_pos[0]+128,
                    negative_item_pos[1]-128 : negative_item_pos[1]+128
                ]
                # negative_item_lm_img = visualize_occ(negative_item_lm, LABEL_TO_COLOR)
                # cv2.imshow(f"Negative sample {i}", negative_item_lm)
                cv2.imwrite(f"logs/negative/{i}.png", negative_item_lm)
                
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            exit()
