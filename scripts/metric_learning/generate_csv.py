import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append(".")

from scripts.utils.coord_conv import position_to_pixel


# 标注数据缺失的场景
# with open("../../logs/scene_annos.txt", encoding="utf-8") as f: # remote
with open("logs/scene_annos.txt", encoding="utf-8") as f: # local
    scene_invalid = f.readlines()
for index, item in enumerate(scene_invalid):
    scene_invalid[index] = item.replace("\n", "")

# 观测数据缺失的样本
# with open("../../logs/scene_observation.txt", encoding="utf-8") as f: # remote
with open("logs/scene_observation.txt", encoding="utf-8") as f:  # local
    obs_invalid = f.readlines()
for index, item in enumerate(obs_invalid):
    obs_invalid[index] = item.replace("\n", "")


if __name__ == "__main__":

    # remote data path
    # data_pth = "e:/datasets/Structure3D/Structured3D" # remote
    # bev_pth = "e:/datasets/Structure3D_bev/Structured3D" # remote
    # map_pth = "e:/datasets/Structure3D_map/Structured3D" # remote
    # output_pth = "e:/datasets/Structure3D_csv/Structured3D" # remote
    # local data path
    data_pth = "e:/datasets/Structure3D/Structured3D" # local
    bev_pth = "e:/datasets/Structure3D_bev/Structured3D" # local
    map_pth = "e:/datasets/Structure3D_map/Structured3D" # local
    output_pth = "e:/datasets/Structure3D_csv/Structured3D" # local

    resolution = 25
    map_occ_size = (1600, 1600)

    # 统计前 100 个场景
    scene_index_list = [f"scene_{num:05}" for num in range(100)]
    for scene_index in tqdm(scene_index_list):
        # 缺少标注的场景作废
        if scene_index in scene_invalid:
            scene_index_list.remove(scene_index)

    # 遍历场景
    for scene_index in tqdm(scene_index_list):

        sample_pos_list = []
        sample_obs_list = []
        sample_map_list = []

        # 当前场景的地图
        scene_map = os.path.join(map_pth, scene_index, "map.npy")
        # 当前场景的边界
        scene_boundary_pth = os.path.join(data_pth, scene_index, "boundary.csv")
        scene_boundary = pd.read_csv(scene_boundary_pth)
        # 场景坐标中心点
        scene_center_point = np.array([scene_boundary['x_center'].item(), scene_boundary['y_center'].item()])
        # 当前场景下的观测数据
        obs_dir = os.path.join(data_pth, scene_index, "2D_rendering")
        obs_list = os.listdir(obs_dir)

        # 遍历观测数据
        for obs_item in obs_list:
            
            # 缺少观测值的样本作废
            if f"{scene_index},{obs_item}" in obs_invalid:
                tqdm.write(f"Jmp obs loss {scene_index} {obs_item}")
                continue
            
            # 真实观测位置
            obs_item_pos = os.path.join(data_pth, scene_index, "2D_rendering", obs_item, "panorama/camera_xyz.txt")
            obs_item_pos = np.genfromtxt(obs_item_pos, delimiter=" ")[:2]
            # 坐标转换
            obs_item_pos = obs_item_pos - scene_center_point
            obs_item_pos = position_to_pixel(obs_item_pos, resolution, map_occ_size).tolist()
            sample_pos_list.append(obs_item_pos)
            
            # 观测数据 BEV
            obs_item_bev = os.path.join(bev_pth, scene_index, "2D_rendering", obs_item, "panorama/full/bev.npy")
            sample_obs_list.append(obs_item_bev)

            # 场景地图
            sample_map_list.append(scene_map)
        
        # 保存到 csv 文件
        scene_output_pth = os.path.join(output_pth, scene_index, f"metric_learning/{scene_index}.csv")
        os.makedirs(os.path.join(output_pth, scene_index, "metric_learning"), exist_ok=True)
        scene_df = pd.DataFrame(
            {
                "gt pos": sample_pos_list,
                "local map": sample_obs_list,
                "global map": sample_map_list
            }
        )
        scene_df.to_csv(scene_output_pth, index=False)