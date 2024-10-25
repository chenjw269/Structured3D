# 20241025
# pytorch 度量学习数据集所需的文件路径表

import os
import platform
import numpy as np
import pandas as pd
from tqdm import tqdm
import concurrent.futures

import sys
sys.path.append(".")

# 坐标转换
from scripts.cad_mapping_v1.coord_conv import position_to_pixel


# 获取系统类型
system_type = platform.system()
# 本地路径和服务器路径
if system_type == 'Windows':
    data_pth = "e:/datasets/Structure3D/Structured3D"
    bev_pth = "e:/datasets/Structure3D_bev/Structured3D"
    map_pth = "e:/datasets/Structure3D_map/Structured3D"
    scene_annos_loss = "logs/scene_annos.txt"
    scene_line_err = "logs/scene_line_err.txt"
    scene_obs_err = "logs/scene_observation.txt"
    output_pth = "e:/datasets/Structure3D_csv/Structured3D"
else:
    data_pth = "/data1/chenjiwei/S3D/zip/Structured3D"
    bev_pth = "/data1/chenjiwei/S3D/Structure3D_bev/Structured3D"
    map_pth = "/data1/chenjiwei/S3D/Structure3D_map/Structured3D"
    scene_annos_loss = "../../logs/scene_annos.txt"
    scene_line_err = "../../logs/scene_line_err.txt"
    scene_obs_err = "../../logs/scene_observation.txt"
    output_pth = "/data1/chenjiwei/S3D/Structure3D_csv/Structured3D"

resolution = 25  # 2.5 cm, 0.025 m / pixel

# 标注数据缺失的场景
with open(scene_annos_loss, encoding="utf-8") as f: # remote
    scene_invalid = f.readlines()
for index, item in enumerate(scene_invalid):
    scene_invalid[index] = item.replace("\n", "")
# 边线错误的场景
with open(scene_line_err, encoding="utf-8") as f:
    scene_invalid_append = f.readlines()
for index, item in enumerate(scene_invalid_append):
    scene_invalid_append[index] = item.replace("\n", "")
scene_invalid = scene_invalid + scene_invalid_append

# 观测数据缺失的样本
with open(scene_obs_err, encoding="utf-8") as f: # remote
    obs_invalid = f.readlines()
for index, item in enumerate(obs_invalid):
    obs_invalid[index] = item.replace("\n", "")


def generate_scene_csv(scene_index):
    
    sample_pos_list = []
    sample_obs_list = []
    sample_map_list = []
    annos = []

    bev_dir = os.path.join(bev_pth, scene_index, "2D_rendering")
    obs_dir = os.path.join(data_pth, scene_index, "2D_rendering")
    scene_map = os.path.join(map_pth, scene_index, "map.npy")
    scene_annos = os.path.join(data_pth, scene_index, "boundary.csv")
    scene_annos_df = pd.read_csv(scene_annos)

    # 遍历观测数据
    obs_list = os.listdir(obs_dir)
    for obs_item in obs_list:
    
        # 缺少观测值的样本作废
        if f"{scene_index},{obs_item}" in obs_invalid:
            tqdm.write(f"Jmp obs loss {scene_index} {obs_item}")
            continue

        # 场景标注
        annos.append(scene_annos)

        # 真实位置
        obs_item_pos = os.path.join(obs_dir, obs_item, "panorama/camera_xyz.txt")
        obs_item_pos = np.genfromtxt(obs_item_pos, delimiter=" ")[:2]
        # 坐标变换
        obs_item_pos = position_to_pixel(obs_item_pos, resolution, scene_annos_df).tolist()
        sample_pos_list.append(obs_item_pos)

        # 局部地图
        obs_item_bev = os.path.join(bev_dir, obs_item, "panorama/full/bev.npy")
        sample_obs_list.append(obs_item_bev)

        # 场景地图
        sample_map_list.append(scene_map)

    # 保存到 csv 文件
    scene_output_pth = os.path.join(output_pth, scene_index, f"inference/{scene_index}.csv")
    os.makedirs(os.path.join(output_pth, scene_index, "inference"), exist_ok=True)
    scene_df = pd.DataFrame(
        {
            "gt pos": sample_pos_list,
            "local map": sample_obs_list,
            "global map": sample_map_list,
            "annos": annos
        }
    )
    scene_df.to_csv(scene_output_pth, index=False)

    return scene_index


if __name__ == "__main__":
    
    scene_index_list = [f"scene_{num:05}" for num in range(3500)] # 前 1000 个场景
    # scene_index_list = [f"scene_{num:05}" for num in range(100)] # 前 100 个场景

    for scene_index in tqdm(scene_index_list):
        # 缺少标注的场景作废
        if scene_index in scene_invalid:
            scene_index_list.remove(scene_index)
    print("Invalid scenes removed")

    with concurrent.futures.ProcessPoolExecutor(max_workers=40) as executor:
        futures = {executor.submit(generate_scene_csv, task): task for task in scene_index_list}

        with tqdm(total=len(futures)) as pbar:
            for future in concurrent.futures.as_completed(futures):
                task_id = futures[future]
                try:
                    result = future.result()
                    # tqdm.write(result)
                except Exception as e:
                    tqdm.write(f"Task {task_id} generated an exception: {e}")

                pbar.update(1)  # 更新进度条


