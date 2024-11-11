import sys
sys.path.append(".")

from s3d import * # s3d 数据集信息

import os # 拼接文件路径
import numpy as np # 读取样本位置
import pandas as pd # 数据目录保存为表
from tqdm import tqdm # 进度条
import concurrent.futures # 多进程
from scripts.cad_mapping_v1.coord_conv import position_to_pixel # 坐标转换


def generate_scene_csv(scene_index):

    sample_pos_list = []
    sample_obs_list = []
    scene_map_list = []
    scene_bound_list = []

    # 场景地图
    scene_map = os.path.join(s3d_map_pth, scene_index, "map.npy")
    # 场景边界
    scene_bound = pd.read_csv(os.path.join(s3d_data_pth, scene_index, "boundary.csv"))

    # bev 观测
    bev_dir = os.path.join(s3d_bev_pth, scene_index, "2D_rendering")
    # 位姿真值
    obs_dir = os.path.join(s3d_data_pth, scene_index, "2D_rendering")

    # 遍历观测数据
    obs_list = os.listdir(obs_dir)
    for obs_item in obs_list:

        # # TODO:缺少观测值的样本作废
        # if f"{scene_index},{obs_item}" in sample_invalid:
        #     tqdm.write(f"Jmp obs loss {scene_index} {obs_item}")
        #     continue

        # csv 每一行存储 bev 路径列表，cad 地图路径，位姿真值列表
        
        # 场景标注
        scene_bound_list.append(scene_bound)

        # 真实位置
        obs_item_pos = os.path.join(obs_dir, obs_item, "panorama/camera_xyz.txt")
        obs_item_pos = np.genfromtxt(obs_item_pos, delimiter=" ")[:2]
        # 坐标变换（csv 中不能存 np.array，只能存 list）
        obs_item_pos = position_to_pixel(obs_item_pos, resolution, scene_bound_df)
        sample_pos_list.append(obs_item_pos.tolist())

        # 观测数据 BEV
        obs_item_bev = os.path.join(bev_dir, obs_item, "panorama/full/bev.npy")
        sample_obs_list.append(obs_item_bev)

        # 场景地图
        sample_map_list.append(scene_map)

    # 保存到 csv 文件
    scene_output_pth = os.path.join(s3d_csv_pth, scene_index, f"metric_learning/{scene_index}.csv")
    os.makedirs(os.path.join(s3d_csv_pth, scene_index, "metric_learning"), exist_ok=True)
    scene_df = pd.DataFrame(
        {
            "gt pos": sample_pos_list,
            "local map": sample_obs_list,
            "global map": sample_map_list,
            "annos": bound_list
        }
    )
    scene_df.to_csv(scene_output_pth, index=False)

    return scene_index


if __name__ == "__main__":

    # Structured3D 包括 3500 个场景
    scene_index_list = [f"scene_{num:05}" for num in range(3500)]

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
