# 创建度量学习数据集数据目录 csv 文件
import sys
sys.path.append(".")

import os
from s3d import *
import pandas as pd
from tqdm import tqdm


def generate_scene_csv(scene_index):
    
    pose_list = []
    bev_list = []
    scene_map_list = []
    scene_bound_list = []

    # 遍历场景中的所有样本
    scene_data_dir = os.path.join(s3d_data_pth, scene_index, "2D_rendering")
    sample_index_list = os.listdir(scene_data_dir)
    for sample_index in sample_index_list:
        # 遍历样本中的所有朝向
        sample_data_dir = os.path.join(scene_data_dir, sample_index, "perspective/full")
        orientation_index_list = os.listdir(sample_data_dir)
        for orientation_index in orientation_index_list:
            # 相机位姿
            pose = os.path.join(
                sample_data_dir, orientation_index, "camera_pose.txt"
            )
            pose_list.append(pose)
            # 单视角 bev
            bev = os.path.join(
                s3d_bev_pth, scene_index, "2D_rendering",
                sample_index, "perspective/full",
                orientation_index, "bev.npy"
            )
            bev_list.append(bev)
            # 场景地图
            scene_map = os.path.join(s3d_map_pth, scene_index, "map.npy")
            scene_map_list.append(scene_map)
            # 场景边界
            scene_bound = os.path.join(s3d_annos_pth, scene_index, "bound.csv")
            scene_bound_list.append(scene_bound)

    # 保存到 csv 文件
    scene_csv_pth = os.path.join(s3d_csv_pth, scene_index, f"metric_pesp.csv")
    os.makedirs(os.path.join(s3d_csv_pth, scene_index), exist_ok=True)
    
    scene_df = pd.DataFrame(
        {
            "gt pose": pose_list,
            "local map": bev_list,
            "global map": scene_map_list,
            "bound": scene_bound_list
        }
    )
    scene_df.to_csv(scene_csv_pth, index=False)

    return scene_index


if __name__ == "__main__":
    
    # Structured3D 包括 3500 个场景
    scene_index_list = [f"scene_{num:05}" for num in range(3500)]

    for scene_index in tqdm(scene_index_list):

        # 
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
