import sys
sys.path.append(".")

import os # 拼接文件路径
import pandas as pd # 数据目录保存为表
from tqdm import tqdm # 进度条
from s3d import * # s3d 数据集信息
import concurrent.futures # 多进程


def generate_scene_csv(scene_index):
    
    sample_pos_list = []
    sample_obs_list = []
    
    # 场景地图
    scene_map = os.path.join(s3d_map_pth, scene_index, "map.npy")
    # 场景边界
    scene_bound = pd.read_csv(os.path.join(s3d_data_pth, scene_index, "boundary.csv"))

    # bev 观测
    bev_dir = os.path.join(s3d_bev_pth, scene_index, "2D_rendering")
    # 位姿真值
    data_dir = os.path.join(s3d_data_pth, scene_index, "2D_rendering")

    # 遍历所有样本
    sample_list = os.listdir(data_dir)
    for sample_item in sample_list:
        sample_dir = os.path.join(data_dir, sample_item, "perspective/full")

        # 遍历所有朝向
        ori_list = os.listdir(sample_dir)        
        for ori_item in ori_list:
            
            # csv 每一行存储位姿真值列表，bev 路径列表，cad 地图路径
            pos = os.path.join(
                s3d_data_pth, scene_index, "2D_rendering", sample_item, "perspective/full", ori_item, "camera_pose.txt"
            )
            sample_pos_list.append(pos)
        
            bev = os.path.join(
                s3d_bev_pth, scene_index, "2D_rendering", sample_item, "perspective/full", ori_item, "bev.npy"
            )
            sample_obs_list.append(bev)
    
    # 保存到 csv 文件
    scene_output_pth = os.path.join(s3d_csv_pth, scene_index, f"metric_v1/{scene_index}_pesp.csv")
    os.makedirs(os.path.join(s3d_csv_pth, scene_index, "metric_v1"), exist_ok=True)
    scene_df = pd.DataFrame(
        {
            "gt pos": [sample_pos_list],
            "local map": [sample_obs_list],
            "global map": [scene_map],
            "annos": [scene_bound]
        }
    )
    scene_df.to_csv(scene_output_pth, index=False)
    
    return scene_index


if __name__ == "__main__":
    
    # Structured3D 包括 3500 个场景
    scene_index_list = [f"scene_{num:05}" for num in range(3500)]
    # 测试：使用前 100 个场景
    scene_index_list = [f"scene_{num:05}" for num in range(100)]
    
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
