import sys
sys.path.append(".")

import os
import cv2
import numpy as np
from s3d import *
from tqdm import tqdm
import concurrent.futures
from scripts.bev_observation.bev_pesp.process_bev import generate_bev # 计算单视角 bev
from scripts.utils.visualize_occ import * # 可视化占用网格


def execute_pesp_bev(task):
    
    # 数据目录
    sample_data_pth = os.path.join(s3d_pesp_data_pth, task)
    # 深度图
    depth = os.path.join(sample_data_pth, "depth.png")
    depth = cv2.imread(depth, cv2.IMREAD_UNCHANGED)
    # 语义分割图
    semantic = os.path.join(sample_data_pth, "semantic.png")
    semantic = cv2.cvtColor(cv2.imread(semantic), cv2.COLOR_BGR2RGB)
    # 相机位姿
    pose = os.path.join(sample_data_pth, "camera_pose.txt")
    pose = np.loadtxt(pose)

    # bev 数据目录
    bev_pth = os.path.join(s3d_bev_pth, task)
    if not os.path.exists(bev_pth):
        os.makedirs(bev_pth)
    # 单视角 bev
    bev = generate_bev(depth, semantic, pose)
    np.save(f"{bev_pth}/bev.npy", bev)
    # 可视化占用网格
    bev_img = visualize_occ(bev, LABEL_TO_COLOR)
    cv2.imwrite(f"{bev_pth}/bev.png", bev_img)

    return task


if __name__ == "__main__":
    
    # # Structured3D 包括 3500 个场景
    # scene_index_list = [f"scene_{num:05}" for num in range(3500)]
    # 取前 100 个场景
    scene_index_list = [f"scene_{num:05}" for num in range(400)]
    
    # 去掉标注数据缺失的场景
    for scene_index in tqdm(scene_index_list):
        if scene_index in scene_invalid:
            scene_index_list.remove(scene_index)
    print("Invalid scenes removed")

    # 观测数据缺失的样本
    with open(s3d_pesp_obs_err, encoding="utf-8") as f:
        obs_loss_list = f.readlines()

    ###########################################
    # 统计计算 bev 的样本
    ###########################################
    task_list = []

    # 遍历场景
    for scene_index in scene_index_list:
        # 场景数据路径
        scene_data_dir = os.path.join(s3d_pesp_data_pth, scene_index, "2D_rendering")
        sample_index_list = os.listdir(scene_data_dir)

        # 遍历样本
        for sample_index in sample_index_list:
            # 样本数据路径
            sample_data_dir = os.path.join(scene_data_dir, sample_index, "perspective/full")
            ori_index_list = os.listdir(sample_data_dir)

            # 遍历朝向
            for ori_index in ori_index_list:
                # 朝向数据路径
                ori_data_dir = os.path.join(sample_data_dir, ori_index)
                
                if f"{scene_index},{sample_index},{ori_index}" in obs_loss_list:
                    continue
                else:
                    task_list.append(
                        f"{scene_index}/2D_rendering/{sample_index}/perspective/full/{ori_index}"
                    )
    
    ###########################################
    # 计算单视角 bev
    ###########################################
    # 用所有样本索引作为参数，构建任务池
    with concurrent.futures.ProcessPoolExecutor(max_workers=128) as executor:
        futures = {executor.submit(execute_pesp_bev, task): task for task in task_list}

        with tqdm(total=len(futures)) as pbar:
            for future in concurrent.futures.as_completed(futures):
                task_id = futures[future]
                try:
                    result = future.result()
                    # tqdm.write(result)
                except Exception as e:
                    tqdm.write(f"Task {task_id} generated an exception: {e}")
                
                pbar.update(1)  # 更新进度条