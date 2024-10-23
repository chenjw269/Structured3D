import os
import cv2
import time
import numpy as np
import concurrent.futures
from tqdm import tqdm

import sys
sys.path.append(".")

from scripts.bev_observation.semantic_bev import generate_semantic_voxel
from scripts.utils.visualize_occ import visualize_occ
from assets.semantic2label import LABEL_TO_COLOR


data_pth = "e:/datasets/Structure3D/Structured3D"
output_pth = "e:/datasets/Structure3D_bev/Structured3D"

def execute_bev_processing(task):
    
    data_dir = os.path.join(data_pth, task)
    output_dir = os.path.join(output_pth, task)
    
    # 深度图路径
    obs_item_depth_full = os.path.join(data_dir, "panorama/full/depth.png")
    # 语义分割图路径
    obs_item_semantic_full = os.path.join(data_dir, "panorama/full/semantic.png")
    # 计算占用网格
    obs_item_bev = generate_semantic_voxel(obs_item_depth_full, obs_item_semantic_full)
    # 可视化占用网格
    obs_item_img = visualize_occ(obs_item_bev, LABEL_TO_COLOR)
    
    # 保存结果
    os.makedirs(os.path.join(output_dir, "panorama/full"), exist_ok=True)
    # BEV 占用网格
    obs_item_occ_output = os.path.join(output_dir, "panorama/full/bev.npy")
    np.save(obs_item_occ_output, obs_item_bev)
    # BEV 图像
    obs_item_img_output = os.path.join(output_dir, "panorama/full/bev.png")
    cv2.imwrite(obs_item_img_output, obs_item_img)

    return task


if __name__ == "__main__":

    # 标注数据缺失的场景
    with open("../../logs/scene_annos.txt", encoding="utf-8") as f: # remote
    # with open("logs/scene_annos.txt", encoding="utf-8") as f: # local

        scene_invalid = f.readlines()
    for index, item in enumerate(scene_invalid):
        scene_invalid[index] = item.replace("\n", "")
        
    # 观测数据缺失的样本
    with open("../../logs/scene_observation.txt", encoding="utf-8") as f: # remote
    # with open("logs/scene_observation.txt", encoding="utf-8") as f:  # local
            obs_invalid = f.readlines()
    for index, item in enumerate(obs_invalid):
        obs_invalid[index] = item.replace("\n", "")

    scene_index_list = [f"scene_{num:05}" for num in range(3500)] # 前 1000 个场景

    #########################################
    # 统计所有场景下的所有样本
    #########################################
    sample_list_total = []
    for scene_index in tqdm(scene_index_list):
        
        # 缺少标注的场景作废
        if scene_index in scene_invalid:
            tqdm.write(f"Jmp annos loss {scene_index}")
            continue
        
        obs_dir = os.path.join(data_pth, scene_index, "2D_rendering")
        obs_list = os.listdir(obs_dir)
        # 遍历观测数据
        for obs_item in obs_list:
            
            # 缺少观测值的样本作废
            if f"{scene_index},{obs_item}" in obs_invalid:
                tqdm.write(f"Jmp obs loss {scene_index} {obs_item}")
                continue
            
            sample_list_total.append(f"{scene_index}/2D_rendering/{obs_item}")

    #########################################
    # 多线程计算 BEV 视图
    #########################################

    # 记录开始时间
    start_time = time.time()

    # 用所有样本索引作为参数，构建任务池
    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
        futures = {executor.submit(execute_bev_processing, task): task for task in sample_list_total}

        with tqdm(total=len(futures)) as pbar:
            for future in concurrent.futures.as_completed(futures):
                task_id = futures[future]
                try:
                    result = future.result()
                    tqdm.write(result)
                except Exception as e:
                    tqdm.write(f"Task {task_id} generated an exception: {e}")
                
                pbar.update(1)  # 更新进度条

                # # 计算已执行时间和剩余时间
                # elapsed_time = time.time() - start_time
                # remaining_time = (elapsed_time / (pbar.n)) * (pbar.total - pbar.n) if pbar.n > 0 else 0
                # tqdm.write(f"Elapsed Time: {elapsed_time:.2f}s, Remaining Time: {remaining_time:.2f}s")