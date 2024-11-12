import sys
sys.path.append(".")

import os
import cv2
import numpy as np
from s3d import *
from tqdm import tqdm
import concurrent.futures
from scripts.bev_pano.process_bev import generate_bev # 创建全景图 bev
from scripts.utils.visualize_occ import visualize_occ # 可视化占用网格
from assets.semantic2label import LABEL_TO_COLOR


def execute_bev_processing(task):
    
    data_dir = os.path.join(s3d_data_pth, task)
    output_dir = os.path.join(s3d_bev_pth, task)
    
    # 深度图路径
    obs_item_depth_full = os.path.join(data_dir, "panorama/full/depth.png")
    # 语义分割图路径
    obs_item_semantic_full = os.path.join(data_dir, "panorama/full/semantic.png")
    # 计算占用网格
    obs_item_bev = generate_bev(obs_item_depth_full, obs_item_semantic_full)
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

    ########################################
    # 统计所有任务
    ########################################    
    # Structured3D 数据集包含 3500 个场景
    scene_index_list = [f"scene_{num:05}" for num in range(3500)]
    # 去掉标注数据缺失的场景
    for scene_index in tqdm(scene_index_list):
        if scene_index in scene_invalid:
            scene_index_list.remove(scene_index)
    print("Invalid scenes removed")

    task_list = []
    # 统计所有场景下的样本
    for scene_index in scene_index_list:

        scene_sample_dir = os.path.join(s3d_data_pth, scene_index, "2D_rendering")
        scene_sample_list = os.listdir(scene_sample_dir)

        # 统计该场景下的所有样本
        for scene_sample in scene_sample_list:

            # 去掉观测数据缺失的样本
            if f"{scene_index},{scene_sample}" in sample_invalid:
                tqdm.write(f"Jmp obs loss {scene_index} {scene_sample}")
                continue
            
            task_list.append(f"{scene_index}/2D_rendering/{scene_sample}")

    ########################################
    # 多线程计算 BEV 视图
    ########################################
    with concurrent.futures.ProcessPoolExecutor(max_workers=40) as executor:
        # 构建任务池
        futures = {executor.submit(execute_bev_processing, task): task for task in task_list}
        # 遍历所有任务
        with tqdm(total=len(futures)) as pbar:
            for future in concurrent.futures.as_completed(futures):
                task_id = futures[future]
                try:
                    result = future.result()
                    # tqdm.write(result)
                except Exception as e:
                    tqdm.write(f"Task {task_id} generated an exception: {e}")

                pbar.update(1)  # 更新进度条
