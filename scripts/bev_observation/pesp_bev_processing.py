# 多线程

import sys
sys.path.append(".")

import os
from s3d import *
import concurrent.futures
from tqdm import tqdm
from scripts.bev.pesp_bev import executing_pesp_bev_processing


if __name__ == "__main__":
    
    #######################################
    # 统计样本
    #######################################
    # Structured3D 包括 3500 个场景
    scene_index_list = [f"scene_{num:05}" for num in range(3500)]
    
    
    # 统计所有用于计算 bev 的样本
    pesp_bev_list = []
    
    for scene_index in scene_index_list:
        # 同一个场景下有多个样本
        scene_data_pth = os.path.join(s3d_data_pth, scene_index, "2D_rendering")
        sample_index_list = os.listdir(scene_data_pth)
        for sample_index in sample_index_list:
            # 同一个样本有多个朝向
            sample_data_pth = os.path.join(scene_data_pth, sample_index, "perspective/full")
            ori_index_list = os.listdir(sample_data_pth)
            for ori_index in ori_index_list:
                
                # 是否为数据缺失的样本
                
                pesp_bev_list.append((scene_index, (sample_index, ori_index)))
    
    print(f"{len(pesp_bev_list)} bev to process")

    
    #######################################
    # 多线程处理 bev
    #######################################
    # 用所有样本索引作为参数，构建任务池
    with concurrent.futures.ProcessPoolExecutor(max_workers=40) as executor:
        futures = {executor.submit(executing_pesp_bev_processing, task): task for task in pesp_bev_list}

        with tqdm(total=len(futures)) as pbar:
            for future in concurrent.futures.as_completed(futures):
                task_id = futures[future]
                try:
                    result = future.result()
                    # tqdm.write(result)
                except Exception as e:
                    tqdm.write(f"Task {task_id} generated an exception: {e}")
                
                pbar.update(1)  # 更新进度条