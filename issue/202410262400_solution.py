# 202410271552
# 

# Solution
# 遍历所有场景，找到采样位置假设时间过长的场景
# 


import sys
sys.path.append(".")

from s3d import * # s3d 数据集信息

import os # 拼接文件路径
import cv2 # 读取场景地图
import time # 测量程序运行时间
import pandas as pd # 读取场景边界
import concurrent.futures # 多进程
from tqdm import tqdm # 进度条
from scripts.inference.position_hypothesis import generate_scene_hypothesis # 采样位置假设


def execute_position_hypothesis(scene_index):

    # 场景坐标边界
    scene_bound = os.path.join(s3d_data_pth, scene_index, "boundary.csv")
    scene_bound = pd.read_csv(scene_bound)

    start_time = time.time()

    # 采样位置假设
    pos_hypothesis = generate_scene_hypothesis(scene_bound)

    end_time = time.time()

    # 如果时间过长，则引发错误
    execution_time = (end_time - start_time) * 1000
    # print(f"程序运行时间: {execution_time} 毫秒")

    if execution_time > 3.5:

        # 引发错误
        # raise StopIteration

        # 场景地图
        scene_map = os.path.join(s3d_map_pth, scene_index, "map.png")
        scene_map = cv2.imread(scene_map)



if __name__ == "__main__":

    # Structured3D 包括 3500 个场景
    scene_index_list = [f"scene_{num:05}" for num in range(3500)]

    for scene_index in tqdm(scene_index_list):
        # 缺少标注的场景作废
        if scene_index in scene_invalid:
            scene_index_list.remove(scene_index)
    print("Invalid scenes removed")

    # # 单线程
    # for scene_index in tqdm(scene_index_list):
    #     execute_position_hypothesis(scene_index)

    # # 多线程
    # with concurrent.futures.ProcessPoolExecutor(max_workers=40) as executor:

    #     futures = {executor.submit(execute_position_hypothesis, task): task for task in scene_index_list}

    #     with tqdm(total=len(futures)) as pbar:
    #         for future in concurrent.futures.as_completed(futures):
    #             task_id = futures[future]
    #             try:
    #                 result = future.result()
    #                 # tqdm.write(result)
    #             except Exception as e:
    #                 tqdm.write(f"Task {task_id} generated an exception: {e}")

    #             pbar.update(1)  # 更新进度条


    # 在单个场景上测试
    
    # scene_00452 execution_time 39.96753692626953 ms
    scene_index = 'scene_00452'
    execute_position_hypothesis(scene_index)
