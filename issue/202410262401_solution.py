# Solution
# 遍历所有场景下的所有样本，找到采样正负样例时间过长的样本

import sys
sys.path.append(".")

from s3d import * # s3d 数据集信息

import os # 拼接文件路径
import cv2 # 读取场景地图
import time # 测量程序运行时间
import numpy as np # 读取样本位置
import pandas as pd # 读取场景边界
from tqdm import tqdm # 进度条
from scripts.cad_mapping_v1.coord_conv import position_to_pixel # 坐标转换
from scripts.metric_learning.neighbor_sample_v0 import generate_neighbor_within # 采样正样例
from scripts.metric_learning.neighbor_sample_v0 import generate_neighbor_between # 采样负样例


def execute_sample_pn(scene_index):
    
    # 场景地图
    scene_map = os.path.join(s3d_map_pth, scene_index, "map.png")
    scene_map = cv2.imread(scene_map)

    # 场景坐标边界
    scene_bound = os.path.join(s3d_data_pth, scene_index, "boundary.csv")
    scene_bound = pd.read_csv(scene_bound)
    scene_bound_sample = np.array([
        [128, int(scene_bound['size_x'].item() / resolution) - 128],
        [128, int(scene_bound['size_y'].item() / resolution) - 128]
    ])
    
    # 遍历样本
    scene_sample_dir = os.path.join(s3d_data_pth, scene_index, "2D_rendering")
    scene_sample_list = os.listdir(scene_sample_dir)
    for sample_item in scene_sample_list:
    
        if f"{scene_index},{sample_item}" in sample_invalid:
            continue
    
        # 样本位置
        sample_item_pos = os.path.join(scene_sample_dir, sample_item, "panorama/camera_xyz.txt")
        sample_item_pos = np.genfromtxt(sample_item_pos, delimiter=" ")[:2]
    
        # 坐标转换
        sample_item_pos = position_to_pixel(sample_item_pos, resolution, scene_bound)

        start_time = time.time()
        # 采样正样例
        positive_radius = 40 # 40 * 0.025 = 1
        positive_sample_pos = generate_neighbor_within(
            position=sample_item_pos, nums=5, radius=positive_radius, p_bound=scene_bound_sample
        )
        end_time = time.time()
        # 如果时间过长，则引发错误
        execution_time = (end_time - start_time)
        if execution_time > 1:
            print(f"{scene_index} {sample_item} err in positive sample")
            raise StopIteration

        start_time = time.time()
        # 采样负样例
        negative_radius_1 = 80 # 80 * 0.025 = 2
        negative_radius_2 = 240 # 240 * 0.025 = 6
        negative_sample_pos = generate_neighbor_between(
            position=sample_item_pos, nums=5,
            radius_1=negative_radius_1, radius_2=negative_radius_2,
            p_bound=scene_bound_sample
        )
        end_time = time.time()
        # 如果时间过长，则引发错误
        execution_time = (end_time - start_time)
        if execution_time > 1:
            print(f"{scene_index} {sample_item} err in negative sample")
            raise StopIteration


if __name__ == "__main__":

    # Structured3D 包括 3500 个场景
    scene_index_list = [f"scene_{num:05}" for num in range(3500)]

    for scene_index in tqdm(scene_index_list):
        # 缺少标注的场景作废
        if scene_index in scene_invalid:
            scene_index_list.remove(scene_index)
    print("Invalid scenes removed")

    # 单线程
    for scene_index in tqdm(scene_index_list):
        execute_sample_pn(scene_index)

