# 不同位姿假设下的观测假设

import sys
sys.path.append(".")

import os
import cv2
from s3d import *
import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts.utils.read_camera_pose import read_camera_pose # 读取样本位姿
from scripts.cad_mapping_v1.coord_conv import position_to_pixel # 坐标转换

from scripts.utils.patches_matching import pose_matching # 匹配最近邻位姿

from scripts.virtual_obs.pesp_vobs import virtual_pesp_obs # 单视角虚拟观测
from scripts.utils.visualize_occ import * # 可视化占用网格


if __name__ == "__main__":

    #####################################
    # 读取数据
    #####################################
    scene_index = "scene_00000"
    sample_index = ("485142", "0")

    # 场景地图
    scene_map = np.load(os.path.join(s3d_map_pth, scene_index, "map.npy"))
    # 读取场景边界
    scene_bound = os.path.join(s3d_annos_pth, scene_index, "boundary.csv")
    scene_bound = pd.read_csv(scene_bound)

    # 样本信息
    sample_pth = os.path.join(s3d_data_pth, scene_index, "2D_rendering", sample_index[0], "perspective/full", sample_index[1])
    # 样本位姿
    sample_pose = np.loadtxt(os.path.join(sample_pth, "camera_pose.txt"))
    sample_pose = read_camera_pose(sample_pose, mode="raw")
    # # 坐标转换
    # sample_position = sample_pose[:2]
    # sample_position = position_to_pixel(sample_position, resolution, scene_bound)
    # sample_pose = np.append(sample_position, sample_pose[2]-90)

    os.makedirs("logs/Virtual obs", exist_ok=True)

    #######################################
    # 采样位姿假设
    #######################################
    p_step = 250 # 以 0.25 m (250 mm) 为步长
    # 由 x 轴坐标范围，得到 x 轴位置假设
    x_hypothesis = np.arange(
        int(scene_bound['x_min'].item()), int(scene_bound['x_max'].item()), p_step)
    # 由 y 轴坐标范围，得到 y 轴位置假设
    y_hypothesis = np.arange(
        int(scene_bound['y_min'].item()), int(scene_bound['y_max'].item()), p_step)
    
    o_step = 15 # 以 15 ° 为步长
    # 由 yaw 角范围，得到朝向假设
    orientation_hypothesis = np.arange(-180, 180, 15)

    # 排列组合所有位姿假设
    pose_hypothesis = []
    for x in x_hypothesis:
        for y in y_hypothesis:
            for ori in orientation_hypothesis:
                pose_hypothesis.append((x, y, ori))
    pose_hypothesis = np.array(pose_hypothesis)
    print(f"{len(pose_hypothesis)} poses hypothesis in total")

    ########################################
    # 位姿假设处的观测假设
    ########################################
    # 找到距离最近的位姿假设
    nearest_index = pose_matching(sample_pose, pose_hypothesis, 1)[0]
    nearest_pose = pose_hypothesis[nearest_index]
    # 坐标转换
    nearest_position = nearest_pose[:2]
    nearest_position = position_to_pixel(nearest_position, resolution, scene_bound)
    nearest_pose = np.append(nearest_position, nearest_pose[2]-90)
    # 获取距离最近的观测假设
    nearest_vobs = virtual_pesp_obs(scene_map, nearest_pose, fov=80)
    nearest_vobs = visualize_occ(nearest_vobs, LABEL_TO_COLOR)
    cv2.imwrite("logs/Virtual obs/nearest_vobs.png", nearest_vobs)

    # 遍历所有位姿假设，获取观测假设
    for pose in tqdm(pose_hypothesis):

        # 坐标转换
        position = pose[:2]
        position = position_to_pixel(position, resolution, scene_bound)
        pose = np.append(position, pose[2]-90)

        # 虚拟观测
        virtual_obs = virtual_pesp_obs(scene_map, pose, fov=80)
        # 可见性检验
        virtual_obs = 
        virtual_obs = visualize_occ(virtual_obs, LABEL_TO_COLOR)

        cv2.imwrite(
            f"logs/Virtual obs/hypothesis/{pose[0]}_{pose[1]}_{pose[2]}.png", virtual_obs)
