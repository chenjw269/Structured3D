# 读取样本的观测和位姿等信息

import sys
sys.path.append(".")

import os
from s3d import *
import numpy as np
import pandas as pd
from scripts.utils.read_camera_pose import read_camera_pose
from scripts.cad_mapping_v1.coord_conv import position_to_pixel


def read_sample_info(scene_index, sample_index, mode="raw"):

    ###############################
    # 场景级别的信息
    ###############################

    # 场景路径
    scene_pth = os.path.join(s3d_data_pth, scene_index)

    # 场景边界
    scene_bound = os.path.join(scene_pth, "boundary.csv")
    scene_bound = pd.read_csv(scene_bound)

    # 场景地图
    scene_map_array = os.path.join(s3d_map_pth, scene_index, "map.npy")
    scene_map_img = os.path.join(s3d_map_pth, scene_index, "map.png")

    ###############################
    # 样本级别的信息
    ###############################

    # 如果是单视角图像
    if isinstance(sample_index, list) or isinstance(sample_index, tuple):
        # 样本路径
        sample_pth = os.path.join(
            s3d_data_pth, scene_index, "2D_rendering", sample_index[0], "perspective/full", sample_index[1]
        )
        
        # 1. 样本位姿
        # 读取位姿
        sample_pose = os.path.join(sample_pth, "camera_pose.txt")
        sample_pose = read_camera_pose(sample_pose, mode=mode)
        # 坐标转换
        sample_position = sample_pose[:2]
        sample_position = position_to_pixel(sample_position, resolution, scene_bound)
        sample_pose = np.append(sample_position, sample_pose[2])
        
        # 2. 样本观测
        # sample_bev = os.path.join(s3d_bev_pth)
        # sample_bev = np.load(sample_bev)
        
    # 如果是全景图图像
    else:
        sample_pth = os.path.join(
            s3d_data_pth, scene_index, "2D_rendering", sample_index, "panorama/full"
        )
        # 读取位姿
        sample_pose = os.path.join(sample_pth)
        sample_pose = np.loadtxt(sample_pose)
        # 坐标转换
        sample_pose = position_to_pixel(sample_pose, resolution, scene_bound)
        sample_pose = np.append(sample_pose, 90)
    
    sample_info = {
        "scene map array": scene_map_array,
        "scene map img": scene_map_img,
        # "sample obs": sample_obs,
        "sample pose": sample_pose,
    }
    
    return sample_info


def read_pesp_sample():
    
    
