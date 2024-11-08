import sys
sys.path.append(".")

import os
import cv2
import copy
from s3d import *
import numpy as np
import pandas as pd

from scripts.utils.read_camera_pose import read_camera_pose # 读取样本位姿
from scripts.cad_mapping_v1.coord_conv import position_to_pixel # 坐标转换

from scripts.virtual_obs.pesp_vobs import * # 虚拟单视角观测
from scripts.utils.visualize_occ import * # 可视化占用网格

from scripts.virtual_obs.ray_casting import ray_casting


if __name__ == "__main__":
    
    scene_index = "scene_00049"
    sample_index = ("2321", "1")

    # 读取样本信息
    sample_pth = os.path.join(s3d_data_pth, scene_index, "2D_rendering", sample_index[0], "perspective/full", sample_index[1])

    # 场景地图
    scene_map = np.load(os.path.join(s3d_map_pth, scene_index, "map.npy"))
    # 场景边界
    scene_bound = os.path.join(s3d_annos_pth, scene_index, "boundary.csv")
    scene_bound = pd.read_csv(scene_bound)
    
    # 样本位姿
    sample_pose = np.loadtxt(os.path.join(sample_pth, "camera_pose.txt"))
    sample_pose = read_camera_pose(sample_pose, mode="raw")
    print(f"{sample_pose}")

    # 坐标转换
    sample_position = sample_pose[:2]
    sample_position = position_to_pixel(sample_position, resolution, scene_bound)
    sample_pose = np.append(sample_position, sample_pose[2]-90)
    print(f"{sample_pose}")

    ############################################
    # 虚拟观测的内容
    ############################################

    fov = 80

    virtual_obs = virtual_pesp_obs(scene_map, sample_pose, fov)
    virtual_obs_img = visualize_occ(virtual_obs, LABEL_TO_COLOR)

    cv2.imshow("Virtual obs without occlusion", virtual_obs_img)

    ############################################
    # 射线检测遮挡
    ############################################
    virtual_obs_mask = ray_casting(virtual_obs, (256, 128), (50, 130), 256)
    virtual_obs = virtual_obs_mask * virtual_obs
    
    virtual_obs_img = visualize_occ(virtual_obs, )

    cv2.imshow("Virtual obs real", virtual_obs_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
