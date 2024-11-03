# 虚拟位姿，受限视野下的观测值

import sys
sys.path.append(".")

import os
import cv2
from s3d import *
import numpy as np
import pandas as pd
from scripts.utils.virtual_obs import virtual_fov # 虚拟视野
from scripts.utils.read_camera_pose import read_camera_pose # 读取相机位姿
from scripts.cad_mapping_v1.coord_conv import position_to_pixel # 坐标变换
from scripts.utils.visualize_occ import visualize_occ # 可视化占用网格


if __name__ == "__main__":
    
    test_scene_index = "scene_00000"
    test_sample_index = "485142"

    # 读取场景标注
    test_scene_bound = os.path.join(s3d_data_pth, test_scene_index, "boundary.csv")
    test_scene_bound = pd.read_csv(test_scene_bound)

    # 读取样本位姿
    test_sample_pose = os.path.join(s3d_data_pth, test_scene_index, "2D_rendering", test_sample_index, "perspective/full/0/camera_pose.txt")
    test_sample_pose = read_camera_pose(test_sample_pose, mode="raw")

    # 坐标转换
    test_sample_position = test_sample_pose[:2]
    test_sample_position = position_to_pixel(test_sample_position, resolution, test_scene_bound)
    test_sample_pose = np.append(test_sample_position, test_sample_pose[2])

    # # 1. 读取场景地图图像
    # test_scene_map = os.path.join(s3d_map_pth, test_scene_index, "map.png")
    # test_scene_map = cv2.imread(test_scene_map)
    # # 虚拟视野观测
    # test_scene_map = virtual_fov(test_scene_map, test_sample_pose, 80, "image")
    
    # 2. 读取场景地图 array
    test_scene_map = os.path.join(s3d_map_pth, test_scene_index, "map.npy")
    test_scene_map = np.load(test_scene_map)
    # 虚拟视野观测
    test_scene_map = virtual_fov(test_scene_map, test_sample_pose, 80, "occ")
    # 可视化占用网格
    test_scene_map = visualize_occ(test_scene_map)

    # 获取局部地图
    test_scene_map = test_scene_map[
        test_sample_position[1] - 128 : test_sample_position[1] + 128,
        test_sample_position[0] - 128 : test_sample_position[0] + 128,
        :
    ]

    # # 将视野旋转一定角度
    # test_scene_map = 

    cv2.imshow("Virtual fov", test_scene_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
