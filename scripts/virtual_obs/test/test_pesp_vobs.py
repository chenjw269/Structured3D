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

from scripts.utils.view_range_utils import generate_ellipse_mask
from scripts.virtual_obs.pesp_vobs import * # 虚拟单视角观测
from scripts.utils.visualize_occ import * # 可视化占用网格


if __name__ == "__main__":

    #################################
    # 场景信息
    #################################
    scene_index = "scene_00000"

    # 场景地图
    scene_map = np.load(os.path.join(s3d_map_pth, scene_index, "map.npy"))
    # 场景边界
    scene_bound = os.path.join(s3d_annos_pth, scene_index, "boundary.csv")
    scene_bound = pd.read_csv(scene_bound)

    #################################
    # 样本信息
    #################################
    # sample_index = ("485142", "0")
    # # 读取样本信息
    # sample_pth = os.path.join(s3d_data_pth, scene_index, "2D_rendering", sample_index[0], "perspective/full", sample_index[1])
    # # 样本位姿
    # sample_pose = np.loadtxt(os.path.join(sample_pth, "camera_pose.txt"))
    # sample_pose = read_camera_pose(sample_pose, mode="raw")
    # print(f"{sample_pose}")
    # # 坐标转换
    # sample_position = sample_pose[:2]
    # sample_position = position_to_pixel(sample_position, resolution, scene_bound)
    # sample_pose = np.append(sample_position, sample_pose[2]-90)
    # print(f"{sample_pose}")

    sample_pose = np.array([256, 558, -270])
    
    ############################################
    # 虚拟观测的视野
    ############################################
    fov = 80

    map_image_cp = copy.copy(scene_map)

    # 将视野外的内容覆盖
    view_range_mask = generate_ellipse_mask(
        image=map_image_cp,
        # 扇形圆心，扇形朝向，扇形角度范围
        center=sample_pose[:2].astype(int), angle=sample_pose[2], fov=fov,
        mode="occ")
    map_image_cp = view_range_mask * map_image_cp

    image_viewrange = visualize_occ(map_image_cp, LABEL_TO_COLOR)
    # 1. 可视化占用网格
    cv2.imshow("View range", image_viewrange)
    # 2. 保存到日志
    cv2.imwrite("logs/view range.png", image_viewrange)

    ##########################################
    # 将视野内容旋转到正方向
    ##########################################
    # cv2 中的正方向为水平向右，转换为正方向竖直向上，需要加上 90 度
    map_image_cp = generate_rotation(
        image=map_image_cp,
        center=sample_pose[:2], angle=(sample_pose[2]+90))

    image_viewcontent = visualize_occ(map_image_cp)
    # 1. 可视化占用网格
    cv2.imshow("View content", image_viewcontent)
    # 2. 保存到日志
    cv2.imwrite("logs/view content.png", image_viewcontent)

    ##########################################
    # 虚拟观测的范围
    ##########################################
    # 获取视野内的部分
    map_image_cp = map_image_cp[
        sample_pose[1].astype(int) - 256: sample_pose[1].astype(int),
        sample_pose[0].astype(int) - 128: sample_pose[0].astype(int) + 128,
    ]
    virtual_view = visualize_occ(map_image_cp)
    # 1. 可视化占用网格
    cv2.imshow("Virtual view 1", virtual_view)
    # 2. 保存到日志
    cv2.imwrite("logs/virtual view 1.png", virtual_view)

    ############################################
    # 虚拟观测的内容
    ############################################
    virtual_obs = virtual_pesp_obs(scene_map, sample_pose, fov)
    virtual_obs = visualize_occ(virtual_obs, LABEL_TO_COLOR)

    # 1. 可视化占用网格
    cv2.imshow("Virtual view 2", virtual_view)
    # 2. 保存到日志
    cv2.imwrite("logs/virtual view 2.png", virtual_view)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
