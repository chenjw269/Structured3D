import sys
sys.path.append(".")

import os
import cv2
import copy
from s3d import *
import numpy as np
import pandas as pd

from scripts.cad_mapping_v1.coord_conv import position_to_pixel # 坐标转换

from scripts.utils.view_range_utils import generate_ellipse_mask # 

from scripts.virtual_obs.pesp_vobs import virtual_pano_obs # 虚拟全景图观测


if __name__ == "__main__":

    #################################
    # 场景信息
    #################################
    scene_index = "scene_00000"
    
    # 场景地图
    scene_map = np.load(os.path.join(s3d_map_pth, scene_index, "map.npy"))
    # 场景边界
    scene_bound = pd.read_csv(os.path.join(s3d_annos_pth, scene_index, "boundary.csv"))

    #################################
    # 样本信息
    #################################
    sample_index = "485142"

    # 读取样本信息
    sample_pth = os.path.join(s3d_data_pth, scene_index, "2D_rendering", sample_index, "panorama/full")
    # 样本位置
    sample_position = np.genfromtxt(os.path.join(sample_pth, ""), delimiter=" ")
    print(f"{sample_position}")
    # 坐标转换
    sample_position = position_to_pixel(sample_position, resolution, scene_bound)

    ############################################
    # 虚拟观测的视野
    ############################################
    view_range = 256
    
    map_image_cp = copy.copy(scene_map)
    # 将视野外的内容覆盖
    view_rect_mask = generate_rectangle_mask(
        image=map_image_cp,
        center=pose[:2].astype(int), angle=pose[2], view_range=view_range,
        mode="occ")
    map_image_cp = view_range_mask * map_image_cp

    ##########################################
    # 将视野内容旋转到正方向
    ##########################################
    # cv2 中的正方向为水平向右，转换为正方向竖直向上，需要加上 90 度
    map_image_cp = generate_rotation(
        image=map_image_cp,
        center=pose[:2], angle=(pose[2]+90)%360)

    ##########################################
    # 虚拟观测的内容
    ##########################################
    # 获取视野内的部分
    map_image_cp = map_image_cp[
        pose[1].astype(int) - 128: pose[1].astype(int) + 128,
        pose[0].astype(int) - 128: pose[0].astype(int) + 128
    ]
    
    cv2.imshow()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
