# 极端情况下的虚拟观测
# 1. 视野范围超出地图坐标边界
# 
import sys
sys.path.append(".")

import os
import copy
from s3d import *
import numpy as np
import pandas as pd

from scripts.utils.read_camera_pose import read_camera_pose # 读取样本位姿
from scripts.cad_mapping_v1.coord_conv import position_to_pixel # 坐标转换

from scripts.utils.view_range_utils import generate_ellipse_mask # 虚拟观测矩形
from scripts.virtual_obs.pesp_vobs import * # 虚拟单视角观测

from scripts.utils.visualize_occ import * # 可视化占用网格


if __name__ == "__main__":
    
    scene_index = "scene_00000"
    sample_index = ("485142", "0")
    
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
    sample_pose[2] = 180 # 朝向 yaw 角
    
    print(f"{sample_pose}")
    # 坐标转换
    sample_position = sample_pose[:2]
    sample_position = position_to_pixel(sample_position, resolution, scene_bound)
    sample_pose = np.append(sample_position, sample_pose[2]-90)
    print(f"{sample_pose}")

    fov = 80

    ############################################
    # 虚拟观测的视野
    ############################################
    map_image_cp = copy.copy(scene_map)

    # 将视野外的内容覆盖
    view_range_mask = generate_ellipse_mask(map_image_cp, sample_pose[:2].astype(int), sample_pose[2], fov, "occ")
    map_image_cp = view_range_mask * map_image_cp
    map_image_cp = visualize_occ(map_image_cp, LABEL_TO_COLOR)

    # 显示视野边界，方便观察
    start_angle = sample_pose[2] - fov/2
    end_angle = sample_pose[2] + fov/2
    cv2.ellipse(map_image_cp, sample_pose[:2].astype(int), (256,256), 0, start_angle, end_angle, (0,0,0), thickness=1)

    cv2.imshow("Virtual Fov", map_image_cp)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
