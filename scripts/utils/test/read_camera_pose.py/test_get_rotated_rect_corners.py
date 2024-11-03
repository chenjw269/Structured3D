# 获取视野范围矩形的角点
import sys
sys.path.append(".")

import os
import cv2
from s3d import *
import numpy as np
import pandas as pd
from scripts.utils.virtual_obs import get_rotated_rect_corners # 获取视野矩形的角点
from scripts.utils.read_camera_pose import read_camera_pose # 读取相机位姿
from scripts.cad_mapping_v1.coord_conv import position_to_pixel # 坐标变换


if __name__ == "__main__":

    test_scene_index = "scene_00000"
    test_sample_index = "485142"

    test_scene_map = os.path.join(s3d_map_pth, test_scene_index, "map.png")
    test_scene_map = cv2.imread(test_scene_map) # 814,814
    
    test_scene_bound = os.path.join(s3d_data_pth, test_scene_index, "boundary.csv")
    test_scene_bound = pd.read_csv(test_scene_bound)

    test_sample_pose = os.path.join(s3d_data_pth, test_scene_index, "2D_rendering", test_sample_index, "perspective/full/0/camera_pose.txt")
    test_sample_pose = read_camera_pose(test_sample_pose, mode="raw")

    # 坐标转换
    test_sample_position = test_sample_pose[:2]
    test_sample_position = position_to_pixel(test_sample_position, resolution, test_scene_bound)
    test_sample_pose = np.append(test_sample_position, test_sample_pose[2])

    rotated_corners = get_rotated_rect_corners(test_sample_pose).astype(int)
    
    for i in range(4):
        test_scene_map = cv2.line(
            test_scene_map, 
            pt1=rotated_corners[i], pt2=rotated_corners[(i+1) % 4],
            color=(0,0,255), thickness=1
        )
    
    cv2.imshow("View range", test_scene_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    