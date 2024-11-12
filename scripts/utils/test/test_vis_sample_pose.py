import sys
sys.path.append(".")

import os
import cv2
from s3d import *
import pandas as pd
from scripts.utils.read_camera_pose import *
from scripts.utils.vis_sample_pose import *
from scripts.cad_mapping_v1.coord_conv import position_to_pixel


if __name__ == "__main__":
    
    test_scene_index = "scene_00000"
    test_sample_index = "485142"

    # 1. 可视化样本位置
    
    # 2. 可视化样本位姿（包括朝向）
    # 读取场景地图
    scene_map = os.path.join(s3d_map_pth, test_scene_index, "map.png")
    scene_map = cv2.imread(scene_map)
    # 读取样本位姿信息
    sample_pos = os.path.join(
        s3d_data_pth, test_scene_index, "2D_rendering", test_sample_index, "perspective/full/0/camera_pose.txt"
    )
    sample_pos = np.loadtxt(sample_pos)
    # 样本位置
    transistion = sample_pos[:3]
    # 样本朝向
    direction = sample_pos[3:6]
    up = sample_pos[6:9]
    # 1. 使用重力方向校正
    # roll, pitch, yaw = compute_euler_angles_wgravity(direction, up)
    # 2. 不使用重力方向校正
    roll, pitch, yaw = compute_euler_angles(direction)
    yaw = - yaw
    
    # 样本在地图图像上的像素位姿
    scene_bound = os.path.join(s3d_data_pth, test_scene_index, "boundary.csv")
    scene_bound = pd.read_csv(scene_bound)
    transistion = position_to_pixel(transistion[:2], resolution=25, scene_boundary=scene_bound)
    pose = np.append(transistion, yaw)
    # 在地图上可视化样本
    poses = np.expand_dims(pose, axis=0) # 增加一维 (函数的输入是位姿列表)
    scene_map = vis_sample_pose(scene_map, poses)

    cv2.imshow("Sample on map", scene_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

