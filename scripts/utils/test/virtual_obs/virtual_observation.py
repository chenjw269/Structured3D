# 虚拟位姿下的观测值

import sys
sys.path.append(".")

import cv2
import copy
import numpy as np
from scripts.utils.read_sample_info import read_sample_info # 读取样本位姿
from scripts.utils.virtual_obs import get_rotated_rect_corners # 从地图上获取视野矩形
from scripts.utils.virtual_obs import virtual_observation # 从地图上获取虚拟观测
from scripts.utils.virtual_obs import virtual_observation # 从地图上获取虚拟观测
from scripts.utils.visualize_occ import * # 可视化占用网格


if __name__ == "__main__":
    
    test_scene_index = "scene_00000"
    test_sample_index = ("485142", "0")
    
    sample_info = read_sample_info(test_scene_index, test_sample_index, mode="gravity")
    # sample_info = read_sample_info(test_scene_index, test_sample_index, mode="raw")

    # 场景地图
    scene_map = sample_info["scene map array"]
    scene_map = np.load(scene_map)

    # 样本位姿
    sample_pose = sample_info["sample pose"]
    
    #########################################
    # 视野范围
    #########################################
    scene_map_cp = visualize_occ(copy.copy(scene_map))
    
    # 可视化样本位置
    scene_map_cp = cv2.circle(scene_map_cp, sample_pose[:2].astype(int), radius=3, color=(0,0,0), thickness=-1)
    scene_map_cp = cv2.circle(scene_map_cp, sample_pose[:2].astype(int), radius=9, color=(0,0,0), thickness=1)
    scene_map_cp = cv2.circle(scene_map_cp, sample_pose[:2].astype(int), radius=15, color=(0,0,0), thickness=1)    
    scene_map_cp = cv2.circle(scene_map_cp, sample_pose[:2].astype(int), radius=21, color=(0,0,0), thickness=1)    

    # 获取视野范围矩形
    view_range_corner = get_rotated_rect_corners(sample_pose, width=128, height=256)
    
    # 可视化视野范围矩形
    for i in range(4):
        scene_map_cp = cv2.line(
            img=scene_map_cp,
            pt1=view_range_corner[i].astype(int), pt2=view_range_corner[(i+1)%4].astype(int),
            color=(255,0,0), thickness=1
        )
    # cv2.imshow("View range rectangle", scene_map_cp)
    cv2.imwrite("logs/view_range.png", scene_map_cp)

    #########################################
    # 完整视野内容
    #########################################
    virtual_obs = virtual_observation(scene_map, sample_pose)
    virtual_obs_img = visualize_occ(virtual_obs, LABEL_TO_COLOR)
    
    # cv2.imshow("Virtual obs", virtual_obs_img)
    cv2.imwrite("logs/full_view.png", virtual_obs_img)

    #########################################
    # 受限视角视野内容
    #########################################
    virtual_obs_fov = virtual_observation(scene_map, sample_pose, 80)
    virtual_obs_fov_img = visualize_occ(virtual_obs_fov, LABEL_TO_COLOR)

    # cv2.imshow("Virtual obs fov", virtual_obs_fov_img)
    cv2.imwrite("logs/fov_view.png", virtual_obs_fov_img)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    