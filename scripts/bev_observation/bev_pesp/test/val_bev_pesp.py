# 验证单视角视图
import sys
sys.path.append(".")

import os
import cv2
import copy
import random
from s3d import *
import numpy as np
import pandas as pd

from scripts.utils.read_camera_pose import read_camera_pose # 读取样本位姿
from scripts.cad_mapping.v1.coord_conv import position_to_pixel # 坐标转换

from scripts.virtual_obs.pesp_vobs import virtual_pesp_viewrect # 虚拟视野矩形
from scripts.virtual_obs.pesp_vobs import virtual_pesp_obs # 虚拟观测
from scripts.utils.visualize_occ import * # 可视化占用网格


if __name__ == "__main__":
    
    ##########################################
    # 随机场景、样本、朝向
    ##########################################
    validation_index_list = []

    # 从全部场景中随机选择 5 个场景
    scene_index_list = [f"scene_{num:05}" for num in range(100)]    
    scene_index_list = random.sample(scene_index_list, 5)

    # 遍历所有场景    
    for scene_index in scene_index_list:
        # 从全部样本中随机选择 2 个样本
        scene_data_pth = os.path.join(s3d_data_pth, scene_index, "2D_rendering")
        sample_index_list = os.listdir(scene_data_pth)
        if len(sample_index_list) > 1:
            sample_index_list = random.sample(sample_index_list, 2)

        # 遍历所有样本        
        for sample_index in sample_index_list:
            # 从全部朝向中随机选择 1 个朝向
            sample_data_pth = os.path.join(scene_data_pth, sample_index, "perspective/full")
            ori_index_list = os.listdir(sample_data_pth)
            ori_index_list = random.sample(ori_index_list, 1)
            
            # 遍历所有朝向
            for ori_index in ori_index_list:
                
                validation_index_list.append(
                    (
                        scene_index,
                        f"{scene_index}/2D_rendering/{sample_index}/perspective/full/{ori_index}"
                    )
                )

    ############################################
    # 验证单视角 bev 和位姿
    ############################################    
    for scene_index, sample_index in validation_index_list:
        
        print(f"Sample {sample_index}")

        # 场景地图
        scene_map = np.load(os.path.join(s3d_map_pth, scene_index, "map.npy"))
        # 场景边界
        scene_bound = pd.read_csv(os.path.join(s3d_annos_pth, scene_index, "boundary.csv"))

        # 样本位姿
        try:
            sample_pose = os.path.join(s3d_data_pth, sample_index, "camera_pose.txt")
            sample_pose = np.loadtxt(sample_pose)
        except Exception as e:
            print(f"Pose file: {sample_pose}")
        sample_pose = read_camera_pose(sample_pose, mode="raw")
        print(f"Sample pose (mm) {sample_pose}")
        
        # 坐标转换
        sample_position = sample_pose[:2]
        sample_position = position_to_pixel(sample_position, resolution, scene_bound)
        sample_pose = np.append(sample_position, (sample_pose[2]-90) % 360)
        print(f"Sample pose (pixel) {sample_pose}")

        #####################################
        # 在地图上标注位姿
        #####################################
        sample_on_map = copy.copy(scene_map)
        sample_on_map = visualize_occ(sample_on_map, LABEL_TO_COLOR)
        sample_on_map = cv2.circle(sample_on_map, sample_position, 3, (0,0,0), -1)
        sample_on_map = cv2.circle(sample_on_map, sample_position, 6, (0,0,0), 1)
        sample_on_map = cv2.circle(sample_on_map, sample_position, 9, (0,0,0), 1)

        ######################################
        # 在地图上标注视野扇形
        ######################################
        # 扇形的开始和结束角度，角度范围从 x 轴正方向开始
        start_angle = sample_pose[2] - 80/2
        end_angle = sample_pose[2] + 80/2
        # 绘制扇形圆弧
        radius = 256
        sample_on_map = cv2.ellipse(
            sample_on_map,
            sample_pose[:2].astype(int), (radius,radius), 0, 
            start_angle, end_angle, color=(0,0,0), thickness=2)
        # 绘制扇形半径
        center_x, center_y = sample_pose[0].astype(int), sample_pose[1].astype(int)
        start_line = (int(center_x + radius * np.cos(np.radians(start_angle))), int(center_y + radius * np.sin(np.radians(start_angle))))
        end_line = (int(center_x + radius * np.cos(np.radians(end_angle))), int(center_y + radius * np.sin(np.radians(end_angle))))
        cv2.line(sample_on_map, (center_x, center_y), start_line, (0, 0, 0), 2)
        cv2.line(sample_on_map, (center_x, center_y), end_line, (0, 0, 0), 2)

        #######################################
        # 在地图上标注视野矩形
        #######################################
        view_on_map = copy.copy(scene_map)
        view_on_map = visualize_occ(view_on_map, LABEL_TO_COLOR)
        view_on_map = virtual_pesp_viewrect(view_on_map, sample_pose, fov=80)
        cv2.imshow("View range rect", view_on_map)

        #######################################
        # 从地图获取虚拟观测
        #######################################
        # 虚拟观测
        sample_virtual_view = virtual_pesp_obs(scene_map, sample_pose, fov=80)
        sample_virtual_view = visualize_occ(sample_virtual_view, LABEL_TO_COLOR)
        # 样本 bev image
        sample_bev_img = os.path.join(s3d_bev_pth, sample_index, "bev.png")
        sample_bev_img = cv2.imread(sample_bev_img)
        # 样本 bev array
        sample_bev_array = os.path.join(s3d_bev_pth, sample_index, "bev.npy")
        sample_bev_array = np.load(sample_bev_array)
        
        ###################################
        # 可视化
        ###################################
        # 样本在地图上的位置
        cv2.imshow("Sample on map", sample_on_map)
        # 虚拟观测
        cv2.imshow("Virtual view", sample_virtual_view)
        # 样本 bev image
        cv2.imshow("Pesp bev img", sample_bev_img)
        # # 样本 bev array
        # cv2.imshow("Pesp bev array", sample_bev_img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
