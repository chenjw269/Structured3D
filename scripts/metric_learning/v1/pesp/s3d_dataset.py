import random
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

import sys
sys.path.append(".")

from s3d import *
from scripts.utils.read_camera_pose import read_camera_pose # 读取位姿
# 坐标转换
from scripts.cad_mapping.v1.coord_conv import position_to_pixel, position_to_pixel_batch
from scripts.metric_learning.neighbor_sample import * # 采样远近邻
from scripts.virtual_obs.pesp_vobs import virtual_pesp_obs_batch # 虚拟观测


class S3DMetricV1(Dataset):
    def __init__(self, csv_pth):
        super(S3DMetricV1, self).__init__()
        
        self.data = pd.read_csv(csv_pth)
        
        self.fov = 80
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):

        # 该场景下所有的样本位姿和 bev
        sample_pose_list = eval(self.data["gt pos"][index]) # 样本位姿
        sample_obs_list = eval(self.data["local map"][index]) # 样本 bev 观测
        # 该场景的地图和边界
        scene_map = np.load(self.data['global map'][index]) # np.array 当前场景的 cad 地图
        scene_bound = pd.read_csv(self.data['annos'][index]) # pd.DataFrame 当前场景的坐标边界

        ########################################
        # 1. 随机选取该场景下的一个样本
        ########################################
        
        # 随机选取其中一个样本
        if len(sample_pose_list) == 1:
            sample_index = 0
        else:
            # randint 取值范围是闭区间，所以右边界要 -1
            sample_index = random.randint(0, len(sample_pose_list)-1)

        ########################################
        # 2. 读取观测数据
        ########################################
        
        # 读取位姿
        sample_pose = sample_pose_list[sample_index]
        sample_pose = np.loadtxt(sample_pose)
        sample_pose = read_camera_pose(sample_pose, mode="raw")
        # 因为
        sample_pose[2] = sample_pose[2] - 90
        # 坐标转换
        sample_position = sample_pose[:2]
        sample_position = position_to_pixel(sample_position, resolution, scene_bound)
        sample_pose_pixel = np.append(sample_position, sample_pose[2])
        
        # 读取观测
        sample_obs = sample_obs_list[sample_index] # -769.331, 2277.85, -179.09909699
        sample_obs = np.load(sample_obs) # np.array 当前样本的 bev 观测

        ########################################
        # 3. 采样近邻和远邻
        ########################################

        near_nums = 5 # 近邻位姿的数量
        near_d_radius = 100 # 0.1 米 / 100 mm 以内为近邻
        near_angle_radius = 15 # 15 度以内为近邻

        far_nums = 5 # 远邻位姿的数量
        far_angle_radius = 30 # 30 度以外为远邻
        far_d_radius_1 = 500 # 0.5 米 / 500 mm 以内为远邻
        # 样本位置到地图边界的最大距离
        d_x_min = sample_pose[0] - scene_bound['x_min'].item()
        d_x_max = scene_bound['x_max'].item() - sample_pose[0]
        d_y_min = sample_pose[1] - scene_bound['y_min'].item()
        d_y_max = scene_bound['y_max'].item() - sample_pose[1]
        d_max = max([d_x_min, d_x_max, d_y_min, d_y_max])
        # 通过比较样本位置和不同边界的距离，确定采样远邻的范围
        far_d_radius_2 = d_max

        bound_list = ((scene_bound['x_min'].item(), scene_bound['x_max'].item()),
                      (scene_bound['y_min'].item(), scene_bound['y_max'].item()))

        # 1. 近邻位姿
        near_pos = gen_position_near_neighbor(
            position=sample_pose[:2], nums=near_nums,
            radius=near_d_radius, bound=bound_list
        )
        near_ori = gen_orientation_near_neighbor(
            orientation=sample_pose[2], nums=near_nums, radius=near_angle_radius
        )
        # 位置坐标转换
        near_pos = position_to_pixel_batch(near_pos, resolution, scene_bound)
        # 位姿 = 位置 + 朝向
        near_pose = np.concatenate(
            (near_pos, np.expand_dims(near_ori, axis=1)), axis=1)
        # 虚拟观测
        positive = virtual_pesp_obs_batch(scene_map, near_pose, fov=self.fov)

        # 2. 位置远邻位姿
        far_pos = gen_position_far_neighbor(
            position=sample_pose[:2], nums=far_nums,
            radius_1=far_d_radius_1, radius_2=far_d_radius_2, bound=bound_list
        )
        near_ori = gen_orientation_near_neighbor(
            orientation=sample_pose[2], nums=far_nums, radius=near_angle_radius
        )
        # 坐标转换
        far_pos = position_to_pixel_batch(far_pos, resolution, scene_bound)
        far_pose = np.concatenate(
            (far_pos, np.expand_dims(near_ori, axis=1)), axis=1)
        # 虚拟观测
        negative_1 = virtual_pesp_obs_batch(scene_map, far_pose, fov=self.fov)

        # 3. 朝向远邻位姿
        near_pos = gen_position_near_neighbor(
            position=sample_pose[:2], nums=far_nums,
            radius=near_d_radius, bound=bound_list
        )
        far_ori = gen_orientation_far_neighbor(
            orientation=sample_pose[2], nums=far_nums, radius=far_angle_radius
        )
        # 坐标转换
        near_pos = position_to_pixel_batch(near_pos, resolution, scene_bound)
        far_pose = np.concatenate(
            (near_pos, np.expand_dims(far_ori, axis=1)), axis=1)
        # 虚拟观测
        negative_2 = virtual_pesp_obs_batch(scene_map, far_pose, fov=self.fov)
        
        negative = np.concatenate((negative_1, negative_2), axis=0)
        
        # 返回数据
        data = {
            "anchor": sample_obs,
            "positive": positive,
            "negative": negative
        }
        return data
        