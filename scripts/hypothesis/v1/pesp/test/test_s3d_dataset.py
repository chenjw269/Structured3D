import sys
sys.path.append(".")

import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm # 进度条
from s3d import *
from scripts.hypothesis.v1.pesp.s3d_dataset import S3DHypothesis
from torch.utils.data import DataLoader

from scripts.utils.read_camera_pose import read_camera_pose # 读取位姿
# 坐标转换
from scripts.cad_mapping.v1.coord_conv import position_to_pixel, position_to_pixel_batch
from scripts.hypothesis.hypothesis_sample import pose_matching # 匹配最近邻位姿
from scripts.virtual_obs.pesp_vobs import virtual_pesp_obs_batch # 虚拟观测
from scripts.utils.visualize_occ import * # 可视化占用网格

from scripts.ft_encoder.cnn_encoder.basic_cnn import FCN


if __name__ == "__main__":
    
    test_csv_pth = "e:/datasets/Structure3D_csv/Structured3D/metric_v1/train.csv"
    # test_csv_pth = "e:/datasets/Structure3D_csv/Structured3D\metric_v1/val.csv"
    # test_csv_pth = "e:/datasets/Structure3D_csv/Structured3D\metric_v1/test.csv"

    test_dataset = S3DHypothesis(test_csv_pth)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    test_batch = next(iter(test_dataloader))
    print(f"Pose {len(test_batch['pose'])}")
    print(f"Observation {len(test_batch['obs'])}")
    # 场景地图
    scene_map = np.load(test_batch['global map'][0])
    print(f"Global map {scene_map.shape}")
    # 场景边界
    scene_bound = pd.read_csv(test_batch['bound'][0])
    # 位姿假设
    pose_hypothesis = np.array(test_batch['hypothesis'][0])
    print(f"Hypothesis {test_batch['hypothesis'].shape}")

    ##############################################
    # 随机选择该场景下的一个样本
    ##############################################    
    index = 0 # 样本索引
    # 真实观测 bev
    obs_image = np.load(test_batch['obs'][index][0])
    obs_image = visualize_occ(obs_image, LABEL_TO_COLOR)
    cv2.imwrite("logs/hypothesis/obs.png", obs_image)
    # 真实位姿
    sample_pose = np.loadtxt(test_batch['pose'][index][0])
    sample_pose = read_camera_pose(sample_pose)
    sample_pose[2] = ((sample_pose[2] - 90) + 360) % 360

    ###############################################
    # 最近邻的位姿假设
    ###############################################
    # 最近邻样例
    nearest_hypothesis = pose_matching(
        pose=sample_pose, hypothesis=pose_hypothesis, k=3
    )
    nearest_hypothesis = pose_hypothesis[nearest_hypothesis]

    # 坐标转换
    nearest_position = position_to_pixel_batch(
        nearest_hypothesis[:, :2], resolution, scene_bound
    )
    nearest_hypothesis[:, :2] = nearest_position
    # 虚拟观测
    virtual_obs = virtual_pesp_obs_batch(
        map_image=scene_map, pose=nearest_hypothesis, fov=80
    )
    for n_idx, i in enumerate(nearest_hypothesis):
        virtual_obs_item = visualize_occ(virtual_obs[n_idx], LABEL_TO_COLOR)
        cv2.imwrite(f"logs/hypothesis/nearest_{n_idx}.png", virtual_obs_item)

    ###############################################
    # 内存占用估计
    ###############################################

    # 1. 样本观测及特征提取
    obs_list = test_batch['obs']
    obs_total = torch.Tensor().cuda()
    for obs_item in obs_list:
        # 增加 channel 维度和 batch 维度
        obs_item = torch.Tensor(np.load(obs_item[0])).unsqueeze(dim=0).unsqueeze(dim=0).cuda()
        # 提取特征
        
        obs_total = torch.concat((obs_total, obs_item), axis=0)

    # 2. 观测假设及特征提取
    model = FCN().cuda()

    batch_size = 256
    N = (pose_hypothesis.shape[0] // batch_size) + 1
    for i in range(N):
        # 批量获取位姿假设
        if (i+1)*batch_size > pose_hypothesis.shape[0]:
            batch_pose_hypothesis = pose_hypothesis[i*batch_size :]
        else:
            batch_pose_hypothesis = pose_hypothesis[i*batch_size : (i+1)*batch_size]
        # 坐标转换
        batch_position = position_to_pixel_batch(
            batch_pose_hypothesis[:, :2], resolution, scene_bound
        )
        batch_pose_hypothesis[:, :2] = batch_position
        # 批量获取观测假设
        batch_virtual_obs = virtual_pesp_obs_batch(scene_map, batch_pose_hypothesis, fov=80)
        batch_virtual_obs = torch.Tensor(batch_virtual_obs).unsqueeze(dim=1).cuda()
        # 批量提取特征
        batch_ft = model(batch_virtual_obs)

        # TODO: 计算相似度
        
        # TODO: 记录相似度

    print(f"Memory {torch.cuda.memory_allocated() / 1024 / 1024}")