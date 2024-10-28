# 20241025
# pytorch s3d 数据集
# 在坐标位置上接近的位置假设，其观测假设与周围局部地图也是相似的
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

import sys
sys.path.append(".")

# 均匀采样位置假设
from scripts.inference.position_hypothesis import generate_scene_hypothesis
# 获取地图上的局部地图
from scripts.utils.extract_local_patches import extract_local_patches


class S3DInference(Dataset):
    
    def __init__(self, csv_pth):
        super(S3DInference, self).__init__()

        self.data = pd.read_csv(csv_pth)
        self.resolution = 25 # 2.5 cm, 0.025 m / pixel

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # 局部地图 Lc,Lc
        local_map = self.data['local map'][index]
        local_map = torch.Tensor(np.load(local_map))
        # 全局地图 Lg,Lg
        global_map = self.data['global map'][index]
        global_map = torch.Tensor(np.load(global_map))
        # 真实位置 2
        gt_pos = eval(self.data['gt pos'][index])
        gt_pos = torch.Tensor(gt_pos)
        
        # 当前场景标注
        scene_annos = self.data['annos'][index]
        scene_annos_df = pd.read_csv(scene_annos)
        # 从场景地图上采样位置假设
        pos_hypothesis = generate_scene_hypothesis(scene_annos_df)

        # 获取位置假设周围的观测假设
        obs_hypothesis = extract_local_patches(global_map, pos_hypothesis)
        
        data = {
            "local map": local_map, # 256,256,
            "gt pos": gt_pos, # 2
            "position hypothesis": pos_hypothesis, # N,2
            "observation hypothesis": obs_hypothesis # N,256,256
        }
        
        return data
