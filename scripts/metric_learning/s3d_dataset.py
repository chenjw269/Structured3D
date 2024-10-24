# 20241019
# 用于对比学习的 Structured3D 数据集
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

import sys
sys.path.append(".")

from scripts.utils.generate_neighbor import * # 随机采样，随机近邻和远邻
from scripts.utils.extract_local_patches import extract_local_patches # 获取地图上的局部地图


class S3D_Dataset(Dataset):
    def __init__(self, csv_pth):
        super(S3D_Dataset, self).__init__()
        
        self.data = pd.read_csv(csv_pth)
        
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
        
        coord_bound = ((0, 1600), (0, 1600)) # 坐标边界
        
        # 随机正样例位置
        positive_radius = 40 # 40 * 0.025 = 1
        positive_nums = 5
        positive_sample_pos = generate_neighbor_within(
            gt_pos, positive_nums,
            positive_radius,
            coord_bound
        )
        positive_lm = extract_local_patches(global_map, positive_sample_pos)
        positive_lm = torch.Tensor(positive_lm)
        # 随机负样例位置
        negative_radius_1 = 100 # 100 * 0.025 = 2.5
        negative_radius_2 = 200 # 200 * 0.025 = 5
        negative_nums = 5
        negative_sample_pos = generate_neighbor_between(
            gt_pos, negative_nums,
            radius_1=negative_radius_1,
            radius_2=negative_radius_2,
            p_bound=coord_bound
        )
        negative_lm = extract_local_patches(global_map, negative_sample_pos)
        negative_lm = torch.Tensor(negative_lm)
        
        data = {
            "local map": local_map,
            "global map": global_map,
            "gt pos": gt_pos,
            "positive": positive_lm,
            # "normal": normal_lm,
            "negative": negative_lm
        }
        
        return data