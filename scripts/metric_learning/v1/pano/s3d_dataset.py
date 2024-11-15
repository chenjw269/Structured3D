# 20241019
# 用于对比学习的 Structured3D 数据集
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

import sys
sys.path.append(".")

from scripts.metric_learning.neighbor_sample_v0 import * # 随机采样，随机近邻和远邻
from scripts.utils.extract_local_patches import extract_local_patches # 获取地图上的局部地图


class S3DPanoramaMC(Dataset):
    def __init__(self, csv_pth):
        super(S3DPanoramaMC, self).__init__()
        
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
        # 场景坐标范围
        scene_bound = np.array([
            [128, int(scene_annos_df['size_x'].item() / self.resolution) - 128],
            [128, int(scene_annos_df['size_y'].item() / self.resolution) - 128]
        ])

        # 样本位置到地图边界的最大距离
        d_x_min = gt_pos[0] - scene_bound[0][0]
        d_x_max = scene_bound[0][1] - gt_pos[0]
        d_y_min = gt_pos[1] - scene_bound[1][0]
        d_y_max = scene_bound[1][1] - gt_pos[1]
        d_max = max([d_x_min, d_x_max, d_y_min, d_y_max])

        # 随机正样例位置
        positive_radius = 60 # 60 * 0.025 = 1.5
        positive_nums = 5
        positive_sample_pos = generate_neighbor_within(
            gt_pos, positive_nums, positive_radius, scene_bound
        )
        positive_lm = extract_local_patches(global_map, positive_sample_pos)
        positive_lm = torch.Tensor(positive_lm)

        # 随机负样例位置
        negative_radius_1 = 140 # 140 * 0.025 = 3.5
        # 如果地图较小，很难获取 3.5 m 以上距离的随机位置，则获取尽可能远的随机位置
        if d_max > negative_radius_1:
            pass
        else:
            negative_radius_1 = d_max
        negative_radius_2 = 280 # 280 * 0.025 = 7
        negative_nums = 5
        negative_sample_pos = generate_neighbor_between(
            gt_pos, negative_nums,
            radius_1=negative_radius_1,
            radius_2=negative_radius_2,
            p_bound=scene_bound
        )
        negative_lm = extract_local_patches(global_map, negative_sample_pos)
        negative_lm = torch.Tensor(negative_lm)

        assert positive_lm.shape == (positive_nums, 256, 256)
        assert negative_lm.shape == (negative_nums, 256, 256)

        data = {
            "local map": local_map,
            "gt pos": gt_pos,
            "positive": positive_lm,
            # "normal": normal_lm,
            "negative": negative_lm
        }
        
        return data