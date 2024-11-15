import sys
sys.path.append(".")

import numpy as np
import pandas as pd
from scripts.hypothesis.hypothesis_sample import * # 采样位姿假设

from torch.utils.data import Dataset


class S3DHypothesis(Dataset):
    def __init__(self, csv_pth):
        super(S3DHypothesis, self).__init__()
        
        self.data = pd.read_csv(csv_pth)
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):

        # 该场景下所有的样本位姿和 bev
        sample_pose = eval(self.data['gt pos'][index]) # 样本位姿
        sample_bev = eval(self.data['local map'][index]) # 样本 bev 观测

        # 该场景的地图和边界
        scene_map = self.data['global map'][index] # 当前场景的 cad 地图
        scene_bound_pth = self.data['annos'][index] # 当前场景的坐标边界

        # 该场景的位姿假设
        scene_bound = pd.read_csv(scene_bound_pth)
        p_bound = ((scene_bound['x_min'].item(), scene_bound['x_max'].item()),
                   (scene_bound['y_min'].item(), scene_bound['y_max'].item()))
        o_bound = (0, 360)
        p_step = 250 # 250 mm / 0.25 m
        o_step = 15 # 15 °
        # 均匀采位姿假设        
        pose_hypothesis = gen_pose_hypothesis(p_bound, o_bound, p_step, o_step)
        
        
        data = {
            "global map": scene_map,
            "bound": scene_bound_pth,
            "hypothesis": pose_hypothesis,
            "pose": sample_pose,
            "obs": sample_bev,
        }

        return data