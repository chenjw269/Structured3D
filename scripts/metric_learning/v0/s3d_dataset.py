import sys
sys.path.append(".")

import torch
from torch.utils.data import Dataset

import pandas as pd
from scripts.virtual_obs.pesp_vobs import virtual_pesp_obs


class S3DMetricLearning(Dataset):

    def __init__(self, csv_pth):
        super(S3DMetricLearning, self).__init__()

        self.data = pd.read_csv(csv_pth)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # 样本位姿

        # 样本单视角 bev

        # 场景地图

        # 正样例：随机在半径为 0.5m，角度为 15° 范围内采样位姿


        # 负样例：随机在半径为 1 到 *距离边界最远距离，角度为 30 ° 到 330 ° 范围内采样位姿



        return super().__getitem__(index)
    