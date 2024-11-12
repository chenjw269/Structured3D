import pandas as pd

from torch.utils.data import Dataset


class S3DMetricV1(Dataset):
    def __init__(self, csv_pth):
        super(S3DMetricV1, self).__init__()
        
        self.data = pd.read_csv(csv_pth)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):

        # 1. 随机选取该场景下的一个样本
        
        # 该场景下所有的样本位姿和 bev
        sample_pos_list = self.data["gt pos"][index]
        sample_obs_list = self.data["local map"][index]
        
        
        # 2. 采样近邻和远邻
        
        # 近邻和远邻位姿
        
        # 坐标转换
        
        # 场景地图
        
        # 获取虚拟观测
        
        # 返回数据

        