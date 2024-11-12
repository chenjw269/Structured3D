import pandas as pd

from torch.utils.data import Dataset


class S3DHypothesis(Dataset):
    def __init__(self, csv_pth):
        super(S3DHypothesis, self).__init__()
        
        self.data = pd.read_csv(csv_pth)
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        # cad 场景地图
        
        # cad 坐标边界
        
        data = {
            "global map": None,
            "bound": None
        }

        return super().__getitem__(index)