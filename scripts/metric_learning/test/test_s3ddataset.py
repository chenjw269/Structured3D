# 20241023
# 测试：用于对比学习的 Structured3D 数据集
from tqdm import tqdm

from torch.utils.data import DataLoader

import sys
sys.path.append(".")

from scripts.metric_learning.s3d_dataset import S3D_Dataset


if __name__ == "__main__":

    # test_csv_pth = "e:/datasets/Structure3D_csv/Structured3D/scene_00000/metric_learning/scene_00000.csv"
    test_csv_pth = "e:/datasets/Structure3D_csv/Structured3D/random_train.csv"
    test_dataset = S3D_Dataset(test_csv_pth)

    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True)

    data = next(iter(test_dataloader))
    local_map = data['local map']
    print(f"Local map {local_map.shape}")
    global_map = data['global map']
    print(f"Global map {global_map.shape}")
    gt_pos = data['gt pos']
    print(f"Gt pos {gt_pos.shape}")
    positive = data['positive']
    print(f"Positive {positive.shape}")
    negative = data['negative']
    print(f"Negative {negative.shape}")

    for _ in tqdm(test_dataloader):
        pass
