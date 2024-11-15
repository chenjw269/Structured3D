import sys
sys.path.append(".")

import cv2
from tqdm import tqdm # 进度条
from scripts.metric_learning.v1.pesp.s3d_dataset import S3DMetricV1
from torch.utils.data import DataLoader

from scripts.utils.visualize_occ import * # 可视化占用网格


if __name__ == "__main__":
    
    test_csv_pth = "e:/datasets/Structure3D_csv/Structured3D/metric_v1/train.csv"
    # test_csv_pth = "e:/datasets/Structure3D_csv/Structured3D\metric_v1/val.csv"
    # test_csv_pth = "e:/datasets/Structure3D_csv/Structured3D\metric_v1/test.csv"

    test_dataset = S3DMetricV1(test_csv_pth)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    test_batch = next(iter(test_dataloader))
    print(f"Anchor {test_batch['anchor'].shape}")
    print(f"Positive {test_batch['positive'].shape}")
    print(f"Negative {test_batch['negative'].shape}")
    
    index = 0
    # 真实观测 bev
    anchor_image = visualize_occ(test_batch['anchor'][index], LABEL_TO_COLOR)
    cv2.imwrite("logs/metric_learning/anchor.png", anchor_image)
    # 正样例
    for p_idx, i in enumerate(range(test_batch['positive'][index].shape[0])):
        positive_image = visualize_occ(test_batch['positive'][index][p_idx], LABEL_TO_COLOR)
        cv2.imwrite(f"logs/metric_learning/positive_{p_idx}.png", positive_image)
    # 负样例
    for p_idx, i in enumerate(range(test_batch['negative'][index].shape[0])):
        negative_image = visualize_occ(test_batch['negative'][index][p_idx], LABEL_TO_COLOR)
        cv2.imwrite(f"logs/metric_learning/negative_{p_idx}.png", negative_image)

    for _ in tqdm(test_dataloader):
        pass
