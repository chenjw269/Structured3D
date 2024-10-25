# 20241023
# 测试：用于对比学习的 Structured3D 数据集
import cv2
from tqdm import tqdm

from torch.utils.data import DataLoader

import sys
sys.path.append(".")

from scripts.metric_learning.s3d_dataset import S3D_Dataset
from scripts.utils.visualize_occ import visualize_occ # 可视化占用网格
from assets.semantic2label import LABEL_TO_COLOR


if __name__ == "__main__":

    test_csv_pth = "e:/datasets/Structure3D_csv/Structured3D/metric_learning_train.csv"
    test_dataset = S3D_Dataset(test_csv_pth)
    print(f"{len(test_dataset)} items in dataset")

    test_batch_size = 2
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

    data = next(iter(test_dataloader))
    local_map = data['local map']
    print(f"Local map {local_map.shape}")
    gt_pos = data['gt pos']
    print(f"Gt pos {gt_pos.shape}")
    positive = data['positive']
    print(f"Positive {positive.shape}")
    negative = data['negative']
    print(f"Negative {negative.shape}")

    # 可视化局部地图、正样例和负样例
    for batch_idx in range(test_batch_size):
        
        # 局部地图
        local_map_item = local_map[batch_idx]
        local_map_item = visualize_occ(local_map_item, LABEL_TO_COLOR)
        cv2.imshow("Local map", local_map_item)

        positive_item = positive[batch_idx]
        for index, i in enumerate(positive_item):
            i = visualize_occ(i, LABEL_TO_COLOR)
            cv2.imshow(f"Positive {index}", i)

        negative_item = negative[batch_idx]
        for index, i in enumerate(negative_item):
            i = visualize_occ(i, LABEL_TO_COLOR)
            cv2.imshow(f"Negative {index}", i)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    for _ in tqdm(test_dataloader):
        pass
