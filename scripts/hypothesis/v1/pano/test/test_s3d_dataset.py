# 20241025
# 测试：用于验证定位的 Structured3D 数据集
import cv2

from tqdm import tqdm

import torch
torch.manual_seed(42) # 设置 CPU 随机数种子
torch.cuda.manual_seed_all(42) # 设置 GPU 随机数种子
from torch.utils.data import DataLoader

import sys
sys.path.append(".")

# pytorch 数据集
from scripts.hypothesis.v1.pano.s3d_dataset import S3DInference
# 可视化占用网格
from scripts.utils.visualize_occ import visualize_occ
from assets.semantic2label import LABEL_TO_COLOR
# 匹配最近邻地图块
from scripts.utils.patches_matching import patches_matching_positive


if __name__ == "__main__":
    
    test_csv_pth = "e:/datasets/Structure3D_csv/Structured3D/matching_inference_test.csv"
    test_dataset = S3DInference(test_csv_pth)
    print(f"{len(test_dataset)} items in dataset")

    test_batch_size = 1
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)
    
    data = next(iter(test_dataloader))
    local_map = data['local map']
    print(f"Local map {local_map.shape}")
    gt_pos = data['gt pos']
    print(f"Gt pos {gt_pos.shape}")
    pos_hypothesis = data['position hypothesis']
    print(f"Position hypothesis {pos_hypothesis.shape}")
    obs_hypothesis = data['observation hypothesis']
    print(f"Observation hypothesis {obs_hypothesis.shape}")
    
    # # 可视化局部地图和最近观测假设
    # for batch_idx in range(test_batch_size):
        
    #     # 局部地图
    #     local_map_item = local_map[batch_idx]
    #     local_map_item = visualize_occ(local_map_item, LABEL_TO_COLOR)
    #     # cv2.imshow("Local map", local_map_item)

    #     # 查找距离最近的位置假设
    #     gt_pos_item = gt_pos[batch_idx]
    #     pos_hypothesis_item = pos_hypothesis[batch_idx]
    #     nearest_indices = patches_matching_positive(gt_pos_item, pos_hypothesis_item, 1)
    #     # 索引到距离最近的观测假设
    #     obs_hypothesis_item = obs_hypothesis[batch_idx]
    #     nearest_obs = obs_hypothesis_item[nearest_indices][0]
    #     # 可视化距离最近的观测假设
    #     nearest_obs = visualize_occ(nearest_obs, LABEL_TO_COLOR)
    #     # cv2.imshow("Nearest obs", nearest_obs)

    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()

    with torch.no_grad():
        for _ in tqdm(test_dataloader):
            pass
