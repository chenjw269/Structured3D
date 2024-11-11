import sys
sys.path.append(".")

from tqdm import tqdm

from torch.utils.data import DataLoader

from scripts.metric_pesp_v0.s3d_dataset import S3DMetricLearning


if __name__ == "__main__":
    
    test_csv = ""
    test_dataset = S3DMetricLearning(test_csv)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)

    # 批数据
    test_batch = next(iter(test_dataloader))
    # 相机位姿
    test_gt_pose = test_batch['gt pose']
    print(test_gt_pose.shape)
    # 单视角 bev
    test_local_map = test_batch['local map']
    print(test_local_map.shape)
    # cad 地图
    test_global_map = test_batch['global map']
    print(test_global_map.shape)
    # 地图边界
    test_bound = test_batch['bound']
    print(test_bound.shape)

    for _ in tqdm(test_dataloader):
        pass
