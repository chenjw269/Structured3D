# 20241028
# 测试训练集、验证集和测试集的数据是否完整
import sys
sys.path.append(".")

from s3d import *
from scripts.metric_learning.v1.pano.s3d_dataset import S3DMetricLearning # s3d 度量学习数据集

import os # 拼接文件路径
from tqdm import tqdm # 进度条
from torch.utils.data import DataLoader # dataloader

# 训练集数据目录
train_dataset = os.path.join(s3d_csv_pth,  f"metric_learning_train.csv")
train_dataset = S3DMetricLearning(train_dataset)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
for _ in tqdm(train_dataloader):
    pass

# 验证集数据目录
val_dataset = os.path.join(s3d_csv_pth, f"metric_learning_val.csv")
val_dataset = S3DMetricLearning(val_dataset)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True)
for _ in tqdm(val_dataloader):
    pass

# 测试集数据目录
test_dataset = os.path.join(s3d_csv_pth, f"metric_learning_test.csv")
test_dataset = S3DMetricLearning(test_dataset)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)
for _ in tqdm(test_dataloader):
    pass
