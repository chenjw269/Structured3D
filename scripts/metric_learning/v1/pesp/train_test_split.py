import sys
sys.path.append(".")

from s3d import * # s3d 数据集信息

import os # 拼接文件路径
import pandas as pd # 数据目录保存为表
from tqdm import tqdm # 进度条


def merge_csv(scene_index_list):

    total_df = pd.DataFrame()
    for scene_index in tqdm(scene_index_list):
        scene_item = os.path.join(
            s3d_csv_pth, scene_index, f"metric_v1/{scene_index}_pesp.csv")
        scene_item_df = pd.read_csv(scene_item)
        total_df = pd.concat((total_df, scene_item_df))

    return total_df

# 数据目录存储文件夹
os.makedirs(os.path.join(s3d_csv_pth, "metric_v1"), exist_ok=True)

##########################################
# 训练 / 验证 / 测试划分
##########################################
# 训练集
# train_list = [f"scene_{num:05}" for num in range(3000)]
train_list = [f"scene_{num:05}" for num in range(80)]
# 排除数据缺失的场景
for scene_index in tqdm(train_list):
    # 缺少标注的场景作废
    if scene_index in scene_invalid:
        train_list.remove(scene_index)
print("Invalid scenes removed from train list")
# 合并数据集的 csv 文件
train_df = merge_csv(train_list)
output_csv_train = os.path.join(s3d_csv_pth, "metric_v1/train.csv")
train_df.to_csv(output_csv_train, index=False)
print(f"Output train csv to {output_csv_train}")

# 验证集
# val_list = [f"scene_{num:05}" for num in range(3000, 3250)]
val_list = [f"scene_{num:05}" for num in range(60, 80)]
for scene_index in tqdm(val_list):
    # 缺少标注的场景作废
    if scene_index in scene_invalid:
        val_list.remove(scene_index)
print("Invalid scenes removed from val list")
val_df = merge_csv(val_list)
output_csv_val = os.path.join(s3d_csv_pth, "metric_v1/val.csv")
val_df.to_csv(output_csv_val, index=False)
print(f"Output val csv to {output_csv_val}")

# 测试集
# test_list = [f"scene_{num:05}" for num in range(3250, 3500)]
test_list = [f"scene_{num:05}" for num in range(80, 100)]
for scene_index in tqdm(test_list):
    # 缺少标注的场景作废
    if scene_index in scene_invalid:
        test_list.remove(scene_index)
print("Invalid scenes removed from test list")
test_df = merge_csv(test_list)
output_csv_test = os.path.join(s3d_csv_pth, "metric_v1/test.csv")
test_df.to_csv(output_csv_test, index=False)
print(f"Output test csv to {output_csv_test}")
