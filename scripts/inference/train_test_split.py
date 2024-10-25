import os
import platform
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append(".")

# 获取系统类型
system_type = platform.system()
# 本地路径和服务器路径
if system_type == 'Windows':
    csv_pth = "e:/datasets/Structure3D_csv/Structured3D"
else:
    csv_pth = "/data1/chenjiwei/S3D/Structure3D_csv/Structured3D"


# 训练 / 测试划分
train_list = [f"scene_{num:05}" for num in range(100)]
test_list = [f"scene_{num:05}" for num in range(100, 105)]

# 输出路径
output_pth = csv_pth
os.makedirs(output_pth, exist_ok=True)

train_df = pd.DataFrame()
for train_item in tqdm(train_list):
    train_item = os.path.join(csv_pth, train_item, f"inference/{train_item}.csv")
    train_item_df = pd.read_csv(train_item)
    train_df = pd.concat((train_df, train_item_df))
output_csv_train = os.path.join(csv_pth, f"inference_train.csv")
train_df.to_csv(output_csv_train, index=False)
print(f"Output train csv to {output_csv_train}")

test_df = pd.DataFrame()
for test_item in tqdm(test_list):
    test_item = os.path.join(csv_pth, test_item, f"inference/{test_item}.csv")
    test_item_df = pd.read_csv(test_item)
    test_df = pd.concat((test_df, test_item_df))
output_csv_test = os.path.join(csv_pth, f"inference_test.csv")
test_df.to_csv(output_csv_test, index=False)
print(f"Output test csv to {output_csv_test}")
