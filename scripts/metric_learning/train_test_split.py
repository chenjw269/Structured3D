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
    scene_annos_loss = "logs/scene_annos.txt"
    scene_line_err = "logs/scene_line_err.txt"
    scene_obs_err = "logs/scene_observation.txt"
    csv_pth = "e:/datasets/Structure3D_csv/Structured3D"
else:
    scene_annos_loss = "../../logs/scene_annos.txt"
    scene_line_err = "../../logs/scene_line_err.txt"
    scene_obs_err = "../../logs/scene_observation.txt"
    csv_pth = "/data1/chenjiwei/S3D"

# 标注数据缺失的场景
with open(scene_annos_loss, encoding="utf-8") as f: # remote
    scene_invalid = f.readlines()
for index, item in enumerate(scene_invalid):
    scene_invalid[index] = item.replace("\n", "")
# 边线错误的场景
with open(scene_line_err, encoding="utf-8") as f:
    scene_invalid_append = f.readlines()
for index, item in enumerate(scene_invalid_append):
    scene_invalid_append[index] = item.replace("\n", "")
scene_invalid = scene_invalid + scene_invalid_append

# 训练 / 测试划分
train_list = [f"scene_{num:05}" for num in range(3000)]
for scene_index in tqdm(train_list):
    # 缺少标注的场景作废
    if scene_index in scene_invalid:
        train_list.remove(scene_index)
print("Invalid scenes removed from train list")
test_list = [f"scene_{num:05}" for num in range(3000, 3500)]
for scene_index in tqdm(test_list):
    # 缺少标注的场景作废
    if scene_index in scene_invalid:
        test_list.remove(scene_index)
print("Invalid scenes removed from test list")

# 输出路径
output_pth = csv_pth
os.makedirs(output_pth, exist_ok=True)

train_df = pd.DataFrame()
for train_item in tqdm(train_list):
    train_item = os.path.join(csv_pth, train_item, f"metric_learning/{train_item}.csv")
    train_item_df = pd.read_csv(train_item)
    train_df = pd.concat((train_df, train_item_df))
output_csv_train = os.path.join(csv_pth, f"metric_learning_train.csv")
train_df.to_csv(output_csv_train, index=False)
print(f"Output train csv to {output_csv_train}")

test_df = pd.DataFrame()
for test_item in tqdm(test_list):
    test_item = os.path.join(csv_pth, test_item, f"metric_learning/{test_item}.csv")
    test_item_df = pd.read_csv(test_item)
    test_df = pd.concat((test_df, test_item_df))
output_csv_test = os.path.join(csv_pth, f"metric_learning_test.csv")
test_df.to_csv(output_csv_test, index=False)
print(f"Output test csv to {output_csv_test}")
