import os
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append(".")


csv_pth = "e:/datasets/Structure3D_csv/Structured3D"

train_list = [f"scene_{num:05}" for num in range(1000)]

# 输出文件夹
output_pth = csv_pth
os.makedirs(output_pth, exist_ok=True)

train_df = pd.DataFrame()
for train_item in tqdm(train_list):
    train_item = os.path.join(csv_pth, train_item, f"metric_learning/{train_item}.csv")
    train_item_df = pd.read_csv(train_item)
    train_df = pd.concat((train_df, train_item_df))
output_csv_train = os.path.join(csv_pth, f"random_train.csv")
train_df.to_csv(output_csv_train, index=False)
print(f"Output train csv to {output_csv_train}")

# test_df = pd.DataFrame()
# for test_item in test_list:
#     test_item = os.path.join(csv_pth, test_item, f"random.csv")
#     test_item_df = pd.read_csv(test_item)
#     test_df = pd.concat((test_df, test_item_df))
# output_csv_test = os.path.join(csv_pth, f"random_test.csv")
# test_df.to_csv(output_csv_test, index=False)
# print(f"Output test csv to {output_csv_test}")
