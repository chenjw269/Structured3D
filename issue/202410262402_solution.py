# Solution
# 

import sys
sys.path.append(".")

from s3d import * # s3d 数据集信息

import time # 测量程序运行时间

import torch # 向量数组
from torch import nn # 神经网络层
import torch.nn.functional as F # 相似度函数


class CNN(nn.Module):
    
    def __init__(self, output_dim=1024):
        super(CNN, self).__init__()
        
        self.features = nn.Sequential(
            # layer 1
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # layer 2
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # layer 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
        )
        # 自适应层，将上一层的数据转换成6x6大小
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        # 全连接层
        self.linear1 = nn.Sequential(
            nn.Linear(256 * 6 * 6, 2 * output_dim),
            nn.ReLU(inplace=True)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(2 * output_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(output_dim),
        )
        for layer in self.linear1:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0.0)
        for layer in self.linear2:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = nn.functional.normalize(x, dim=1)
        x = self.linear2(x)
        x = nn.functional.normalize(x, dim=1)
        
        return x

model = CNN().cuda()
sim_func = F.cosine_similarity

# 假设观测假设为 (3025, 256, 256)，观测值 BEV 为 (256, 256)
obs_hypothesis = torch.rand((3025, 1, 256, 256)).cuda()
local_map = torch.rand((1, 256, 256)).cuda()

print()

# 第一种方式，与每个观测假设的特征，分别计算相似度
# 时间过长（11.344250440597534 s）
# 增加 with torch.no_grad (4.917964220046997s)

start_time = time.time()

with torch.no_grad():
    similarity = torch.Tensor().cuda()
    for index in range(obs_hypothesis.shape[0]):
        sim_item = sim_func(
            model(obs_hypothesis[index].unsqueeze(dim=0)),
            model(local_map.unsqueeze(dim=0))
        )
        similarity = torch.concat((similarity, sim_item), dim=0)

end_time = time.time()
execution_time = (end_time - start_time)
print(execution_time)

# 第二种方式，将观测值 BEV 复制到和观测假设同维度，一起计算相似度
# 内存消耗过大

# start_time = time.time()

# local_map = local_map.unsqueeze(dim=0).expand(3025,1,256,256)

# local_map_ft = model(local_map)
# obs_hypothesis_ft = model(obs_hypothesis)
# similarity = sim_func(obs_hypothesis_ft, local_map_ft)

# end_time = time.time()
# execution_time = (end_time - start_time)
# print(execution_time)

# 第三种方式，分别提取观测值 BEV 特征和观测假设特征，分别计算相似度
# 时间过长 (16.0518741607666)
# 增加 with torch.no_grad (0.643230676651001)

start_time = time.time()

with torch.no_grad():
    local_map_ft = model(local_map.unsqueeze(dim=0))
    obs_hypothesis_ft = model(obs_hypothesis)

    similarity = sim_func(local_map_ft, obs_hypothesis_ft)

end_time = time.time()
execution_time = (end_time - start_time)
print(execution_time)
