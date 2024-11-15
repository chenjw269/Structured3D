import torch
from torch import nn


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

class FCN(nn.Module):
    
    def __init__(self, output_dim=1024):
        super(FCN, self).__init__()
        
        self.features = nn.Sequential(
            # layer 1
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2), # 去掉池化实现全卷积
            # layer 2
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2), # 去掉池化实现全卷积
            # layer 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        # 自适应层，将上一层的数据转换成6x6大小
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        # 全连接层
        self.linear = nn.Sequential(
            nn.Linear(256 * 6 * 6, 2 * output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2 * output_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(output_dim)
        )
        for layer in self.linear:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)

        return x
