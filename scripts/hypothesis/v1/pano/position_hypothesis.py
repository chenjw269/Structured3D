# 20241025
# 从地图上采样位置假设

# 202410251254
# 202410251322

# 28 min，合格

import os
import cv2
import torch
import numpy as np
import pandas as pd

import sys
sys.path.append(".")

from s3d import *

# 坐标转换
from scripts.cad_mapping_v1.coord_conv import *


def sample_position_hypothesis(x_range, y_range, step, boundary):
    """从坐标范围内采样位置假设

    Args:
        x_range (list(int, int)): x 轴坐标范围
        y_range (list(int, int)): y 轴坐标范围
        step (int): 移动步长
        boundary (int): 边缘宽度 / 窗口大小

    Returns:
        np.array: 位置假设
    """
    padding = max(step, boundary)
    
    x_range = [
        x_range[0] - padding,
        x_range[1] + padding
    ]
    y_range = [
        y_range[0] - padding,
        y_range[1] + padding
    ]
    
    Iy, Ix = torch.meshgrid(
        torch.arange(x_range[0], x_range[1], step),
        torch.arange(y_range[0], y_range[1], step),
    )
    samples_loc = torch.stack([Ix, Iy], dim=-1).reshape(-1, 2)

    return samples_loc

def generate_scene_hypothesis(scene_info):
    """在指定场景下采样位置假设

    Args:
        scene_info (pd.DataFrame): 场景标注

    Returns:
        np.array: 位置假设
    """
    # 场景边界坐标
    # 1. 直接通过像素值设置坐标边界
    scene_min_p = (256, 256)
    scene_max_p = (
        int(scene_info['size_y'][0] / resolution - 256),
        int(scene_info['size_x'][0] / resolution - 256),
    )
    
    scene_p = np.array([scene_min_p, scene_max_p])
    # # 2. 通过标注的 x y 坐标范围确定坐标边界
    # scene_min_p = (scene_info['x_min'][0], scene_info['y_min'][0])
    # scene_max_p = (scene_info['x_max'][0], scene_info['y_max'][0])
    # # 归一化，坐标中心对齐到原点
    # scene_center_p = (scene_info['center_x'][0], scene_info['center_y'][0])
    # scene_min_p = (scene_min_p[0]-scene_center_p[0], scene_min_p[1]-scene_center_p[1])
    # scene_max_p = (scene_max_p[0]-scene_center_p[0], scene_max_p[1]-scene_center_p[1])
    # # 坐标变换
    # scene_min_p = position_to_pixel(scene_min_p, resolution, scene_info)
    # scene_max_p = position_to_pixel(scene_max_p, resolution, scene_info)
    
    # scene_p = np.array([scene_min_p, scene_max_p])

    # x y 轴坐标范围
    scene_x_range = sorted(scene_p[:,0].tolist())
    scene_y_range = sorted(scene_p[:,1].tolist())

    # 从坐标范围内采样位置假设
    hypothesis_loc = sample_position_hypothesis(scene_x_range, scene_y_range, 10, 0)

    return hypothesis_loc
