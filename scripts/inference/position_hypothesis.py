# 20241025
# 从地图上采样位置假设

# 202410251254
# 202410251322

# 28 min，合格

import os
import cv2
import torch
import platform
import numpy as np
import pandas as pd

import sys
sys.path.append(".")

# 坐标转换
from scripts.cad_mapping_v1.coord_conv import position_to_pixel_mapping


# 获取系统类型
system_type = platform.system()
# 本地路径和服务器路径
if system_type == 'Windows':
    data_pth = "e:/datasets/Structure3D/Structured3D"
    map_pth = "e:/datasets/Structure3D_map/Structured3D"
    scene_annos_loss = "logs/scene_annos.txt"
    scene_line_err = "logs/scene_line_err.txt"
    output_pth = "e:/datasets/Structure3D_map/Structured3D"
else:
    data_pth = "/data1/chenjiwei/S3D/zip/Structured3D"
    map_pth = "/data1/chenjiwei/S3D/Structure3D_map/Structured3D"
    scene_annos_loss = "../../logs/scene_annos.txt"
    scene_line_err = "../../logs/scene_line_err.txt"
    output_pth = "/data1/chenjiwei/S3D/Structure3D_map/Structured3D"

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
        scene_pth (str): 场景标注路径

    Returns:
        np.array: 位置假设
    """

    resolution = 25  # 2.5 cm, 0.025 m / pixel

    # 场景边界坐标
    scene_min_p = (scene_info['x_min'].item(), scene_info['y_min'].item())
    scene_max_p = (scene_info['x_max'].item(), scene_info['y_max'].item())
    scene_p = np.array([scene_min_p, scene_max_p])
    # 坐标变换
    scene_p = position_to_pixel_mapping(scene_p, resolution, scene_info)
    # 场景实际坐标范围
    scene_x_range = sorted(scene_p[:,0].tolist())
    scene_y_range = sorted(scene_p[:,1].tolist())
    # 从坐标范围内采样位置假设
    hypothesis_loc = sample_position_hypothesis(scene_x_range, scene_y_range, 10, 0)

    return hypothesis_loc


if __name__ == "__main__":
    
    scene_index = "scene_00000"
    
    # 场景边界数据
    scene_info = os.path.join(data_pth, scene_index, "boundary.csv")
    scene_info = pd.read_csv(scene_info)
    
    resolution = 25  # 2.5 cm, 0.025 m / pixel
    
    # 场景地图
    scene_map = os.path.join(map_pth, scene_index, "map.png")
    scene_map = cv2.imread(scene_map)
    # 场景边界坐标
    scene_min_p = (scene_info['x_min'].item(), scene_info['y_min'].item())
    scene_max_p = (scene_info['x_max'].item(), scene_info['y_max'].item())
    scene_p = np.array([scene_min_p, scene_max_p])
    # 坐标变换
    scene_p = position_to_pixel_mapping(scene_p, resolution, scene_info)
    # 场景实际坐标范围
    scene_x_range = sorted(scene_p[:,0].tolist())
    scene_y_range = sorted(scene_p[:,1].tolist())
    # 从场景中采样位置假设
    hypothesis_loc = generate_scene_hypothesis(scene_info)
    # 可视化位置假设
    for i in hypothesis_loc:
        i = i.tolist()
        scene_map = cv2.circle(scene_map, i, 1, (0,0,255), -1)
    cv2.imshow("Visualize hypothesis", scene_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
