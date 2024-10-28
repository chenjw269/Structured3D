import sys
sys.path.append(".")

from s3d import * # s3d 数据集信息
import os # 拼接文件路径
import cv2 # 读取地图
import pandas as pd # 读取场景边界信息
from scripts.cad_mapping_v1.scene_boundary_statistics import *


if __name__ == "__main__":
    
    scene_index = "scene_02600"
    
    # 计算场景的坐标边界
    execute_scene_boundary(scene_index)
    
    # 读取场景的坐标边界
    scene_bound = os.path.join(s3d_data_pth, scene_index, f"boundary.csv")
    scene_bound = pd.read_csv(scene_bound)

    # 场景地图
    scene_map = os.path.join(s3d_map_pth, scene_index, "map.png")
    scene_map = cv2.imread(scene_map)
    print(scene_map.shape)

    # 物体坐标极值
    x_min = scene_bound['x_min'][0]
    x_max = scene_bound['x_max'][0]
    y_min = scene_bound['y_min'][0]
    y_max = scene_bound['y_max'][0]
    print(f"X axis range ({x_min}, {x_max}), Y axis range ({y_min}, {y_max})")

    # 地图尺寸
    x_range = scene_bound['size_x'][0]
    y_range = scene_bound['size_y'][0]
    print(f"X range (0, {x_range}), Y range (0, {y_range})")
    print(f"Map size {int(x_range / resolution)}, {int(y_range / resolution)}")
