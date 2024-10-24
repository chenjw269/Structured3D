# 20241019
# 从地图上采样位置假设
import os
import cv2
import numpy as np
import pandas as pd

import sys
sys.path.append(".")

from scripts.metric_learning.position_hypothesis import position_hypothesis
from scripts.utils.coord_conv import position_to_pixel


if __name__ == "__main__":

    data_pth = "e:/datasets/Structure3D/Structured3D"
    map_pth = "e:/datasets/Structure3D_map/Structured3D"

    scene_index = "scene_00000"

    resolution = 25
    map_occ_size = (1600, 1600)

    # 读取场景地图
    scene_map = os.path.join(map_pth, scene_index, "map.png")
    scene_map = cv2.imread(scene_map)
    # 读取场景坐标边界
    scene_boundary = os.path.join(data_pth, scene_index, "boundary.csv")
    scene_boundary = pd.read_csv(scene_boundary)

    # 场景坐标中心点
    scene_center_point = np.array([scene_boundary['x_center'].item(), scene_boundary['y_center'].item()])
    # 场景边界最小值
    scene_min_point = np.array([scene_boundary['x_min'].item(), scene_boundary['y_min'].item()])
    scene_min_point_norm = scene_min_point - scene_center_point
    scene_min_point_pixel = position_to_pixel(scene_min_point_norm, resolution, map_occ_size)
    # 场景边界最大值
    scene_max_point = np.array([scene_boundary['x_max'].item(), scene_boundary['y_max'].item()])
    scene_max_point_norm = scene_max_point - scene_center_point
    scene_max_point_pixel = position_to_pixel(scene_max_point_norm, resolution, map_occ_size)

    # 在边界内采样位置假设
    sample_loc = position_hypothesis(
        x_range=sorted([scene_min_point_pixel[0], scene_max_point_pixel[0]]),
        y_range=sorted([scene_min_point_pixel[1], scene_max_point_pixel[1]]),
        step=10, boundary=128
    )
    
    # 在地图上可视化位置假设
    for i in sample_loc:
        i = i.tolist()
        scene_map = cv2.circle(scene_map, i, 1, (0,0,255), -1)
    
    cv2.imshow("Visualize hypothesis", scene_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
