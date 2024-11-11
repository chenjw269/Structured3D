# 20241027
# 在地图上可视化样本位置

import os

import sys
sys.path.append(".")

from s3d import *

# 读取样本位置
import numpy as np
# 读取场景标注
import pandas as pd
# 坐标转换
from scripts.cad_mapping_v1.coord_conv import position_to_pixel
# 读取地图图片，可视化
import cv2


def visualize_sample_loc(scene_index):

    scene_annos = os.path.join(s3d_data_pth, scene_index, "boundary.csv")
    scene_annos = pd.read_csv(scene_annos)

    scene_map = os.path.join(s3d_map_pth, scene_index, "map.png")
    scene_map = cv2.imread(scene_map)

    scene_data_pth = os.path.join(s3d_data_pth, scene_index, "2D_rendering")
    for sample_item in os.listdir(scene_data_pth):
        # 读取样本位置
        sample_item_loc = os.path.join(scene_data_pth, sample_item, "panorama/camera_xyz.txt")
        sample_item_loc = np.genfromtxt(sample_item_loc)[:2]
        # 坐标转换
        sample_item_loc = position_to_pixel(sample_item_loc, resolution, scene_annos)
        # 可视化样本位置
        scene_map = cv2.circle(scene_map, sample_item_loc, 3, (0,0,0), -1)
    
    cv2.imshow("Sample on map", scene_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
