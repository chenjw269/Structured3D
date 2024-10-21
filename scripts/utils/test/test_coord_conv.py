import cv2
import numpy as np

import sys
sys.path.append(".")

from scripts.utils.coord_conv import position_to_pixel # 坐标转换
from scripts.utils.visualize_occ import visualize_occ # 可视化占用网格
from assets.semantic2label import LABEL_TO_COLOR


if __name__ == "__main__":

    resolution = 25 # 2.5 cm, 0.025 m / pixel
    map_occ_size = (1600, 1200) # x 轴范围为 (-20m, 20m) y 轴范围为 (-15m, 15m)

    # 样本真实位置
    sample_pos = "e:/datasets/Structure3D/Structured3D/scene_00000/2D_rendering/485142/panorama/camera_xyz.txt"
    sample_pos = np.genfromtxt(sample_pos, delimiter=" ")[:2]

    # 坐标转换后的像素位置
    sample_pos = position_to_pixel(sample_pos, resolution, map_occ_size)

    # 周围真值局部地图
    scene_occ = "e:/datasets/Structure3D_map/Structured3D/scene_00000/map.npy"
    scene_occ = np.load(scene_occ)
    local_map_gt = scene_occ[
        sample_pos[0] - 128: sample_pos[0] + 128,
        sample_pos[1] - 128: sample_pos[1] + 128
    ]
    local_map_gt_img = visualize_occ(local_map_gt, LABEL_TO_COLOR)

    # 观测值计算的局部地图
    local_map_obs = "e:/datasets/Structure3D_bev/Structured3D/scene_00000/2D_rendering/485142/panorama/full/bev.png"
    local_map_obs = cv2.imread(local_map_obs)

    cv2.imshow("Local map gt", local_map_gt_img)
    cv2.imshow("Local map obs", local_map_obs)

    cv2.waitKey(0)
    cv2.destroyAllWindows()