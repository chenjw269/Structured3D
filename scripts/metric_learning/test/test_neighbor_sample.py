import os
import numpy as np
from tqdm import tqdm


if __name__ == "__main__":

    data_pth = "e:/datasets/Structure3D/Structured3D"

    # Structured3D 包括 3500 个场景
    scene_index_list = [f"scene_{num:05}" for num in range(3500)]

    resolution = 25 # 2.5 cm, 0.025 m / pixel
    map_occ_size = (1600, 1200) # x 轴范围为 (-20m, 20m) y 轴范围为 (-15m, 15m)

    # 遍历场景
    for scene_index in tqdm(scene_index_list):
        
        scene_data_pth = os.path.join(data_pth, scene_index, "2D_rendering")

        # 遍历样本
        scene_sample_list = os.listdir(scene_data_pth)
        for sample_item in scene_sample_list:
            # 读取样本位置
            sample_pos = os.path.join(scene_data_pth, sample_item, "panorama/camera_xyz.txt")
            sample_pos = np.genfromtxt(sample_pos, delimiter=" ")
            # 坐标转换
            

            # 采样随机位置

            # 在地图上可视化随机位置

            # 随机位置周围的局部地图

