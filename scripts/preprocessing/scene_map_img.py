# 20241018
# 构建场景的 CAD 地图
import os
import cv2
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append(".")

# 读取地图
from scripts.utils.read_area_wall import read_area_wall
from scripts.utils.read_obj import read_obj
# 填充占用网格
from scripts.utils.draw_polygon import draw_polygon
from scripts.utils.draw_polygon import fill_polygon
# 可视化占用网格
from assets.semantic2label import SEMANTIC_TO_LABEL, LABEL_TO_COLOR
from scripts.utils.generate_colors import generate_colors
from scripts.utils.visualize_occ import visualize_occ


if __name__ == "__main__":
    
    data_pth = "e:/datasets/Structure3D/Structured3D"
    output_pth = "e:/datasets/Structure3D_map/Structured3D"
    
    # # Structured3D 包括 3500 个场景
    # scene_index_list = [f"scene_{num:05}" for num in range(3500)]
    # 统计前 100 个场景
    scene_index_list = [f"scene_{num:05}" for num in range(100)]
    
    # label_color_dict = generate_colors(13)
    # print(label_color_dict)
    
    resolution = 25 # 2.5 cm, 0.025 m / pixel
    map_occ_size = (1600, 1200) # x 轴范围为 (-20m, 20m) y 轴范围为 (-15m, 15m)

    # 遍历场景
    for scene_index in tqdm(scene_index_list):

        # 地图标注数据
        annos = os.path.join(data_pth, scene_index, "annotation_3d.json")
        with open(annos, encoding="utf-8") as f:
            annos = json.load(f)
        # 物体地图数据
        obj_map = os.path.join(data_pth, scene_index, "bbox_3d.json")
        with open(obj_map, encoding="utf-8") as f:
            obj_map = json.load(f)
        # 物体边界数据
        scene_info = os.path.join(data_pth, scene_index, "boundary.csv")
        scene_info = pd.read_csv(scene_info)
        # 实例语义映射
        ins2sem = os.path.join(data_pth, scene_index, "ins_semantic.json")
        with open(ins2sem, encoding="utf-8") as f:
            ins2sem = json.load(f)

        # 从地图标注中读取顶点
        junctions = np.array([junc['coordinate'][:2] for junc in annos['junctions']])
        junctions = junctions - np.array([scene_info['x_center'][0], scene_info['y_center'][0]])
        # 从地图标注中读取区域
        areawall_polygons = read_area_wall(annos)
        
        # 创建地图占用网格
        map_occ = np.zeros(map_occ_size, dtype=np.uint8)
        
        #########################################
        # 绘制区域的边缘墙面
        #########################################
        color_wall = 1
        thickness = 2
        for (polygon, poly_type) in areawall_polygons:
            if poly_type in ["window", "door"]:
                continue
            map_occ = draw_polygon(map_occ, polygon, junctions, color_wall, thickness, resolution, map_occ_size)

        #########################################
        # 擦除 window 和 door 的边缘
        #########################################
        # color_empty = 0
        # thickness = 2
        # for (polygon, poly_type) in areawall_polygons:
        #     if poly_type not in ["door"]: # 擦除门边缘
        #         continue
        #     map_occ = draw_polygon(map_occ, polygon, junctions, color_empty, thickness, resolution, map_occ_size)

        #########################################
        # 绘制家具物体
        #########################################
        scene_obj = read_obj(obj_map, ins2sem)
        
        map_occ = fill_polygon(map_occ, scene_obj, SEMANTIC_TO_LABEL, resolution, map_occ_size)

        # 将占用网格转为图像
        map_img = visualize_occ(map_occ, LABEL_TO_COLOR)

        # # 上下翻转图像
        # map_occ = np.flipud(map_occ)

        # # 可视化结果
        # cv2.imshow("CAD Map", map_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        # plt.figure(figsize=(10, 10))
        # plt.imshow(map_img)
        # plt.show()

        # 保存结果
        scene_output_dir = os.path.join(output_pth, scene_index)
        os.makedirs(scene_output_dir, exist_ok=True)
        # np.array
        scene_output_pth_array = os.path.join(output_pth, scene_index, "map.npy")
        np.save(scene_output_pth_array, map_occ)
        # 图片
        scene_output_pth_img = os.path.join(output_pth, scene_index, "map.png")
        cv2.imwrite(scene_output_pth_img, map_img)
