# 20241018
# 构建场景的 CAD 地图
import os
import cv2
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append(".")

# 读取地图
from scripts.utils.read_area_wall import read_area_wall
from scripts.utils.read_obj import read_obj
# 填充占用网格
from scripts.utils.draw_polygon import draw_polygon
# 可视化占用网格
from assets.semantic2label import SEMANTIC_TO_LABEL
from scripts.utils.generate_colors import generate_colors
from scripts.utils.visualize_occ import visualize_occ


if __name__ == "__main__":
    
    data_pth = "e:/datasets/Structure3D/Structured3D"
    
    # Structured3D 包括 3500 个场景
    scene_index_list = [f"scene_{num:05}" for num in range(3500)]
    
    label_color_dict = generate_colors(13)
    print(label_color_dict)
    
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
        # 实例语义映射
        ins2sem = os.path.join(data_pth, scene_index, "ins_semantic.json")
        with open(ins2sem, encoding="utf-8") as f:
            ins2sem = json.load(f)

        # 从地图标注中读取顶点
        junctions = np.array([junc['coordinate'][:2] for junc in annos['junctions']])
        # 从地图标注中读取区域
        areawall_polygons = read_area_wall(annos)
        
        # 创建地图占用网格
        map_occ = np.zeros([800, 800], dtype=np.uint8)
        
        #########################################
        # 绘制区域的边缘墙面
        #########################################
        color_wall = 1
        thickness = 3
        for (polygon, poly_type) in areawall_polygons:
            if poly_type in ["window", "door"]:
                continue
            map_occ = draw_polygon(map_occ, polygon, junctions, color_wall, thickness)

        #########################################
        # 擦除 window 和 door 的边缘
        #########################################
        color_empty = 0
        thickness = 3
        for (polygon, poly_type) in areawall_polygons:
            if poly_type not in ["door"]: # 擦除门边缘
                continue
            map_occ = draw_polygon(map_occ, polygon, junctions, color_empty, thickness)

        #########################################
        # 绘制家具物体
        #########################################
        scene_obj = read_obj(obj_map, ins2sem)
        
        for instance, ins_semantic in scene_obj:
            # 查表，得到该语义类别对应的索引
            if ins_semantic in SEMANTIC_TO_LABEL.keys():
                semantic_label = SEMANTIC_TO_LABEL[ins_semantic]
            else:
                continue
            # 填充 2d 占用网格
            cv2.fillPoly(map_occ, [instance], color=semantic_label)

        # 将占用网格转为图像
        map_img = visualize_occ(map_occ, label_color_dict)

        # # 上下翻转图像
        # map_occ = np.flipud(map_occ)

        # 可视化结果
        plt.figure(figsize=(10, 10))
        plt.imshow(map_img)
        plt.show()

        # 保存结果
