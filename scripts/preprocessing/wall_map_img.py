# 20241018
# 从地图标注数据中，获取墙壁地图
import os
import cv2
import json
import numpy as np
from tqdm import tqdm

import sys
sys.path.append(".")

from scripts.utils.read_area_wall import read_area_wall
from scripts.utils.read_outer_wall import read_outer_wall


if __name__ == "__main__":
    
    data_pth = "e:/datasets/Structure3D/Structured3D"
    
    # Structured3D 包括 3500 个场景
    scene_index_list = [f"scene_{num:05}" for num in range(3500)]
    
    # 遍历场景
    for scene_index in tqdm(scene_index_list):

        # 读取地图标注
        annos = os.path.join(data_pth, scene_index, "annotation_3d.json")
        with open(annos, encoding="utf-8") as f:
            annos = json.load(f)

        # 从地图标注中读取外墙
        outwall_polygon = read_outer_wall(annos)
        # 从地图标注中读取区域
        areawall_polygons = read_area_wall(annos)

        # 创建地图图像
        image = np.ones([800, 800]) * 255

        junctions = np.array([junc['coordinate'][:2] for junc in annos['junctions']])

        #########################################
        # 绘制外墙
        #########################################
        # 为了让多边形形成一个环，把头节点添加到尾部
        outwall_polygon = np.array(outwall_polygon + [outwall_polygon[0], ])
        # 查表得到节点坐标
        outwall_polygon = junctions[outwall_polygon]
        # 将多边形的顶点转换成 1 cm/pixel
        for idx, vertex in enumerate(outwall_polygon):
            vertex = [int(vertex[0] / 10), int(vertex[1] / 10)]
            outwall_polygon[idx] = vertex
        # 各坐标加上偏移量 400 pixel
        for idx, vertex in enumerate(outwall_polygon):
            vertex = [int(vertex[0] + 400), int(vertex[1] + 400)]
            outwall_polygon[idx] = vertex
        # 颜色 (B, G, R) 和线的厚度
        color = (0, 0, 0)   # 黑色
        thickness = 3 # 宽度为 3
        # 在图像上绘制线条
        for idx in range(outwall_polygon.shape[0] - 1):
            start_point = outwall_polygon[idx].astype(int)
            end_point = outwall_polygon[idx + 1].astype(int)
            image = cv2.line(image, start_point, end_point, color, thickness)
    
        #########################################
        # 绘制区域的边缘墙面
        # 不绘制 window 和 door，只绘制房间类型
        #########################################
        for (polygon, poly_type) in areawall_polygons:
            # 为了让多边形形成一个环，把头节点添加到尾部
            polygon = np.array(polygon + [polygon[0], ])
            # 查表得到节点坐标
            polygon = junctions[polygon]
            # 将多边形的顶点转换成 1 cm/pixel
            for idx, vertex in enumerate(polygon):
                vertex = [int(vertex[0] / 10), int(vertex[1] / 10)]
                polygon[idx] = vertex
            # 各坐标加上偏移量 400 pixel
            for idx, vertex in enumerate(polygon):
                vertex = [int(vertex[0] + 400), int(vertex[1] + 400)]
                polygon[idx] = vertex
            # 绘制
            # 颜色 (B, G, R) 和线的厚度
            if poly_type in ["window", "door"]:
                continue
            else:
                color = (0, 0, 0)   # 黑色
            thickness = 3 # 宽度为 3
            # 在图像上绘制线条
            for idx in range(polygon.shape[0] - 1):
                start_point = polygon[idx].astype(int)
                end_point = polygon[idx + 1].astype(int)
                image = cv2.line(image, start_point, end_point, color, thickness)

        #########################################
        # 其次擦除 window 和 door 的边缘
        #########################################
        for (polygon, poly_type) in areawall_polygons:
            # 为了让多边形形成一个环，把头节点添加到尾部
            polygon = np.array(polygon + [polygon[0], ])
            # 查表得到节点坐标
            polygon = junctions[polygon]
            # 将多边形的顶点转换成 1 cm/pixel
            for idx, vertex in enumerate(polygon):
                vertex = [int(vertex[0] / 10), int(vertex[1] / 10)]
                polygon[idx] = vertex
            # 各坐标加上偏移量 400 pixel
            for idx, vertex in enumerate(polygon):
                vertex = [int(vertex[0] + 400), int(vertex[1] + 400)]
                polygon[idx] = vertex
            # 绘制
            # 颜色 (B, G, R) 和线的厚度
            if poly_type in ["door"]: # 擦除门边缘
            # if poly_type in ["window", "door"]: # 擦除门窗边缘
                color = (255, 255, 255) # 白色
            else:
                continue
            thickness = 3 # 宽度为 3
            # 在图像上绘制线条
            for idx in range(polygon.shape[0] - 1):
                start_point = polygon[idx].astype(int)
                end_point = polygon[idx + 1].astype(int)
                image = cv2.line(image, start_point, end_point, color, thickness)

        # 上下翻转图像
        image = np.flipud(image)

        # 可视化结果
        cv2.imshow('Image with Line', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # # 保存结果
        # wall_map_img = os.path.join()
