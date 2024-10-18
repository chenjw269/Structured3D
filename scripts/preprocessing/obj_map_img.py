# 20241018
# 将 json 物体地图转换为图像
import os
import cv2
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append(".")

from assets.semantic2label import SEMANTIC_TO_LABEL
from scripts.utils.visualize_occ import visualize_occ
from misc.utils import get_corners_of_bb3d_no_index

def generate_colors(k):
    # 使用colormap生成k个均匀分布的颜色
    colors = plt.cm.get_cmap('hsv', k)  # 选择hsv色彩空间，保证颜色差异较大
    rgb_colors = [colors(i)[:3] for i in range(k)]  # 提取RGB值
    rgb_colors = (np.array(rgb_colors) * 255).astype(int)  # 转换为0-255范围内的整数
    rgb_colors = {
        index+1 : color.tolist() for index, color in enumerate(rgb_colors)
    }
    rgb_colors[0] = (255,255,255) # 空格为白色
    return rgb_colors


if __name__ == "__main__":

    data_pth = "e:/datasets/Structure3D/Structured3D"

    # Structured3D 包括 3500 个场景
    scene_index_list = [f"scene_{num:05}" for num in range(3500)]

    label_color_dict = generate_colors(13)
    print(label_color_dict)

    # 遍历场景
    for scene_index in tqdm(scene_index_list):

        # 创建地图
        image = np.zeros([800, 800], dtype=np.uint8)

        # 物体地图数据
        obj_map = os.path.join(data_pth, scene_index, "bbox_3d.json")
        with open(obj_map, encoding="utf-8") as f:
            obj_map = json.load(f)
        # 实例语义映射
        ins2sem = os.path.join(data_pth, scene_index, "ins_semantic.json")
        with open(ins2sem, encoding="utf-8") as f:
            ins2sem = json.load(f)

        # 遍历该场景下的实例
        for obj in obj_map:
            
            # 查表，将实例映射到语义类别，如果不是有意义的类别则跳过
            ins_id = str(obj["ID"])
            if ins_id in ins2sem.keys():
                ins_semantic = ins2sem[ins_id]
                # 排序，得到该实例最可能的语义类别
                ins_semantic = max(ins_semantic, key=ins_semantic.get)
                # 不是有意义的类别则跳过
                if ins_semantic in ["prop", "unknown", "structure"]:
                    continue
            else:
                continue

            # 获取实例的边界
            basis = np.array(obj['basis'])
            coeffs = np.array(obj['coeffs'])
            centroid = np.array(obj['centroid'])

            corners = get_corners_of_bb3d_no_index(basis, coeffs, centroid)
            polygon = corners[[0, 1, 2, 3, 0], :2]

            # 将多边形的顶点转换成 1 cm/pixel
            for idx, vertex in enumerate(polygon):
                vertex = [int(vertex[0] / 10), int(vertex[1] / 10)]
                polygon[idx] = vertex
            # 各坐标加上偏移量 400 pixel
            for idx, vertex in enumerate(polygon):
                vertex = [int(vertex[0] + 400), int(vertex[1] + 400)]
                polygon[idx] = vertex
            polygon = polygon.astype(int)

            # TODO: 根据语义类别，和 3d 边界，填充占用网格
            # 查表，得到该语义类别对应的索引
            if ins_semantic in SEMANTIC_TO_LABEL.keys():
                semantic_label = SEMANTIC_TO_LABEL[ins_semantic]
            else:
                continue
            # 填充 2d 占用网格
            cv2.fillPoly(image, [polygon], color=semantic_label)
            
        # 将占用网格转为图像
        image_rgb = visualize_occ(image, label_color_dict)

        # # 可视化结果
        # plt.figure(figsize=(10, 10))
        # plt.imshow(image_rgb)
        # plt.show()

        # 保存结果
        obj_map_occ = os.path.join(data_pth, scene_index, "map/obj_occ.npy")
        np.save(obj_map_occ, image)
        obj_map_img = os.path.join(data_pth, scene_index, "map/obj_occ.png")
        cv2.imwrite(image_rgb, obj_map_img)
