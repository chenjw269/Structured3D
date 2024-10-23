import os
import cv2
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import concurrent.futures

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
from scripts.utils.visualize_occ import visualize_occ


data_pth = "e:/datasets/Structure3D/Structured3D" # remote
output_pth = "e:/datasets/Structure3D_map/Structured3D" # remote
data_pth = "e:/datasets/Structure3D/Structured3D" # local
output_pth = "e:/datasets/Structure3D_map/Structured3D" # local

resolution = 25  # 2.5 cm, 0.025 m / pixel
map_occ_size = (1600, 1600)  # x 轴范围为 (-20m, 20m) y 轴范围为 (-20m, 20m)

# 标注数据缺失的场景
with open("logs/scene_annos.txt", encoding="utf-8") as f:
    scene_invalid = f.readlines()
for index, item in enumerate(scene_invalid):
    scene_invalid[index] = item.replace("\n", "")

def execute_semantic_mapping(scene_index):

    # 地图标注数据
    annos = os.path.join(data_pth, scene_index, "annotation_3d.json")
    with open(annos, encoding="utf-8") as f:
        annos = json.load(f)
    # 物体地图数据
    obj_map = os.path.join(data_pth, scene_index, "bbox_3d.json")
    with open(obj_map, encoding="utf-8") as f:
        obj_map = json.load(f)
    # 场景边界数据
    scene_info = os.path.join(data_pth, scene_index, "boundary.csv")
    scene_info = pd.read_csv(scene_info)
    # 实例语义映射
    ins2sem = os.path.join(data_pth, scene_index, "ins_semantic.json")
    with open(ins2sem, encoding="utf-8") as f:
        ins2sem = json.load(f)

    ##########################
    # 坐标对齐
    ##########################
    # 从地图标注中读取顶点
    junctions = np.array([junc['coordinate'][:2] for junc in annos['junctions']])
    # 所有顶点进行平移，将场景中心对齐到 (0,0)
    junctions = junctions - np.array([scene_info['x_center'][0], scene_info['y_center'][0]])

    # 创建地图占用网格
    map_occ = np.zeros(map_occ_size, dtype=np.uint8)

    #########################################
    # 绘制区域的边缘墙面
    #########################################
    # 从地图标注中读取区域
    areawall_polygons = read_area_wall(annos)

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

    # # 可视化结果
    # # cv2
    # cv2.imshow("CAD Map", map_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # # pyplot
    # plt.figure(figsize=(10, 10))
    # plt.imshow(map_img)
    # plt.show()

    # 保存结果
    scene_output_dir = os.path.join(output_pth, scene_index)
    os.makedirs(scene_output_dir, exist_ok=True)
    # 占用网格 np.array
    scene_output_pth_array = os.path.join(output_pth, scene_index, "map.npy")
    np.save(scene_output_pth_array, map_occ)
    # 图片 png
    scene_output_pth_img = os.path.join(output_pth, scene_index, "map.png")
    cv2.imwrite(scene_output_pth_img, map_img)

    return scene_index


if __name__ == '__main__':

    scene_index_list = [f"scene_{num:05}" for num in range(3500)]
    # scene_index_list = [f"scene_{num:05}" for num in range(100)] # 前 100 个场景

    for scene_index in tqdm(scene_index_list):
        # 缺少标注的场景作废
        if scene_index in scene_invalid:
            scene_index_list.remove(scene_index)

    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
        futures = {executor.submit(execute_semantic_mapping, task): task for task in scene_index_list}

        with tqdm(total=len(futures)) as pbar:
            for future in concurrent.futures.as_completed(futures):
                task_id = futures[future]
                try:
                    result = future.result()
                    tqdm.write(result)
                except Exception as e:
                    tqdm.write(f"Task {task_id} generated an exception: {e}")

                pbar.update(1)  # 更新进度条

