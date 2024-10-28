# 20241024
# 统计场景的坐标边界
import os
import cv2
import json
import platform
import numpy as np
import pandas as pd
from tqdm import tqdm
import concurrent.futures


import sys
sys.path.append(".")

# 坐标转换
from scripts.cad_mapping_v1.coord_conv import position_to_pixel_mapping
# 读取地图
from scripts.utils.read_area_wall import read_area_wall
from scripts.utils.read_obj import read_obj
# 填充占用网格
from scripts.utils.draw_polygon import draw_polygon
from scripts.utils.draw_polygon import fill_polygon
# 可视化占用网格
from assets.semantic2label import SEMANTIC_TO_LABEL, LABEL_TO_COLOR
from scripts.utils.visualize_occ import visualize_occ

# 获取系统类型
system_type = platform.system()
# 本地路径和服务器路径
if system_type == 'Windows':
    data_pth = "e:/datasets/Structure3D/Structured3D"
    scene_annos_loss = "logs/scene_annos.txt"
    scene_line_err = "logs/scene_line_err.txt"
    output_pth = "e:/datasets/Structure3D_map/Structured3D"
else:
    data_pth = "/data1/chenjiwei/S3D/zip/Structured3D"
    scene_annos_loss = "../../logs/scene_annos.txt"
    scene_line_err = "../../logs/scene_line_err.txt"
    output_pth = "/data1/chenjiwei/S3D/Structure3D_map/Structured3D"

resolution = 25  # 2.5 cm, 0.025 m / pixel

# 标注数据缺失的场景
with open(scene_annos_loss, encoding="utf-8") as f:
    scene_invalid = f.readlines()
for index, item in enumerate(scene_invalid):
    scene_invalid[index] = item.replace("\n", "")
# 边线错误的场景
with open(scene_line_err, encoding="utf-8") as f:
    scene_invalid_append = f.readlines()
for index, item in enumerate(scene_invalid_append):
    scene_invalid_append[index] = item.replace("\n", "")
scene_invalid = scene_invalid + scene_invalid_append


def execute_semantic_mapping(scene_index, mode='save'):

    ####################################
    # 读取数据
    ####################################
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
    
    #########################################
    # 绘制 CAD 地图
    #########################################
    # 地图尺寸，因为转换为图像之后 xy 轴会交换，所以这里交换 xy 轴创建 array
    scene_size = [
        int(scene_info['size_y'].item() / resolution),
        int(scene_info['size_x'].item() / resolution)
    ]
    # 创建地图占用网格
    map_occ = np.zeros(scene_size, dtype=np.uint8)

    #########################################
    # 绘制区域的边缘墙面
    #########################################
    # 从地图标注中读取区域多边形
    areawall_polygons = read_area_wall(annos)
    # 从地图标注中读取顶点
    junctions = np.array([junc['coordinate'][:2] for junc in annos['junctions']])
    # 所有顶点进行坐标变换
    junctions = position_to_pixel_mapping(junctions, resolution, scene_info)
    # 绘制区域多边形
    color_wall = 1
    thickness = 2
    for (polygon, poly_type) in areawall_polygons:
        if poly_type in ["window", "door"]:
            continue
        map_occ = draw_polygon(map_occ, polygon, junctions, color_wall, thickness)

    #########################################
    # 绘制家具物体
    #########################################
    # 从地图标注中读取物体多边形
    scene_obj = read_obj(obj_map, ins2sem)
    for index, obj in enumerate(scene_obj):
        scene_obj[index][0] = position_to_pixel_mapping(obj[0], resolution, scene_info)
    # 绘制物体多边形
    map_occ = fill_polygon(map_occ, scene_obj, SEMANTIC_TO_LABEL)

    #########################################
    # 保存结果
    #########################################
    # 将占用网格转为图像
    map_img = visualize_occ(map_occ, LABEL_TO_COLOR)

    if mode == "show":
        # 可视化结果
        cv2.imshow("CAD Map", map_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif mode == "save":
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

if __name__ == "__main__":

    # Structured3D 包括 3500 个场景
    scene_index_list = [f"scene_{num:05}" for num in range(3500)]
    # 统计前 100 个场景
    # scene_index_list = [f"scene_{num:05}" for num in range(100)]

    for scene_index in tqdm(scene_index_list):
        # 缺少标注的场景作废
        if scene_index in scene_invalid:
            scene_index_list.remove(scene_index)
    print("Invalid scenes removed")

    with concurrent.futures.ProcessPoolExecutor(max_workers=40) as executor:

        futures = {executor.submit(execute_semantic_mapping, task): task for task in scene_index_list}

        with tqdm(total=len(futures)) as pbar:
            for future in concurrent.futures.as_completed(futures):
                task_id = futures[future]
                try:
                    result = future.result()
                    # tqdm.write(result)
                except Exception as e:
                    tqdm.write(f"Task {task_id} generated an exception: {e}")

                pbar.update(1)  # 更新进度条
