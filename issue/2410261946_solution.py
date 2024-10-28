# 202410271020
############################################################
# Solution
# 问题产生的原因是，场景标注中并没有包含所有的 outwall
# scene26000 场景中有多个 outwall，但标注文件中只提供了一个
# 获取场景边界应该通过遍历标注中所有物体，而非 outwall
############################################################

import os
import math
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append(".")

from s3d import *
# 读取地图
from scripts.utils.read_obj import read_obj

from scripts.utils.lines_to_polygons import convert_lines_to_vertices
from scripts.cad_mapping_v1.cad_mapping import execute_semantic_mapping

# 从场景标注中，读取所有的 outwall，并统计坐标边界
def read_scene_boundary(scene_index):

    #################################
    # 读取场景建筑外墙（outwall）标注
    #################################
    # 读取场景标注文件
    with open(os.path.join(s3d_data_pth, scene_index, "annotation_3d.json")) as f:
        scene_annos = json.load(f)
    # 读取标注中的建筑外墙（outwall）
    outwall_planes = []
    for semantic in scene_annos['semantics']:
        if semantic['type'] == 'outwall':
            outwall_planes.append(semantic['planeID'])
    if len(outwall_planes) == 0:
        raise StopIteration

    #################################
    # 读取场景孔洞（lines_hole）标注
    #################################
    # extract hole vertices
    lines_holes = []
    for semantic in scene_annos['semantics']:
        if semantic['type'] in ['window', 'door']:
            for planeID in semantic['planeID']: # where 找到非零的索引
                lines_holes.extend(np.where(np.array(scene_annos['planeLineMatrix'][planeID]))[0].tolist())
    lines_holes = np.unique(lines_holes)

    #################################
    # 为场景外墙构建多边形
    #################################
    # 场景中的地面顶点 junctions on the floor
    junctions = np.array([junc['coordinate'] for junc in scene_annos['junctions']])
    junction_floor = np.where(np.isclose(junctions[:, -1], 0))[0]
    # 外墙多边形在平面图上的边线
    # outerwall_polygon 为外墙多边形的顶点索引
    outwall_planes_floor = []
    for outwall in outwall_planes:
        outwall_floor = []
        for planeID in outwall:
            # 根据平面标识符 planeID 查找属于该平面的线标识符 lineIDs
            lineIDs = np.where(np.array(scene_annos['planeLineMatrix'][planeID]))[0].tolist()
            # 将属于门窗的 lineIDs 排除
            lineIDs = np.setdiff1d(lineIDs, lines_holes)
            junction_pairs = [np.where(np.array(scene_annos['lineJunctionMatrix'][lineID]))[0].tolist() for lineID in lineIDs]
            for junction_item in junction_pairs:
                if len(junction_item) == 2:
                    start, end = junction_item
                else:
                    continue
                if start in junction_floor and end in junction_floor:
                    outwall_floor.append([start, end])
        try:
            outwall_polygon = convert_lines_to_vertices(outwall_floor)[0]
            outwall_planes_floor.append(outwall_polygon)
        except:
            continue

    #################################
    # 读取场景物体标注
    #################################
    # 地图标注中的物体数据
    obj_map = os.path.join(s3d_data_pth, scene_index, "bbox_3d.json")
    with open(obj_map, encoding="utf-8") as f:
        obj_map = json.load(f)
    # 实例语义映射
    ins2sem = os.path.join(s3d_data_pth, scene_index, "ins_semantic.json")
    with open(ins2sem, encoding="utf-8") as f:
        ins2sem = json.load(f)
    # 读取场景中的物体
    scene_obj = read_obj(obj_map, ins2sem)

    #################################
    # 统计坐标边界
    #################################
    # 统计场景的坐标范围
    scene_info = {
        "x_min": [math.inf], "x_max": [-math.inf],
        "y_min": [math.inf], "y_max": [-math.inf]
    }
    
    # 遍历场景外墙的多边形，更新边界
    for outwall_polygon in outwall_planes_floor:
        # 在尾部插入头，令多边形首尾相连
        outwall_polygon = np.array(outwall_polygon + [outwall_polygon[0], ])
        # 顶点表：场景中的所有顶点
        junctions = np.array([junc['coordinate'][:2] for junc in scene_annos['junctions']])
        # 通过顶点表和顶点索引，查表得到顶点坐标
        outwall_polygon = junctions[outwall_polygon]

        scene_info["x_min"][0] = min(scene_info["x_min"][0], outwall_polygon[:,0].min())
        scene_info["x_max"][0] = max(scene_info["x_max"][0], outwall_polygon[:,0].max())
        scene_info["y_min"][0] = min(scene_info["y_min"][0], outwall_polygon[:,1].min())
        scene_info["y_max"][0] = max(scene_info["y_max"][0], outwall_polygon[:,1].max())

    # 遍历场景物体的多边形，更新边界
    for obj in tqdm(scene_obj):
        scene_info["x_min"][0] = min(obj[0][:,0].min(), scene_info["x_min"][0])
        scene_info["x_max"][0] = max(obj[0][:,0].max(), scene_info["x_max"][0])
        scene_info["y_min"][0] = min(obj[0][:,1].min(), scene_info["y_min"][0])
        scene_info["y_max"][0] = max(obj[0][:,1].max(), scene_info["y_max"][0])

    scene_info["center_x"] = (scene_info["x_min"][0] + scene_info["x_max"][0]) / 2
    scene_info["center_y"] = (scene_info["y_min"][0] + scene_info["y_max"][0]) / 2
    scene_info["size_x"] = scene_info["x_max"][0] - scene_info["x_min"][0] + 12800
    scene_info["size_y"] = scene_info["y_max"][0] - scene_info["y_min"][0] + 12800

    #################################
    # 保存场景的边界信息
    #################################
    output_pth = os.path.join(s3d_data_pth, scene_index, "boundary.csv")
    scene_info_df = pd.DataFrame(scene_info)
    scene_info_df.to_csv(output_pth, index=False)


if __name__ == "__main__":

    scene_index = "scene_02600"

    # 可视化
    # 获取场景边界
    read_scene_boundary(scene_index)
    # 创建 cad 地图
    execute_semantic_mapping(scene_index, mode="show")
