# 20241019
# 统计所有场景的坐标边界
import os
import json
import platform
import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append(".")

from scripts.utils.lines_to_polygons import convert_lines_to_vertices


if __name__ == "__main__":

    # 获取系统类型
    system_type = platform.system()
    # 本地路径和服务器路径
    if system_type == 'Windows':
        data_pth = "e:/datasets/Structure3D/Structured3D"
        scene_annos_loss = "logs/scene_annos.txt"
    else:
        data_pth = "/data1/chenjiwei/S3D/zip/Structured3D"
        scene_annos_loss = "../../logs/scene_annos.txt"

    # # Structured3D 包括 3500 个场景
    # scene_index_list = [f"scene_{num:05}" for num in range(3500)]
    # 统计前 1000 个场景
    scene_index_list = [f"scene_{num:05}" for num in range(1000)]

    # 标注数据缺失的场景
    with open(scene_annos_loss, encoding="utf-8") as f: # remote
        scene_invalid = f.readlines()
    for index, item in enumerate(scene_invalid):
        scene_invalid[index] = item.replace("\n", "")

    # 统计场景的坐标范围
    x_max_total = -float('inf')
    x_min_total = float('inf')
    y_max_total = -float('inf')
    y_min_total = float('inf')

    # 遍历场景
    for scene_index in tqdm(scene_index_list):
        # 缺少标注的场景作废
        if scene_index in scene_invalid:
            tqdm.write(f"Jmp annos loss {scene_index}")
            continue
        else:
            tqdm.write(f'Current scene: {scene_index}')  # 输出当前场景

        # 读取场景标注文件
        with open(os.path.join(data_pth, scene_index, "annotation_3d.json"), encoding="utf-8") as f:
            annos = json.load(f)

        # 读取标注中的建筑外墙（outwall）
        for semantic in annos['semantics']:
            # outwall
            if semantic['type'] == 'outwall':
                outerwall_planes = semantic['planeID']

        # extract hole vertices
        lines_holes = []
        for semantic in annos['semantics']:
            if semantic['type'] in ['window', 'door']:
                for planeID in semantic['planeID']: # where 找到非零的索引
                    lines_holes.extend(np.where(np.array(annos['planeLineMatrix'][planeID]))[0].tolist())
        lines_holes = np.unique(lines_holes)

        # junctions on the floor
        junctions = np.array([junc['coordinate'] for junc in annos['junctions']])
        junction_floor = np.where(np.isclose(junctions[:, -1], 0))[0]

        # 外墙多边形在平面图上的边线
        # outerwall_polygon 为外墙多边形的顶点索引
        outerwall_floor = []
        for planeID in outerwall_planes:
            # 根据平面标识符 planeID 查找属于该平面的线标识符 lineIDs
            lineIDs = np.where(np.array(annos['planeLineMatrix'][planeID]))[0].tolist()
            # 将属于门窗的 lineIDs 排除
            lineIDs = np.setdiff1d(lineIDs, lines_holes)
            junction_pairs = [np.where(np.array(annos['lineJunctionMatrix'][lineID]))[0].tolist() for lineID in lineIDs]
            for junction_item in junction_pairs:
                if len(junction_item) == 2:
                    start, end = junction_item
                else:
                    continue
                if start in junction_floor and end in junction_floor:
                    outerwall_floor.append([start, end])
        try:
            outerwall_polygon = convert_lines_to_vertices(outerwall_floor)[0]
        except:
            tqdm.write(f"Jmp {scene_index} due to line error")
            continue

        ############################################
        # 统计场景的坐标范围
        ############################################
        scene_info = {}
        
        # 所有顶点坐标列表
        junctions = np.array([junc['coordinate'][:2] for junc in annos['junctions']])
        # 在尾部插入头，令多边形首尾相连
        outerwall_polygon = np.array(outerwall_polygon + [outerwall_polygon[0], ])
        # 通过顶点索引，查找顶点坐标
        outerwall_polygon = junctions[outerwall_polygon]
        # 每个场景中，xy 坐标的最大最小值
        x_min_scene = outerwall_polygon[:,0].min()
        scene_info["x_min"] = [x_min_scene]
        x_max_scene = outerwall_polygon[:,0].max()
        scene_info["x_max"] = [x_max_scene]
        y_min_scene = outerwall_polygon[:,1].min()
        scene_info["y_min"] = [y_min_scene]
        y_max_scene = outerwall_polygon[:,1].max()
        scene_info["y_max"] = [y_max_scene]
        # 移动一定的距离，将场景的中心调整为原点
        x_center = (x_min_scene + x_max_scene) / 2
        scene_info["x_center"] = [x_center]
        y_center = (y_min_scene + y_max_scene) / 2
        scene_info["y_center"] = [y_center]
        x_min_scene = x_min_scene - x_center
        y_min_scene = y_min_scene - y_center
        x_max_scene = x_max_scene - x_center
        y_max_scene = y_max_scene - y_center

        # 保存场景的边界信息
        output_pth = os.path.join(data_pth, scene_index, "boundary.csv")
        scene_info_df = pd.DataFrame(scene_info)
        scene_info_df.to_csv(output_pth, index=False)

        # 更新统计信息中，两个坐标轴坐标的最小值和最大值
        if x_min_scene < x_min_total:
            x_min_total = x_min_scene
        if y_min_scene < y_min_total:
            y_min_total = y_min_scene
        if x_max_scene > x_max_total:
            x_max_total = x_max_scene
        if y_max_scene > y_max_total:
            y_max_total = y_max_scene

    print(f"X min {x_min_total}")
    print(f"X max {x_max_total}")
    print(f"Y min {y_min_total}")
    print(f"Y max {y_max_total}")

# 未调整地图中心之前
# X min -19617.20801
# X max 17946.719
# Y min -14615.176
# Y max 12605.0
# 调整地图中心之后
# X min -17202.0
# X max 17202.0
# Y min -12605.0
# Y max 12605.0
# 可见地图应该覆盖 x 轴 -20m 到 20m 范围，和 y 轴 -15m 到 15m 范围