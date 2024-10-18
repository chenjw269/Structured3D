import numpy as np

import sys
sys.path.append(".")

from scripts.utils.lines_to_polygons import convert_lines_to_vertices


def read_outer_wall(annos):
    
    # 读取 outwall 类别的平面
    for semantic in annos['semantics']:
        if semantic['type'] == 'outwall':
            outerwall_planes = semantic['planeID']

    # extract hole vertices
    lines_holes = []
    for semantic in annos['semantics']:
        if semantic['type'] in ['window', 'door']:
            for planeID in semantic['planeID']:
                lines_holes.extend(np.where(np.array(annos['planeLineMatrix'][planeID]))[0].tolist())
    lines_holes = np.unique(lines_holes)

    # 顶点 / 交叉点
    junctions = np.array([junc['coordinate'] for junc in annos['junctions']])
    # 地面顶点，即坐标的最后一维为 0 的顶点
    junction_floor = np.where(np.isclose(junctions[:, -1], 0))[0]

    # 为对象创建多边形
    outerwall_floor = []
    for planeID in outerwall_planes:
        # 根据查找 planeID 查找 lineIDs
        lineIDs = np.where(np.array(annos['planeLineMatrix'][planeID]))[0].tolist()
        # 将属于门窗的 lineIDs 排除
        lineIDs = np.setdiff1d(lineIDs, lines_holes)
        # 查表得到顶点对
        junction_pairs = [np.where(np.array(annos['lineJunctionMatrix'][lineID]))[0].tolist() for lineID in lineIDs]
        # 检查顶点是否都属于地面顶点
        for start, end in junction_pairs:
            if start in junction_floor and end in junction_floor:
                outerwall_floor.append([start, end])

    # 将多个顶点对拼接成一条曲线
    outerwall_polygon = convert_lines_to_vertices(outerwall_floor)[0]

    return outerwall_polygon
