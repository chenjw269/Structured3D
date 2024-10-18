import numpy as np

import sys
sys.path.append(".")

from scripts.utils.lines_to_polygons import convert_lines_to_vertices


def read_area_wall(annos):
    
    # 读取 door window 和各类房间
    planes = []
    for semantic in annos['semantics']:
        for planeID in semantic['planeID']:
            if annos['planes'][planeID]['type'] == 'floor':
                planes.append({'planeID': planeID, 'type': semantic['type']})

    # 为对象创建多边形
    polygons = []
    for plane in planes:
        # 根据对象的 planeID，找到对应的 lineIDs
        lineIDs = np.where(np.array(annos['planeLineMatrix'][plane['planeID']]))[0].tolist()
        # 根据对象的 lineIDs，找到对应的 junction pairs 顶点对
        junction_pairs = [np.where(np.array(annos['lineJunctionMatrix'][lineID]))[0].tolist() for lineID in lineIDs]
        # 把顶点对转换成一个首尾相连的顶点环
        polygon = convert_lines_to_vertices(junction_pairs)
        polygons.append([polygon[0], plane['type']])

    return polygons
