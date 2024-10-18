import numpy as np

import sys
sys.path.append(".")

from misc.utils import get_corners_of_bb3d_no_index


def read_obj(objs, ins2sem, ):
    
    obj_ret = []
    
    # 遍历该场景下的实例
    for obj in objs:
        
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

        obj_ret.append([polygon, ins_semantic])

    return obj_ret
