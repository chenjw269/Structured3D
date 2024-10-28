import numpy as np


def read_scene_outwall(scene_annos):
    
    #################################
    # 读取场景建筑外墙（outwall）标注
    #################################
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
    # 读取标注中的门窗 window door
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
    
    for outwall_polygon in outwall_planes_floor:
        # 在尾部插入头，令多边形首尾相连
        outwall_polygon = np.array(outwall_polygon + [outwall_polygon[0], ])
        # 顶点表：场景中的所有顶点
        junctions = np.array([junc['coordinate'][:2] for junc in scene_annos['junctions']])
        # 通过顶点表和顶点索引，查表得到顶点坐标
        outwall_polygon = junctions[outwall_polygon]
    
    return outwall_planes_floor

def read_scene_obj(scene_annos):
    

# 从场景标注中，读取 outwall 和所有物体，统计坐标边界
def read_scene_boundary(scene_annos):
    
    