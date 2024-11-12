import sys
sys.path.append(".")


from s3d import * # s3d 数据集信息
import os # 拼接文件路径
import json # 读取场景标注 json 文件
import math # 无穷大 无穷小 作为初始边界
import numpy as np # 实现读取中的查表操作
import pandas as pd # 保存场景边界信息
import concurrent.futures # 多进程
from tqdm import tqdm # 进度条
from scripts.utils.lines_to_polygons import convert_lines_to_vertices # 多边形转换
from scripts.utils.read_obj import read_obj # 读取标注中的物体


# 统计场景坐标函数
def execute_scene_boundary(scene_index):

    #################################
    # 读取场景标注文件
    #################################
    # 读取场景标注文件
    with open(os.path.join(s3d_data_pth, scene_index, "annotation_3d.json")) as f:
        scene_annos = json.load(f)
    # 地图标注中的物体数据
    obj_map = os.path.join(s3d_data_pth, scene_index, "bbox_3d.json")
    with open(obj_map, encoding="utf-8") as f:
        obj_map = json.load(f)
    # 实例语义映射
    ins2sem = os.path.join(s3d_data_pth, scene_index, "ins_semantic.json")
    with open(ins2sem, encoding="utf-8") as f:
        ins2sem = json.load(f)

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
    for obj in scene_obj:
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
    output_pth = os.path.join(s3d_annos_pth, scene_index, "boundary.csv")
    scene_info_df = pd.DataFrame(scene_info)
    scene_info_df.to_csv(output_pth, index=False)


if __name__ == "__main__":
    
    # Structured3D 包括 3500 个场景
    scene_index_list = [f"scene_{num:05}" for num in range(3500)]
    
    for scene_index in tqdm(scene_index_list):
        # 缺少标注的场景作废
        if scene_index in scene_invalid:
            scene_index_list.remove(scene_index)
    print("Invalid scenes removed")

    with concurrent.futures.ProcessPoolExecutor(max_workers=40) as executor:

        futures = {executor.submit(execute_scene_boundary, task): task for task in scene_index_list}

        with tqdm(total=len(futures)) as pbar:
            for future in concurrent.futures.as_completed(futures):
                task_id = futures[future]
                try:
                    result = future.result()
                    # tqdm.write(result)
                except Exception as e:
                    tqdm.write(f"Task {task_id} generated an exception: {e}")

                pbar.update(1)  # 更新进度条

