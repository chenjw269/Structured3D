# 20241018
# 根据官方 api 获取的墙壁地图

import os
import cv2
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.plotting import plot_polygon

from misc.colors import semantics_cmap


def convert_lines_to_vertices(lines):
    """convert line representation to polygon vertices
    """
    polygons = []
    lines = np.array(lines)

    polygon = None
    while len(lines) != 0:
        # 多边形中的第一条边，直接取线集的第一条线
        if polygon is None:
            polygon = lines[0].tolist()
            lines = np.delete(lines, 0, 0)

        # 从线集中，查找
        lineID, juncID = np.where(lines == polygon[-1])
        vertex = lines[lineID[0], 1 - juncID[0]]
        lines = np.delete(lines, lineID, 0)

        if vertex in polygon:
            polygons.append(polygon)
            polygon = None
        else:
            polygon.append(vertex)

    return polygons

def visualize_floorplan(args):
    """visualize floorplan
    """
    with open(os.path.join(args.path, f"scene_{args.scene:05d}", "annotation_3d.json")) as file:
        annos = json.load(file)

    # 读取 door window 和各类房间
    planes = []
    for semantic in annos['semantics']:
        for planeID in semantic['planeID']:
            if annos['planes'][planeID]['type'] == 'floor':
                planes.append({'planeID': planeID, 'type': semantic['type']})

    # 顶点 / 交叉点
    junctions = np.array([junc['coordinate'] for junc in annos['junctions']])

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

    # 测试，对象及其多边形
    # polygons = [[[14, 16, 44, 36], 'living room']]
    # polygons = [[[0, 18, 30, 34, 38], 'bedroom']]

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)

    # 创建地图图像
    image = np.ones([800, 800]) * 255

    junctions = np.array([junc['coordinate'][:2] for junc in annos['junctions']])
    #########################################
    # 首先绘制区域的边缘墙面
    # 不绘制 window 和 door，只绘制房间类型
    #########################################
    for (polygon, poly_type) in polygons:
        # 为了让多边形形成一个环，把头节点添加到尾部
        polygon = np.array(polygon + [polygon[0], ])
        # 查表得到节点坐标
        polygon = junctions[polygon]
        # 将多边形的顶点转换成 1 cm/pixel
        for idx, vertex in enumerate(polygon):
            vertex = [int(vertex[0] / 10), int(vertex[1] / 10)]
            polygon[idx] = vertex
        # 各坐标加上偏移量 400 pixel
        for idx, vertex in enumerate(polygon):
            vertex = [int(vertex[0] + 400), int(vertex[1] + 400)]
            polygon[idx] = vertex
        # 绘制
        # 颜色 (B, G, R) 和线的厚度
        if poly_type in ["window", "door"]:
            continue
        else:
            color = (0, 0, 0)   # 黑色
        thickness = 3
        # 在图像上绘制线条
        for idx in range(polygon.shape[0] - 1):
            start_point = polygon[idx].astype(int)
            end_point = polygon[idx + 1].astype(int)
            image = cv2.line(image, start_point, end_point, color, thickness)

    #########################################
    # 其次擦除 window 和 door 的边缘
    #########################################
    for (polygon, poly_type) in polygons:
        # 为了让多边形形成一个环，把头节点添加到尾部
        polygon = np.array(polygon + [polygon[0], ])
        # 查表得到节点坐标
        polygon = junctions[polygon]
        # 将多边形的顶点转换成 1 cm/pixel
        for idx, vertex in enumerate(polygon):
            vertex = [int(vertex[0] / 10), int(vertex[1] / 10)]
            polygon[idx] = vertex
        # 各坐标加上偏移量 400 pixel
        for idx, vertex in enumerate(polygon):
            vertex = [int(vertex[0] + 400), int(vertex[1] + 400)]
            polygon[idx] = vertex
        # 绘制
        # 颜色 (B, G, R) 和线的厚度
        if poly_type in ["door"]:
        # if poly_type in ["window", "door"]:
            color = (255, 255, 255) # 白色
        else:
            continue
        thickness = 3
        # 在图像上绘制线条
        for idx in range(polygon.shape[0] - 1):
            start_point = polygon[idx].astype(int)
            end_point = polygon[idx + 1].astype(int)
            image = cv2.line(image, start_point, end_point, color, thickness)

    # 上下翻转图像
    image = np.flipud(image)

    # 显示图像
    cv2.imshow('Image with Line', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # plt.axis('equal')
    # plt.axis('off')
    # plt.show()
    

def parse_args():
    parser = argparse.ArgumentParser(
        description="Structured3D Floorplan Visualization")
    parser.add_argument("--path", default="E:/datasets/Structure3D/Structured3D",
                        help="dataset path", metavar="DIR")
    parser.add_argument("--scene", default=0,
                        help="scene id", type=int)
    return parser.parse_args()


def main():
    args = parse_args()

    args.path = "E:/datasets/Structure3D/Structured3D"
    args.scene = 0

    visualize_floorplan(args)


if __name__ == "__main__":
    main()
