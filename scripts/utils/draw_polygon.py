import cv2
import numpy as np


def draw_polygon(image, polygon, junctions, color, thickness):
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
    # 在图像上绘制线条
    for idx in range(polygon.shape[0] - 1):
        start_point = polygon[idx].astype(int)
        end_point = polygon[idx + 1].astype(int)
        image = cv2.line(image, start_point, end_point, color, thickness)

    return image
