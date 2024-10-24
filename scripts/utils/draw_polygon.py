import cv2
import numpy as np


# def draw_polygon(image, polygon, junctions, color, thickness, resolution, map_occ_size):
#     # 为了让多边形形成一个环，把头节点添加到尾部
#     polygon = np.array(polygon + [polygon[0], ])
#     # 查表得到节点坐标
#     polygon = junctions[polygon]
    
#     ###########################################
#     # 坐标转换
#     ###########################################
#     # 多边形顶点坐标，从 mm 转换成 pixel
#     for idx, vertex in enumerate(polygon):
#         vertex = [int(vertex[0] / resolution), int(vertex[1] / resolution)]
#         polygon[idx] = vertex
#     # 各坐标加上偏移量 400 pixel
#     for idx, vertex in enumerate(polygon):
#         vertex = [
#             int(vertex[0] + map_occ_size[0]/2),
#             int(vertex[1] + map_occ_size[1]/2)
#         ]
#         polygon[idx] = vertex
#     # 上下翻转
#     for idx, vertex in enumerate(polygon):
#         vertex = [
#             int(vertex[0]),
#             int(map_occ_size[1] - vertex[1])
#         ]
#         polygon[idx] = vertex

#     ###########################################
#     # 填充网格
#     ###########################################
#     # 在图像上绘制线条
#     for idx in range(polygon.shape[0] - 1):
#         start_point = polygon[idx].astype(int)
#         end_point = polygon[idx + 1].astype(int)
#         image = cv2.line(image, start_point, end_point, color, thickness)

#     return image

def draw_polygon(image, polygon, junctions, color, thickness):

    ###########################################
    # 获取顶点坐标
    ###########################################
    # 为了让多边形形成一个环，把头节点添加到尾部
    polygon = np.array(polygon + [polygon[0], ])
    # 查表得到节点坐标
    polygon = junctions[polygon]

    ###########################################
    # 填充网格
    ###########################################
    # 在图像上绘制线条
    for idx in range(polygon.shape[0] - 1):
        start_point = polygon[idx].astype(int)
        end_point = polygon[idx + 1].astype(int)
        image = cv2.line(image, start_point, end_point, color, thickness)

    return image

# def fill_polygon(image, objects, semantic_label_dict, resolution, map_occ_size):

#     ###########################################
#     # 坐标转换
#     ###########################################
#     # 多边形顶点坐标，从 mm 转换成 pixel
#     for idx, vertex in enumerate(objects):
#         vertex = vertex[0]
#         vertex = (vertex / resolution).astype(int)
#         objects[idx][0] = vertex
#     # 各坐标加上偏移量 400 pixel
#     for idx, vertex in enumerate(objects):
#         vertex = vertex[0]
#         vertex = vertex + (np.array(map_occ_size)/2).astype(int)
#         objects[idx][0] = vertex
#     # 上下翻转
#     for idx, vertex in enumerate(objects):
#         vertex = vertex[0]
#         vertex[:, 1] = map_occ_size[1] - vertex[:, 1]
#         objects[idx][0] = vertex

#     ###########################################
#     # 填充网格
#     ###########################################
#     for instance, ins_semantic in objects:
#         # 查表，得到该语义类别对应的索引
#         if ins_semantic in semantic_label_dict.keys():
#             semantic_label = semantic_label_dict[ins_semantic]
#         else:
#             continue
#         # 填充 2d 占用网格
#         image = cv2.fillPoly(image, [instance], color=semantic_label)
    
#     return image

def fill_polygon(image, objects, semantic_label_dict):

    ###########################################
    # 填充网格
    ###########################################
    for instance, ins_semantic in objects:
        # 查表，得到该语义类别对应的索引
        if ins_semantic in semantic_label_dict.keys():
            semantic_label = semantic_label_dict[ins_semantic]
        else:
            continue
        # 填充 2d 占用网格
        image = cv2.fillPoly(image, [instance], color=semantic_label)
    
    return image
