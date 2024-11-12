# 在地图上可视化样本的位置
import cv2
import copy
import numpy as np


def vis_sample_positions(map_img, positions):

    map_img_vis = copy.copy(map_img)
    for p in positions:
        # 样本位置
        map_img_vis = cv2.circle(map_img_vis, p, radius=3, color=(0,0,0), thickness=-1)
        # 显示几个同心圆
        map_img_vis = cv2.circle(map_img_vis, p, radius=6, color=(0,0,0), thickness=1)
        map_img_vis = cv2.circle(map_img_vis, p, radius=6, color=(0,0,0), thickness=1)
    
    return map_img_vis

def vis_sample_pose(map_img, poses):
    
    map_img_vis = copy.copy(map_img)
    for index in range(poses.shape[0]):
        p = poses[index]
        
        # 样本位置
        map_img_vis = cv2.circle(map_img_vis, p[:2].astype(int), radius=3, color=(0,0,0), thickness=-1)
        # 显示几个同心圆
        map_img_vis = cv2.circle(map_img_vis, p[:2].astype(int), radius=9, color=(0,0,0), thickness=1)
        map_img_vis = cv2.circle(map_img_vis, p[:2].astype(int), radius=18, color=(0,0,0), thickness=1)
        map_img_vis = cv2.circle(map_img_vis, p[:2].astype(int), radius=27, color=(0,0,0), thickness=1)
        # 显示朝向
        start_point = p[:2].astype(int) # 箭头起点
        angle = p[2] # 朝向角度（度数）
        length = 100  # 箭头长度
        radians = np.radians(angle)  # 将角度转换为弧度
        end_point = (
            int(start_point[0] + length * np.cos(radians)),
            int(start_point[1] + length * np.sin(radians))
        )
        color = (0, 0, 0)  # 箭头颜色（蓝色）
        thickness = 2        # 箭头粗细
        tip_length = 0.05  # 箭头大小（10%）
        cv2.arrowedLine(map_img_vis, start_point, end_point, color, thickness, tipLength=tip_length)

    return map_img_vis
