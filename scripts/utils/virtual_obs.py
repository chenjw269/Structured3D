# 虚拟观测：提供场景 cad 地图和相机位姿，获取虚拟视野

import cv2
import math
import copy
import numpy as np
from scipy.ndimage import map_coordinates


#################################################
# 虚拟观测，模拟已知位姿下的观测
#################################################

def get_rotated_rect_corners(pose, width=256, height=256):
    
    center = pose[:2]
    angle = pose[2]
    
    angle_rad = np.radians(angle) # 将朝向角度转换为弧度
    cos_angle = np.cos(angle_rad) # 余弦
    sin_angle = np.sin(angle_rad) # 正弦

    # 原始视野矩形的四个角
    # 1. 周围环视正方形区域
    # corners = np.array([
    #     [-width / 2, -height / 2],
    #     [width / 2, -height / 2],
    #     [width / 2, height / 2],
    #     [-width / 2, height / 2]
    # ])
    # 2. 前方矩形区域
    corners = np.array([
        [-width / 2, 0],
        [width / 2, 0],
        [width / 2, height],
        [-width / 2, height],
    ])

    # 旋转视野矩形的四个角
    rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
    rotated_corners = np.dot(corners, rotation_matrix) + center

    return rotated_corners

def generate_rotation(image, angle):
    
    # 图像中心
    center = (image.shape[1] // 2, image.shape[0] // 2)
    
    # 获取旋转矩阵，旋转角度为顺时针旋转
    scale = 1.0  # 缩放比例
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    
    # 执行旋转变换
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

    return rotated_image

#################################################
# 虚拟视野，模拟受限视角下的观测
#################################################

def generate_ellipse_mask(image, center, angle, fov, mode="occ"):

    # 图像形状
    height, width = image.shape[0], image.shape[1] # 256,256
    # 创建掩码
    mask = np.zeros((height, width))
    # # 扇形的中心，也就是图像的中心
    # center = (int(height/2), int(width/2))
    # 扇形 x y 轴的半径，为图像对角线长度的一半
    radius = int(math.sqrt(
        pow(height, 2) + pow(width/2, 2)
    ))
    radius = (radius, radius)
    # 扇形的开始和结束角度，角度范围从 x 轴正方向开始
    start_angle = angle - fov/2
    end_angle = angle + fov/2
    # 绘制扇形
    mask = cv2.ellipse(mask, center, radius, 0, start_angle, end_angle, 1, -1)

    if mode == "occ":
        return mask
    elif mode == "image":
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)  

    return mask


def virtual_fov(map_image, pose, fov, mode="occ"):
    
    map_image_cp = copy.copy(map_image)
    
    mask_image = generate_ellipse_mask(map_image, pose[:2].astype(int), pose[2], fov, mode=mode)
    map_image_cp = mask_image * map_image_cp

    return map_image_cp

#################################################
# 虚拟观测主函数
#################################################

def virtual_observation(map_image, pose, fov=180):
    
    corners = get_rotated_rect_corners(pose, width=128, height=256)

    # 生成矩形内的网格
    x_min, y_min = np.floor(np.min(corners, axis=0)).astype(int)
    x_max, y_max = np.ceil(np.max(corners, axis=0)).astype(int)

    # 生成截取范围的网格坐标
    grid_x, grid_y = np.mgrid[0:map_image.shape[0], 0:map_image.shape[1]]
    grid_coords = np.column_stack((grid_y.ravel(), grid_x.ravel()))  # 注意顺序为 (y, x)

    # 创建需要插值的坐标
    mask = ((grid_coords[:, 0] >= y_min) & (grid_coords[:, 0] <= y_max) &
            (grid_coords[:, 1] >= x_min) & (grid_coords[:, 1] <= x_max))
    valid_coords = grid_coords[mask]

    # 从地图中获取相应的像素值
    observed_area = map_coordinates(map_image, [valid_coords[:, 0], valid_coords[:, 1]], order=1)

    # 创建输出图像
    output_shape = (y_max - y_min + 1, x_max - x_min + 1)
    observed_image = np.zeros(output_shape)
    observed_image[valid_coords[:, 0] - y_min, valid_coords[:, 1] - x_min] = observed_area

    # 按照位姿中的朝向进行旋转
    # cv2 中的正方向为水平向右，转换为正方向竖直向上
    observed_image = generate_rotation(observed_image, (pose[2]+90))

    # 按照位姿中的视野进行覆盖
    fov_center = np.array([observed_image.shape[1]/2, observed_image.shape[0], -90])
    observed_image_fov = virtual_fov(observed_image, fov_center, fov)

    return observed_image_fov