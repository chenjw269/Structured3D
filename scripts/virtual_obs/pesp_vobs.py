import cv2
import math
import copy
import numpy as np
from scipy.ndimage import map_coordinates

from scripts.utils.view_range_utils import generate_ellipse_mask # 地图扇形掩码
from scripts.utils.view_range_utils import generate_rotation # 地图旋转


def generate_viewrange_rectangle(pose, width=256, height=256):
    """获取视野范围矩形

    Args:
        pose (np.array): 观测位姿
        width (int, optional): 视野矩形宽度. Defaults to 256.
        height (int, optional): 视野矩形高度. Defaults to 256.

    Returns:
        np.array: 视野矩形角点
    """
    center = pose[:2]
    angle = pose[2]
    
    # 原始视野矩形的四个角
    # 单视角视图的视野为前方矩形区域
    corners = np.array([
        [0, -width / 2],
        [0, width / 2],
        [height, width / 2],
        [height, -width / 2],
    ])

    angle_rad = np.radians(-angle) # 将朝向角度转换为弧度
    cos_angle = np.cos(angle_rad) # 余弦
    sin_angle = np.sin(angle_rad) # 正弦

    # 旋转视野矩形的四个角
    rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
    rotated_corners = np.dot(corners, rotation_matrix) + center

    return rotated_corners.astype(int)

def virtual_pesp_viewrect(map_image, pose, fov):
    
    map_image_cp = copy.copy(map_image)

    # #########################################
    # # 虚拟观测的视野
    # #########################################
    # # 将视野外的内容覆盖
    # view_range_mask = generate_ellipse_mask(
    #     image=map_image_cp,
    #     center=pose[:2].astype(int), angle=pose[2], fov=fov, mode="occ")
    # map_image_cp = view_range_mask * map_image_cp

    ##########################################
    # 虚拟观测的范围
    ##########################################
    # 获取视野的边界
    viewrange_corners = generate_viewrange_rectangle(pose, width=256, height=256)

    # 绘制视野矩形
    for i in range(4):
        map_image_cp = cv2.line(
            map_image_cp,
            pt1=viewrange_corners[i], pt2=viewrange_corners[(i+1)%4],
            color=(0,0,255), thickness=3
        )

    # 绘制获取该视野需要截取的范围
    x_min, y_min = np.floor(np.min(viewrange_corners, axis=0)).astype(int) # x y 坐标的最小值
    x_max, y_max = np.ceil(np.max(viewrange_corners, axis=0)).astype(int) # x y 坐标的最大值
    
    map_image_cp = cv2.line(map_image_cp, (x_min, y_min), (x_max, y_min), color=(0,255,0), thickness=3)
    map_image_cp = cv2.line(map_image_cp, (x_max, y_min), (x_max, y_max), color=(0,255,0), thickness=3)
    map_image_cp = cv2.line(map_image_cp, (x_max, y_max), (x_min, y_max), color=(0,255,0), thickness=3)
    map_image_cp = cv2.line(map_image_cp, (x_min, y_max), (x_min, y_min), color=(0,255,0), thickness=3)

    return map_image_cp

def virtual_pesp_obs_v0(map_image, pose, fov=80):
    """根据相机位姿，从 cad 地图上获取虚拟观测 bev

    Args:
        map_image (np.array): cad 地图
        pose (np.array): 相机 2d 位姿 (x,y,yaw)
        fov (float): 视野范围

    Returns:
        np.array: 虚拟观测 bev
    """

    map_image_cp = copy.copy(map_image)

    #########################################
    # 虚拟观测的视野
    #########################################
    # 将视野外的内容覆盖
    view_range_mask = generate_ellipse_mask(
        image=map_image_cp,
        center=pose[:2].astype(int), angle=pose[2], fov=fov, mode="occ")
    map_image_cp = view_range_mask * map_image_cp

    ##########################################
    # 虚拟观测的范围
    ##########################################
    # 获取视野内的部分
    viewrange_corners = generate_viewrange_rectangle(pose, width=256, height=256)
    
    x_min, y_min = np.floor(np.min(viewrange_corners, axis=0)).astype(int) # x y 坐标的最小值
    x_max, y_max = np.ceil(np.max(viewrange_corners, axis=0)).astype(int) # x y 坐标的最大值
    
    # 生成截取范围的网格坐标
    grid_x, grid_y = np.mgrid[0:map_image.shape[0], 0:map_image.shape[1]]
    grid_coords = np.column_stack((grid_y.ravel(), grid_x.ravel()))  # 注意顺序为 (y, x)

    # 创建需要插值的坐标
    mask = ((grid_coords[:, 0] >= y_min) & (grid_coords[:, 0] <= y_max) &
            (grid_coords[:, 1] >= x_min) & (grid_coords[:, 1] <= x_max))
    valid_coords = grid_coords[mask]

    # 从地图中获取相应的像素值
    observed_area = map_coordinates(map_image_cp, [valid_coords[:, 0], valid_coords[:, 1]], order=1)

    # 创建输出图像
    output_shape = (y_max - y_min + 1, x_max - x_min + 1)
    observed_image = np.zeros(output_shape)
    observed_image[valid_coords[:, 0] - y_min, valid_coords[:, 1] - x_min] = observed_area

    ##########################################
    # 旋转虚拟观测
    ##########################################
    # cv2 中的正方向为水平向右，转换为正方向竖直向上，需要加上 90 度
    observed_image = generate_rotation(observed_image, (pose[2]+90))
    
    return observed_image


def virtual_pesp_obs(map_image, pose, fov=80):
    """根据相机位姿，从 cad 地图上获取虚拟观测 bev

    Args:
        map_image (np.array): cad 地图
        pose (np.array): 相机 2d 位姿 (x,y,yaw)
        fov (float): 视野范围

    Returns:
        np.array: 虚拟观测 bev
    """
    map_image_cp = copy.copy(map_image)

    #########################################
    # 虚拟观测的视野
    #########################################

    if isinstance(pose, np.ndarray):
        position = pose[:2].astype(int)
    else:
        position = pose[:2]

    # 将视野外的内容覆盖
    view_range_mask = generate_ellipse_mask(
        image=map_image_cp,
        center=position, angle=pose[2], fov=fov, mode="occ")
    map_image_cp = view_range_mask * map_image_cp

    ##########################################
    # 将视野内容旋转到正方向
    ##########################################
    # cv2 中的正方向为水平向右，转换为正方向竖直向上，需要加上 90 度
    map_image_cp = generate_rotation(
        image=map_image_cp,
        center=pose[:2], angle=(pose[2]+90))

    ##########################################
    # 虚拟观测的范围
    ##########################################
    # 获取视野内的部分
    map_image_cp = map_image_cp[
        pose[1].astype(int) - 256: pose[1].astype(int),
        pose[0].astype(int) - 128: pose[0].astype(int) + 128
    ]
    
    return map_image_cp

def virtual_pesp_obs_batch(map_image, pose, fov=80):
    """批量地，根据相机位姿，从 cad 地图上获取虚拟观测 bev

    Args:
        map_image (np.array): cad 地图
        pose (np.array): 批量相机 2d 位姿 (n,x,y,yaw)
        fov (float): 视野范围

    Returns:
        np.array: 批量虚拟观测 bev
    """
    obs_total = []
    n = pose.shape[0]
    
    for i in range(n):
        obs_item = virtual_pesp_obs(map_image, pose[i], fov)
        obs_total.append(obs_item)
    
    obs_total = np.stack(obs_total)
    
    return obs_total