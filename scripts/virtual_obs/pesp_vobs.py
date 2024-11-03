import cv2
import math
import copy
import numpy as np
from scipy.ndimage import map_coordinates


def generate_rectangle_mask(image, center, angle, fov, mode="occ"):
    pass

def generate_ellipse_mask(image, center, angle, fov, mode="occ"):
    """图像扇形掩码

    Args:
        image (np.array): 输入图像
        center (np.array): 扇形圆心
        angle (float): 扇形朝向
        fov (float): 扇形角度范围
        mode (str, optional): 掩码的对象（占用网格/图像）. Defaults to "occ".

    Returns:
        np.array: 图像掩码
    """
    # 图像形状
    height, width = image.shape[0], image.shape[1] # 256,256
    # 创建掩码
    mask = np.zeros((height, width))

    # # 扇形 x y 轴的半径，为图像对角线长度，以覆盖全部图像内容
    # radius = int(math.sqrt(
    #     pow(height, 2) + pow(width/2, 2)
    # ))
    # 扇形 x y 轴的半径，为 0.025 * 256 = 6.4 m
    radius = 256
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
        [-width / 2, 0],
        [width / 2, 0],
        [width / 2, height],
        [-width / 2, height],
    ])

    angle_rad = np.radians(angle) # 将朝向角度转换为弧度
    cos_angle = np.cos(angle_rad) # 余弦
    sin_angle = np.sin(angle_rad) # 正弦

    # 旋转视野矩形的四个角
    rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
    rotated_corners = np.dot(corners, rotation_matrix) + center

    return rotated_corners.astype(int)

def generate_rotation(image, angle):
    
    # 图像中心
    center = (image.shape[1] // 2, image.shape[0] // 2)
    
    # 获取旋转矩阵，旋转角度为顺时针旋转
    scale = 1.0  # 缩放比例
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    
    # 执行旋转变换
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

    return rotated_image

def virtual_pesp_obs(map_image, pose, fov):

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
    viewrange_corners = generate_viewrange_rectangle(pose, width=128, height=256)
    
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
    observed_area = map_coordinates(map_image, [valid_coords[:, 0], valid_coords[:, 1]], order=1)

    # 创建输出图像
    output_shape = (y_max - y_min + 1, x_max - x_min + 1)
    observed_image = np.zeros(output_shape)
    observed_image[valid_coords[:, 0] - y_min, valid_coords[:, 1] - x_min] = observed_area

    # 旋转虚拟观测
    # cv2 中的正方向为水平向右，转换为正方向竖直向上，需要加上 90 度
    observed_image = generate_rotation(observed_image, (pose[2]+90))
    
    return observed_image