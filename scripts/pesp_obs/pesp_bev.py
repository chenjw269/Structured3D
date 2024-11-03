# 20241029
# 视野受限条件下的 BEV
import cv2
import math
import numpy as np


# 视野范围大小
view_range = 60
view_range = 90
view_range = 120


def get_field_of_view(array, start_angle, field_of_view, center=(128, 128)):
    height, width, _ = array.shape # 256,256,3
    center_x, center_y = center
    
    # 转换角度为弧度
    start_angle_rad = np.radians(start_angle)
    field_of_view_rad = np.radians(field_of_view)
    
    # 计算视野的边界角度
    end_angle_rad = start_angle_rad + field_of_view_rad
    
    # 计算视野的坐标范围
    angle_range = np.linspace(start_angle_rad, end_angle_rad, num=width)
    
    # 计算对应的 x, y 坐标
    x = center_x + (height / 2) * np.cos(angle_range)
    y = center_y + (height / 2) * np.sin(angle_range)
    
    # 确保坐标在范围内
    x = np.clip(x, 0, height - 1).astype(int)
    y = np.clip(y, 0, width - 1).astype(int)

    # 提取视野区域
    fov_image = np.zeros((height, width, 3), dtype=array.dtype)
    for i in range(len(x) - 1):
        rr, cc = np.linspace(y[i], y[i+1], num=height).astype(int), np.linspace(x[i], x[i+1], num=height).astype(int)
        fov_image[rr, cc] = array[rr, cc]

    return fov_image


def generate_ellipse_mask(image, angle, fov):

    # 图像形状
    height, width = image.shape[0], image.shape[1] # 256,256
    # 创建掩码
    mask = np.zeros((height, width))
    # 扇形的中心，也就是图像的中心
    center = (int(height/2), int(width/2))
    # 扇形 x y 轴的半径，为图像对角线长度的一半
    radius = int(math.sqrt(
        pow(height/2, 2) + pow(width/2, 2)
    ))
    radius = (radius, radius)
    # 扇形的开始和结束角度，角度范围从 x 轴正方向开始
    start_angle = angle - fov/2
    end_angle = angle + fov/2
    # 绘制扇形
    mask = cv2.ellipse(mask, center, radius, 0, start_angle, end_angle, 1, -1)
    
    return mask


def generate_rotation(image, angle):
    
    # 图像中心
    center = (image.shape[1] // 2, image.shape[0] // 2)
    
    # 获取旋转矩阵，旋转角度为顺时针旋转
    scale = 1.0  # 缩放比例
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    
    # 执行旋转变换
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

    return rotated_image