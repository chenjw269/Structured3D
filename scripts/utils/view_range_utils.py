import cv2
import numpy as np


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

def generate_rectangle_mask(image, center, angle, view_range, mode="occ"):
    """图像矩形掩码

    Args:
        image (np.array): 输入图像
        center (np.array): 矩形中心
        angle (float): 矩形朝向
        view_range (float): 视野范围
        mode (str, optional): 掩码的对象（占用网格/图像）. Defaults to "occ".
    """
    # 图像形状
    height, width = image.shape[0], image.shape[1] # 256,256
    # 创建掩码
    mask = np.zeros((height, width))
    
    # 旋转前的矩形顶点
    corners = np.array([
        [-view_range / 2, -view_range / 2],
        [-view_range / 2, view_range / 2],
        [view_range / 2, view_range / 2],
        [view_range / 2, -view_range / 2],
    ])
    
    # 旋转后的矩形顶点
    angle_rad = np.radians(-angle) # 将朝向角度转换为弧度
    cos_angle = np.cos(angle_rad) # 余弦
    sin_angle = np.sin(angle_rad) # 正弦
    rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
    rotated_corners = np.dot(corners, rotation_matrix) + center
    
    # 将矩形范围内的掩码设为 1
    x_min, y_min = np.floor(np.min(rotated_corners, axis=0)).astype(int) # x y 坐标的最小值
    x_max, y_max = np.ceil(np.max(rotated_corners, axis=0)).astype(int) # x y 坐标的最大值
    mask[x_min: x_max, y_min: y_max] = 1

    if mode == "occ":
        return mask
    elif mode == "image":
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    return mask

def generate_rotation(image, center, angle):
    
    # cv2.getRotationMatrix2D 要求 center 元素为 float
    center = center.astype(np.float64)
    
    # 获取旋转矩阵，旋转角度为顺时针旋转
    scale = 1.0  # 缩放比例
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    
    # 执行旋转变换
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST)

    return rotated_image
