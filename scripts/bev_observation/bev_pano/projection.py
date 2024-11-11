# 相机投影和反投影
import numpy as np


def depth_pano_proj(depth):
    """从深度图计算点云
    全景图遵循球面坐标系，水平方向的视野为 360 度，竖直方向的视野为上下各 pi/2
    知道像素点的深度和像素坐标，就可以换算三维坐标

    Args:
        depth (np.array): 深度图图像

    Returns:
        np.array: 深度图点云
    """
    # 获取深度图的维度
    H, W = depth.shape
    # 生成每个像素点的 (u, v) 坐标
    u = np.arange(W)
    v = np.arange(H)
    # 创建网格
    u, v = np.meshgrid(u, v)

    # 计算球面坐标系的经度和纬度
    theta = 2 * np.pi * u / W  # 经度 [0, 2π]
    phi = np.pi * v / H - np.pi / 2  # 纬度 [-π/2, π/2]

    # 计算 3D 坐标
    x = depth * np.cos(phi) * np.cos(theta)
    y = depth * np.cos(phi) * np.sin(theta)
    z = depth * np.sin(phi)

    y = -y  # 左右翻转

    # 将 (x, y, z) 组合成 3D 点云的形式
    points_3d = np.stack((x, y, z), axis=-1)

    return points_3d