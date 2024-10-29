import numpy as np


# 将深度图转换为点云
def depth_to_3d(depth_map):
    """将全景深度图转换成点云，单位与深度值单位相同

    Args:
        depth_map (np.array): 深度图矩阵

    Returns:
        np.array: 深度图点云
    """
    # 获取深度图的维度
    H, W = depth_map.shape
    # 生成每个像素点的 (u, v) 坐标
    u = np.arange(W)
    v = np.arange(H)
    # 创建网格
    u, v = np.meshgrid(u, v)

    # 计算球面坐标系的经度和纬度
    theta = 2 * np.pi * u / W  # 经度 [0, 2π]
    phi = np.pi * v / H - np.pi / 2  # 纬度 [-π/2, π/2]

    # 计算 3D 坐标
    x = depth_map * np.cos(phi) * np.cos(theta)
    y = depth_map * np.cos(phi) * np.sin(theta)
    y = -y  # 左右翻转
    z = depth_map * np.sin(phi)

    # 将 (x, y, z) 组合成 3D 点云的形式
    points_3d = np.stack((x, y, z), axis=-1)

    return points_3d
