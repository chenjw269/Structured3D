# 20241018
# 数据增强，单个位置观测值计算的 BEV 过于稀疏，进行插值使得更稠密
import numpy as np
from sklearn.neighbors import NearestNeighbors


def points_noise(points):
    # 设置扰动的范围，调整 scale 以控制新点与原点的距离
    scale = 0.01

    # 生成新的点云
    new_points = []
    for _ in range(5):
        new_points_item = points + np.random.normal(scale=scale, size=points.shape)
        new_points.append(new_points_item)
    new_points = np.array(new_points).reshape(-1, 3)

    # 合并原始点云和新生成的点
    dense_points = np.vstack((points, new_points))

    return dense_points

def points_interpolation(points):
    """对点云进行插值，使得点云更稠密

    Args:
        points (np.array): 点云数组
    """
    # 设置近邻点的数量 K
    K = 5
    
    # KNN找到每个点的最近邻
    nbrs = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(points)
    distances, indices = nbrs.kneighbors(points)

    # 初始化插值点的列表
    interpolated_points = []

    # 对每个点，计算其与每个近邻点之间的中点
    for i in range(len(points)):
        for j in range(1, K):  # 从 1 开始，忽略自己与自己的邻居
            mid_point = (points[i] + points[indices[i, j]]) / 2
            interpolated_points.append(mid_point)

    # 将插值点转换为 numpy 数组
    interpolated_points = np.array(interpolated_points)

    # 合并原始点云和插值生成的点
    dense_points = np.vstack((points, interpolated_points))

    return dense_points
