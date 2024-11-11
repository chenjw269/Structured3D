# bev 处理函数
import sys
sys.path.append(".")

import time
import numpy as np
from tqdm import tqdm
from scripts.bev_pano.projection import depth_pano_proj # 深度图投影到点云


def generate_bev(depth, semantic):

    #############################################
    # 反投影：将深度图反投影为点云
    #############################################
    start_time = time.time()
    
    # 深度值从 mm 换算为 m
    depth = depth / 1000
    # 计算深度图点云
    depth_pc = depth_pano_proj(depth)

    # 调整点云坐标系为：z 轴竖直向上，x 轴水平向右，y 轴垂直向里
    # 旋转矩阵
    R = np.array([[0, 1, 0],   # X 轴 -> Y 轴
                [1, 0, 0],   # Y 轴 -> X 轴
                [0, 0, -1]]) # Z 轴 -> -Z 轴 (反转方向)
    depth_pc = np.dot(depth_pc, R.T)

    end_time = time.time()
    execution_time = end_time - start_time
    tqdm.write(f"Time for computing PC {execution_time}")

    ##############################################
    # 栅格化：遍历不同语义，栅格化点云，填充网格
    ##############################################
    start_time = time.time()

    # 创建空占用网格
    resolution = 0.025 # 单格大小为 2.5m x 2.5cm
    semantic_voxel_size = 256 # 256 x 256 → 6.4m x 6.4m
    semantic_voxel = np.zeros([semantic_voxel_size, semantic_voxel_size])
    # 遍历不同语义类别    
    semantic_type_list = list(SEMANTIC_TO_LABEL.keys())
    for semantic_type in semantic_type_list:
        
        # 1. 找到该语义类别的点云
        # 查表，得到该语义类别对应的分割颜色
        semantic_color = SEMANTIC_TO_COLOR[semantic_type]
        # 查找该语义类别的点云索引
        semantic_index = np.all(semantic == semantic_color, axis=-1)
        # 索引该语义类别对应的全部点云
        semantic_depth = depth_pc[semantic_index]
        # 如果没有该类别的点云，则跳转到下一个语义类别
        if semantic_depth.shape[0] == 0:
            continue
        
        # 2. 点云栅格化
        # semantic_depth = points_noise(semantic_depth) # 1. 通过噪声点云稠密化
        semantic_depth = points_interpolation(semantic_depth) # 2. 通过插值点云稠密化
        # 将点云格式由 numpy 数组转换为 Open3D 点云对象
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(semantic_depth)
        # 将点云栅格化为体素
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size=resolution)
        
        # 3. 遍历栅格，填充占用网格
        voxels = voxel_grid.get_voxels()
        for voxel in voxels:
            # 体素左下角坐标
            voxel_origin = voxel.grid_index * resolution + voxel_grid.origin
            # 计算体素中心坐标（点云坐标）在 bev 网格上的坐标
            voxel_index_0 = int(voxel_origin[0] // resolution) + int(semantic_voxel_size / 2)
            voxel_index_1 = int(voxel_origin[1] // resolution) + int(semantic_voxel_size / 2)
            # 填充体素网格（越界检查，只选择体素网格坐标范围内的点云）
            if 0 <= voxel_index_0 < semantic_voxel_size \
                and 0 <= voxel_index_1 < semantic_voxel_size:
                semantic_voxel[voxel_index_1][voxel_index_0] = SEMANTIC_TO_LABEL[semantic_type]

    end_time = time.time()
    execution_time = end_time - start_time
    tqdm.write(f"Time for computing OCC {execution_time}")

    return semantic_voxel
