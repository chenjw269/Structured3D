# bev 处理函数
import sys
sys.path.append(".")

import time
import numpy as np
import open3d as o3d
from tqdm import tqdm
from scripts.bev_pesp.projection import parse_camera_intrinsic # 计算相机内参
from scripts.bev_pesp.projection import depth_pesp_proj # 深度图投影到点云
from scripts.bev_pesp.projection import gravity_align_rotation # 重力方向对齐矩阵
from scripts.bev_pesp.projection import gravity_align_yaw # 重力方向对齐朝向
from scripts.bev.points_aug import * # 点云稠密化方法
from assets.semantic2label import * # 语义-颜色对应关系


def generate_bev(depth, semantic, camera_pose):

    #############################################
    # 从 fov 计算相机内参
    #############################################
    # 获取图像的尺寸
    height, width = depth.shape
    # 从相机位姿计算内参
    K = parse_camera_intrinsic(camera_pose, height, width)

    #############################################
    # 反投影：将深度图反投影为点云
    #############################################
    start_time = time.time()
    
    # 计算深度图点云
    depth_pc = depth_pesp_proj(depth, K)

    # # 与重力方向对齐
    # view_direction = camera_pose[3:6]
    # gravity_direction = camera_pose[6:9]
    # rm_gravity = gravity_align_rotation(view_direction, gravity_direction)
    # # 旋转点云
    # depth_pc = np.dot(depth_pc, rm_gravity)

    # # 与真实朝向对齐
    # R = gravity_align_yaw(view_direction, gravity_direction)
    # depth_pc = np.dot(depth_pc, R.T)

    # 旋转矩阵
    R = np.array([[1, 0, 0],   # X 轴 -> X 轴
                [0, 0, -1],   # Y 轴 -> -Y 轴
                [0, 1, 0]])  # Z 轴 -> Z 轴
    # 调整点云坐标系为：z 轴竖直向上，x 轴水平向右，y 轴垂直向里
    depth_pc = np.dot(depth_pc, R.T)

    end_time = time.time()
    execution_time = end_time - start_time
    # tqdm.write(f"Time for computing PC {execution_time}")

    ##############################################
    # 栅格化：遍历不同语义，栅格化点云，填充网格
    ##############################################
    start_time = time.time()

    # 创建空占用网格
    resolution = 0.025 # 单格大小为 2.5m x 2.5cm
    semantic_voxel_size = 256 # 256 x 256 → 6.4m x 6.4m
    # 增大 occ 范围，以避免旋转后丢失信息
    # semantic_voxel_size = 512 # 512 x 512 → 12.8m x 12.8m
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
        # 可选：下采样点云（使点云更稀疏），有助于显示效果
        # point_cloud = point_cloud.voxel_down_sample(voxel_size)
        # 将点云栅格化为体素
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size=resolution)
        
        # 3. 遍历栅格，填充占用网格
        # 遍历体素，填充占用网格
        voxels = voxel_grid.get_voxels()
        for voxel in voxels:
            # 体素左下角坐标
            voxel_origin = voxel.grid_index * resolution + voxel_grid.origin
            # 计算体素坐标
            # 坐标转换：点云坐标→体素坐标
            voxel_index_0 = int(voxel_origin[0] // resolution) + int(semantic_voxel_size / 2)
            voxel_index_1 = int(voxel_origin[1] // resolution) + semantic_voxel_size
            # voxel_index_1 = int(voxel_origin[1] // resolution) + int(semantic_voxel_size / 2)
            # 填充体素网格，只选择体素网格范围内的点云
            if 0 <= voxel_index_0 < semantic_voxel_size \
                and 0 <= voxel_index_1 < semantic_voxel_size:
                semantic_voxel[voxel_index_1][voxel_index_0] = SEMANTIC_TO_LABEL[semantic_type]

    end_time = time.time()
    execution_time = end_time - start_time
    # tqdm.write(f"Time for computing OCC {execution_time}")

    return semantic_voxel
