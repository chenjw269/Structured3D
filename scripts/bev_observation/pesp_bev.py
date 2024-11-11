import sys
sys.path.append(".")

import os
import cv2
import time
import open3d as o3d
import numpy as np
from s3d import *
from scripts.bev.points_aug import * # 点云稠密化方法
from assets.semantic2label import * # 语义-颜色对应关系
from scripts.utils.read_camera_pose import rotation_matrix_gravity


def parse_camera_rpy(camera_info):
    
    view_direction = camera_info[3:6]
    up_direction = camera_info[6:9]
    
    # 1. 归一化视线方向和上方向
    view_direction = view_direction / np.linalg.norm(view_direction)
    up_direction = up_direction / np.linalg.norm(up_direction)

    # 2. 计算 yaw 角
    yaw = np.arctan2(view_direction[0], view_direction[2])

    # 3. 计算 pitch 角
    pitch = np.arcsin(-view_direction[1])  # 视线方向在 y 轴的分量

    # 4. 计算 roll 角
    # 先计算右方向
    right_direction = np.cross(up_direction, view_direction)
    right_direction = right_direction / np.linalg.norm(right_direction)

    # 重新计算上方向以确保正交性
    up_direction = np.cross(view_direction, right_direction)

    # 计算 roll 角
    roll = np.arctan2(up_direction[2], up_direction[1])  # 上方向在 y-z 平面的分量

    # 输出角度（将弧度转换为角度）
    yaw_deg = np.degrees(yaw)
    pitch_deg = np.degrees(pitch)
    roll_deg = np.degrees(roll)

    return roll_deg, pitch_deg, yaw_deg

def parse_camera_direction(camera_info):
    
    view_direction = camera_info[3:6]
    up_direction = camera_info[6:9]
    
    # 1. 归一化视线方向和上方向
    view_direction = view_direction / np.linalg.norm(view_direction)
    up_direction = up_direction / np.linalg.norm(up_direction)

    # 2. 计算右方向（视线方向和上方向的叉积）
    right_direction = np.cross(up_direction, view_direction)
    right_direction = right_direction / np.linalg.norm(right_direction)

    # 3. 重新计算上方向，保证正交
    up_direction = np.cross(view_direction, right_direction)
    
    # #####################################
    # # 消除 roll pitch yaw 旋转
    # #####################################
    # rotation_matrix = np.array([
    #     right_direction,
    #     up_direction,
    #     -view_direction  # 注意这里取负号，使得z轴朝外
    # ]).T

    #####################################
    # 消除 roll pitch 旋转
    #####################################    
    yaw_angle = np.arctan2(view_direction[0], view_direction[2]) + np.pi/2

    # 5. 构建 yaw 旋转矩阵
    rotation_matrix = np.array([
        [np.cos(yaw_angle), 0, np.sin(yaw_angle)],
        [0,                 1, 0],
        [-np.sin(yaw_angle), 0, np.cos(yaw_angle)]
    ])

    return rotation_matrix

def parse_camera_intrinsic(camera_info, height=720, width=1280):
    """从图像的形状和相机视野计算相机的内参

    Args:
        camera_info (np.array): 相机的视野等信息
        height (int, optional): 图像高度. Defaults to 720.
        width (int, optional): 图像宽度. Defaults to 1280.

    Returns:
        np.array: 相机内参矩阵
    """
    # half-angles of the horizontal and vertical fields of view 视野
    xfov = camera_info[9]
    yfov = camera_info[10]

    K = np.diag([1, 1, 1]) # 对角单位矩阵
    K[0, 2] = width / 2
    K[1, 2] = height / 2
    K[0, 0] = K[0, 2] / np.tan(xfov) # 图像半宽度，除以横向视野的正切值
    K[1, 1] = K[1, 2] / np.tan(yfov) # 图像半高度，除以纵向视野的正切值
    
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, K[0][0], K[1][1], K[0][2], K[1][2])

    return intrinsic.intrinsic_matrix

def depth_pesp_proj(depth, ins):
    """从深度图计算点云

    Args:
        depth (np.array): 深度图图像
        ins (np.array): 相机内参矩阵

    Returns:
        np.array: 深度图点云
    """
    height, width = depth.shape
    
    fx = ins[0][0]
    fy = ins[1][1]
    cx = ins[0][2]
    cy = ins[1][2]
    
    point_cloud = []
    
    for v in range(height):
        for u in range(width):
            
            z = depth[v, u] # 深度值，单位为毫米
            z = z / 1000.0 # 转换单位为米
            
            x = (u - cx) * z / fx # 将像素的横坐标，换算为世界 x 坐标
            y = (v - cy) * z / fy # 将像素的纵坐标，换算为世界 y 坐标
            
            point_cloud.append((x, y, z))
    
    point_cloud = np.array(point_cloud).reshape(height, width, 3)

    return point_cloud

def generate_pesp_bev(depth, semantic, camera_pose):

    start_time = time.time()

    #############################################
    # 反投影：将深度图反投影为点云
    #############################################
    # 获取图像的尺寸
    height, width = depth.shape
    # 从相机位姿计算内参
    K = parse_camera_intrinsic(camera_pose, height, width)
    # 计算深度图点云
    depth_pc = depth_pesp_proj(depth, K)

    ##############################################
    # 对齐
    ##############################################
    view_direction = camera_pose[3:6]
    gravity_direction = camera_pose[6:9]
    rm_gravity = rotation_matrix_gravity(view_direction, gravity_direction)
    depth_pc = np.dot(depth_pc, rm_gravity)

    # 旋转矩阵
    R = np.array([[1, 0, 0],   # X 轴 -> X 轴
                [0, -1, 0],   # Y 轴 -> -Y 轴
                [0, 0, 1]])  # Z 轴 -> Z 轴
    # 调整点云坐标系为：z 轴竖直向上，x 轴水平向右，y 轴垂直向里
    depth_pc = np.dot(depth_pc, R.T)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Processing depth pc {execution_time}")

    start_time = time.time()

    ##############################################
    # 栅格化：遍历不同语义，栅格化点云，填充网格
    ##############################################
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
            voxel_index_1 = int(voxel_origin[1] // resolution) + semantic_voxel_size # int(semantic_voxel_size / 2)
            # 填充体素网格，只选择体素网格范围内的点云
            if 0 <= voxel_index_0 < semantic_voxel_size \
                and 0 <= voxel_index_1 < semantic_voxel_size:
                semantic_voxel[voxel_index_1][voxel_index_0] = SEMANTIC_TO_LABEL[semantic_type]

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Processing occ {execution_time}")

    return semantic_voxel


def executing_pesp_bev_processing(scene_index, sample_index):

    # 场景数据目录
    sample_data_pth = os.path.join(
        s3d_data_pth, scene_index, "2D_rendering", sample_index[0], "perspective/full", sample_index[1]
    )
    # 深度图
    depth = os.path.join(sample_data_pth, "depth.png")
    depth = cv2.imread(depth, cv2.IMREAD_UNCHANGED)

    # 语义分割图
    semantic = os.path.join(sample_data_pth, "semantic.png")
    # 默认是 BGR 通道读取
    semantic = cv2.imread(semantic)
    # 将 BGR 转换为 RGB
    semantic = cv2.cvtColor(semantic, cv2.COLOR_BGR2RGB)

    # 相机位姿
    pose = os.path.join(sample_data_pth, "camera_pose.txt")
    pose = np.loadtxt(pose)

    # 创建 bev
    bev = generate_pesp_bev(depth, semantic, pose)

    return bev
