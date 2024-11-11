# 相机投影和反投影
import math
import numpy as np
import open3d as o3d


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
    单视角视图遵循针孔相机模型
    知道像素点的深度和像素坐标，就可以换算三维坐标

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

def compute_yaw_from_view(view_direction):

    tx = view_direction[0]
    ty = view_direction[1]

    # 计算 yaw 角（单位：弧度）
    yaw = math.atan2(tx, ty)

    # 如果需要转换为角度：
    yaw_deg = math.degrees(yaw)
    
    return yaw_deg

def gravity_align_rotation(direction, up):
    """通过重力方向对相机坐标系进行对齐

    Args:
        direction (np.array): 相机朝向向量
        up (np.array): 重力方向向量（竖直向上）

    Returns:
        np.array: 校准相机坐标系的旋转矩阵
    """
    # 归一化视角方向向量
    direction = direction / np.linalg.norm(direction)
    # print(f"Original direction {direction}")
    # 归一化重力方向向量
    up = up / np.linalg.norm(up)
    # print(f"Original gravity {up}")
    # 计算右向量 (right vector)
    right = np.cross(direction, up)
    right /= np.linalg.norm(right)
    
    # # 重新计算相机朝向
    # corrected_direction = -np.cross(right, up)
    # print(f"Corrected direction {corrected_direction}")
    # 重新计算重力方向
    corrected_up = -np.cross(direction, right) # 校正后的重力方向向量（竖直向上）
    # print(f"Corrected gravity {corrected_up}")

    # 创建旋转矩阵
    # rotation_matrix = np.array([right, up, corrected_direction]) # 将原本的相机朝向对准到新的相机朝向
    rotation_matrix = np.array([right, corrected_up, direction]) # 将原本的重力方向对准到新的重力方向）

    return rotation_matrix

def gravity_align_yaw(direction, up):
    """通过重力方向对 roll pitch yaw 进行对齐

    Args:
        direction (np.array): 相机朝向向量
        up (np.array): 重力方向向量（竖直向上）

    Returns:
        float: roll pitch yaw
    """
    # 归一化视角方向向量
    direction = direction / np.linalg.norm(direction)
    # print(f"Original direction {direction}")
    # 归一化重力方向向量
    up = up / np.linalg.norm(up)
    # print(f"Original gravity {up}")
    # 计算右向量 (right vector)
    right = np.cross(up, direction)
    right /= np.linalg.norm(right)
    
    # # 重新计算相机朝向
    corrected_direction = np.cross(right, up)
    # # print(f"Corrected direction {corrected_direction}")
    # 重新计算重力方向
    # corrected_up = -np.cross(direction, right) # 校正后的重力方向向量（竖直向上）
    # print(f"Corrected gravity {corrected_up}")
    
    # 创建旋转矩阵
    rotation_matrix = np.array([right, corrected_direction, up]) # 将原本的相机朝向对准到新的相机朝向
    # rotation_matrix = np.array([right, corrected_up, direction]) # 将原本的重力方向对准到新的重力方向）
    

    # 提取欧拉角 (yaw, pitch, roll)
    yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]) + np.pi/2
    pitch = np.arcsin(-rotation_matrix[2, 0])
    roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])

    # 转换为度
    yaw_deg = np.degrees(yaw) # 绕 Y 轴的偏航角
    pitch_deg = np.degrees(pitch) # 绕 X 轴的俯仰角
    roll_deg = np.degrees(roll) # 绕 Z 轴的滚转角

    # print(f"Direction {corrected_direction}")
    print(f"Roll-pitch-yaw {roll_deg} {pitch_deg} {yaw_deg}")

    # yaw 计算绕 Z 轴旋转矩阵
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1]])
    # pitch 计算绕 Y 轴旋转矩阵
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
               [0, 1, 0],
               [-np.sin(pitch), 0, np.cos(pitch)]])
    # 计算绕 X 轴旋转矩阵
    Rx = np.array([[1, 0, 0],
               [0, np.cos(roll), -np.sin(roll)],
               [0, np.sin(roll), np.cos(roll)]])

    # 总旋转矩阵
    R = np.dot(Rz, np.dot(Ry, Rx))

    # return roll, pitch, yaw
    # return R
    return Rz

