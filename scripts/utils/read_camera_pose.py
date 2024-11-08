# 读取相机位姿
import math
import numpy as np


def compute_euler_angles(direction):
    """计算相机位姿中的朝向角

    Args:
        direction (np.array): 相机朝向向量

    Returns:
        float: xyz 轴各自的朝向角
    """

    # 归一化朝向向量
    direction = direction / np.linalg.norm(direction)

    # 计算欧拉角
    yaw = np.arctan2(direction[1], direction[0])
    pitch = np.arcsin(-direction[2])
    roll = 0.0

    # 转换为度
    yaw = np.degrees(yaw)
    pitch = np.degrees(pitch)
    roll = np.degrees(roll)
    
    print(f"Direction {direction}")
    print(f"Roll-pitch-yaw {roll} {pitch} {yaw}")

    return roll, pitch, yaw


def rotation_matrix_gravity(direction, up):
    # 归一化方向向量
    direction = direction / np.linalg.norm(direction)
    print(f"Original direction {direction}")

    # 归一化重力方向向量
    up = up / np.linalg.norm(up)
    print(f"Original gravity {up}")

    # 计算右向量 (right vector)
    right = np.cross(direction, up)
    right /= np.linalg.norm(right)
    
    # # 重新计算相机朝向
    # corrected_direction = -np.cross(right, up)
    # print(f"Corrected direction {corrected_direction}")
    # 重新计算
    corrected_up = -np.cross(direction, right) # 校正后的重力方向向量（竖直向上）
    print(f"Corrected gravity {corrected_up}")

    # 创建旋转矩阵（将原本的重力方向对准到新的重力方向）
    # rotation_matrix = np.array([right, up, corrected_direction])
    rotation_matrix = np.array([right, corrected_up, direction])
    
    
    return rotation_matrix

def compute_euler_angles_wgravity(direction, up):
    """在重力方向校正下，计算相机位姿中的朝向角

    Args:
        direction (np.array): 相机朝向向量
        up (np.array): 重力方向向量（竖直向上）

    Returns:
        float: xyz 轴各自的朝向角
    """
    
    # 归一化方向向量
    direction = direction / np.linalg.norm(direction)
    print(direction)

    # 归一化重力方向向量
    up = up / np.linalg.norm(up)

    # 计算右向量 (right vector)
    right = np.cross(up, direction)
    right /= np.linalg.norm(right)
    
    # 重新计算相机朝向
    corrected_direction = np.cross(right, up)
    # print(corrected_direction)

    # 创建旋转矩阵
    rotation_matrix = np.array([right, corrected_direction, up])
    # print(rotation_matrix)

    # 提取欧拉角 (yaw, pitch, roll)
    yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    pitch = np.arcsin(-rotation_matrix[2, 0])
    roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])

    # 转换为度
    yaw = np.degrees(yaw) # 绕 Y 轴的偏航角
    pitch = np.degrees(pitch) # 绕 X 轴的俯仰角
    roll = np.degrees(roll) # 绕 Z 轴的滚转角

    print(f"Direction {corrected_direction}")
    print(f"Roll-pitch-yaw {roll} {pitch} {yaw}")

    return roll, pitch, yaw

def compute_yaw_from_view(view_direction):
    """从相机朝向向量计算 yaw 角

    Args:
        view_direction (np.array): 相机朝向方向的单位向量

    Returns:
        float: 相机朝向的 yaw 角
    """
    tx = view_direction[0]
    ty = view_direction[1]

    # 计算 yaw 角（单位：弧度）
    yaw = math.atan2(tx, ty)

    # 如果需要转换为角度：
    yaw_deg = math.degrees(yaw)
    
    return yaw_deg

def read_camera_pose(pose, mode="raw"):

    assert mode in ["raw", "gravity"], "Error: mode should be either 'raw' or 'gravity'."

    # pose = np.loadtxt(pose)

    transistion = pose[:2] # 2d 位置

    t = pose[3:6]
    u = pose[6:9]
    
    # 1. 直接计算朝向
    if mode == "raw":
        yaw = compute_yaw_from_view(t)
    # 2. 根据重力方向调整计算朝向
    elif mode == "gravity":
        roll, pitch, yaw = compute_euler_angles_wgravity(t, u)
    else:
        yaw = None
        print("Unhandled mode")

    pose = np.append(transistion, yaw)
    
    return pose
