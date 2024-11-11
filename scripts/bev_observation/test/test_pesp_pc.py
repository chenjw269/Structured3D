# 测试如何将单视角点云与地平面对齐
import sys
sys.path.append(".")

from scripts.bev.pesp_bev import parse_camera_rpy # 从位姿读取 roll pitch yaw
from scripts.bev.pesp_bev import parse_camera_intrinsic # 计算内参
from scripts.bev.pesp_bev import depth_pesp_proj # 深度图点云
from scripts.utils.read_camera_pose import rotation_matrix_gravity



import os
import cv2
from s3d import *
import open3d as o3d
import numpy as np


def normalize(vector):
    return vector / np.linalg.norm(vector)

def parse_camera_info(camera_info, height, width):
    """ extract intrinsic and extrinsic matrix
    """
    lookat = normalize(camera_info[3:6])
    up = normalize(camera_info[6:9])

    W = lookat
    U = np.cross(W, up)
    V = np.cross(W, U)

    rot = np.vstack((U, V, W))

    trans = camera_info[:3]

    xfov = camera_info[9]
    yfov = camera_info[10]

    K = np.diag([1, 1, 1])

    K[0, 2] = width / 2
    K[1, 2] = height / 2

    K[0, 0] = K[0, 2] / np.tan(xfov)
    K[1, 1] = K[1, 2] / np.tan(yfov)

    return rot, trans, K

def create_pcd_perspective(rgb_image_path, depth_image_path, camera_path):    
    color = cv2.imread(rgb_image_path)
    depth = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

    height, width = color.shape[:2]

    camera_info = np.loadtxt(camera_path)

    rot, trans, K = parse_camera_info(camera_info, height, width)

    trans = np.array(trans) / 1000

    color_o3d = o3d.geometry.Image(color)
    depth_o3d = o3d.geometry.Image(depth)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d, depth_scale=1.0, depth_trunc=10.0, convert_rgb_to_intensity=False)

    extrinsic = np.ones(4)
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rot.T
    extrinsic[:3, -1] = trans
    extrinsic = np.linalg.inv(extrinsic)

    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, K[0][0], K[1][1], K[0][2], K[1][2])
    pointcloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic, extrinsic)

    return pointcloud


if __name__ == "__main__":

    # scene_index = "scene_00000"
    # sample_index = ("485142", "0")

    # # 场景数据目录
    # sample_data_pth = os.path.join(
    #     s3d_data_pth, scene_index, "2D_rendering", sample_index[0], "perspective/full", sample_index[1]
    # )

    ##############################################
    # 显示点云
    ##############################################
    # # 深度图
    # depth = os.path.join(sample_data_pth, "depth.png")
    # depth = cv2.imread(depth, cv2.IMREAD_UNCHANGED)
    # # 相机位姿
    # pose = os.path.join(sample_data_pth, "camera_pose.txt")
    # pose = np.loadtxt(pose)
    # # 获取图像的尺寸
    # height, width = depth.shape
    # # 从相机位姿计算内参
    # K = parse_camera_intrinsic(pose, height, width)
    # # 计算深度图点云
    # depth_pc = depth_pesp_proj(depth, K).reshape(-1, 3)
    # # rgb 图
    # rgb = os.path.join(sample_data_pth, "rgb_rawlight.png")
    # rgb = (cv2.imread(rgb) / 255).reshape(-1, 3)
    
    depth = "e:/datasets/Structure3D/Structured3D/scene_00000/2D_rendering/485142/perspective/full/0/depth.png"
    rgb = "e:/datasets/Structure3D/Structured3D/scene_00000/2D_rendering/485142/perspective/full/0/rgb_rawlight.png"
    pose = "e:/datasets/Structure3D/Structured3D/scene_00000/2D_rendering/485142/perspective/full/0/camera_pose.txt"
    
    pcd = create_pcd_perspective(rgb, depth, pose)
    # # 显示深度图点云
    # o3d.visualization.draw_geometries([pcd])
    
    ###########################################
    # 显示 xyz 轴
    ###########################################
    # 创建点集，记录所有线段起点和终点
    points = np.array([
        [0, 0, 0],  # 原点
        [10, 0, 0],  # X 轴箭头
        [0, 10, 0],  # Y 轴箭头
        [0, 0, 10]   # Z 轴箭头
    ])
    # 创建线段集，记录起点和终点索引
    lines = np.array([
        [0, 1],  # X 轴
        [0, 2],  # Y 轴
        [0, 3]   # Z 轴
    ])
    # Open3d LineSet
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    # 设置颜色
    colors = np.array([
        [1, 0, 0],  # X 轴 (红色)
        [0, 1, 0],  # Y 轴 (绿色)
        [0, 0, 1]   # Z 轴 (蓝色)
    ])
    line_set.colors = o3d.utility.Vector3dVector(colors)
    # # 可视化
    # o3d.visualization.draw_geometries([line_set])
        
    # # 计算 roll pitch yaw
    # view_direction = pose[3:6]
    # gravity_direction = pose[6:9]
    # rm_gravity = rotation_matrix_gravity(view_direction, gravity_direction)
    # depth_pc = depth_pc @ rm_gravity

    # 消除 roll
    
    # 消除 pitch
    
    # 消除 yaw

    # 创建Open3D点云对象
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(depth_pc)
    # pcd.colors = o3d.utility.Vector3dVector(rgb)
    o3d.visualization.draw_geometries([line_set, pcd])
