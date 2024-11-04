import numpy as np
import open3d as o3d


def normalize(vector):
    # 归一化
    return vector / np.linalg.norm(vector)

def parse_camera_info(camera_info, height=720, width=1280):
    
    # view direction 视线朝向的方向
    view_direction = normalize(camera_info[3:6])
    # up direction 重力校正方向
    up_direction = normalize(camera_info[6:9])
    
    W = view_direction
    U = np.cross(W, up_direction) # 视线方向和重力方向叉积，右方向
    V = np.cross(W, U) # 视线方向和右方向叉积，纠正后的重力方向

    # 返回值 1：朝向
    rot = np.vstack((U, V, W))

    # 返回值 2：位置    
    # eye viewpoint of the camera 视点位置
    trans = camera_info[:3]
    
    # half-angles of the horizontal and vertical fields of view 视野
    xfov = camera_info[9]
    yfov = camera_info[10]

    # 返回值 3：内参
    K = np.diag([1, 1, 1])

    K[0, 2] = width / 2
    K[1, 2] = height / 2

    K[0, 0] = K[0, 2] / np.tan(xfov)
    K[1, 1] = K[1, 2] / np.tan(yfov)

    return rot, trans, K

    
if __name__ == "__main__":

    height, width = 720, 1280

    camera_info = "e:/datasets/Structure3D/Structured3D/scene_00000/2D_rendering/485142/perspective/full/0/camera_pose.txt"
    camera_info = np.loadtxt(camera_info)
    
    rot, trans, K = parse_camera_info(camera_info)

    extrinsic = np.ones(4)
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rot.T
    extrinsic[:3, -1] = trans
    extrinsic = np.linalg.inv(extrinsic)

    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, K[0][0], K[1][1], K[0][2], K[1][2])
    
    
    print(intrinsic.intrinsic_matrix)
