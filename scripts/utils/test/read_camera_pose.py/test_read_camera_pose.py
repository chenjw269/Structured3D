import sys
sys.path.append(".")

import numpy as np
from scripts.utils.read_camera_pose import *


if __name__ == "__main__":
    
    # test_camera_pose = "e:/datasets/Structure3D/Structured3D/scene_00000/2D_rendering/485142/perspective/full/0/camera_pose.txt"
    # test_camera_pose = np.loadtxt(test_camera_pose)
    
    # t = test_camera_pose[3:6]
    # u = test_camera_pose[6:9]

    # roll_raw, pitch_raw, yaw_raw = compute_euler_angles(t)
    # print(roll_raw, pitch_raw, yaw_raw)
    
    # roll, pitch, yaw = compute_euler_angles_wgravity(t, u)
    # print(roll, pitch, yaw)

    test_camera_pose = "e:/datasets/Structure3D/Structured3D/scene_00000/2D_rendering/485142/perspective/full/0/camera_pose.txt"

    test_camera_pose_raw = read_camera_pose(test_camera_pose, "raw")
    print(test_camera_pose_raw)

    test_camera_pose_gravity = read_camera_pose(test_camera_pose, "gravity")
    print(test_camera_pose_gravity)
