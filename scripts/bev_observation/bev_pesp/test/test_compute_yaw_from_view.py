import sys
sys.path.append(".")

import numpy as np
from scripts.bev_pesp.projection import compute_yaw_from_view


if __name__ == "__main__":
    
    camera_pose = "e:/datasets/Structure3D/Structured3D/scene_00000/2D_rendering/485142/perspective/full/0/camera_pose.txt"
    camera_pose = np.loadtxt(camera_pose)
    
    view_direction = camera_pose[3:6]
    print(view_direction)
    
    yaw = compute_yaw_from_view(view_direction)
    
    print(yaw)
