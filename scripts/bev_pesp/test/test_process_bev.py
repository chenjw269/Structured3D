import sys
sys.path.append(".")

import os
import cv2
import numpy as np
from s3d import *
from scripts.bev_pesp.process_bev import * # 单视角视图 bev
from scripts.utils.visualize_occ import * # 可视化占用网格


if __name__ == "__main__":
    
    test_scene_index = "scene_00000"
    test_sample_index = ("485142", "0")

    # 数据目录
    test_sample_data_pth = os.path.join(
        s3d_data_pth, test_scene_index, "2D_rendering", test_sample_index[0], "perspective/full", test_sample_index[1]
    )
    # 深度图
    test_depth = os.path.join(test_sample_data_pth, "depth.png")
    test_depth = cv2.imread(test_depth, cv2.IMREAD_UNCHANGED)
    # 语义分割图
    test_semantic = os.path.join(test_sample_data_pth, "semantic.png")
    test_semantic = cv2.imread(test_semantic)
    test_semantic = cv2.cvtColor(test_semantic, cv2.COLOR_BGR2RGB)    
    # 相机位姿
    test_pose = os.path.join(test_sample_data_pth, "camera_pose.txt")
    test_pose = np.loadtxt(test_pose)

    # 单视角视图 bev
    test_bev = generate_bev(test_depth, test_semantic, test_pose)
    # 可视化占用网格
    test_bev = visualize_occ(test_bev, LABEL_TO_COLOR)
    
    # # 1. 可视化占用网格
    # cv2.imshow("Pesp bev", test_bev)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # 2. 保存到日志
    cv2.imwrite("logs/pesp bev.png", test_bev)
