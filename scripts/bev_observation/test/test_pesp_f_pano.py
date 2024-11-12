import sys
sys.path.append(".")

import os
import cv2
import copy
from s3d import *
import numpy as np
from scripts.utils.read_sample_info import read_sample_info # 读取样本位姿
from scripts.utils.view_range_utils import generate_ellipse_mask # 扇形视野
from scripts.utils.view_range_utils import generate_rotation # 旋转视野
from scripts.bev.pesp_f_pano import executing_pespfpano_processing # 从全景图 bev 获取单视角 bev
from scripts.utils.visualize_occ import *


if __name__ == "__main__":
    
    test_scene_index = "scene_00000"
    test_sample_index = ("485142", "0")
    
    # 读取样本信息
    sample_info = read_sample_info(test_scene_index, test_sample_index, mode="gravity")

    # 全景图 bev
    sample_pano_bev = os.path.join(
        s3d_bev_pth, test_scene_index, "2D_rendering", test_sample_index[0], "panorama/full/bev.npy"
    )
    sample_pano_bev = np.load(sample_pano_bev)
    
    # 样本位姿
    sample_pose = sample_info["sample pose"]

    fov = 80
    
    ############################################
    # 虚拟观测的视野
    ############################################
    pano_bev_cp = copy.copy(sample_pano_bev)

    # 将视野外的内容覆盖
    pano_center = np.array([pano_bev_cp.shape[0] / 2, pano_bev_cp.shape[1] / 2])
    view_range_mask = generate_ellipse_mask(
        image=pano_bev_cp,
        center=pano_center.astype(int), angle=sample_pose[2], fov=fov, mode="occ")
    pano_bev_cp = view_range_mask * pano_bev_cp

    pano_bev_cp = visualize_occ(pano_bev_cp, LABEL_TO_COLOR)
    cv2.imshow("Virtual Fov", pano_bev_cp)

    ##############################################
    # 虚拟观测的内容
    ##############################################
    pano_bev_cp = copy.copy(sample_pano_bev)

    pesp_bev = executing_pespfpano_processing(pano_bev_cp, sample_pose, fov=fov, mode="occ")
    pesp_bev = visualize_occ(pesp_bev, LABEL_TO_COLOR)
    cv2.imshow("Pesp bev", pesp_bev)

    cv2.waitKey(0)
    cv2.destroyAllWindows()