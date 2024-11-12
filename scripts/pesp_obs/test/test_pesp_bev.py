import sys
sys.path.append(".")

import os
import cv2 
import matplotlib.pyplot as plt
from s3d import *
from scripts.pesp_obs.pesp_bev import *
# 可视化 bev
from scripts.utils.visualize_occ import visualize_occ
from assets.semantic2label import LABEL_TO_COLOR


if __name__ == "__main__":

    test_scene_index = "scene_00000"
    test_sample_index = "485142"

    # # 读取 BEV 图像
    # test_bev = os.path.join(
    #     s3d_bev_pth, test_scene_index, "2D_rendering", test_sample_index, "panorama/full/bev.png"
    # )
    # test_bev = cv2.imread(test_bev)
    # 读取 BEV array
    test_bev = os.path.join(
        s3d_bev_pth, test_scene_index, "2D_rendering", test_sample_index, "panorama/full/bev.npy"
    )
    test_bev = np.load(test_bev)

    #######################################
    # 测试视野受限情况下的 BEV
    #######################################
    # # 0 度朝向，120 度视野，生成掩码
    # test_image = np.zeros((256,256))
    # test_mask = generate_ellipse_mask(test_image, 0, 120)
    # # # 增加到 3 通道，可用于覆盖 rgb 图像
    # # test_mask = np.repeat(test_mask[:, :, np.newaxis], 3, axis=2)    
    # # 根据掩码，对 BEV 进行覆盖
    # test_bev = test_mask * test_bev

    # # 可视化 bev
    # test_bev = visualize_occ(test_bev, LABEL_TO_COLOR)

    # cv2.imshow("Fov BEV", test_bev)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #######################################
    # 测试一定朝向下的 BEV
    #######################################
    # 45 度朝向旋转 BEV
    test_bev = generate_rotation(test_bev, 45)

    # 可视化 BEV
    test_bev = visualize_occ(test_bev, LABEL_TO_COLOR)

    cv2.imshow("Rotated BEV", test_bev)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #######################################
    # 测试视野受限情况下，一定朝向的 BEV
    #######################################
    # 90 度朝向，120 度视野，生成掩码
    test_image = np.zeros((256,256))
    test_mask = generate_ellipse_mask(test_image, 90, 120)
    # 根据掩码，对 BEV 进行覆盖
    test_bev = test_mask * test_bev

    # 45 度朝向旋转 BEV
    test_bev = generate_rotation(test_bev, 45)
    
    # 可视化 BEV
    test_bev = visualize_occ(test_bev, LABEL_TO_COLOR)
    
    cv2.imshow("Perspective BEV", test_bev)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
