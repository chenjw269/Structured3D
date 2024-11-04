import sys
sys.path.append(".")

import cv2
import copy
from s3d import *
import numpy as np
from scripts.utils.read_sample_info import read_sample_info # 读取样本位姿
from scripts.utils.view_range_utils import generate_ellipse_mask
from scripts.virtual_obs.pesp_vobs import * # 虚拟单视角观测
from scripts.utils.visualize_occ import * # 可视化占用网格


if __name__ == "__main__":

    test_scene_index = "scene_00000"
    test_sample_index = ("485142", "0")

    # 读取样本信息
    sample_info = read_sample_info(test_scene_index, test_sample_index, mode="gravity")

    # 场景地图
    scene_map = sample_info["scene map array"]
    scene_map = np.load(scene_map)

    # 样本位姿
    sample_pose = sample_info["sample pose"]
    print(sample_pose)

    fov = 80
    
    # ############################################
    # # 虚拟观测的视野
    # ############################################
    # map_image_cp = copy.copy(scene_map)

    # # 将视野外的内容覆盖
    # view_range_mask = generate_ellipse_mask(map_image_cp, sample_pose[:2].astype(int), sample_pose[2], fov, "occ")
    # map_image_cp = view_range_mask * map_image_cp
    # map_image_cp = visualize_occ(map_image_cp, LABEL_TO_COLOR)

    # # 显示视野边界，方便观察
    # start_angle = sample_pose[2] - fov/2
    # end_angle = sample_pose[2] + fov/2
    # cv2.ellipse(map_image_cp, sample_pose[:2].astype(int), (256,256), 0, start_angle, end_angle, (0,0,0), thickness=1)

    # cv2.imshow("Virtual Fov", map_image_cp)

    # ############################################
    # # 虚拟观测的范围
    # ############################################
    # map_image_cp = copy.copy(scene_map)
    # map_image_cp = visualize_occ(map_image_cp, LABEL_TO_COLOR)

    # # 显示视野边界，方便观察
    # view_range_rect = generate_viewrange_rectangle(sample_pose, width=256, height=256)
    # for i in range(4):
    #     map_image_cp = cv2.line(map_image_cp, view_range_rect[i], view_range_rect[(i+1)%4], (0,0,255), thickness=1)
    # cv2.imshow("View Range ", map_image_cp)

    ############################################
    # 虚拟观测的内容
    ############################################

    virtual_obs = virtual_pesp_obs(scene_map, sample_pose, fov)
    virtual_obs = visualize_occ(virtual_obs, LABEL_TO_COLOR)

    cv2.imshow("Virutal Obs", virtual_obs)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
