import cv2
import math
import numpy as np

import sys
sys.path.append(".")

import time
from scripts.pesp_mc.pesp_bev import generate_ellipse_mask # 视野扇形掩码
# 可视化 bev
from scripts.utils.visualize_occ import visualize_occ
from assets.semantic2label import LABEL_TO_COLOR


def bev_to_equal_angle(ori_bev, fov_split):

    height, width = ori_bev.shape
    
    # 计算分区数量
    assert 360 % fov_split == 0 # 360 应当可以被角度间距整除
    split_nums = 360 // fov_split
    # 将 bev 按照相等的角度间距，划分为多个分区
    angle_bev_list = []
    for i in range(split_nums):
        # 获取视野掩码
        angle_mask = generate_ellipse_mask(test_bev, i*fov_split, fov_split)
        # 覆盖视野之外的部分
        angle_bev = ori_bev * angle_mask

        angle_bev_list.append(angle_bev)
    # 将每个分区旋转到 y 轴正方向，获取视锥范围内的所有像素
    center = (ori_bev.shape[1] // 2, ori_bev.shape[0] // 2)
    for i in range(split_nums):
        # 旋转每个分区
        scale = 1.0  # 缩放比例
        angle_rot = i * fov_split + 90 # 旋转角度
        rotation_matrix = cv2.getRotationMatrix2D(center, angle_rot, scale) # 旋转矩阵
        angle_bev = cv2.warpAffine(angle_bev_list[i], rotation_matrix, (ori_bev.shape[1], ori_bev.shape[0])) # 执行旋转变换
        # 获取有信息的部分
        width_range = int(height/2 * math.tan(math.radians(fov_split/2)))
        angle_bev = angle_bev[
            :128,
            center[1]-width_range : center[1]+width_range
        ]
        angle_bev_list[i] = angle_bev
    
    # # 1. 拼接每个分区的 bev 视锥
    # total_angle_bev = np.concatenate(angle_bev_list, axis=1)
    # 2. 将每个分区的 bev 视锥按照 array 形式返回
    total_angle_bev = np.stack(angle_bev_list, axis=0)
    
    return split_nums, total_angle_bev


# 读取 bev
test_bev = "e:/datasets/Structure3D_bev/Structured3D/scene_00000/2D_rendering/485142/panorama/full/bev.npy"
test_bev = np.load(test_bev)
print(f"Original bev {test_bev.shape}")

start_time = time.time()
split_nums, test_bev = bev_to_equal_angle(test_bev, 45)
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time {execution_time}")
print(f"Equal angle bev {test_bev.shape}, Split nums {split_nums}")
# test_bev = visualize_occ(test_bev, LABEL_TO_COLOR)

# cv2.imshow("Angle bev", test_bev)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
