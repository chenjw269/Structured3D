
import cv2
import numpy as np

import sys
sys.path.append(".")

from scripts.pesp_mc.pesp_bev import generate_ellipse_mask # 视野扇形掩码
# 可视化 bev
from scripts.utils.visualize_occ import visualize_occ
from assets.semantic2label import LABEL_TO_COLOR


# 读取 bev
test_bev = "e:/datasets/Structure3D_bev/Structured3D/scene_00000/2D_rendering/485142/panorama/full/bev.npy"
test_bev = np.load(test_bev)

# 将 bev 按照相等的角度间距，划分为多个分区
angle_split = 90
assert 360 % angle_split == 0 # 360 应当可以被角度间距整除
split_nums = 360 // angle_split

angle_bev_list = []
for i in range(split_nums):
    
    angle_mask = generate_ellipse_mask(test_bev, i*angle_split, angle_split)
    angle_bev = test_bev * angle_mask
    
    angle_bev_img = visualize_occ(angle_bev, LABEL_TO_COLOR)
    angle_bev_list.append(angle_bev_img)
    cv2.imwrite(f"logs/equal_angle_split/split_{i}.png", angle_bev_img)
    
    # 视野范围标上红线，方便理解
    

# 旋转，获取该分区视锥范围内的像素
# 图像中心
center = (test_bev.shape[1] // 2, test_bev.shape[0] // 2)
for i in range(split_nums):

    scale = 1.0  # 缩放比例
    angle_rot = i * angle_split # 旋转角度
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_rot, scale) # 旋转矩阵
    # 执行旋转变换
    angle_bev_img = cv2.warpAffine(angle_bev_list[i], rotation_matrix, (test_bev.shape[1], test_bev.shape[0]))
    angle_bev_img = angle_bev_img[:,int(angle_bev_img.shape[1]/2):]
    # 保存图像
    cv2.imwrite(f"logs/equal_angle_split/rotated_{i}.png", angle_bev_img)
    angle_bev_list[i] = angle_bev_img
    
    # 视野范围标上红线，方便理解
    

# 拼接得到等角度间距的 bev 表示
total_angle_bev = np.concatenate(angle_bev_list, axis=1)
cv2.imwrite(f"logs/equal_angle_split/total_angle_bev.png", total_angle_bev)
# 视野范围标上红线，方便理解
for i in range(1, 4):
    total_angle_bev = cv2.line(total_angle_bev, (i*128,0), (i*128,255), (0,0,255), 1)
cv2.imwrite(f"logs/equal_angle_split/total_angle_bev_fov.png", total_angle_bev)
