# 20241017
# 将深度图中的深度值归一化到 0 到 255 的范围，进行可视化

import cv2
import numpy as np


def normalize_depth(depth_img):

    # Step 1: 找到深度图的最小值和最大值
    min_val = np.min(depth_img)
    max_val = np.max(depth_img)

    # Step 2: 将深度图归一化到 [0, 1]
    normalized_depth_map = (depth_img - min_val) / (max_val - min_val)

    # Step 3: 将归一化的值映射到 [0, 255] 范围
    scaled_depth_map = (normalized_depth_map * 255).astype(np.uint8)

    return scaled_depth_map

if __name__ == "__main__":

    # depth_img_pth = "e:/datasets/Structure3D/Structured3D/scene_00000/2D_rendering/485142/perspective/full/0/depth.png"
    depth_img_pth = "e:/datasets/Structure3D/Structured3D/scene_00000/2D_rendering/485142/panorama/full/depth.png"
    depth_img = cv2.imread(depth_img_pth)

    depth_img = normalize_depth(depth_img)

    # 显示灰度图
    # cv2.imshow("Depth image", normalized_depth_map)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 保存灰度图
    cv2.imwrite("normalized_depth.png", depth_map)
