import math
import numpy as np


def patches_matching_positive(pos, split_centers, k=1):

    # 计算和每个窗口中心之间的距离
    patch_distance_list = []
    for patch_center in split_centers:
        patch_distance = math.sqrt(pow(pos[0] - patch_center[0], 2) \
                                    + pow(pos[1] - patch_center[1], 2))
        patch_distance_list.append(patch_distance)

    # 选择距离最小的 k 个窗口
    patch_distance_list = np.array(patch_distance_list)
    patch_indices = (np.argsort(patch_distance_list)).tolist()
    top_indices = patch_indices[:k]


    return top_indices
