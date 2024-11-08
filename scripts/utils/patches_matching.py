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

def pose_matching(pose, hypothesis, k=1):
    """从位姿假设中，找到与目标位姿距离最近的位姿

    Args:
        pose (np.array): 目标位姿
        hypothesis (np.array): 多个位姿假设
        k (int, optional): 查找 top k 近邻位姿假设. Defaults to 1.

    Returns:
        list: top k 近邻位姿的索引
    """
    # 计算样本位姿和每个位姿假设之间的距离
    hypo_distance_list = []
    for hypo_pose in hypothesis:
        # 两个位姿之间的距离为：位置距离 + 朝向之差
        hypo_distance = math.sqrt(pow(pose[0] - hypo_pose[0], 2) \
                                    + pow(pose[1] - hypo_pose[1], 2)) \
                                        + abs(hypo_pose[2] - pose[2])
        hypo_distance_list.append(hypo_distance)
    
    # 选择距离最小的 k 个位姿假设
    sort_indices = np.argsort(np.array(hypo_distance_list)).tolist()
    top_indices = sort_indices[:k]
    
    return top_indices
