import math
import numpy as np


def gen_position_hypothesis(bound, step=250):
    
    # 由 x 轴坐标范围，得到 x 轴位置假设
    x_hypothesis = np.arange(
        int(bound[0][0]), int(bound[0][1]), step)
    # 由 y 轴坐标范围，得到 y 轴位置假设
    y_hypothesis = np.arange(
        int(bound[1][0]), int(bound[1][1]), step)
    
    return x_hypothesis, y_hypothesis

def gen_orientation_hypothesis(bound, step):
    """均匀采样朝向假设

    Args:
        bound (list(int, int)): 朝向边界
        step (int): 采样步长

    Returns:
        np.array: 朝向假设
    """
    # 由 yaw 角范围，得到朝向假设
    orientation_hypothesis = np.arange(bound[0], bound[1], step)

    return orientation_hypothesis

def gen_pose_hypothesis(p_bound, o_bound, p_step, o_step):
    """均匀采样位姿假设

    Args:
        p_bound (pd.DataFrame): 位置边界
        o_bound (list(int, int)): 朝向边界
        p_step (int): 位置采样步长
        o_step (int): 朝向采样步长

    Returns:
        np.array: 位姿假设
    """
    x_hypothesis, y_hypothesis = gen_position_hypothesis(
        bound=p_bound, step=p_step
    )
    o_hypothesis = gen_orientation_hypothesis(
        bound=o_bound, step=o_step
    )
    
    # 排列组合，得到所有位姿假设
    pose_hypothesis = []
    for x in x_hypothesis:
        for y in y_hypothesis:
            for ori in o_hypothesis:
                pose_hypothesis.append((x, y, ori))
    pose_hypothesis = np.array(pose_hypothesis)

    return pose_hypothesis

def position_matching(position, hypothesis, k=1):
    """从位姿假设中，找到与目标位姿距离最近的位姿

    Args:
        position (np.array): 目标位置
        hypothesis (np.array): 多个位置假设
        k (int, optional): 查找 top k 近邻位姿假设. Defaults to 1.

    Returns:
        list: top k 近邻位置的索引
    """
    # 计算样本位姿和每个位姿假设之间的距离
    hypo_distance_list = []
    for hypo_pos in hypothesis:
        # 两个位姿之间的距离为：位置距离 + 朝向之差
        hypo_distance = math.sqrt(pow(position[0] - hypo_pos[0], 2) \
                                    + pow(position[1] - hypo_pos[1], 2)) \
                                        + abs(hypo_pos[2] - position[2])
        hypo_distance_list.append(hypo_distance)
    
    # 选择距离最小的 k 个位姿假设
    sort_indices = np.argsort(np.array(hypo_distance_list)).tolist()
    top_indices = sort_indices[:k]
    
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
