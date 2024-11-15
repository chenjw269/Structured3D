import math
import random
import numpy as np


def gen_position_near_neighbor(position, nums, radius, bound):
    """采样近邻位置：在指定半径范围内随机采样 2d 位置

    Args:
        position (np.array): 采样位置中心
        nums (int): 采样位置数量
        radius (float): 采样位置范围
        bound (list((int, int), (int, int))): 坐标边界

    Returns:
        np.array: 采样得到的随机近邻位置
    """
    assert bound[0][0] < bound[0][1] # x 轴坐标范围的下界小于上界
    assert bound[1][0] < bound[1][1] # y 轴坐标范围的下界小于上界
    
    random_points = []
    
    while True:
        # 随机生成半径范围内的半径
        r = radius * math.sqrt(random.random())

        # 随机生成0到2π之间的角度
        theta = random.uniform(0, 2 * math.pi)

        # 极坐标转换为笛卡尔坐标
        x = int(position[0] + r * math.cos(theta))
        y = int(position[1] + r * math.sin(theta))

        # 检查坐标是否越界
        x_available = (bound[0][0] < x < bound[0][1])
        y_available = (bound[1][0] < y < bound[1][1])
        if (x_available and y_available):
            random_points.append((x, y))

        if len(random_points) == nums:
            return np.array(random_points)
    
def gen_position_far_neighbor(position, nums, radius_1, radius_2, bound):
    
    assert bound[0][0] < bound[0][1] # x 轴坐标范围的下界小于上界
    assert bound[1][0] < bound[1][1] # y 轴坐标范围的下界小于上界
    
    random_points = []
    
    while True:
        # 随机生成半径范围内的半径
        r = math.sqrt(random.uniform(radius_1**2, radius_2**2))
        
        # 随机生成0到2π之间的角度
        theta = random.uniform(0, 2 * math.pi)
        
        # 极坐标转换为笛卡尔坐标
        x = int(position[0] + r * math.cos(theta))
        y = int(position[1] + r * math.sin(theta))

        # 检查坐标是否越界
        x_available = (bound[0][0] < x < bound[0][1])
        y_available = (bound[1][0] < y < bound[1][1])
        if (x_available and y_available):
            random_points.append((x, y))

        if len(random_points) == nums:
            return np.array(random_points)

def gen_orientation_near_neighbor(orientation, nums, radius):
    
    
    random_angle = []
    
    while True:
        # 随机生成半径范围内的朝向
        o = random.uniform(-radius, radius)
        
        # 将坐标归一化到 0 到 360 度范围
        random_angle_item = (orientation + o + 360) % 360
        random_angle.append(random_angle_item)

        if len(random_angle) == nums:
            return np.array(random_angle)

def gen_orientation_far_neighbor(orientation, nums, radius):
    
    
    random_angle = []
    
    while True:
        # 随机生成半径范围内的朝向
        o = random.uniform(radius, 360 - radius)
        
        # 将坐标归一化到 0 到 360 度范围
        random_angle_item = (orientation + o) % 360
        random_angle.append(random_angle_item)
        
        if len(random_angle) == nums:
            return random_angle
