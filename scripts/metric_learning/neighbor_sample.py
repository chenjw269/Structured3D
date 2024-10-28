# 20241019
# 在地图上采样正负样例
import math
import random


# 采样近邻：在指定半径范围内采样随机点
def generate_neighbor_within(position, nums, radius, p_bound):
    """在指定范围内，随机采样 2d 平面上的坐标

    Args:
        position (list(int, int)): 采样范围的位置中心
        nums (int): 采样位置的数量
        radius (float): 采样位置的范围
        p_bound (list((int, int), (int, int))): 采样位置的坐标边界

    Returns:
        list(int, int): 采样得到的随机位置
    """
    assert p_bound[0][0] < p_bound[0][1] # x 轴坐标范围的下界小于上界
    assert p_bound[1][0] < p_bound[1][1] # y 轴坐标范围的下界小于上界
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
        x_available = (p_bound[0][0] < x < p_bound[0][1])
        y_available = (p_bound[1][0] < y < p_bound[1][1])
        if (x_available and y_available):
            random_points.append((x, y))

        if len(random_points) == nums:
            return random_points

# 采样远邻：在两个指定半径范围之间采样随机点
def generate_neighbor_between(position, nums, radius_1, radius_2, p_bound):
    """生成两个范围之间的随机点

    Args:
        position (list(int, int)): 参考位置
        radius_1 (int): 较小半径范围
        radius_2 (int): 较大半径范围
        p_bound (list(int, int)): 坐标范围
    """
    assert p_bound[0][0] < p_bound[0][1] # x 轴坐标范围的下界小于上界
    assert p_bound[1][0] < p_bound[1][1] # y 轴坐标范围的下界小于上界
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
        x_available = (p_bound[0][0] < x < p_bound[0][1])
        y_available = (p_bound[1][0] < y < p_bound[1][1])
        if (x_available and y_available):
            random_points.append((x, y))
    
        if len(random_points) == nums:
            return random_points

# 采样远邻：在指定半径范围之外采样随机点
def generate_neighbor_outside(position, nums, radius, p_bound=(0,800)):
    """生成范围以外的随机点

    Args:
        position (list(int, int)): 参考位置
        nums (int): 随机点数量
        radius (int): 半径范围
        p_bound (list(int, int)): 坐标范围
    """
    assert p_bound[0] < p_bound[1]
    random_points = []
    
    while True:
        # 随机生成半径范围内的半径
        r = math.sqrt(random.uniform(radius**2, (2*radius)**2))
        
        # 随机生成0到2π之间的角度
        theta = random.uniform(0, 2 * math.pi)
    
        # 极坐标转换为笛卡尔坐标
        x = int(position[0] + r * math.cos(theta))
        y = int(position[1] + r * math.sin(theta))
        
        # 检查坐标是否越界
        if (p_bound[0] < x < p_bound[1] and p_bound[0] < y < p_bound[1]):
            random_points.append((x, y))
    
        if len(random_points) == nums:
            return random_points
