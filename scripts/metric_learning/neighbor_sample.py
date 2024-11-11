import random


def gen_position_near_neighbor(position, nums, radius, bound):
    
    
def gen_position_far_neighbor(position, nums, radius_1, radius_2, bound):
    
    assert bound[0][0] < bound[0][1] # x 轴坐标范围的下界小于上界
    assert bound[1][0] < bound[1][1] # y 轴坐标范围的下界小于上界
    
    neighbor = []
    
    while True:
        

def gen_orientation_near_neighbor(orientation, nums, radius, bound):
    
    assert bound[0] < bound[1] # 角度范围下界小于上界
    
    neighbor = []
    
    while True:
        # 随机生成半径范围内的朝向
        o = random.uniform()
        
        # 检查坐标是否越界
        
        if len(neighbor) == nums:
            return neighbor

def gen_orientation_far_neighbor(orientation, nums, radius, bound):
    
    assert bound[0] < bound[1] # 角度范围下界小于上界
    
    neighbor = []