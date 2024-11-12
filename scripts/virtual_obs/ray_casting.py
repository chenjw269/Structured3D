# 由于遮挡问题，观测结果和地图上对应位姿虚拟观测不同的样本
# Sample scene_00049/2D_rendering/2321/perspective/full/1
# Sample pose (mm) [ 3337.65      -5284.52       -152.8631655]
# Sample pose (pixel) [610.        691.        117.1368345]

import numpy as np

def ray_casting(fov_map, start_pos, fov_angle, max_distance=256):
    """
    对局部地图进行射线投射，检测视锥内被遮挡的部分。
    
    fov_map: (256, 256)  带有墙壁占用信息的鸟瞰图
    start_pos: (x, y)    相机的 2D 位姿位置
    fov_angle: (min_angle, max_angle)  相机视野角度范围
    max_distance: 最远的可视距离

    返回：被遮挡的区域
    """
    # 视锥扇形的角度范围，假设相机的朝向为零度
    visible_area = np.zeros_like(fov_map)
    
    # 遍历视野角度范围内的每一条射线
    for angle in np.arange(fov_angle[0], fov_angle[1], 0.01):  # 调整精度
        x, y = start_pos
        
        meet_wall = 0
        
        for distance in range(1, max_distance):
            # 计算射线的目标位置
            ray_x = int(x + distance * np.cos(angle))
            ray_y = int(y + distance * np.sin(angle))

            # 检查是否越界
            if 0 <= ray_x < fov_map.shape[0] and 0 <= ray_y < fov_map.shape[1]:
                # 如果射线碰到了墙壁，则更新遮挡标志位
                if fov_map[ray_x, ray_y] == 1:

                    # 更新遮挡标志位
                    if meet_wall < 2:   # 前 n 次遇到墙壁，暂时可见，之后的均不可见
                        meet_wall += 1
                        visible_area[ray_x, ray_y] = 1
                    else:               # 不是前 n 次遇到墙壁，不可见
                        # visible_area[ray_x, ray_y] = 0
                        break
                # comment: 不需要，因为掩码本身初始化为 0
                # elif meet_wall == 1:    # 如果已经遇到了墙壁，则后面的区域全部遮挡
                #     visible_area[ray_x, ray_y] = 0
                #     break
                else:                   # 如果还没遇到墙壁，则完全可见
                    visible_area[ray_x, ray_y] = 1

            else:
                break  # 超过边界，停止投射

    return visible_area

def process_virtual_observation(cad_map, start_pos, fov_angle):
    """
    处理虚拟观测地图，去除被遮挡的部分。

    cad_map: (256, 256)  CAD地图的虚拟观测区域
    start_pos: (x, y)    相机的 2D 位姿位置
    fov_angle: (min_angle, max_angle)  相机视野角度范围
    """
    # 进行射线投射，检测被遮挡区域
    visible_area = ray_casting(cad_map, start_pos, fov_angle)
    
    # 使用遮挡检测结果更新CAD地图
    processed_map = cad_map * visible_area
    return processed_map

