import numpy as np


def position_to_pixel(ori_pos, resolution, map_occ_size):
    if not isinstance(ori_pos, np.ndarray):
        ori_pos = np.array(ori_pos)
    
    # 根据比例尺进行换算，从 mm 转换成 pixel
    ori_pos = (ori_pos / resolution).astype(int)
    # 根据地图尺寸进行偏移
    ori_pos = ori_pos + (np.array(map_occ_size)/2).astype(int)
    # 上下翻转
    ori_pos[1] = map_occ_size[1] - ori_pos[1]
    # 交换 xy 坐标
    ori_pos = np.array([ori_pos[1], ori_pos[0]])
    
    return ori_pos

# def position_to_pixel_norm(ori_pos, resolution, map_occ_size,)
