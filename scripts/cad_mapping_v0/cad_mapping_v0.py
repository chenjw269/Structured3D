# 20241024
# 前 1000 个场景的坐标范围为 x:(-20m, 20m) y:(-20,20)
# 将所有场景的标注映射到尺寸为 (1600,1600) 的 2D 网格，分辨率为 2.5 cm/pixel
# (1600 px,1600 px) 对应于 (40m, 40m)
# 根据建图方式，得到相应的样本位置到地图上位置的坐标变换方式
import copy
import numpy as np


def cad_mapping():
    pass


def position_to_pixel(ori_pos, resolution, map_occ_size):
    """将空间坐标换算到地图平面坐标

    Args:
        ori_pos (list(float, float)): 空间坐标
        resolution (int): 分辨率
        map_occ_size (list(int, int)): 地图尺寸

    Returns:
        list(int, int): 地图平面坐标
    """
    tgt_pos = copy.copy(ori_pos)

    if not isinstance(tgt_pos, np.ndarray):
        tgt_pos = np.array(tgt_pos)

    # 根据比例尺进行换算，从 mm 转换成 pixel
    tgt_pos = (tgt_pos / resolution).astype(int)
    # 根据地图尺寸进行偏移
    tgt_pos = tgt_pos + (np.array(map_occ_size)/2).astype(int)
    # 上下翻转
    tgt_pos[1] = map_occ_size[1] - tgt_pos[1]
    # 交换 xy 坐标
    tgt_pos = np.array([tgt_pos[1], tgt_pos[0]])

    return tgt_pos


if __name__ == "__main__":
    
    scene_index_list = []