# 20241024
# 根据场景坐标边界，对场景进行坐标变换

# 202410242129
# 202410242141
# 12 min，优秀

import copy
import numpy as np


def position_to_pixel(ori_pos, resolution, scene_boundary):
    
    tgt_pos = copy.copy(ori_pos)
    if not isinstance(tgt_pos, np.ndarray):
        tgt_pos = np.array(tgt_pos)
    
    # 1. 地图的原点（0，0）为坐标中心
    scene_center = np.array([scene_boundary['center_x'].item(), scene_boundary['center_y'].item()])
    tgt_pos = tgt_pos - scene_center

    # 2. 坐标除以分辨率，从 mm 转换为 pixel
    tgt_pos = (tgt_pos / resolution).astype(int)

    # 3. 根据地图尺寸，平移像素坐标
    scene_size = np.array([
        scene_boundary['size_x'].item() / resolution,
        scene_boundary['size_y'].item() / resolution
    ])
    tgt_pos = (tgt_pos + scene_size / 2).astype(int)

    # 4. 上下翻转 y 轴
    tgt_pos[1] = scene_size[1] - tgt_pos[1]
    # # 交换 xy 坐标
    # tgt_pos = np.array([tgt_pos[1], tgt_pos[0]])

    return np.array(tgt_pos)

def position_to_pixel_mapping(ori_pos, resolution, scene_boundary):
    """对比样本坐标的坐标转换过程，一方面是要进行批量转换，一方面是不用交换 xy 轴

    Args:
        ori_pos (np.array): 空间坐标系下的坐标点（地图标注的顶点和物体点）
        resolution (int): 分辨率，将空间坐标转换为 2d 地图坐标，单位为 mm / pixel
        scene_boundary (list(int, int)): 地图尺寸

    Returns:
        np.array: 转换后，在 2d 地图上的坐标点
    """
    tgt_pos = copy.copy(ori_pos)
    if not isinstance(tgt_pos, np.ndarray):
        tgt_pos = np.array(tgt_pos)
    
    # normalization，将坐标中心修改为原点
    scene_center = np.array([scene_boundary['center_x'].item(), scene_boundary['center_y'].item()])
    tgt_pos = tgt_pos - scene_center
    # 根据比例尺进行换算，从 mm 转换成 pixel
    tgt_pos = (tgt_pos / resolution).astype(int)
    # 根据地图尺寸进行偏移
    scene_size = np.array([
        int(scene_boundary['size_x'].item() / resolution),
        int(scene_boundary['size_y'].item() / resolution)
    ])
    tgt_pos = (tgt_pos + scene_size / 2).astype(int)
    # 上下翻转
    tgt_pos[:,1] = scene_size[1] - tgt_pos[:,1]
    # # 交换 xy 坐标
    # tgt_pos = tgt_pos[:, [1,0]]

    return tgt_pos


def position_to_pixel_batch(ori_pos, resolution, scene_boundary):
    """批量将真实世界位置坐标转换到像素坐标

    Args:
        ori_pos (np.array): 真实世界位置坐标
        resolution (int): 距离到像素的缩放比例，mm/pixel
        scene_boundary (dict): 场景边界信息

    Returns:
        np.array: 像素坐标
    """
    # 复制位置数组，从内存上进行隔离
    tgt_pos = copy.copy(ori_pos)
    if not isinstance(tgt_pos, np.ndarray):
        tgt_pos = np.array(tgt_pos)
    
    # normalization，将坐标中心修改为原点
    scene_center = np.array([scene_boundary['center_x'].item(), scene_boundary['center_y'].item()])
    tgt_pos = tgt_pos - scene_center

    # 缩放 (根据比例 mm/pixel，从 mm 转换成 pixel)
    tgt_pos = (tgt_pos / resolution).astype(int)

    # 平移 (根据地图尺寸平移，使得原点对应地图中心)
    scene_size = np.array([
        int(scene_boundary['size_x'].item() / resolution),
        int(scene_boundary['size_y'].item() / resolution)
    ])
    tgt_pos = (tgt_pos + scene_size / 2).astype(int)

    # 上下翻转 (图像坐标系 y 轴是自上而下的，所以需要翻转 y 轴)
    tgt_pos[:,1] = scene_size[1] - tgt_pos[:,1]
    
    # # RM: 交换 xy 坐标
    # tgt_pos = tgt_pos[:, [1,0]]

    return tgt_pos