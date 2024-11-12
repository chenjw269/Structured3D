
import copy


from scripts.utils.view_range_utils import generate_rectangle_mask # 地图扇形掩码
from scripts.utils.view_range_utils import generate_rotation # 地图旋转


def virtual_pano_obs(map_image, pose, view_range=256):
    """根据相机位姿，从 cad 地图上获取虚拟观测 bev

    Args:
        map_image (np.array): cad 地图
        pose (np.array): 相机 2d 位姿 (x,y,yaw)
        fov (float): 视野范围

    Returns:
        np.array: 虚拟观测 bev
    """

    map_image_cp = copy.copy(map_image)

    #########################################
    # 虚拟观测的视野
    #########################################
    # 将视野外的内容覆盖
    view_range_mask = generate_rectangle_mask(
        image=map_image_cp,
        center=pose[:2].astype(int), angle=pose[2], view_range=view_range,
        mode="occ")
    map_image_cp = view_range_mask * map_image_cp

    ##########################################
    # 将视野内容旋转到正方向
    ##########################################
    # cv2 中的正方向为水平向右，转换为正方向竖直向上，需要加上 90 度
    map_image_cp = generate_rotation(
        image=map_image_cp,
        center=pose[:2], angle=(pose[2]+90))

    ##########################################
    # 虚拟观测的内容
    ##########################################
    # 获取视野内的部分
    map_image_cp = map_image_cp[
        pose[1].astype(int) - 128: pose[1].astype(int) + 128,
        pose[0].astype(int) - 128: pose[0].astype(int) + 128
    ]
    
    ##########################################
    # 可见性处理
    ##########################################
    
    return map_image_cp
