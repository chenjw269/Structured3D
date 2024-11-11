#########################################################################
# 从全景图 bev 中获取单视角 bev
# 已知单视角视图的朝向，从全景图 bev 的该朝向下获取一定视野的内容，作为单视角 bev
#########################################################################
import copy
import numpy as np
from scripts.utils.view_range_utils import generate_ellipse_mask # 扇形视野
from scripts.utils.view_range_utils import generate_rotation # 旋转视野


def executing_pespfpano_processing(pano_bev, pose, fov=80, mode="occ"):

    pano_bev_cp = copy.copy(pano_bev)

    #########################################
    # 虚拟观测的视野
    #########################################
    # 以全景图 bev 的中心为旋转中心
    pano_center = np.array([pano_bev.shape[0] / 2, pano_bev.shape[1] / 2])
    # 以全景图朝向为扇形方向，覆盖视野之外的 bev
    view_range_mask = generate_ellipse_mask(
        image=pano_bev,
        center=pano_center.astype(int), angle=pose[2], fov=fov, mode=mode)
    pano_bev_cp = view_range_mask * pano_bev_cp

    ##########################################
    # 旋转虚拟观测
    ##########################################
    # cv2 中的正方向为水平向右，转换为正方向竖直向上，需要加上 90 度
    pano_bev_cp = generate_rotation(pano_bev_cp, (pose[2]+90))
    
    return pano_bev_cp
