import cv2
import open3d as o3d

import sys
sys.path.append(".")

from scripts.bev_observation.depth_to_3d import depth_to_3d # 深度图转点云
from scripts.bev_observation.points_aug import * # 点云稠密化方法


def generate_semantic_voxel(depth_pth, semantic_pth):
    
    ########################################
    # 读取数据：语义分割图、深度图点云
    ########################################
        
    # 读取语义分割图
    # 默认是 BGR 通道读取
    semantic_img = cv2.imread(semantic_pth)
    # 将 BGR 转换为 RGB
    semantic_img = cv2.cvtColor(semantic_img, cv2.COLOR_BGR2RGB)
    
    # 读取深度图
    # S3D 以 16 位整数存储深度
    depth_img = cv2.imread(depth_pth, cv2.IMREAD_UNCHANGED).astype(np.float32)
    # 将深度值为0的地方设置为NaN，表示无效的深度数据
    # depth_img[depth_img == 0] = np.nan
    # S3D 深度单位为 mm
    depth_array = np.array(depth_img) / 1000.0
    # 将深度图转换成点云
    depth_pc = depth_to_3d(depth_array)
    
    # 调整点云坐标系为：z 轴竖直向上，x 轴水平向右，y 轴垂直向里
    # 旋转矩阵
    R = np.array([[0, 1, 0],   # X 轴 -> Y 轴
                [1, 0, 0],   # Y 轴 -> X 轴
                [0, 0, -1]]) # Z 轴 -> -Z 轴 (反转方向)
    depth_pc = np.dot(depth_pc, R.T)
    
    ######################################
    # 栅格化：遍历不同语义，栅格化点云，填充网格
    ######################################
    # 创建空占用网格
    # 单格大小为 0.01m x 0.01m
    semantic_voxel_size = 600 # 600 x 600 → 6m x 6m
    semantic_voxel = np.zeros([semantic_voxel_size, semantic_voxel_size])

    # 遍历不同语义类别    
    semantic_type_list = list(LABEL_TO_COLOR.keys())  # 所有的语义类别
    for i in NOT_VIS_TYPE:
        semantic_type_list.remove(i)
    
    # semantic_type_list = VIS_TYPE
    
    for semantic_type in semantic_type_list:
        # 查表，该语义类别对应的分割颜色
        semantic_color = LABEL_TO_COLOR[semantic_type]
        # 选择该语义类别对应的点云
        semantic_index = np.where(np.all(semantic_img == semantic_color, axis=-1))
        semantic_depth = depth_pc[semantic_index]
        # 如果没有该类别的点云，则跳转到下一个语义类别
        if semantic_depth.shape[0] == 0:
            continue
        # 点云稠密化
        # semantic_depth = points_noise(semantic_depth)
        # semantic_depth = points_interpolation(semantic_depth)
        # 将 numpy 数组转换为 Open3D 点云对象
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(semantic_depth)
        # 体素大小
        voxel_size = 0.01
        # 可选：下采样点云（使点云更稀疏），有助于显示效果
        # point_cloud = point_cloud.voxel_down_sample(voxel_size)
        # 将点云栅格化为体素
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size=voxel_size)
        # 遍历体素，填充占用网格
        voxels = voxel_grid.get_voxels()
        for voxel in voxels:
            # 体素左下角坐标
            voxel_origin = voxel.grid_index * voxel_size + voxel_grid.origin
            # 计算体素坐标
            # 坐标转换：点云坐标→体素坐标
            voxel_index_0 = int(voxel_origin[0] // 0.01) + int(semantic_voxel_size / 2)
            voxel_index_1 = int(voxel_origin[1] // 0.01) + int(semantic_voxel_size / 2)
            # 填充体素网格
            # 体素网格范围多大，就用多大范围内的点云
            if 0 <= voxel_index_0 < semantic_voxel_size \
                and 0 <= voxel_index_1 < semantic_voxel_size:

                # 语义标签 +1（从 1 开始），因为默认值是 0（不能从 0 开始）
                semantic_voxel[voxel_index_1][voxel_index_0] = (semantic_type_list.index(semantic_type) + 1)

    return semantic_voxel
