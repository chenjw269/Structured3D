# 20241018
# 根据语义标签 → 颜色的映射，将 2d 占用网格可视化

import numpy as np
from PIL import Image


def visualize_occ(semantic_voxel, semantic_color_dict):
    
    # 创建占用网格
    semantic_voxel_size = (semantic_voxel.shape[0], semantic_voxel.shape[1])
    semantic_voxel_img = np.ones([semantic_voxel_size[0], semantic_voxel_size[1], 3])
    
    # 遍历所有的语义类别
    for semantic_type in semantic_color_dict.keys():
        
        semantic_color = semantic_color_dict[semantic_type] # 语义类别对应颜色
        semantic_voxel_index = (semantic_voxel == int(semantic_type)) # 占用网格索引
        semantic_voxel_img[semantic_voxel_index] = np.array(semantic_color) / 255 # 填充占用网格

    semantic_voxel_img = (semantic_voxel_img * 255).astype(np.uint8)
    # semantic_voxel_img = (Image.fromarray(semantic_voxel_img)).convert("RGB")
    
    return semantic_voxel_img
