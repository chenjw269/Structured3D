import sys
sys.path.append(".")

from s3d import *

import os
import numpy as np
from tqdm import tqdm
from PIL import Image


def check_bev(path):
    
    # 检查 array
    array_pth = os.path.join(path, "bev.npy")
    if not os.path.exists(array_pth):
        return False
    array_data = np.load(array_pth)
    if array_data.shape != (256, 256):
        return False
    
    # 检查 png
    img_pth = os.path.join(path, "bev.png")
    if not os.path.exists(img_pth):
        return False
    img_data = np.array(Image.open(img_pth))
    if img_data.shape != (256, 256, 3):
        return False

    return True

if __name__ == "__main__":
    
    # # Structured3D 包括 3500 个场景
    # scene_index_list = [f"scene_{num:05}" for num in range(3500)]
    # 取前 100 个场景
    scene_index_list = [f"scene_{num:05}" for num in range(400, 800)]

    # 去掉标注数据缺失的场景
    for scene_index in tqdm(scene_index_list):
        if scene_index in scene_invalid:
            scene_index_list.remove(scene_index)
    print("Invalid scenes removed")

    # 遍历场景
    for scene_index in tqdm(scene_index_list):
        # 场景数据路径
        scene_data_dir = os.path.join(s3d_pesp_data_pth, scene_index, "2D_rendering")
        sample_index_list = os.listdir(scene_data_dir)
        
        # 遍历样本
        for sample_index in sample_index_list:
            # 样本数据路径
            sample_data_dir = os.path.join(scene_data_dir, sample_index, "perspective/full")
            ori_index_list = os.listdir(sample_data_dir)

            # 遍历朝向
            for ori_index in ori_index_list:
                # 朝向数据路径
                ori_data_dir = os.path.join(sample_data_dir, ori_index)
                
                # 对应的 bev 文件
                bev_path = os.path.join(
                    s3d_bev_pth, scene_index, "2D_rendering",
                    sample_index, "perspective/full", ori_index)
                if check_bev(bev_path):
                    pass
                else:
                    tqdm.write(f"File loss {scene_index}/{sample_index}/{ori_index}")
                    # return False
