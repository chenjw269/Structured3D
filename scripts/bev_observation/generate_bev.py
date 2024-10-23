# 20241019
# 从观测数据计算 BEV 占用网格
import os
import cv2
from tqdm import tqdm

import sys
sys.path.append(".")

from scripts.utils.generate_colors import generate_colors
from scripts.bev_observation.semantic_bev import generate_semantic_voxel
from assets.semantic2label import LABEL_TO_COLOR, SEMANTIC_TO_LABEL
from scripts.utils.visualize_occ import visualize_occ


if __name__ == "__main__":

    data_pth = "e:/datasets/Structure3D/Structured3D"
    output_pth = "e:/datasets/Structure3D_bev/Structured3D"

    # Structured3D 包括 3500 个场景
    scene_index_list = [f"scene_{num:05}" for num in range(3500)]
    # # 统计前 100 个场景
    # scene_index_list = [f"scene_{num:05}" for num in range(100)]

    # # 随机生成语义类别对应颜色
    # label_color_dict = generate_colors(13)
    # print(label_color_dict)
    
    # 标注数据缺失的场景
    with open("logs/scene_annos.txt", encoding="utf-8") as f:
        scene_invalid = f.readlines()
    for index, item in enumerate(scene_invalid):
        scene_invalid[index] = item.replace("\n", "")
        
    # 观测数据缺失的样本
    with open("logs/scene_observation.txt", encoding="utf-8") as f:
        obs_invalid = f.readlines()
    for index, item in enumerate(obs_invalid):
        obs_invalid[index] = item.replace("\n", "")


    # 遍历场景
    scene_index_list = scene_index_list[:1500]
    for scene_index in tqdm(scene_index_list, desc="Scenes"):
        
        # 缺少标注的场景作废
        if scene_index in scene_invalid:
            tqdm.write(f"Jmp annos loss {scene_index}")
            continue
        
        # 当前场景下的观测数据
        obs_dir = os.path.join(data_pth, scene_index, "2D_rendering")
        obs_list = os.listdir(obs_dir)
        # 结果保存位置
        output_dir = os.path.join(output_pth, scene_index, "2D_rendering")
        
        # 遍历观测数据
        for obs_item in tqdm(obs_list, desc="Obs", leave=False):

            # 缺少观测值的样本作废
            if f"{scene_index},{obs_item}" in obs_invalid:
                tqdm.write(f"Jmp obs loss {scene_index} {obs_item}")
                continue

            # 深度图路径
            obs_item_depth_full = os.path.join(obs_dir, obs_item, "panorama/full/depth.png")
            # 语义分割图路径
            obs_item_semantic_full = os.path.join(obs_dir, obs_item, "panorama/full/semantic.png")
            # 计算占用网格
            obs_item_bev = generate_semantic_voxel(obs_item_depth_full, obs_item_semantic_full)
            # 可视化占用网格
            obs_item_img = visualize_occ(obs_item_bev, LABEL_TO_COLOR)
            
            # # 可视化结果
            # cv2.imshow("BEV Image", obs_item_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # 保存结果
            os.makedirs(os.path.join(output_dir, obs_item, "panorama/full"), exist_ok=True)
            obs_item_img_output = os.path.join(output_dir, obs_item, "panorama/full/bev.png")
            cv2.imwrite(obs_item_img_output, obs_item_img)
