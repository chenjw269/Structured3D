# 20241017
# 统计场景中，每个实例对应的语义类别
import os
import cv2
import json
import numpy as np
from tqdm import tqdm

import sys
sys.path.append(".")

from assets.color2semantic import COLOR_TO_SEMANTIC


if __name__ == "__main__":

    data_pth = "e:/datasets/Structure3D/Structured3D"

    # Structured3D 包括 3500 个场景
    scene_index_list = [f"scene_{num:05}" for num in range(3500)]
    # # 统计前 100 个场景
    # scene_index_list = [f"scene_{num:05}" for num in range(100)]

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
    for scene_index in tqdm(scene_index_list[706:]):

        # 缺少标注的场景作废
        if scene_index in scene_invalid:
            tqdm.write(f"Jmp annos loss {scene_index}")
            continue

        # 记录当前场景下实例与语义类别的对应关系
        scene_ins2sem_dict = {}

        # 观测数据
        obs_dir = os.path.join(data_pth, scene_index, "2D_rendering")
        obs_list = os.listdir(obs_dir)
        for obs_item in tqdm(obs_list, leave=False):

            # 缺少观测值的样本作废
            if f"{scene_index},{obs_item}" in obs_invalid:
                tqdm.write(f"Jmp obs loss {scene_index} {obs_item}")
                continue

            # 检查文件是否完整：rgb 图、语义分割图、深度图

            # 全景图，完整场景
            obs_item_dir_full = os.path.join(obs_dir, obs_item, "panorama/full")
            
            # 实例分割图像
            obs_item_instance_full= os.path.join(obs_item_dir_full, "instance.png")
            # unchanged 格式，读取 int 类型索引
            obs_item_instance_full = cv2.imread(obs_item_instance_full, cv2.IMREAD_UNCHANGED)
            # 图像中所有的实例，最末一个是无意义的 65535
            obs_item_instance_list = np.unique(obs_item_instance_full)[:-1]
            
            # 语义分割图像
            obs_item_semantic_full = os.path.join(obs_item_dir_full, "semantic.png")
            # 默认是 BGR 通道读取，将 BGR 转换为 RGB
            obs_item_semantic_full = cv2.imread(obs_item_semantic_full)
            obs_item_semantic_full = cv2.cvtColor(obs_item_semantic_full, cv2.COLOR_BGR2RGB)

            # 遍历图像中的实例，统计每个实例的语义类别及像素点数量
            for instance_id in obs_item_instance_list:
                # 如果已有此实例的记录
                if str(instance_id) in scene_ins2sem_dict.keys():
                    pass
                # 如果没有此实例的记录，则创建一个记录
                else:
                    scene_ins2sem_dict[str(instance_id)] = {}
                
                # 实例对应的像素位置
                instance_index = [obs_item_instance_full == instance_id][0]
                # 实例对应的语义颜色
                instance_semantic = obs_item_semantic_full[instance_index]
                # 使用 np.unique 计算每种颜色（每一行）出现的次数
                colors, counts = np.unique(instance_semantic, axis=0, return_counts=True)

                # 每种颜色的语义类别，及其像素个数，存储到该实例数据下
                for index, c in enumerate(colors):
                    semantic_type = COLOR_TO_SEMANTIC[tuple(c.tolist())]

                    if semantic_type in scene_ins2sem_dict[str(instance_id)].keys():
                        scene_ins2sem_dict[str(instance_id)][semantic_type] += int(counts[index])
                    else:
                        scene_ins2sem_dict[str(instance_id)][semantic_type] = int(counts[index])
                
        scene_ins2sem_pth = os.path.join(data_pth, scene_index, "ins_semantic.json")
        with open(scene_ins2sem_pth, "w") as f:
            json.dump(scene_ins2sem_dict, f, indent=4)

