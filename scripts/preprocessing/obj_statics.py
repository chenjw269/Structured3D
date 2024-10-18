# 20241018
# 统计每个场景下物体的类别和出现次数
import os
import json
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # 统计每种类别物体出现的次数
    obj_nums = {}
    
    data_pth = "e:/datasets/Structure3D/Structured3D"

    # # Structured3D 包括 3500 个场景
    # scene_index_list = [f"scene_{num:05}" for num in range(3500)]
    # 统计前 100 个场景
    scene_index_list = [f"scene_{num:05}" for num in range(100)]

    # 遍历场景
    for scene_index in tqdm(scene_index_list):
        # print(scene_index)
        scene_data_pth = os.path.join(data_pth, scene_index)
        
        # 物体地图数据
        obj_map = os.path.join(scene_data_pth, "bbox_3d.json")
        with open(obj_map, encoding="utf-8") as f:
            obj_map = json.load(f)
        # 实例语义映射
        ins2sem = os.path.join(scene_data_pth, "ins_semantic.json")
        with open(ins2sem, encoding="utf-8") as f:
            ins2sem = json.load(f)

        # 遍历所有物体，统计出现次数
        for obj in obj_map:
            
            # 查表，将实例映射到语义类别，如果不是有意义的类别则跳过
            ins_id = str(obj["ID"])
            if ins_id in ins2sem.keys():
                ins_semantic = ins2sem[ins_id]
                # 排序，得到该实例最可能的语义类别
                ins_semantic = max(ins_semantic, key=ins_semantic.get)
                # 不是有意义的类别则跳过
                if ins_semantic in ["prop", "unknown", "structure"]:
                    continue
                
            else:
                continue

            # 统计物体出现次数
            if ins_semantic in obj_nums.keys():
                obj_nums[ins_semantic] += 1
            else:
                obj_nums[ins_semantic] = 1

    # 统计物体出现的次数
    keys = list(obj_nums.keys())
    values = list(obj_nums.values())
    # 绘制柱状图
    plt.figure(figsize=(16, 6))
    plt.bar(keys, values, color='skyblue')
    # 旋转X轴标签，调整字体大小
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.xlabel('Semantic types')
    plt.ylabel('Occurrence nums')
    plt.title('Bar Chart of Semantic Types by Occurrence')
    plt.tight_layout()  # 自动调整布局以适应旋转的标签
    plt.show()

    # 统计出现次数最多的 20 类物体
    top_20_keys = sorted(obj_nums, key=obj_nums.get, reverse=True)[:20]
    print(top_20_keys)
