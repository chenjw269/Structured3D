# 20241022
# 统计标注数据缺失的场景，记录为表
import os
import json
from tqdm import tqdm


def check_json_file(file_path):
    try:
        with open(file_path, 'r') as f:
            json.load(f)
        return True
    except Exception as Exec:
        return False


if __name__ == "__main__":

    # 记录到日志文件
    f = open("../logs/scene_annos.txt", "w", encoding="utf-8") # remote
    data_pth = "/data1/chenjiwei/S3D/zip/Structured3D" # remote

    # f = open("logs/scene_annos.txt", "w", encoding="utf-8") # local
    # data_pth = "e:/datasets/Structure3D/Structured3D" # local

    # Structured3D 包括 3500 个场景
    scene_index_list = [f"scene_{num:05}" for num in range(3500)]

    # 遍历场景
    for scene_index in tqdm(scene_index_list):

        # 场景地图
        scene_annotation = os.path.join(data_pth, scene_index, "annotation_3d.json")
        anno_valid = check_json_file(scene_annotation)

        # 物体包围盒
        scene_bbox = os.path.join(data_pth, scene_index, "bbox_3d.json")
        bbox_valid = check_json_file(scene_bbox)

        if anno_valid and bbox_valid:
            pass
        else:
            print(f"Annos loss {scene_index}")
            f.write(f"{scene_index}\n")

    f.close()
