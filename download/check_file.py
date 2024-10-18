# 20241017
# 检查文件完整性
import os
import json
from PIL import Image
from tqdm import tqdm


def check_json_file(file_path):
    try:
        with open(file_path, 'r') as f:
            json.load(f)
        return True
    except Exception as Exec:
        raise Exception("Json error.")

def check_image_file(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()  # 验证图片文件是否损坏
        return True
    except (IOError, SyntaxError) as e:
        
        with open("", "a+") as f:
            f.write(f"{file_path}\n")
        
        raise Exception("Image error.")

if __name__ == "__main__":
    data_pth = "e:/datasets/Structure3D/Structured3D"

    # Structured3D 包括 3500 个场景
    scene_index_list = [f"scene_{num:05}" for num in range(3500)]

    # 遍历场景
    for scene_index in tqdm(scene_index_list):
        
        # 场景地图
        scene_annotation = os.path.join(data_pth, scene_index, "annotation_3d.json")
        check_json_file(scene_annotation)
        # 物体包围盒
        scene_bbox = os.path.join(data_pth, scene_index, "bbox_3d.json")
        check_json_file(scene_bbox)
        
        # 观测数据
        obs_dir = os.path.join(data_pth, scene_index, "2D_rendering")
        obs_list = os.listdir(obs_dir)
        for obs_item in obs_list:

            # 全景图，简化场景
            # obs_item_dir_simple = os.path.join(obs_dir, obs_item, "panorama/simple")

            # obs_item_rgb_simple = os.path.join(obs_item_dir_simple, "rgb_rawlight.png")
            # check_image_file(obs_item_rgb_simple)
            # obs_item_depth_simple = os.path.join(obs_item_dir_simple, "depth.png")
            # check_image_file(obs_item_depth_simple)
            # obs_item_semantic_simple = os.path.join(obs_item_dir_simple, "semantic.png")
            # check_image_file(obs_item_semantic_simple)
            
            # 全景图，完整场景
            obs_item_dir_full = os.path.join(obs_dir, obs_item, "panorama/full")
            obs_item_rgb_full = os.path.join(obs_item_dir_full, "rgb_rawlight.png")
            check_image_file(obs_item_rgb_full)
            obs_item_depth_full = os.path.join(obs_item_dir_full, "depth.png")
            check_image_file(obs_item_depth_full)
            obs_item_semantic_full = os.path.join(obs_item_dir_full, "semantic.png")
            check_image_file(obs_item_semantic_full)
            obs_item_instance_full= os.path.join(obs_item_dir_full, "instance.png")
            check_image_file(obs_item_instance_full)
            
            # 单视角图像，简化场景
            
            # 单视角图像，完整场景