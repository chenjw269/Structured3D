# 20241017
# 检查文件完整性
import os
import cv2
import json
from PIL import Image
from tqdm import tqdm


def check_image_file(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()  # 验证图片文件是否损坏
        return True
    except Exception as Exec:
        return False

def check_image_panorama(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()  # 验证图片文件是否损坏
        
        # 读取图像
        image = Image.open(file_path)
        # 获取图像形状
        width, height = image.size
        # 检查形状
        if (height, width) == (512, 1024):
            return True
        else:
            return False

    except Exception as Exec:
        return False


if __name__ == "__main__":

    # 记录到日志文件
    # f_record = open("../logs/scene_observation.txt", "w", encoding="utf-8") # remote
    # data_pth = "/data1/chenjiwei/S3D/zip/Structured3D" # remote
    f_record = open("logs/scene_observation.txt", "w", encoding="utf-8") # local
    data_pth = "e:/datasets/Structure3D/Structured3D" # local

    # Structured3D 包括 3500 个场景
    scene_index_list = [f"scene_{num:05}" for num in range(3500)]

    # 标注数据缺失的场景
    # with open("../logs/scene_annos.txt", encoding="utf-8") as f: # remote
    with open("logs/scene_annos.txt", encoding="utf-8") as f: # local
            scene_invalid = f.readlines()
    for index, item in enumerate(scene_invalid):
        scene_invalid[index] = item.replace("\n", "")

    # 遍历场景
    for scene_index in tqdm(scene_index_list):

        # 缺少标注的场景作废
        if scene_index in scene_invalid:
            continue
        
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
            rgb_valid = check_image_panorama(obs_item_rgb_full)
            obs_item_depth_full = os.path.join(obs_item_dir_full, "depth.png")
            depth_valid = check_image_panorama(obs_item_depth_full)
            obs_item_semantic_full = os.path.join(obs_item_dir_full, "semantic.png")
            semantic_valid = check_image_panorama(obs_item_semantic_full)
            obs_item_instance_full= os.path.join(obs_item_dir_full, "instance.png")
            instance_valid = check_image_panorama(obs_item_instance_full)
            obs_item_pos_full = os.path.join(obs_dir, obs_item, "panorama/camera_xyz.txt")
            pos_valid = os.path.exists(obs_item_pos_full)
            
            if rgb_valid and depth_valid and semantic_valid and instance_valid and pos_valid:
                pass
            else:
                tqdm.write(f"Obs loss {scene_index} {obs_item}")
                f_record.write(f"{scene_index},{obs_item}\n")
            
            # 单视角图像，简化场景
            
            # 单视角图像，完整场景

    f_record.close()
