####################################
# 20241106
# 检查单视角视图的观测数据是否完整
####################################

import sys
sys.path.append(".")

import os
from PIL import Image
from tqdm import tqdm
from s3d import *

# 检查位姿文件 txt
def check_txt(path):
    """检查 txt 文件

    Args:
        path (str): txt 文件路径

    Returns:
        bool: txt 文件是否存在
    """
    return os.path.exists(path)

# 检查图像
def check_img(path):
    
    try:
        with Image.open(path) as img:
            img.verify()  # 验证图片文件是否损坏
        # 读取图像
        image = Image.open(path)
        # 获取图像形状
        width, height = image.size
        # 检查形状
        if (width, height) == (1280, 720):
            return True
        else:
            return False

    except Exception as Exec:
        return False

# 检查

if __name__ == "__main__":
    
    # 记录到日志文件
    f_record = open(s3d_pesp_obs_err, "a", encoding="utf-8")
    
    # # Structured3D 包括 3500 个场景
    # scene_index_list = [f"scene_{num:05}" for num in range(3500)]
    # 取前 100 个场景
    scene_index_list = [f"scene_{num:05}" for num in range(400)]
    
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
                
                # 深度图
                depth = os.path.join(ori_data_dir, "depth.png")
                depth_valid = check_img(depth)
                # 语义分割图
                semantic = os.path.join(ori_data_dir, "semantic.png")
                semantic_valid = check_img(semantic)
                # rgb 图
                rgb = os.path.join(ori_data_dir, "rgb_rawlight.png")
                rgb_valid = check_img(rgb)
                # 位姿
                pose = os.path.join(ori_data_dir, "camera_pose.txt")
                pose_valid = os.path.exists(pose)

                if rgb_valid and depth_valid and semantic_valid and pose_valid:
                    pass
                else:
                    tqdm.write(f"Pesp obs loss {scene_index}/{sample_index}/{ori_index}")
                    f_record.write(f"{scene_index},{sample_index},{ori_index}\n")

    f_record.close()
