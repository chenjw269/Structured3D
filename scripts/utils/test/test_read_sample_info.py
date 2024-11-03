# 读取样本的位姿等信息
import sys
sys.path.append(".")

from scripts.utils.read_sample_info import read_sample_info


if __name__ == "__main__":
    
    test_scene_index = "scene_00000"
    test_sample_index = ("485142", "0")

    # 读取样本信息
    sample_info = read_sample_info(test_scene_index, test_sample_index, mode="gravity")

    print(f"Infomation in sample {sample_info.keys()}")

    print(f"Sample pose {sample_info['sample pose']}")
