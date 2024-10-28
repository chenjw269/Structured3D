# 202410271115
# test: visualize_sample_loc
# 测试：在地图上可视化样本位置

import sys
sys.path.append(".")

from scripts.cad_mapping_v1.visualize_sample_loc import visualize_sample_loc

if __name__ == "__main__":

    scene_index = "scene_02600"
    visualize_sample_loc(scene_index)
