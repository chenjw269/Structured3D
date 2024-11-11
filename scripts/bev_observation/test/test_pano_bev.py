
import sys
sys.path.append(".")

import cv2
from scripts.bev.pano_bev import executing_pano_bev_processing
from scripts.utils.visualize_occ import *


if __name__ == "__main__":

    test_scene_index = "scene_00000"
    test_sample_index = "485142"

    bev = executing_pano_bev_processing(test_scene_index, test_sample_index)

    bev_img = visualize_occ(bev, LABEL_TO_COLOR)

    # 1. 可视化单视角 bev
    cv2.imshow("Pesp bev", bev_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # # 2. 保存单视角 bev
    # cv2.imwrite("logs/pesp_bev.png", bev_img)

