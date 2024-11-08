# 多个朝向假设下的观测假设

import sys
sys.path.append(".")

import cv2
import numpy as np

from scripts.virtual_obs.pesp_vobs import virtual_pesp_obs # 单视角虚拟观测
from scripts.utils.visualize_occ import * # 可视化占用网格


if __name__ == "__main__":

    # 读取场景地图

    # 读取样本位姿
    
    # 以 15 度为步长，得到朝向假设
    orientation_hypothesis = np.arange(0, 360, 15)

    # 单视角虚拟观测
    for orientation in orientation_hypothesis:

        virtual_obs = virtual_pesp_obs()
        virtual_obs = visualize_occ(virtual_obs, LABEL_TO_COLOR)

        cv2.imwrite(f"logs/Virtual obs/{orientation}.png", )
