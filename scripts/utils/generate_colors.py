import numpy as np
import matplotlib.pyplot as plt


def generate_colors(k):
    # 使用colormap生成k个均匀分布的颜色
    colors = plt.cm.get_cmap('hsv', k)  # 选择hsv色彩空间，保证颜色差异较大
    rgb_colors = [colors(i)[:3] for i in range(k)]  # 提取RGB值
    rgb_colors = (np.array(rgb_colors) * 255).astype(int)  # 转换为0-255范围内的整数
    rgb_colors = {
        index+1 : color.tolist() for index, color in enumerate(rgb_colors)
    }
    rgb_colors[0] = (255,255,255) # 空格为白色
    rgb_colors[1] = (0, 0, 0) # 墙壁为黑色
    return rgb_colors