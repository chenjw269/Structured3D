import torch
import numpy as np

def extract_local_patches(image, points, patch_size=256):
    # 图像的高和宽
    H, W = image.shape
    # 半边的大小
    half_patch = patch_size // 2
    # 存储结果的数组
    patches = []
    
    if isinstance(points, np.ndarray):
        points = [points.tolist()]
    if isinstance(points, torch.Tensor):
        points = [points.tolist()]

    for point in points:
        x, y = point
        # 确保不超出边界，计算局部区域的起始和结束坐标
        x_start = max(0, x - half_patch)
        y_start = max(0, y - half_patch)
        x_end = min(H, x + half_patch)
        y_end = min(W, y + half_patch)
        
        # 计算实际区域大小（如果超出边界则补齐）
        patch = np.zeros((patch_size, patch_size))
        patch_x_start = half_patch - (x - x_start)
        patch_y_start = half_patch - (y - y_start)
        
        # 复制局部图像区域到 patch 中
        patch[patch_x_start:patch_x_start + (x_end - x_start), patch_y_start:patch_y_start + (y_end - y_start)] = image[x_start:x_end, y_start:y_end]
        patches.append(patch)

    # 返回形状为 (N, 256, 256) 的数组
    return np.array(patches)

if __name__ == "__main__":

    # 示例数据
    image = np.random.rand(800, 800)  # 生成一个 800x800 的图像
    points = np.array([[100, 100], [400, 400], [700, 700]])  # 一组位置点

    # 提取局部图像
    patches = extract_local_patches(image, points)
    print(patches.shape)  # 输出结果的形状 (N, 256, 256)
