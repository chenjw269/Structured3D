# 20241019
# 从地图上采样位置假设
import torch


def position_hypothesis(x_range, y_range, step, boundary):
    
    padding = max(step, boundary)
    
    x_range = [
        x_range[0] - padding,
        x_range[1] + padding
    ]
    y_range = [
        y_range[0] - padding,
        y_range[1] + padding
    ]
    
    Iy, Ix = torch.meshgrid(
        torch.arange(x_range[0], x_range[1], step),
        torch.arange(y_range[0], y_range[1], step),
    )
    samples_loc = torch.stack([Ix, Iy], dim=-1).reshape(-1, 2)

    return samples_loc
