# 20241019
# 从地图上采样位置假设
import torch


def position_hypothesis(x_range, y_range, step):
    
    x_range = [
        x_range[0] - step,
        x_range[1] + step
    ]
    y_range = [
        y_range[0] - step,
        y_range[1] + step
    ]
    
    Iy, Ix = torch.meshgrid(
        torch.arange(x_range[0], x_range[1], step),
        torch.arange(y_range[0], y_range[1], step),
    )
    samples_loc = torch.stack([Ix, Iy], dim=-1).reshape(-1, 2)

    return samples_loc


if __name__ == "__main__":

    window_size = 256
    slide_step = 10


