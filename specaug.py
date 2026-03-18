import random
import torch

def time_mask(x: torch.Tensor, max_width: int = 30, num_masks: int = 2):
    B, T, F = x.shape
    for _ in range(num_masks):
        w = random.randint(0, max_width)
        if w == 0 or T - w <= 0:
            continue
        t0 = random.randint(0, T - w)
        x[:, t0:t0+w, :] = 0
    return x

def freq_mask(x: torch.Tensor, max_width: int = 15, num_masks: int = 2):
    B, T, F = x.shape
    for _ in range(num_masks):
        w = random.randint(0, max_width)
        if w == 0 or F - w <= 0:
            continue
        f0 = random.randint(0, F - w)
        x[:, :, f0:f0+w] = 0
    return x

def spec_augment(x: torch.Tensor,
                 t_max: int = 30, t_masks: int = 2,
                 f_max: int = 15, f_masks: int = 2):
    x = time_mask(x, max_width=t_max, num_masks=t_masks)
    x = freq_mask(x, max_width=f_max, num_masks=f_masks)
    return x