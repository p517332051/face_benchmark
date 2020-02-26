import torch
import numpy as np

def divs_tensors(device,tensors, targets, divs_nums):
    ImageList_div = torch.chunk(tensors.to(device), divs_nums)
    start_slice = 0
    targets_div = []
    for index, cast_tensor in enumerate(ImageList_div):
        batch_size = cast_tensor.shape[0]
        if targets!=None:
            targets_div.append(
                torch.Tensor(targets[start_slice:start_slice + batch_size]).to(device).long()
            )
        start_slice += batch_size
    return ImageList_div, targets_div