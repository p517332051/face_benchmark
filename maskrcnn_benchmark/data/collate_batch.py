# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.structures.image_list import to_image_list,image_list_tensor
import numpy as np
__all__ = ["BatchCollator","BatchCollatorTriplet"]


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = to_image_list(transposed_batch[0], self.size_divisible)
        targets = transposed_batch[1]
        img_ids = transposed_batch[2]
        return images, targets, img_ids


class BatchCollatorFace(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = image_list_tensor(transposed_batch[0])
        targets = transposed_batch[1]
        img_ids = transposed_batch[2]
        return images, targets, img_ids

class BatchCollatorTriplet(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        #img_a, img_p, img_n,c1,c2
        transposed_batch = list(zip(*batch))

        img_a = image_list_tensor(transposed_batch[0])
        img_p = image_list_tensor(transposed_batch[1])
        img_n = image_list_tensor(transposed_batch[2])

        c1    = transposed_batch[3]
        c2 = transposed_batch[4]
        return img_a, img_p, img_n,c1,c2
def BUILD_BatchCollator(cfg):

    if  "Triplet" in cfg.MODEL.META_ARCHITECTURE:
        return BatchCollatorTriplet(cfg.DATALOADER.SIZE_DIVISIBILITY)
    elif "Face" in cfg.MODEL.META_ARCHITECTURE:
        return BatchCollatorFace(cfg.DATALOADER.SIZE_DIVISIBILITY)
    else:
        return BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)


class BBoxAugCollator(object):
    """
    From a list of samples from the dataset,
    returns the images and targets.
    Images should be converted to batched images in `im_detect_bbox_aug`
    """

    def __call__(self, batch):
        return list(zip(*batch))

