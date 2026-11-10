"""Model inputs of the classification experiment (Section 3.4).

Every condition sees the whole 4:3 frame resized to 256 x 192, without a centre crop.

    training    random crop of 80 to 100% of the frame area at the 4:3 aspect ratio,
                resized to 256 x 192, horizontal flip, colour jitter
    evaluation  plain resize to 256 x 192

Rows are dicts read from the split files: ``crop_path`` (image), ``label_id``
(1 vacant, 0 non-vacant), ``example_id``, ``source``, and for the conditions with a
mask the path of an 8-bit PNG of the frame's size in which 255 marks the house
(``mask_path`` for the top-1 mask, ``mask_shuf_path`` for the shuffled-mask control).

For those conditions the mask is the fourth input channel. It gets the same crop and
flip as the image (nearest-neighbour resampling) and the values -1 (background) and
+1 (house). ``mask_only`` sets the three image channels to zero.
"""
from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

MEAN = [0.485, 0.456, 0.406]   # ImageNet
STD = [0.229, 0.224, 0.225]
FRAME43 = [192, 256]           # (height, width) of the model input


def rgb_transforms():
    """(training transform, evaluation transform) for the three-channel conditions."""
    train_tf = T.Compose([
        T.RandomResizedCrop(tuple(FRAME43), scale=(0.8, 1.0), ratio=(4 / 3, 4 / 3)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.03),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    eval_tf = T.Compose([
        T.Resize(tuple(FRAME43)),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    return train_tf, eval_tf


class ImageDataset(Dataset):
    """Three-channel input: the image only."""

    def __init__(self, rows: list[dict[str, str]], transform):
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(row["crop_path"]).convert("RGB")
        x = self.transform(image)
        y = np.float32(int(row["label_id"]))
        return x, y, row["example_id"], row["source"]


class MaskTransform:
    """Joint transform of an image and its mask into a four-channel tensor."""

    def __init__(self, train: bool, mask_col: str = "mask_path", mask_only: bool = False):
        self.train, self.mask_col, self.mask_only = train, mask_col, mask_only
        self.jitter = T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.03)
        self.out = FRAME43
        self.rrc = T.RandomResizedCrop(FRAME43, scale=(0.8, 1.0), ratio=(4 / 3, 4 / 3))

    def __call__(self, img: Image.Image, mask: Image.Image) -> torch.Tensor:
        if self.train:
            i, j, h, w = self.rrc.get_params(img, self.rrc.scale, self.rrc.ratio)
            img = F.resized_crop(img, i, j, h, w, self.out, InterpolationMode.BILINEAR, antialias=True)
            mask = F.resized_crop(mask, i, j, h, w, self.out, InterpolationMode.NEAREST)
            if random.random() < 0.5:
                img, mask = F.hflip(img), F.hflip(mask)
            img = self.jitter(img)
        else:
            img = F.resize(img, FRAME43, InterpolationMode.BILINEAR, antialias=True)
            mask = F.resize(mask, FRAME43, InterpolationMode.NEAREST)
        x = F.normalize(F.to_tensor(img), MEAN, STD)
        if self.mask_only:
            x = torch.zeros_like(x)
        m = torch.from_numpy((np.asarray(mask, dtype=np.float32) > 127).astype(np.float32))[None] * 2.0 - 1.0
        return torch.cat([x, m], dim=0)


class MaskDataset(Dataset):
    """Four-channel input: the image and the mask named by the transform's ``mask_col``."""

    def __init__(self, rows, transform: MaskTransform):
        self.rows, self.transform = rows, transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        r = self.rows[index]
        img = Image.open(r["crop_path"]).convert("RGB")
        mask = Image.open(r[self.transform.mask_col]).convert("L")
        if mask.size != img.size:
            raise ValueError(f"mask size {mask.size} != image size {img.size}: {r[self.transform.mask_col]}")
        return self.transform(img, mask), np.float32(int(r["label_id"])), r["example_id"], r["source"]
