"""Model inputs of the classification experiment (Section 3.4).

The whole 4:3 frame is resized to 256 x 192. Training adds a random crop of 80 to 100%
of the frame, a horizontal flip and colour jitter. A mask goes in as a fourth channel
with -1 for background and +1 for the house, and gets the same crop and flip.
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
    """(train, eval) transforms for the image-only conditions."""
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
    """Image only, three channels."""

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
    """Same crop and flip for an image and its mask, stacked into four channels."""

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
    """Image plus the mask in transform.mask_col, four channels."""

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
