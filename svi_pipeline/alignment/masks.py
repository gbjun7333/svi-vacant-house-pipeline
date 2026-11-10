"""House masks of one image, bit-packed in one .npz file (Section 3.3).

A mask index is the position of the mask in that file.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np


def load_all_masks(npz_path: Path) -> Tuple[List[np.ndarray], List[float], List[Tuple[int, ...]]]:
    """Masks (H x W bool arrays), scores and predicted boxes of one image."""
    d = np.load(npz_path)
    num = int(d["num_masks"])
    if num == 0:
        return [], [], []
    H, W = int(d["height"]), int(d["width"])
    bits = d["mask_bits"]
    nb = H * W
    masks = [np.unpackbits(bits[i])[:nb].reshape(H, W).astype(bool) for i in range(num)]
    scores = [float(s) for s in d["scores"]]
    bboxes = [tuple(int(v) for v in bb) for bb in d["bboxes_xyxy"]]
    return masks, scores, bboxes


def load_mask_stack(npz_path: Path) -> np.ndarray:
    """All masks of one image as a bool array [n, H, W]."""
    data = np.load(npz_path)
    num_masks = int(data["num_masks"])
    height = int(data["height"])
    width = int(data["width"])
    stack = np.zeros((num_masks, height, width), dtype=bool)
    for idx in range(num_masks):
        stack[idx] = np.unpackbits(data["mask_bits"][idx])[: height * width].reshape(height, width).astype(bool)
    return stack


def tight_bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """(x1, y1, x2, y2) extent of a mask, x2 and y2 exclusive."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("empty mask")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def save_mask_npz(save_path: Path, masks: np.ndarray, boxes: np.ndarray, scores: np.ndarray,
                  height: int, width: int) -> None:
    """Write the masks of one image ([n, H, W] bool)."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    n = masks.shape[0]
    if n == 0:
        np.savez_compressed(save_path, num_masks=np.int32(0), height=np.int32(height), width=np.int32(width),
                            mask_bits=np.zeros((0, 0), dtype=np.uint8), bboxes_xyxy=np.zeros((0, 4), dtype=np.int32),
                            areas=np.zeros((0,), dtype=np.int32), scores=np.zeros((0,), dtype=np.float32))
        return
    masks_flat = masks.reshape(n, -1)
    np.savez_compressed(save_path, num_masks=np.int32(n), height=np.int32(height), width=np.int32(width),
                        mask_bits=np.packbits(masks_flat, axis=1), bboxes_xyxy=boxes.astype(np.int32),
                        areas=masks.sum(axis=(1, 2)).astype(np.int32), scores=scores.astype(np.float32))
