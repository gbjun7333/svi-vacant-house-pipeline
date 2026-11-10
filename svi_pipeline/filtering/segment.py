"""SegFormer-B2 (Cityscapes) segmentation of one image and its class ratios (Section 3.2)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SEGFORMER_MODEL_ID = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"

# Cityscapes train ids
CLASS_INDICES = {
    "road": 0, "sidewalk": 1, "building": 2, "wall": 3, "fence": 4,
    "pole": 5, "traffic_light": 6, "traffic_sign": 7, "vegetation": 8,
    "terrain": 9, "sky": 10, "person": 11, "rider": 12, "car": 13,
    "truck": 14, "bus": 15, "train": 16, "motorcycle": 17, "bicycle": 18,
}

VEHICLE_CLASSES = [
    CLASS_INDICES["car"], CLASS_INDICES["truck"], CLASS_INDICES["bus"],
]


def load_model(device: torch.device):
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    processor = SegformerImageProcessor.from_pretrained(SEGFORMER_MODEL_ID)
    model = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_MODEL_ID)
    model.eval()
    model.to(device)
    return model, processor


def segment_single_image(
    model, processor, image_path: Path, device: torch.device
) -> Optional[Dict]:
    """Class ratios of one image, or None if it cannot be read or segmented."""
    try:
        img = Image.open(image_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        upsampled = F.interpolate(
            logits,
            size=(img.size[1], img.size[0]),
            mode="bilinear",
            align_corners=False,
        )
        mask = upsampled.argmax(dim=1).squeeze(0).cpu().numpy()
        total_pixels = mask.size
        h, w = mask.shape

        building_mask = mask == CLASS_INDICES["building"]
        result = {
            "building_ratio": float(building_mask.sum() / total_pixels),
            "vegetation_ratio": float((mask == CLASS_INDICES["vegetation"]).sum() / total_pixels),
            "sky_ratio": float((mask == CLASS_INDICES["sky"]).sum() / total_pixels),
            "road_ratio": float((mask == CLASS_INDICES["road"]).sum() / total_pixels),
            "sidewalk_ratio": float((mask == CLASS_INDICES["sidewalk"]).sum() / total_pixels),
        }

        # central 10% x 10% of the frame
        cx, cy = w // 2, h // 2
        margin_x, margin_y = max(1, w // 20), max(1, h // 20)
        center_region = building_mask[
            cy - margin_y : cy + margin_y,
            cx - margin_x : cx + margin_x,
        ]
        center_pixels = center_region.size
        result["center_building_ratio"] = float(center_region.sum() / center_pixels) if center_pixels > 0 else 0.0

        vehicle_mask = np.isin(mask, VEHICLE_CLASSES)
        result["vehicle_ratio"] = float(vehicle_mask.sum() / total_pixels)
        overlap = np.logical_and(vehicle_mask, building_mask)
        result["vehicle_building_overlap"] = float(overlap.sum() / total_pixels)

        del inputs, outputs, logits, upsampled, mask, building_mask, vehicle_mask
        return result

    except Exception:
        return None
