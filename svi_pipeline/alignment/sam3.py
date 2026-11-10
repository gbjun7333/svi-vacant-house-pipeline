"""SAM 3 house and fence masks, and the images excluded at this step (Section 3.3, Table C.2).

SAM 3 runs through Hugging Face transformers with the text prompts "house" and "fence".
SAM3_MODEL can point to another model id or a local snapshot folder.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..device import get_device
from .masks import save_mask_npz

CONF = 0.5
PROMPTS = ("house", "fence")

# exclusion thresholds, ratios are mask area over image area
MAX_HOUSES = 8                 # too_many at or above
MAX_FENCE_RATIO = 0.5          # fence_dominant at or above
MIN_MEAN_HOUSE_RATIO = 0.03    # distant_small below, mean over the house masks
MIN_HOUSE_RATIO = 0.05         # distant_small below, all house masks together


class Sam3Runner:
    def __init__(self, model_id: str | None = None, device: torch.device | None = None):
        from transformers import Sam3Model, Sam3Processor

        self.model_id = model_id or os.environ.get("SAM3_MODEL", "facebook/sam3")
        self.device = device or get_device()
        self.processor = Sam3Processor.from_pretrained(self.model_id)
        self.model = Sam3Model.from_pretrained(self.model_id).to(self.device).eval()
        self._text = {}
        for p in PROMPTS:
            t = self.processor(text=p, return_tensors="pt")
            self._text[p] = {k: v.to(self.device) for k, v in t.items() if k in ("input_ids", "attention_mask")}

    def pixel_values(self, image: Image.Image) -> torch.Tensor:
        """Same preprocessing as the reference SAM 3 processor."""
        from torchvision.transforms import v2

        tf = v2.Compose([
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(1008, 1008)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        x = v2.functional.to_image(image)
        return tf(x).unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> dict:
        """{prompt: (masks [n, H, W], boxes [n, 4], scores [n])}."""
        w, h = image.size
        px = self.pixel_values(image)
        vision = self.model.get_vision_features(pixel_values=px)
        out = {}
        for p in PROMPTS:
            o = self.model(vision_embeds=vision, **self._text[p])
            scores = o.pred_logits[0].float().sigmoid()
            if getattr(o, "presence_logits", None) is not None:
                scores = scores * o.presence_logits[0].float().sigmoid()
            scores = scores.reshape(-1)
            keep = scores > CONF
            if int(keep.sum()) == 0:
                out[p] = (np.zeros((0, h, w), dtype=bool), np.zeros((0, 4), dtype=np.float32),
                          np.zeros((0,), dtype=np.float32))
                continue
            logits = o.pred_masks[0][keep].float()
            logits = torch.nn.functional.interpolate(logits.unsqueeze(0), size=(h, w), mode="bilinear",
                                                     align_corners=False).squeeze(0)
            masks = (logits > 0).cpu().numpy()
            boxes = o.pred_boxes[0][keep].float().cpu().numpy() * np.array([w, h, w, h], dtype=np.float32)
            out[p] = (masks, boxes.astype(np.float32), scores[keep].cpu().numpy().astype(np.float32))
        return out


def house_filter(num_houses: int, house_ratio: float, fence_ratio: float) -> tuple[str, str]:
    """("pass", "pass") or ("fail", reason) for one image (Table C.2)."""
    if num_houses == 0:
        return "fail", "no_house"
    if num_houses >= MAX_HOUSES:
        return "fail", "too_many"
    if fence_ratio >= MAX_FENCE_RATIO:
        return "fail", "fence_dominant"
    if (house_ratio / num_houses) < MIN_MEAN_HOUSE_RATIO or house_ratio < MIN_HOUSE_RATIO:
        return "fail", "distant_small"
    return "pass", "pass"


def run_one(runner: Sam3Runner, image_path: Path, house_npz: Path) -> dict:
    """Segment one image, save its house masks, return the exclusion result."""
    t0 = time.time()
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    res = runner.predict(img)
    hm, hb, hs = res["house"]
    fm, _fb, _fs = res["fence"]
    area = float(w * h)
    num_h, num_f = int(hm.shape[0]), int(fm.shape[0])
    house_ratio = float(hm.sum()) / area if num_h else 0.0
    fence_ratio = float(fm.sum()) / area if num_f else 0.0
    result, reason = house_filter(num_h, house_ratio, fence_ratio)
    save_mask_npz(house_npz, hm, hb, hs, h, w)
    return {"width": w, "height": h, "num_houses": num_h, "house_ratio": round(house_ratio, 6),
            "num_fences": num_f, "fence_ratio": round(fence_ratio, 6), "filter_result": result,
            "filter_reason": reason, "sec": round(time.time() - t0, 3)}
