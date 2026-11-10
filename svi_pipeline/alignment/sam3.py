"""Building instance masks with SAM 3 and the image exclusions at this step
(Section 3.3, Supplementary Table C.2).

SAM 3 is run through Hugging Face ``transformers`` (``Sam3Model``, weights
``facebook/sam3``) with two text prompts, "house" and "fence". For each prompt:

    score = sigmoid(pred_logits) * sigmoid(presence_logits), instances with score > 0.5 are kept
    mask  = mask logits resized bilinearly to the image (align_corners=False), thresholded at 0
    box   = predicted xyxy box scaled to pixels

The house masks are saved (``masks.save_mask_npz``). An image is excluded when

    no_house        no house instance
    too_many        8 or more house instances
    fence_dominant  fence masks cover half of the frame or more
    distant_small   mean house-mask area below 3% of the frame, or total below 5%

where a ratio is the summed mask area of the prompt divided by the image area.

``SAM3_MODEL`` may point to another model id or to a local snapshot folder.
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

MAX_HOUSES = 8                 # this many house instances or more -> too_many
MAX_FENCE_RATIO = 0.5          # fence_ratio at or above this -> fence_dominant
MIN_MEAN_HOUSE_RATIO = 0.03    # house_ratio / num_houses below this -> distant_small
MIN_HOUSE_RATIO = 0.05         # house_ratio below this -> distant_small


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
        """Image transform of the reference SAM 3 processor: uint8, resize to
        1008 x 1008, float in [0, 1], normalise with mean 0.5 and std 0.5."""
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
        """{prompt: (masks [n, H, W] bool, boxes [n, 4] float32, scores [n] float32)}."""
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
    """Segment one image, save its house masks and return the exclusion decision."""
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
