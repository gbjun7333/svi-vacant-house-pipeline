from __future__ import annotations

import os

import torch


def get_device() -> torch.device:
    """CUDA, then MPS, then CPU. SVI_DEVICE forces one, e.g. SVI_DEVICE=cpu."""
    forced = os.environ.get("SVI_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
