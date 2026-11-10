"""Building Capture Rate, the share of images whose building ratio is at least tau (Section 3.2, Eq. 1)."""
from __future__ import annotations

from typing import Dict, Sequence

BCR_THRESHOLDS = [0.01, 0.03, 0.05, 0.10]
PRINCIPAL_THRESHOLD = 0.05


def building_capture_rate(records: Sequence[Dict], threshold: float) -> float:
    if not records:
        return 0.0
    return sum(1 for r in records if r["building_ratio"] >= threshold) / len(records)


def threshold_key(threshold: float) -> str:
    """0.05 -> "tau_5pct"."""
    return f"tau_{int(threshold * 100)}pct"
