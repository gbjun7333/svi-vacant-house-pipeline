"""Building Capture Rate (Section 3.2, Equation 1).

BCR(tau) is the share of images whose building-pixel ratio r_i is at least tau:

    BCR(tau) = (1 / N) * sum_i 1(r_i >= tau)

The paper reports tau = 1%, 3%, 5% and 10% and uses 5% as the principal threshold.
"""
from __future__ import annotations

from typing import Dict, Sequence

BCR_THRESHOLDS = [0.01, 0.03, 0.05, 0.10]
PRINCIPAL_THRESHOLD = 0.05


def building_capture_rate(records: Sequence[Dict], threshold: float) -> float:
    """BCR of a set of per-image records with a ``building_ratio`` field (0.0 if empty)."""
    if not records:
        return 0.0
    return sum(1 for r in records if r["building_ratio"] >= threshold) / len(records)


def threshold_key(threshold: float) -> str:
    """Key used in result files, e.g. 0.05 -> "tau_5pct"."""
    return f"tau_{int(threshold * 100)}pct"
