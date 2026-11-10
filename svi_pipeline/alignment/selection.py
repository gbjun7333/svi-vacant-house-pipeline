"""Selection of the target-house mask among the candidates of an image (Section 3.3).

The paper compares each spatial factor alone, their product and four heuristic
baselines against human labels (Section 4.4). Overlap with the parcel band performed
best and is the rule the pipeline uses.

A candidate is a dict with idx, f_overlap and s (see ``scoring``).
"""
from __future__ import annotations

from typing import Dict, List


def band_gate(cands: List[Dict]) -> List[Dict]:
    """Candidates that overlap the parcel band horizontally."""
    return [c for c in cands if c["f_overlap"] > 0]


def pick_top1_overlap(cands: List[Dict]) -> Dict:
    """Largest overlap. Ties go to the larger combined score, then the lower index."""
    return sorted(cands, key=lambda c: (-c["f_overlap"], -c["s"], c["idx"]))[0]
