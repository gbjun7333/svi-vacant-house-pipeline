"""Top-1 house mask of an image, by overlap with the parcel band (Section 3.3)."""
from __future__ import annotations

from typing import Dict, List


def band_gate(cands: List[Dict]) -> List[Dict]:
    """Candidates that overlap the parcel band."""
    return [c for c in cands if c["f_overlap"] > 0]


def pick_top1_overlap(cands: List[Dict]) -> Dict:
    """Largest overlap. Ties go to the larger combined score, then the lower index."""
    return sorted(cands, key=lambda c: (-c["f_overlap"], -c["s"], c["idx"]))[0]
