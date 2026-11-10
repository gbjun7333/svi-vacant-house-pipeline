"""Image filters and the filter configuration of each operating group (Section 3.2, Table 1).

Filters are coded as in Table 1 of the paper:

    A  distance            panorama farther than 30 m from the parcel
    B  vegetation occlusion
    C  close-up capture
    D  vehicle occlusion
    E  abnormal angle
    F  central building

Filter A is applied when the images are acquired (Section 3.1), so it is not part
of this module. Filters B to F work on the ratios returned by ``segment``. Each
function returns True when the image passes.

This module has no third-party dependencies.
"""
from __future__ import annotations

from typing import Dict, Tuple


# Filters and thresholds per operating group (Table 1).
GROUP_FILTERS = {
    "Green_Rural":   {"filters": ["B", "F"],      "params": {"B_max": 0.50, "F_min": 0.10}},
    "Green_Urban":   {"filters": ["B", "D", "F"], "params": {"B_max": 0.50, "D_max": 0.02, "F_min": 0.05}},
    "OpenSky_Rural": {"filters": ["E", "F"],      "params": {"F_min": 0.15}},
    "OpenSky_Urban": {"filters": ["E", "F"],      "params": {"F_min": 0.15}},
    "BuiltUp_Rural": {"filters": ["C", "F"],      "params": {"C_max": 0.90, "F_min": 0.03}},
    "BuiltUp_Urban": {"filters": ["C", "F"],      "params": {"C_max": 0.85, "F_min": 0.03}},
}

GROUP_NAMES = [
    "Green_Rural", "Green_Urban",
    "OpenSky_Rural", "OpenSky_Urban",
    "BuiltUp_Rural", "BuiltUp_Urban",
]


def filter_b(r: Dict, threshold: float = 0.40) -> bool:
    """B, vegetation occlusion: pass when vegetation_ratio <= threshold."""
    return r["vegetation_ratio"] <= threshold


def filter_c(r: Dict, min_ratio: float = 0.05, max_ratio: float = 0.80) -> bool:
    """C, close-up capture: pass when min_ratio <= building_ratio <= max_ratio."""
    return min_ratio <= r["building_ratio"] <= max_ratio


def filter_d(r: Dict, threshold: float = 0.10) -> bool:
    """D, vehicle occlusion: pass when vehicle_building_overlap < threshold."""
    vbo = r.get("vehicle_building_overlap")
    if vbo is None:
        return True
    return vbo < threshold


def filter_e(r: Dict, sky_min: float = 0.01, road_min: float = 0.01, bldg_exception: float = 0.50) -> bool:
    """E, abnormal angle: fail when both sky and road are nearly absent.

    An image whose building_ratio is at least ``bldg_exception`` always passes.
    """
    if r["building_ratio"] >= bldg_exception:
        return True
    if r["sky_ratio"] < sky_min and r["road_ratio"] < road_min:
        return False
    return True


def filter_f(r: Dict, threshold: float = 0.05) -> bool:
    """F, central building: pass when center_building_ratio >= threshold."""
    cbr = r.get("center_building_ratio")
    if cbr is None:
        return True
    return cbr >= threshold


def apply_group_filters(r: Dict, group_name: str) -> Tuple[bool, Dict[str, bool]]:
    """Apply the filters of an operating group to one image.

    Returns (passed, result of each applied filter). A group name that is not in
    ``GROUP_FILTERS`` gets filters B to F with their default thresholds.
    """
    if group_name not in GROUP_FILTERS:
        results = {
            "B": filter_b(r), "C": filter_c(r), "D": filter_d(r),
            "E": filter_e(r), "F": filter_f(r),
        }
        return all(results.values()), results

    config = GROUP_FILTERS[group_name]
    filters = config["filters"]
    params = config["params"]

    results = {}
    for f_name in filters:
        if f_name == "B":
            results["B"] = filter_b(r, threshold=params.get("B_max", 0.40))
        elif f_name == "C":
            results["C"] = filter_c(r, max_ratio=params.get("C_max", 0.80))
        elif f_name == "D":
            results["D"] = filter_d(r, threshold=params.get("D_max", 0.10))
        elif f_name == "E":
            results["E"] = filter_e(r)
        elif f_name == "F":
            results["F"] = filter_f(r, threshold=params.get("F_min", 0.05))

    passed = all(results.values())
    return passed, results
