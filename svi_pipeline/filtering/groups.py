"""Image groups, administrative divisions and operating groups (Sections 3.2 and 4.2).

Every image belongs to one *image group* (K-means on the vegetation and sky ratios)
and one *administrative division* (a property of its parcel). Their combination is
the *operating group* that selects the filter configuration in ``filters``.

    image group     Built-up, Green, Open-sky
    division        Urban, Rural
    operating group e.g. "BuiltUp_Urban", shown as "Built-up × Urban"
"""
from __future__ import annotations

BUILT_UP = "BuiltUp"
GREEN = "Green"
OPEN_SKY = "OpenSky"
IMAGE_GROUPS = (BUILT_UP, GREEN, OPEN_SKY)

URBAN = "Urban"
RURAL = "Rural"
DIVISIONS = (URBAN, RURAL)

IMAGE_GROUP_LABELS = {BUILT_UP: "Built-up", GREEN: "Green", OPEN_SKY: "Open-sky"}


def operating_group(image_group: str, division: str) -> str:
    """Key of the operating group, e.g. ``operating_group("Green", "Rural") == "Green_Rural"``."""
    return f"{image_group}_{division}"


# Order used in Table 1 of the paper.
OPERATING_GROUPS = tuple(
    operating_group(g, d) for g in (BUILT_UP, OPEN_SKY, GREEN) for d in DIVISIONS
)

OPERATING_GROUP_LABELS = {
    operating_group(g, d): f"{IMAGE_GROUP_LABELS[g]} × {d}"
    for g in IMAGE_GROUPS
    for d in DIVISIONS
}
