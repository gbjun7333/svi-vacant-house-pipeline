"""Image groups, administrative divisions and operating groups (Section 3.2).

An operating group is the image group of an image plus the division of its parcel,
e.g. BuiltUp_Urban. It decides which filters apply.
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
    return f"{image_group}_{division}"


# order of Table 1
OPERATING_GROUPS = tuple(
    operating_group(g, d) for g in (BUILT_UP, OPEN_SKY, GREEN) for d in DIVISIONS
)

OPERATING_GROUP_LABELS = {
    operating_group(g, d): f"{IMAGE_GROUP_LABELS[g]} × {d}"
    for g in IMAGE_GROUPS
    for d in DIVISIONS
}
