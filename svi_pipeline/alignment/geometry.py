"""Parcel band of an image (Section 3.3).

Local east-north metres with the camera at the origin. heading_deg is a compass
bearing (0 = north, 90 = east). Column x looks along the bearing
theta = heading + (x / W - 0.5) * fov, and the parcel band is the column range
of the parcel's vertices.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

from shapely.geometry import MultiPolygon, Polygon

CAR_FWD_PAD_M = 0.5      # vertices closer than this in front of the camera are ignored
EARTH_R_M = 6378137.0


def ll_to_xy(lat: float, lng: float, lat0: float, lng0: float) -> Tuple[float, float]:
    """(east, north) offset in metres from (lat0, lng0)."""
    x = math.radians(lng - lng0) * EARTH_R_M * math.cos(math.radians(lat0))
    y = math.radians(lat - lat0) * EARTH_R_M
    return x, y


def polygon_to_local(geom: Dict, lat0: float, lng0: float):
    """GeoJSON Polygon or MultiPolygon (lon/lat) as a shapely geometry in local metres."""
    def conv(coords):
        return [ll_to_xy(c[1], c[0], lat0, lng0) for c in coords]

    t = geom["type"]
    if t == "Polygon":
        return Polygon(conv(geom["coordinates"][0]),
                       [conv(r) for r in geom["coordinates"][1:]])
    if t == "MultiPolygon":
        return MultiPolygon([Polygon(conv(pc[0]), [conv(r) for r in pc[1:]])
                             for pc in geom["coordinates"]])
    raise ValueError(t)


def vertex_column(x_world, y_world, pano_xy, heading_deg, fov_h_deg, W) -> Optional[float]:
    """Image column of a ground point. None when it is behind the camera or within CAR_FWD_PAD_M of it."""
    dx = x_world - pano_xy[0]
    dy = y_world - pano_xy[1]
    h = math.radians(heading_deg)
    # east = +x, north = +y
    fwd = dx * math.sin(h) + dy * math.cos(h)
    right = dx * math.cos(h) - dy * math.sin(h)
    if fwd <= CAR_FWD_PAD_M:
        return None
    offset = math.atan2(right, fwd)     # theta - heading
    return W * (0.5 + offset / math.radians(fov_h_deg))


def parcel_band(target_polygon, pano_xy, heading_deg, fov_h, W):
    """(u_min, u_max) of the projected parcel, clamped to the image.

    None when no vertex is in front of the camera.
    """
    if isinstance(target_polygon, MultiPolygon):
        exteriors = [list(p.exterior.coords) for p in target_polygon.geoms]
    else:
        exteriors = [list(target_polygon.exterior.coords)]
    us = []
    for ring in exteriors:
        for x, y in ring:
            u = vertex_column(x, y, pano_xy, heading_deg, fov_h, W)
            if u is not None:
                us.append(u)
    if not us:
        return None
    return max(0.0, min(us)), min(float(W), max(us))


def target_distance_m(target_polygon, pano_xy) -> float:
    """Distance from the camera to the parcel centroid."""
    c = target_polygon.centroid
    return math.sqrt((c.x - pano_xy[0]) ** 2 + (c.y - pano_xy[1]) ** 2)
