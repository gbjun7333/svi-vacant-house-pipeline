"""Projection of a parcel into an image (Section 3.3).

Coordinates are metres in a local east-north frame whose origin is the camera
position. The camera sits at the origin, ``CAMERA_HEIGHT_M`` above the ground, with
no tilt or roll, and looks along ``heading_deg`` (compass bearing, 0 = north,
90 = east). Parcel vertices are projected with a pinhole model, and the horizontal
extent of the projected parcel is the *parcel band* in which the target house is
expected to appear.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

from shapely.geometry import MultiPolygon, Polygon

CAMERA_HEIGHT_M = 2.5
CAR_FWD_PAD_M = 0.5      # vertices closer than this in front of the camera are ignored
EARTH_R_M = 6378137.0


# ---------------------------------------------------------------------------
# longitude/latitude -> local metres
# ---------------------------------------------------------------------------

def ll_to_xy(lat: float, lng: float, lat0: float, lng0: float) -> Tuple[float, float]:
    """Metre offset (east, north) of (lat, lng) from the origin (lat0, lng0)."""
    x = math.radians(lng - lng0) * EARTH_R_M * math.cos(math.radians(lat0))
    y = math.radians(lat - lat0) * EARTH_R_M
    return x, y


def polygon_to_local(geom: Dict, lat0: float, lng0: float):
    """GeoJSON Polygon or MultiPolygon (EPSG:4326) as a shapely geometry in local metres."""
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


# ---------------------------------------------------------------------------
# pinhole projection
# ---------------------------------------------------------------------------

def focal_px(width_px: int, fov_h_deg: float) -> float:
    """Horizontal focal length in pixels."""
    return (width_px / 2.0) / math.tan(math.radians(fov_h_deg) / 2.0)


def project_world_to_image(x_world, y_world, pano_xy, cam_h, heading_deg,
                           fov_h_deg, W, H) -> Optional[Tuple[float, float]]:
    """Pixel (u, v) of the ground point (x_world, y_world, 0).

    Returns None when the point is behind the camera or closer than
    ``CAR_FWD_PAD_M`` in front of it.
    """
    dx = x_world - pano_xy[0]
    dy = y_world - pano_xy[1]
    dz = 0.0 - cam_h
    h = math.radians(heading_deg)
    # world axes: east = +x, north = +y, up = +z
    fwd = (math.sin(h), math.cos(h), 0.0)
    right = (math.cos(h), -math.sin(h), 0.0)
    up = (0.0, 0.0, 1.0)
    X = dx * right[0] + dy * right[1] + dz * right[2]
    Y = dx * up[0] + dy * up[1] + dz * up[2]
    Z = dx * fwd[0] + dy * fwd[1] + dz * fwd[2]
    if Z <= CAR_FWD_PAD_M:
        return None
    fx = (W / 2.0) / math.tan(math.radians(fov_h_deg) / 2.0)
    u = (W / 2.0) + fx * (X / Z)
    v = (H / 2.0) - fx * (Y / Z)
    return u, v


def parcel_band(target_polygon, pano_xy, cam_h, heading_deg, fov_h, W, H):
    """Parcel band (u_min, u_max): horizontal pixel extent of the projected parcel.

    Every exterior vertex in front of the camera is projected. The band runs from the
    smallest to the largest u, each end clamped to the image (u_min to at least 0,
    u_max to at most W). Returns None when no vertex is in front of the camera.
    """
    if isinstance(target_polygon, MultiPolygon):
        exteriors = [list(p.exterior.coords) for p in target_polygon.geoms]
    else:
        exteriors = [list(target_polygon.exterior.coords)]
    us = []
    for ring in exteriors:
        for x, y in ring:
            p = project_world_to_image(x, y, pano_xy, cam_h, heading_deg, fov_h, W, H)
            if p is None:
                continue
            u, _ = p
            us.append(u)
    if not us:
        return None
    return max(0.0, min(us)), min(float(W), max(us))


def target_distance_m(target_polygon, pano_xy) -> float:
    """Distance in metres from the camera to the parcel centroid."""
    c = target_polygon.centroid
    return math.sqrt((c.x - pano_xy[0]) ** 2 + (c.y - pano_xy[1]) ** 2)
