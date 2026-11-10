"""Image groups by K-means (Section 3.2).

Images are clustered on two segmentation ratios, [vegetation_ratio, sky_ratio],
standardised with a StandardScaler. The building ratio is left out of the features
on purpose, because the Building Capture Rate that evaluates the groups is defined
on it. One model is fitted per platform with k = 3. ``models/`` holds the two
models fitted in the study (Naver and Kakao images), and ``assign_cluster``
applies a fitted model to images without refitting.

This module needs numpy only. ``fit_cluster_model`` additionally needs scikit-learn.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .groups import BUILT_UP, GREEN, OPEN_SKY


# ---------------------------------------------------------------------------
# fitted model: load and apply
# ---------------------------------------------------------------------------

def load_cluster_model(path: str | Path) -> Dict:
    """Load a fitted clustering model.

    Required keys: ``scaler_mean``, ``scaler_std``, ``kmeans_centers`` (in scaled
    space) and ``cluster_names`` (cluster index as a string -> image group).
    """
    with open(path, "r", encoding="utf-8") as f:
        model = json.load(f)
    for key in ("scaler_mean", "scaler_std", "kmeans_centers", "cluster_names"):
        if key not in model:
            raise ValueError(f"cluster model missing required key: {key}")
    return model


def assign_cluster(vegetation_ratio: float, sky_ratio: float, model: Dict) -> Tuple[int, str]:
    """Nearest fitted centroid for one image. Returns (cluster index, image group).

    Same arithmetic as ``StandardScaler.transform`` followed by ``KMeans.predict``:
    scale by the stored mean and standard deviation, then take the centroid with the
    smallest Euclidean distance.
    """
    mean = np.asarray(model["scaler_mean"], dtype=float)
    std = np.asarray(model["scaler_std"], dtype=float)
    centers = np.asarray(model["kmeans_centers"], dtype=float)

    features = np.asarray([vegetation_ratio, sky_ratio], dtype=float)
    scaled = (features - mean) / std

    dists = np.linalg.norm(centers - scaled, axis=1)
    cluster_id = int(np.argmin(dists))
    return cluster_id, model["cluster_names"][str(cluster_id)]


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------

def name_clusters(cluster_stats: Dict[int, Dict]) -> Dict[int, str]:
    """Name each cluster after its dominant scene attribute.

    The cluster with the highest mean vegetation ratio is Green. Among the others,
    the one with the highest mean sky ratio is Open-sky. The rest are Built-up.
    ``cluster_stats[c]`` needs ``vegetation_mean`` and ``sky_mean``.
    """
    clusters = list(cluster_stats.keys())
    name_map: Dict[int, str] = {}

    veg_sorted = sorted(clusters, key=lambda c: cluster_stats[c]["vegetation_mean"], reverse=True)
    sky_sorted = sorted(clusters, key=lambda c: cluster_stats[c]["sky_mean"], reverse=True)

    assigned = set()

    green_c = veg_sorted[0]
    name_map[green_c] = GREEN
    assigned.add(green_c)

    for c in sky_sorted:
        if c not in assigned:
            name_map[c] = OPEN_SKY
            assigned.add(c)
            break

    for c in clusters:
        if c not in assigned:
            name_map[c] = BUILT_UP

    return name_map


def fit_cluster_model(records: Sequence[Dict], n_clusters: int = 3, seed: int = 42) -> Tuple[Dict, List[int]]:
    """Fit StandardScaler + K-means on [vegetation_ratio, sky_ratio].

    ``records`` are per-image dicts with ``vegetation_ratio``, ``sky_ratio`` and
    ``building_ratio``. Returns the model as a JSON-serialisable dict (the format
    ``load_cluster_model`` reads) and the cluster index of every record.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    features = np.array([[r["vegetation_ratio"], r["sky_ratio"]] for r in records])

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_labels = kmeans.fit_predict(features_scaled)

    cluster_stats = {}
    for c in range(n_clusters):
        indices = [i for i, label in enumerate(cluster_labels) if label == c]
        cluster_data = [records[i] for i in indices]
        cluster_stats[c] = {
            "count": len(cluster_data),
            "vegetation_mean": float(np.mean([r["vegetation_ratio"] for r in cluster_data])),
            "sky_mean": float(np.mean([r["sky_ratio"] for r in cluster_data])),
            "building_mean": float(np.mean([r["building_ratio"] for r in cluster_data])),
        }

    name_map = name_clusters(cluster_stats)
    model = {
        "n_clusters": n_clusters,
        "cluster_stats": {str(k): v for k, v in cluster_stats.items()},
        "cluster_names": {str(k): v for k, v in name_map.items()},
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_std": scaler.scale_.tolist(),
        "kmeans_centers": kmeans.cluster_centers_.tolist(),
    }
    return model, [int(c) for c in cluster_labels]

