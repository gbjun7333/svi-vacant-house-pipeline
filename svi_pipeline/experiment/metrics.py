"""Evaluation measures of the classification experiment (Section 3.4).

Images are scored by the model, parcels by the mean probability of their images.
F1 uses a threshold tuned on the validation images of the fold: the threshold with
the highest image-level F1 among the validation probabilities (the smallest one when
several tie). The same threshold is applied to images and parcels of the test split.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUROC as the share of (vacant, non-vacant) pairs ranked correctly, ties counting half."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return math.nan
    wins = 0.0
    total = len(pos) * len(neg)
    for p in pos:
        wins += float((p > neg).sum())
        wins += 0.5 * float((p == neg).sum())
    return float(wins / total)


def metrics_at(labels: np.ndarray, probs: np.ndarray, thr: float) -> dict:
    """Counts, accuracy, precision, recall and F1 at a threshold (prob >= thr is vacant), and AUROC."""
    preds = (probs >= thr).astype(np.int64)
    labels = labels.astype(np.int64)
    tp = int(((preds == 1) & (labels == 1)).sum()); tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum()); fn = int(((preds == 0) & (labels == 1)).sum())
    prec = safe_div(tp, tp + fp); rec = safe_div(tp, tp + fn)
    return {"count": int(len(labels)), "accuracy": safe_div(tp + tn, len(labels)), "precision": prec, "recall": rec,
            "f1": safe_div(2 * prec * rec, prec + rec), "auroc": compute_auc(labels, probs),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def tune_threshold(labels: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    """(threshold, F1) with the highest F1 over the distinct probabilities, ties to the smallest."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.unique(probs):
        f1 = metrics_at(labels, probs, float(t))["f1"]
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def parcel_level(parcel_keys, labels, probs) -> pd.DataFrame:
    """One row per parcel: label, mean probability of its images, number of images."""
    df = pd.DataFrame({"parcel_key": list(parcel_keys), "label_id": [int(v) for v in labels], "prob": probs})
    return df.groupby("parcel_key").agg(label_id=("label_id", "first"), prob=("prob", "mean"), n_views=("prob", "size"))
