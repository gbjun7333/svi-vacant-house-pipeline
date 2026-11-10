"""Input conditions, folds and seeds of the classification experiment (Section 3.4, Table 3)."""
SEEDS = [13, 17, 23, 29, 31]   # fold k uses SEEDS[k]
FOLDS = range(5)
# files = split files the condition reads, mask_col = column with the mask PNG (fourth input channel)
CONDS = {
    "raw": {"files": "raw", "mask_col": None, "mask_only": False},
    "filtered": {"files": "filtered", "mask_col": None, "mask_only": False},
    "aligned": {"files": "aligned", "mask_col": "mask_path", "mask_only": False},
    "aligned_shuf": {"files": "aligned", "mask_col": "mask_shuf_path", "mask_only": False},
    "mask_only": {"files": "aligned", "mask_col": "mask_path", "mask_only": True},
}
ORDER = ["raw", "filtered", "aligned", "aligned_shuf", "mask_only"]
CONDITION_LABELS = {
    "raw": "Raw SVI",
    "filtered": "+ Group-specific filtering",
    "aligned": "+ Target-house alignment",
    "aligned_shuf": "Shuffled-mask control",
    "mask_only": "Mask-only control",
}
