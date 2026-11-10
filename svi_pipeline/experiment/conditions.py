"""Input conditions, folds and seeds of the classification experiment (Section 3.4, Table 3).

    raw            image, all images of the parcel            Raw SVI
    filtered       image, images that passed filtering        + Group-specific filtering
    aligned        image + top-1 house mask (4th channel)     + Target-house alignment
    aligned_shuf   image + mask of another house in the frame Shuffled-mask control
    mask_only      top-1 mask without the image               Mask-only control

``files`` names the split files a condition reads, ``mask_col`` the column with the
mask PNG that becomes the fourth input channel.
"""
SEEDS = [13, 17, 23, 29, 31]   # fold k is trained with SEEDS[k]
FOLDS = range(5)
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
