"""Task-suitability filtering (Section 3.2).

groups    image groups, administrative divisions, operating groups
segment   SegFormer-B2 class ratios of one image (needs torch and transformers)
cluster   image groups by K-means
filters   filters B to F and the filter configuration of each operating group
bcr       Building Capture Rate
models/   the K-means models fitted in the study, one per platform
"""
