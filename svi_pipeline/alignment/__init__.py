"""Target-house alignment (Section 3.3).

masks       instance-mask files (.npz)
sam3        SAM 3 building instance masks and the exclusions of Table C.2
geometry    projection of the parcel into the image, parcel band
scoring     center alignment, overlap, area consistency, combined score
selection   top-1 mask by overlap with the parcel band
"""
