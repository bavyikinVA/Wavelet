"""Unified numerical policy for Wavelets Analysis Studio."""
from __future__ import annotations

import numpy as np

COMPUTE_DTYPE = np.float32
COMPUTE_DTYPE_NAME = "float32"

# Initial scientific tolerances. They are intentionally explicit and tested.
CWT_RTOL = 5e-5
CWT_ATOL = 5e-6
ENVELOPE_RTOL = 1e-4
ENVELOPE_ATOL = 1e-6
EXTREMA_POSITION_TOLERANCE_PX = 1
EXTREMA_MATCH_RATIO_MIN = 0.999
KNN_DISTANCE_RTOL = 1e-4
KNN_DISTANCE_ATOL = 1e-5
KNN_NEIGHBOR_MATCH_RATIO_MIN = 0.99


def as_compute_array(value, *, copy: bool = False) -> np.ndarray:
    return np.array(value, dtype=COMPUTE_DTYPE, copy=copy)


def as_compute_scales(scales) -> np.ndarray:
    return np.asarray(scales, dtype=COMPUTE_DTYPE)


def dtype_name(value) -> str:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        dtype = np.asarray(value).dtype
    return str(dtype)
