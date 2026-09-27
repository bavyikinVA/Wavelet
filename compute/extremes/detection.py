"""Local extrema detection with optional distance/prominence filtering.

The project historically used a strict 3-sample rule::

    x[i] > x[i-1] and x[i] > x[i+1]

(and the analogous rule for minima).  ``scipy.signal.find_peaks`` is used when
``distance`` or ``prominence`` filtering is requested, while
``plateau_size=(None, 1)`` preserves that strict no-plateau policy.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy.signal import find_peaks


def normalize_peak_parameters(distance=1, prominence=0.0) -> tuple[int, float]:
    """Validate and normalize user-facing peak parameters.

    ``distance`` is measured in samples/pixels along the analysed 1-D line.
    ``prominence`` is measured in the same coefficient units as the signal.
    Values ``1`` and ``0`` respectively reproduce the historical strict-local
    extrema rule without extra filtering.
    """
    try:
        distance_value = int(distance)
    except (TypeError, ValueError) as exc:
        raise ValueError("distance должен быть целым числом >= 1") from exc
    if distance_value < 1:
        raise ValueError("distance должен быть >= 1")

    try:
        prominence_value = float(prominence)
    except (TypeError, ValueError) as exc:
        raise ValueError("prominence должен быть числом >= 0") from exc
    if not math.isfinite(prominence_value) or prominence_value < 0.0:
        raise ValueError("prominence должен быть конечным числом >= 0")
    return distance_value, prominence_value


def _peak_kwargs(distance: int, prominence: float) -> dict:
    # plateau_size <= 1 keeps the old strict rule: flat tops are not extrema.
    kwargs = {"plateau_size": (None, 1)}
    if distance > 1:
        kwargs["distance"] = distance
    if prominence > 0.0:
        kwargs["prominence"] = prominence
    return kwargs


def _strict_vectorized(coefs: np.ndarray, row_var: bool, col_var: bool,
                       max_var: bool, min_var: bool):
    """Fast legacy-equivalent path for distance=1/prominence=0."""
    points_max_by_row = []
    points_min_by_row = []
    points_max_by_column = []
    points_min_by_column = []

    if row_var and (max_var or min_var) and coefs.shape[1] >= 3:
        left = coefs[:, :-2]
        center = coefs[:, 1:-1]
        right = coefs[:, 2:]
        if max_var:
            yy, xx = np.where((center > left) & (center > right))
            points_max_by_row = [[int(x + 1), int(y)] for y, x in zip(yy, xx)]
        if min_var:
            yy, xx = np.where((center < left) & (center < right))
            points_min_by_row = [[int(x + 1), int(y)] for y, x in zip(yy, xx)]

    if col_var and (max_var or min_var) and coefs.shape[0] >= 3:
        up = coefs[:-2, :]
        center = coefs[1:-1, :]
        down = coefs[2:, :]
        if max_var:
            yy, xx = np.where((center > up) & (center > down))
            points_max_by_column = [[int(x), int(y + 1)] for y, x in zip(yy, xx)]
        if min_var:
            yy, xx = np.where((center < up) & (center < down))
            points_min_by_column = [[int(x), int(y + 1)] for y, x in zip(yy, xx)]

    return (
        points_max_by_row,
        points_max_by_column,
        points_min_by_row,
        points_min_by_column,
    )


def find_extrema_2d(coefs, row_var=True, col_var=True, max_var=True,
                     min_var=True, *, distance=1, prominence=0.0):
    """Find strict local extrema along rows and/or columns.

    The filtered path intentionally processes one 1-D view at a time.  This
    keeps peak memory bounded for production images up to 2500x2000: no full
    row/column copies or dense peak-property matrices are created.
    """
    matrix = np.asarray(coefs)
    if matrix.ndim != 2:
        raise ValueError("Поиск экстремумов ожидает двумерную матрицу коэффициентов")

    distance, prominence = normalize_peak_parameters(distance, prominence)
    if distance == 1 and prominence == 0.0:
        return _strict_vectorized(matrix, row_var, col_var, max_var, min_var)

    kwargs = _peak_kwargs(distance, prominence)
    points_max_by_row = []
    points_min_by_row = []
    points_max_by_column = []
    points_min_by_column = []

    if row_var and (max_var or min_var):
        for y, signal in enumerate(matrix):
            if max_var:
                peaks, _ = find_peaks(signal, **kwargs)
                points_max_by_row.extend([int(x), int(y)] for x in peaks)
            if min_var:
                peaks, _ = find_peaks(-signal, **kwargs)
                points_min_by_row.extend([int(x), int(y)] for x in peaks)

    if col_var and (max_var or min_var):
        # ``matrix[:, x]`` is a view.  SciPy may make a small contiguous 1-D
        # work buffer internally, but we never materialize a 2500x2000 transpose.
        for x in range(matrix.shape[1]):
            signal = matrix[:, x]
            if max_var:
                peaks, _ = find_peaks(signal, **kwargs)
                points_max_by_column.extend([int(x), int(y)] for y in peaks)
            if min_var:
                peaks, _ = find_peaks(-signal, **kwargs)
                points_min_by_column.extend([int(x), int(y)] for y in peaks)

    return (
        points_max_by_row,
        points_max_by_column,
        points_min_by_row,
        points_min_by_column,
    )
