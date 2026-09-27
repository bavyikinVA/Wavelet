"""PCHIP envelopes over previously detected extrema.

The public dense interpolator is useful for scientific visualisation.  The
production point pipeline needs only local extrema of those envelopes; for that
case we exploit PCHIP's shape-preserving monotonicity and do not rasterize a
full envelope for every line.  This matters for 2500x2000 coefficient maps.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator


def _clean_support(support, length: int) -> np.ndarray:
    support = np.asarray(support, dtype=np.intp).reshape(-1)
    if support.size == 0:
        return np.empty(0, dtype=np.intp)
    support = support[(support >= 0) & (support < int(length))]
    if support.size == 0:
        return np.empty(0, dtype=np.intp)
    return np.unique(support)


def interpolate_envelope(signal, support):
    """Return a dense PCHIP envelope with constant boundary tails.

    The PCHIP curve passes through all supplied extrema and is shape-preserving
    between them.  Outside the first/last support point the nearest support
    value is held constant; endpoints are therefore *not* invented extrema.

    ``None`` means there is no support.  One support point defines a constant
    line.  As with ordinary extrema interpolation, this is not a mathematical
    guarantee that an upper curve bounds every signal sample from above (or a
    lower curve from below).
    """
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError("Огибающая строится для одномерной линии")
    support = _clean_support(support, signal.size)
    if support.size == 0:
        return None

    values = np.asarray(signal[support], dtype=np.float64)
    if support.size == 1:
        return np.full(signal.size, values[0], dtype=np.float64)

    result = np.empty(signal.size, dtype=np.float64)
    first = int(support[0])
    last = int(support[-1])
    result[:first] = values[0]
    result[last + 1:] = values[-1]

    interpolator = PchipInterpolator(
        support.astype(np.float64, copy=False),
        values,
        extrapolate=False,
    )
    grid = np.arange(first, last + 1, dtype=np.float64)
    result[first:last + 1] = interpolator(grid)
    # Enforce exact knot values after floating-point polynomial evaluation.
    # Besides being mathematically faithful to interpolation, this avoids
    # 1-ulp pseudo-peaks where a PCHIP knot meets a constant boundary tail.
    result[support] = values
    return result


def _group_points(points, line_count: int, direction: str):
    """Group (x, y) extrema once instead of rescanning all points per line."""
    groups = [[] for _ in range(int(line_count))]
    if direction == "row":
        for x, y in points:
            line = int(y)
            if 0 <= line < line_count:
                groups[line].append(int(x))
    elif direction == "col":
        for x, y in points:
            line = int(x)
            if 0 <= line < line_count:
                groups[line].append(int(y))
    else:
        raise ValueError("direction должен быть 'row' или 'col'")

    return [
        np.unique(np.asarray(group, dtype=np.intp))
        if group else np.empty(0, dtype=np.intp)
        for group in groups
    ]


def _pchip_envelope_extrema(signal, support, *, lower: bool = False):
    """Return extrema of a PCHIP envelope without dense rasterisation.

    PCHIP is monotone on every interval between consecutive support nodes when
    the node values are monotone, and it does not overshoot.  Therefore local
    extrema of the envelope can occur only at support knots or on flat runs of
    equal-valued support knots.  For a flat run we return the same midpoint
    coordinate that dense ``scipy.signal.find_peaks`` would select.

    This is mathematically equivalent to building the dense PCHIP and then
    searching its extrema, while reducing work from O(line_length) interpolation
    per line to O(number_of_support_extrema).
    """
    signal = np.asarray(signal)
    support = _clean_support(support, signal.size)
    if support.size < 3:
        # Constant tails mean neither boundary support can be a strict interior
        # envelope extremum.  With fewer than three support runs none exists.
        return np.empty(0, dtype=np.intp)

    values = np.asarray(signal[support], dtype=np.float64)

    # Compress exact-equality plateaus.  Exact equality is intentional: only an
    # exactly flat PCHIP interval becomes a dense plateau.
    run_starts = [0]
    for i in range(1, values.size):
        if values[i] != values[i - 1]:
            run_starts.append(i)
    run_starts = np.asarray(run_starts, dtype=np.intp)
    run_ends = np.empty_like(run_starts)
    run_ends[:-1] = run_starts[1:] - 1
    run_ends[-1] = values.size - 1

    if run_starts.size < 3:
        return np.empty(0, dtype=np.intp)

    run_values = values[run_starts]
    if lower:
        mask = ((run_values[1:-1] < run_values[:-2]) &
                (run_values[1:-1] < run_values[2:]))
    else:
        mask = ((run_values[1:-1] > run_values[:-2]) &
                (run_values[1:-1] > run_values[2:]))

    selected_runs = np.flatnonzero(mask) + 1
    if selected_runs.size == 0:
        return np.empty(0, dtype=np.intp)

    result = np.empty(selected_runs.size, dtype=np.intp)
    for out_idx, run_idx in enumerate(selected_runs):
        start_x = int(support[run_starts[run_idx]])
        end_x = int(support[run_ends[run_idx]])
        result[out_idx] = (start_x + end_x) // 2
    return result


def get_row_envelopes(coefs, max_points, min_points):
    """PCHIP upper/lower envelope extrema in (x, y) coordinates.

    Extrema are grouped by row exactly once.  The dense PCHIP curve remains
    available through :func:`interpolate_envelope`, but the point pipeline uses
    the equivalent sparse extraction above to stay scalable at 2500x2000.
    """
    coefs = np.asarray(coefs)
    if coefs.ndim != 2:
        raise ValueError("Огибающие ожидают двумерную матрицу коэффициентов")

    rows = coefs.shape[0]
    max_groups = _group_points(max_points, rows, "row")
    min_groups = _group_points(min_points, rows, "row")

    upper_max_points = []
    lower_min_points = []
    for row, row_data in enumerate(coefs):
        upper = _pchip_envelope_extrema(row_data, max_groups[row], lower=False)
        lower = _pchip_envelope_extrema(row_data, min_groups[row], lower=True)
        upper_max_points.extend((int(x), row) for x in upper)
        lower_min_points.extend((int(x), row) for x in lower)
    return upper_max_points, lower_min_points


def get_column_envelopes(coefs, max_points, min_points):
    """Column analogue of :func:`get_row_envelopes`, preserving (x, y)."""
    coefs = np.asarray(coefs)
    if coefs.ndim != 2:
        raise ValueError("Огибающие ожидают двумерную матрицу коэффициентов")

    cols = coefs.shape[1]
    max_groups = _group_points(max_points, cols, "col")
    min_groups = _group_points(min_points, cols, "col")

    upper_max_points = []
    lower_min_points = []
    for col in range(cols):
        signal = coefs[:, col]
        upper = _pchip_envelope_extrema(signal, max_groups[col], lower=False)
        lower = _pchip_envelope_extrema(signal, min_groups[col], lower=True)
        upper_max_points.extend((col, int(y)) for y in upper)
        lower_min_points.extend((col, int(y)) for y in lower)
    return upper_max_points, lower_min_points
