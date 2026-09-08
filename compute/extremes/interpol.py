import numpy as np
from scipy.signal import find_peaks


def interpolate_envelope(signal, support):
    """Linear interpolation with constant tails; None means no support.

    One support point defines a constant line with no peaks.
    This interpolant is not a guaranteed upper/lower bound on the signal.
    """
    support = np.unique(np.asarray(support, dtype=int))
    if support.size == 0:
        return None
    return np.interp(np.arange(len(signal)), support, np.asarray(signal)[support])


def get_row_envelopes(coefs, max_points, min_points):
    """Return maxima/minima of interpolated envelopes in (x, y) coordinates."""
    upper_max_points = []
    lower_min_points = []
    for row, row_data in enumerate(coefs):
        upper = interpolate_envelope(row_data, [x for x, y in max_points if y == row])
        lower = interpolate_envelope(row_data, [x for x, y in min_points if y == row])
        if upper is not None:
            indices, _ = find_peaks(upper)
            upper_max_points.extend((x, row) for x in indices)
        if lower is not None:
            indices, _ = find_peaks(-lower)
            lower_min_points.extend((x, row) for x in indices)
    return upper_max_points, lower_min_points


def get_column_envelopes(coefs, max_points, min_points):
    """Apply the row policy to transposed data, preserving (x, y) coordinates."""
    upper, lower = get_row_envelopes(
        np.asarray(coefs).T,
        [(y, x) for x, y in max_points],
        [(y, x) for x, y in min_points],
    )
    return ([(y, x) for x, y in upper], [(y, x) for x, y in lower])
