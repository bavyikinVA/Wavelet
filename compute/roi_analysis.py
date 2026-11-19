"""Masked local 1D extrema and sparse envelope extrema for independent ROIs.

Local peak detection works on each *connected interval of the mask along the
analysed axis*.  Mirror padding extends values from inside that same interval
and therefore never imports CWT samples from outside the ROI.  The padded
samples do not become output coordinates.  Short segments (< 3 pixels) have
no reliable 1D peak and produce no extrema.  Every ROI is analyzed separately,
including pixels that belong to multiple overlapping regions.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from compute.extremes.detection import normalize_peak_parameters, _peak_kwargs
from compute.extremes.interpol import _pchip_envelope_extrema


def _runs(bits):
    """Yield [start, stop) for true intervals; contiguous in ONE analysis line."""
    padded = np.empty(bits.size + 2, dtype=np.int8)
    padded[0] = padded[-1] = 0
    padded[1:-1] = bits
    changes = np.flatnonzero(np.diff(padded))
    return zip(changes[::2], changes[1::2])


def _as_points(points):
    arr = np.asarray(points, dtype=np.int32)
    return arr.reshape(-1, 2) if arr.size else np.empty((0, 2), dtype=np.int32)


def masked_extrema(coefs, mask, *, distance=1, prominence=0.0,
                   row_var=True, col_var=True, max_var=True, min_var=True):
    """Return pmaxr, pmaxc, pminr, pminc for this particular ROI."""
    coefs = np.asarray(coefs)
    mask = np.asarray(mask, dtype=bool)
    if coefs.ndim != 2 or coefs.shape != mask.shape:
        raise ValueError("ROI и коэффициенты CWT должны иметь одинаковую форму (H, W)")
    distance, prominence = normalize_peak_parameters(distance, prominence)
    opts = _peak_kwargs(distance, prominence)
    outputs = [[], [], [], []]
    for direction, active in (("row", row_var), ("col", col_var)):
        if not active:
            continue
        line_count = coefs.shape[0] if direction == "row" else coefs.shape[1]
        for index in range(line_count):
            bits = mask[index, :] if direction == "row" else mask[:, index]
            signal = coefs[index, :] if direction == "row" else coefs[:, index]
            for start, stop in _runs(bits):
                if stop - start < 3:
                    continue
                local = np.asarray(signal[start:stop], dtype=np.float64)
                if not np.isfinite(local).all():
                    raise ValueError("CWT содержит NaN/Inf внутри выделенной ROI")
                # reflect: one neighbour inside ROI at both ends, rather than
                # zeros or samples from the surrounding tissue.
                padded = np.pad(local, (1, 1), mode="reflect")
                for is_max, target in ((True, 0 if direction == "row" else 1),
                                       (False, 2 if direction == "row" else 3)):
                    if (max_var if is_max else min_var):
                        found, _ = find_peaks(padded if is_max else -padded, **opts)
                        for pos in found:
                            pos -= 1
                            if 0 <= pos < len(local):
                                coordinate = int(start + pos)
                                outputs[target].append((coordinate, index) if direction == "row"
                                                       else (index, coordinate))
    return tuple(_as_points(points) for points in outputs)


def masked_envelopes(coefs, mask, maxima, minima, *, direction="row"):
    """Sparse local extrema of PCHIP envelopes, separately for each mask run."""
    coefs = np.asarray(coefs)
    mask = np.asarray(mask, dtype=bool)
    if coefs.ndim != 2 or mask.shape != coefs.shape:
        raise ValueError("ROI и коэффициенты CWT должны иметь одинаковую форму")
    if direction not in ("row", "col"):
        raise ValueError("direction должен быть 'row' или 'col'")
    line_count = coefs.shape[0] if direction == "row" else coefs.shape[1]
    grouped = []
    for points in (maxima, minima):
        group = [[] for _ in range(line_count)]
        for x, y in _as_points(points):
            line = y if direction == "row" else x
            value = x if direction == "row" else y
            if 0 <= line < line_count:
                group[line].append(value)
        grouped.append([np.asarray(values, dtype=np.int32) for values in group])
    results = [[], []]
    for line in range(line_count):
        bits = mask[line, :] if direction == "row" else mask[:, line]
        signal = coefs[line, :] if direction == "row" else coefs[:, line]
        for start, stop in _runs(bits):
            interval = signal[start:stop]
            for k in (0, 1):
                supports = grouped[k][line]
                selected = supports[(supports >= start) & (supports < stop)] - start
                positions = _pchip_envelope_extrema(interval, selected, lower=bool(k))
                for pos in positions:
                    coordinate = int(start + pos)
                    results[k].append((coordinate, line) if direction == "row"
                                      else (line, coordinate))
    return _as_points(results[0]), _as_points(results[1])
