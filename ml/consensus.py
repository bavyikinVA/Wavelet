"""Spatial fusion of DBSCAN anomaly layers from different wavelet scales."""
from __future__ import annotations

import numpy as np


def build_anomaly_consensus(layers, *, shape, grid_size=16, anomaly_threshold=0.75):
    """Fuse several spatial DBSCAN layers into one stability map.

    Each input layer is a mapping with ``points`` (N,2) and ``scores`` (N,).
    For every grid cell and every layer we compute the fraction of points whose
    anomaly score is >= ``anomaly_threshold``. The final stability value is the
    mean across *all selected layers*; a layer with no points in a cell
    contributes zero. Thus a high result requires the same area to remain
    anomalous across scales, rather than merely being anomalous once.
    """
    if not layers:
        raise ValueError("Для карты устойчивости нужны хотя бы два DBSCAN-слоя")
    if len(layers) < 2:
        raise ValueError("Для карты устойчивости выберите минимум два DBSCAN-слоя")
    h, w = map(int, shape[:2])
    grid_size = max(2, int(grid_size))
    threshold = float(anomaly_threshold)
    rows = int(np.ceil(h / grid_size))
    cols = int(np.ceil(w / grid_size))
    cells = rows * cols
    aggregate = np.zeros(cells, dtype=np.float32)
    coverage = np.zeros(cells, dtype=np.int16)

    for layer in layers:
        points = np.asarray(layer["points"], dtype=float)
        scores = np.asarray(layer["scores"], dtype=float)
        if len(points) != len(scores):
            raise ValueError("Число координат и anomaly score в слое не совпадает")
        if len(points) == 0:
            continue
        cx = np.clip((points[:, 0] // grid_size).astype(int), 0, cols - 1)
        cy = np.clip((points[:, 1] // grid_size).astype(int), 0, rows - 1)
        flat = cy * cols + cx
        count = np.zeros(cells, dtype=np.int32)
        anomalous = np.zeros(cells, dtype=np.int32)
        np.add.at(count, flat, 1)
        np.add.at(anomalous, flat, scores >= threshold)
        present = count > 0
        ratios = np.zeros(cells, dtype=np.float32)
        ratios[present] = anomalous[present] / count[present]
        aggregate += ratios
        coverage += present.astype(np.int16)

    stability = aggregate / float(len(layers))
    occupied = coverage > 0
    ids = np.flatnonzero(occupied)
    if not len(ids):
        return {
            "points": np.empty((0, 2), dtype=float),
            "scores": np.empty(0, dtype=np.float32),
            "coverage": np.empty(0, dtype=np.float32),
            "grid_size": grid_size,
            "threshold": threshold,
        }
    gy, gx = np.divmod(ids, cols)
    points = np.column_stack(((gx + .5) * grid_size, (gy + .5) * grid_size)).astype(float)
    points[:, 0] = np.minimum(points[:, 0], w - 1)
    points[:, 1] = np.minimum(points[:, 1], h - 1)
    return {
        "points": points,
        "scores": stability[occupied],
        "coverage": coverage[occupied].astype(np.float32) / float(len(layers)),
        "grid_size": grid_size,
        "threshold": threshold,
    }
