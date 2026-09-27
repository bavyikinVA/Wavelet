"""Compact in-memory representation of KNN results.

Production KNN uses dense NumPy arrays with shape ``(N, k)`` instead of one
Python dictionary/list object per point and per neighbor.  Legacy checkpoint
payloads remain readable through :func:`neighbor_arrays`.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np


COMPACT_KNN_FORMAT = "compact_arrays_v1"


def make_compact_neighbors(indices, distances, angles=None) -> dict:
    """Return a validated compact KNN payload.

    ``indices`` is int32; distances and optional angles are float32.  Arrays
    are contiguous so downstream vectorized calculations and serialization do
    not need millions of Python objects.
    """
    idx = np.ascontiguousarray(indices, dtype=np.int32)
    dist = np.ascontiguousarray(distances, dtype=np.float32)
    if idx.ndim != 2 or dist.ndim != 2 or idx.shape != dist.shape:
        raise ValueError("KNN indices/distances must be 2-D arrays of equal shape")

    result = {
        "format": COMPACT_KNN_FORMAT,
        "indices": idx,
        "distances": dist,
    }
    if angles is not None:
        ang = np.ascontiguousarray(angles, dtype=np.float32)
        if ang.shape != idx.shape:
            raise ValueError("KNN angles must have the same shape as indices")
        result["angles"] = ang
    return result


def is_compact_neighbors(neighbors) -> bool:
    return (
        isinstance(neighbors, Mapping)
        and neighbors.get("format") == COMPACT_KNN_FORMAT
        and isinstance(neighbors.get("indices"), np.ndarray)
        and isinstance(neighbors.get("distances"), np.ndarray)
    )


def neighbor_arrays(neighbors, *, point_count: int | None = None):
    """Return ``indices, distances, angles`` arrays for compact or legacy data.

    Legacy format is ``{point_id: {indices: [...], distances: [...], ...}}``.
    It is converted only when old checkpoints/results are read.  New
    production runs never build that object-heavy representation.
    """
    if is_compact_neighbors(neighbors):
        idx = np.asarray(neighbors["indices"], dtype=np.int32)
        dist = np.asarray(neighbors["distances"], dtype=np.float32)
        angles = neighbors.get("angles")
        if angles is not None:
            angles = np.asarray(angles, dtype=np.float32)
        return idx, dist, angles

    if not isinstance(neighbors, Mapping) or not neighbors:
        n = int(point_count or 0)
        empty_i = np.empty((n, 0), dtype=np.int32)
        empty_f = np.empty((n, 0), dtype=np.float32)
        return empty_i, empty_f, None

    numeric_keys = []
    for key in neighbors:
        try:
            numeric_keys.append(int(key))
        except (TypeError, ValueError):
            continue
    inferred_n = max(numeric_keys, default=-1) + 1
    n = max(int(point_count or 0), inferred_n)

    k = 0
    has_angles = False
    for key in numeric_keys:
        item = neighbors.get(key, neighbors.get(str(key), {}))
        k = max(k, len(item.get("indices", [])))
        has_angles = has_angles or "angles" in item

    indices = np.full((n, k), -1, dtype=np.int32)
    distances = np.full((n, k), np.nan, dtype=np.float32)
    angles = np.full((n, k), np.nan, dtype=np.float32) if has_angles else None

    for key in numeric_keys:
        item = neighbors.get(key, neighbors.get(str(key), {}))
        row_idx = np.asarray(item.get("indices", []), dtype=np.int32)
        row_dist = np.asarray(item.get("distances", []), dtype=np.float32)
        width = min(k, len(row_idx), len(row_dist))
        if width:
            indices[key, :width] = row_idx[:width]
            distances[key, :width] = row_dist[:width]
        if angles is not None:
            row_angles = np.asarray(item.get("angles", []), dtype=np.float32)
            awidth = min(k, len(row_angles))
            if awidth:
                angles[key, :awidth] = row_angles[:awidth]

    return indices, distances, angles
