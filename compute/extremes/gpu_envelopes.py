"""Vectorized sparse GPU extraction of the project's PCHIP-envelope extrema.

The production point pipeline never needs to rasterize a dense PCHIP curve.
For shape-preserving PCHIP, local extrema of the interpolant can occur only at
support knots (or at the midpoint of an exactly flat run of support knots).

Older GPU code processed every row/column separately and called ``cp.asnumpy``
for every line.  That produced thousands of tiny launches and host/device
synchronizations.  This implementation performs grouping, plateau compression,
local-extrema selection and coordinate reconstruction as a few vectorized CuPy
operations for the whole 2-D map, then transfers the final compact Nx2 result
once.
"""
from __future__ import annotations

import numpy as np

from compute.backend_policy import classify_gpu_exception


class GPUEnvelopeProcessor:
    def __init__(self, max_gpu_memory_mb=6000):
        self.max_gpu_memory = int(max_gpu_memory_mb) * 1024 * 1024
        self._cached_host_id = None
        self._cached_shape = None
        self._cached_gpu = None

    def _coefs_gpu(self, coefs):
        import cupy as cp

        if isinstance(coefs, cp.ndarray):
            arr = coefs
            if arr.dtype != cp.float32:
                arr = arr.astype(cp.float32, copy=False)
            return arr

        host = np.asarray(coefs)
        cache_key = id(coefs)
        if (
            self._cached_gpu is not None
            and self._cached_host_id == cache_key
            and self._cached_shape == tuple(host.shape)
        ):
            return self._cached_gpu

        arr = cp.asarray(host, dtype=cp.float32)
        self._cached_host_id = cache_key
        self._cached_shape = tuple(host.shape)
        self._cached_gpu = arr
        return arr

    @staticmethod
    def _normalize_points(points):
        arr = np.asarray(points, dtype=np.int32)
        if arr.size == 0:
            return np.empty((0, 2), dtype=np.int32)
        return np.ascontiguousarray(arr.reshape(-1, 2), dtype=np.int32)

    @staticmethod
    def _sparse_points_gpu(coefs_gpu, points, direction: str, *, lower: bool):
        import cupy as cp

        host_points = GPUEnvelopeProcessor._normalize_points(points)
        if len(host_points) < 3:
            return np.empty((0, 2), dtype=np.int32)

        pts = cp.asarray(host_points, dtype=cp.int32)
        if direction == "row":
            line = pts[:, 1]
            coord = pts[:, 0]
            line_limit = int(coefs_gpu.shape[0])
            coord_limit = int(coefs_gpu.shape[1])
        elif direction == "col":
            line = pts[:, 0]
            coord = pts[:, 1]
            line_limit = int(coefs_gpu.shape[1])
            coord_limit = int(coefs_gpu.shape[0])
        else:
            raise ValueError("direction должен быть 'row' или 'col'")

        valid = (
            (line >= 0) & (line < line_limit)
            & (coord >= 0) & (coord < coord_limit)
        )
        pts = pts[valid]
        if pts.size == 0:
            return np.empty((0, 2), dtype=np.int32)

        if direction == "row":
            line = pts[:, 1]
            coord = pts[:, 0]
        else:
            line = pts[:, 0]
            coord = pts[:, 1]

        # One scalar key is enough because coordinates are non-negative image
        # indices and coord < coord_limit.  Sorting by this key groups each line
        # and orders its support positions in one GPU sort.
        key = line.astype(cp.int64) * np.int64(coord_limit) + coord.astype(cp.int64)
        order = cp.argsort(key)
        line = line[order]
        coord = coord[order]

        # Remove accidental duplicate support coordinates.  CPU _clean_support
        # uses np.unique, so this preserves exact parity.
        unique_mask = cp.empty(line.size, dtype=cp.bool_)
        unique_mask[0] = True
        if line.size > 1:
            unique_mask[1:] = (line[1:] != line[:-1]) | (coord[1:] != coord[:-1])
        line = line[unique_mask]
        coord = coord[unique_mask]
        if line.size < 3:
            return np.empty((0, 2), dtype=np.int32)

        if direction == "row":
            values = coefs_gpu[line, coord]
        else:
            values = coefs_gpu[coord, line]

        # A new run starts when either the line changes or the exact support
        # value changes.  Exact equality intentionally mirrors the CPU rule:
        # only an exactly flat PCHIP interval is a plateau.
        run_start_mask = cp.empty(line.size, dtype=cp.bool_)
        run_start_mask[0] = True
        run_start_mask[1:] = (
            (line[1:] != line[:-1]) | (values[1:] != values[:-1])
        )
        starts = cp.flatnonzero(run_start_mask).astype(cp.int32, copy=False)
        if starts.size < 3:
            return np.empty((0, 2), dtype=np.int32)

        ends = cp.empty_like(starts)
        ends[:-1] = starts[1:] - 1
        ends[-1] = line.size - 1

        run_line = line[starts]
        run_values = values[starts]
        if run_line.size < 3:
            return np.empty((0, 2), dtype=np.int32)

        same_line = (
            (run_line[1:-1] == run_line[:-2])
            & (run_line[1:-1] == run_line[2:])
        )
        if lower:
            extrema_mask = (
                same_line
                & (run_values[1:-1] < run_values[:-2])
                & (run_values[1:-1] < run_values[2:])
            )
        else:
            extrema_mask = (
                same_line
                & (run_values[1:-1] > run_values[:-2])
                & (run_values[1:-1] > run_values[2:])
            )

        selected = cp.flatnonzero(extrema_mask).astype(cp.int32, copy=False) + 1
        if selected.size == 0:
            return np.empty((0, 2), dtype=np.int32)

        start_coord = coord[starts[selected]]
        end_coord = coord[ends[selected]]
        midpoint = (start_coord + end_coord) // 2
        selected_line = run_line[selected]

        if direction == "row":
            out = cp.stack((midpoint, selected_line), axis=1)
        else:
            out = cp.stack((selected_line, midpoint), axis=1)

        return cp.asnumpy(out).astype(np.int32, copy=False)

    def _get(self, coefs, max_points, min_points, direction):
        try:
            coefs_gpu = self._coefs_gpu(coefs)
            if coefs_gpu.ndim != 2:
                raise ValueError("Огибающие ожидают двумерную матрицу коэффициентов")

            upper = self._sparse_points_gpu(
                coefs_gpu, max_points, direction, lower=False
            )
            lower = self._sparse_points_gpu(
                coefs_gpu, min_points, direction, lower=True
            )
            return upper, lower
        except Exception as exc:
            mapped = classify_gpu_exception(exc)
            if mapped is None:
                raise
            raise mapped from exc

    def get_row_envelopes_gpu(self, coefs, max_points, min_points):
        return self._get(coefs, max_points, min_points, "row")

    def get_col_envelopes_gpu(self, coefs, max_points, min_points):
        return self._get(coefs, max_points, min_points, "col")
