"""Sparse GPU extraction of extrema of the project's PCHIP envelopes.

The production pipeline does not rasterize the dense PCHIP curve.  Because
PCHIP is shape-preserving and monotone between monotone support values, local
extrema of the interpolant occur only at support knots or exact-value support
plateaus.  The GPU implementation therefore applies the same sparse rule as
the CPU implementation directly to support coordinates and values.
"""
from __future__ import annotations

import numpy as np

from compute.backend_policy import classify_gpu_exception


def _group_points(points, line_count: int, direction: str):
    groups = [[] for _ in range(int(line_count))]
    arr = np.asarray(points, dtype=np.int32).reshape(-1, 2) if len(points) else np.empty((0, 2), np.int32)
    coord_idx, line_idx = (0, 1) if direction == "row" else (1, 0)
    for point in arr:
        line = int(point[line_idx])
        if 0 <= line < line_count:
            groups[line].append(int(point[coord_idx]))
    return [np.unique(np.asarray(g, dtype=np.int32)) if g else np.empty(0, np.int32) for g in groups]


class GPUEnvelopeProcessor:
    def __init__(self, max_gpu_memory_mb=6000):
        self.max_gpu_memory = int(max_gpu_memory_mb) * 1024 * 1024

    @staticmethod
    def _sparse_line_gpu(signal_gpu, support, *, lower=False):
        import cupy as cp

        support = np.asarray(support, dtype=np.int32)
        if support.size < 3:
            return np.empty(0, dtype=np.int32)
        support_gpu = cp.asarray(support)
        values = signal_gpu[support_gpu]

        # Start a run whenever the exact support value changes. Exact equality
        # matches the CPU rule and dense PCHIP plateau semantics.
        changes = cp.empty(values.size, dtype=cp.bool_)
        changes[0] = True
        changes[1:] = values[1:] != values[:-1]
        starts = cp.flatnonzero(changes)
        if starts.size < 3:
            return np.empty(0, dtype=np.int32)
        ends = cp.empty_like(starts)
        ends[:-1] = starts[1:] - 1
        ends[-1] = values.size - 1
        run_values = values[starts]

        if lower:
            mask = (run_values[1:-1] < run_values[:-2]) & (run_values[1:-1] < run_values[2:])
        else:
            mask = (run_values[1:-1] > run_values[:-2]) & (run_values[1:-1] > run_values[2:])
        selected = cp.flatnonzero(mask) + 1
        if selected.size == 0:
            return np.empty(0, dtype=np.int32)

        start_x = support_gpu[starts[selected]]
        end_x = support_gpu[ends[selected]]
        return cp.asnumpy((start_x + end_x) // 2).astype(np.int32, copy=False)

    def _get(self, coefs, max_points, min_points, direction):
        try:
            import cupy as cp
            coefs_gpu = coefs if isinstance(coefs, cp.ndarray) else cp.asarray(coefs, dtype=cp.float32)
            if coefs_gpu.dtype != cp.float32:
                coefs_gpu = coefs_gpu.astype(cp.float32, copy=False)
            if coefs_gpu.ndim != 2:
                raise ValueError("Огибающие ожидают двумерную матрицу коэффициентов")

            line_count = coefs_gpu.shape[0] if direction == "row" else coefs_gpu.shape[1]
            max_groups = _group_points(max_points, line_count, direction)
            min_groups = _group_points(min_points, line_count, direction)
            upper, lower = [], []
            for line in range(line_count):
                signal = coefs_gpu[line] if direction == "row" else coefs_gpu[:, line]
                up = self._sparse_line_gpu(signal, max_groups[line], lower=False)
                lo = self._sparse_line_gpu(signal, min_groups[line], lower=True)
                if direction == "row":
                    upper.extend((int(x), line) for x in up)
                    lower.extend((int(x), line) for x in lo)
                else:
                    upper.extend((line, int(y)) for y in up)
                    lower.extend((line, int(y)) for y in lo)
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
