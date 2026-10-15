"""Extrema detection with a vectorized CUDA fast path.

Filtered extrema use the bounded-memory SciPy reference on host arrays.
Calling cupyx.scipy.signal.find_peaks individually on thousands of GPU lines
can stall the pipeline; CPU processing also preserves exact SciPy semantics.
"""
from __future__ import annotations

import numpy as np

from compute.backend_policy import classify_gpu_exception
from compute.extremes.detection import find_extrema_2d, normalize_peak_parameters


class ExtremesFinder:
    @staticmethod
    def find_extremes_gpu(coefs_gpu, row_var=True, col_var=True,
                          max_var=True, min_var=True, *,
                          distance=1, prominence=0.0):
        try:
            import cupy as cp

            distance, prominence = normalize_peak_parameters(distance, prominence)
            input_is_gpu = isinstance(coefs_gpu, cp.ndarray)
            host_reference = None if input_is_gpu else np.asarray(coefs_gpu)

            if distance == 1 and prominence == 0.0:
                device_coefs = (coefs_gpu if input_is_gpu else cp.asarray(host_reference, dtype=cp.float32))
                if device_coefs.dtype != cp.float32:
                    device_coefs = device_coefs.astype(cp.float32, copy=False)
                if device_coefs.ndim != 2:
                    raise ValueError("Поиск экстремумов ожидает двумерную матрицу коэффициентов")
                result = ExtremesFinder._strict_gpu(
                    device_coefs, row_var, col_var, max_var, min_var
                )
                coefs_numpy = cp.asnumpy(device_coefs) if input_is_gpu else host_reference
            else:
                # Use one host transfer at most: SciPy is substantially more
                # predictable than per-line GPU peak-finding for filtered data.
                coefs_numpy = cp.asnumpy(coefs_gpu) if input_is_gpu else host_reference
                if coefs_numpy.ndim != 2:
                    raise ValueError("Поиск экстремумов ожидает двумерную матрицу коэффициентов")
                cpu_points = find_extrema_2d(
                    coefs_numpy, row_var, col_var, max_var, min_var,
                    distance=distance, prominence=prominence,
                )
                result = tuple(
                    np.asarray(points, dtype=np.int32).reshape(-1, 2)
                    for points in cpu_points
                )
            return (coefs_numpy, *result)
        except Exception as exc:
            mapped = classify_gpu_exception(exc)
            if mapped is None:
                raise
            raise mapped from exc

    @staticmethod
    def _to_host_points(x_gpu, y_gpu):
        import cupy as cp

        if x_gpu.size == 0:
            return np.empty((0, 2), dtype=np.int32)
        pts = cp.stack((x_gpu, y_gpu), axis=1).astype(cp.int32, copy=False)
        return cp.asnumpy(pts).astype(np.int32, copy=False)

    @staticmethod
    def _strict_gpu(coefs_gpu, row_var, col_var, max_var, min_var):
        import cupy as cp

        empty = np.empty((0, 2), dtype=np.int32)
        pmaxr = pmaxc = pminr = pminc = empty

        if row_var and (max_var or min_var) and coefs_gpu.shape[1] >= 3:
            left = coefs_gpu[:, :-2]
            center = coefs_gpu[:, 1:-1]
            right = coefs_gpu[:, 2:]
            if max_var:
                yy, xx = cp.where((center > left) & (center > right))
                pmaxr = ExtremesFinder._to_host_points(xx + 1, yy)
            if min_var:
                yy, xx = cp.where((center < left) & (center < right))
                pminr = ExtremesFinder._to_host_points(xx + 1, yy)

        if col_var and (max_var or min_var) and coefs_gpu.shape[0] >= 3:
            up = coefs_gpu[:-2, :]
            center = coefs_gpu[1:-1, :]
            down = coefs_gpu[2:, :]
            if max_var:
                yy, xx = cp.where((center > up) & (center > down))
                pmaxc = ExtremesFinder._to_host_points(xx, yy + 1)
            if min_var:
                yy, xx = cp.where((center < up) & (center < down))
                pminc = ExtremesFinder._to_host_points(xx, yy + 1)

        return pmaxr, pmaxc, pminr, pminc

