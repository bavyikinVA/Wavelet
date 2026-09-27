"""GPU implementation of the project's 1-D extrema semantics.

The GPU path intentionally mirrors :mod:`compute.extremes.detection`:
strict local extrema for the default policy and ``find_peaks`` with the same
``distance``, ``prominence`` and no-plateau policy for filtered detection.
"""
from compute.backend_policy import classify_gpu_exception
from compute.extremes.detection import normalize_peak_parameters


class ExtremesFinder:
    @staticmethod
    def find_extremes_gpu(coefs_gpu, row_var=True, col_var=True,
                          max_var=True, min_var=True, *,
                          distance=1, prominence=0.0):
        """Find extrema on GPU with the same scientific contract as CPU.

        Returns ``(coefs_numpy, max_row, max_col, min_row, min_col)`` for
        compatibility with the application pipeline.  Peak detection itself is
        executed on the GPU; the final coordinates are transferred to host
        memory because downstream caches/exports use host ``int32`` arrays.
        """
        try:
            import cupy as cp
            from cupyx.scipy.signal import find_peaks

            distance, prominence = normalize_peak_parameters(distance, prominence)
            if not isinstance(coefs_gpu, cp.ndarray):
                coefs_gpu = cp.asarray(coefs_gpu, dtype=cp.float32)
            elif coefs_gpu.dtype != cp.float32:
                coefs_gpu = coefs_gpu.astype(cp.float32, copy=False)

            if coefs_gpu.ndim != 2:
                raise ValueError("Поиск экстремумов ожидает двумерную матрицу коэффициентов")

            if distance == 1 and prominence == 0.0:
                result = ExtremesFinder._strict_gpu(
                    coefs_gpu, row_var, col_var, max_var, min_var
                )
            else:
                kwargs = {"plateau_size": (None, 1)}
                if distance > 1:
                    kwargs["distance"] = distance
                if prominence > 0.0:
                    kwargs["prominence"] = prominence
                result = ExtremesFinder._filtered_gpu(
                    coefs_gpu, row_var, col_var, max_var, min_var,
                    find_peaks=find_peaks, kwargs=kwargs,
                )

            return (cp.asnumpy(coefs_gpu), *result)
        except Exception as exc:
            mapped = classify_gpu_exception(exc)
            if mapped is None:
                raise
            raise mapped from exc

    @staticmethod
    def _strict_gpu(coefs_gpu, row_var, col_var, max_var, min_var):
        import cupy as cp

        pmaxr, pmaxc, pminr, pminc = [], [], [], []
        if row_var and (max_var or min_var) and coefs_gpu.shape[1] >= 3:
            left, center, right = coefs_gpu[:, :-2], coefs_gpu[:, 1:-1], coefs_gpu[:, 2:]
            if max_var:
                yy, xx = cp.where((center > left) & (center > right))
                pmaxr = [[int(x + 1), int(y)] for y, x in zip(cp.asnumpy(yy), cp.asnumpy(xx))]
            if min_var:
                yy, xx = cp.where((center < left) & (center < right))
                pminr = [[int(x + 1), int(y)] for y, x in zip(cp.asnumpy(yy), cp.asnumpy(xx))]

        if col_var and (max_var or min_var) and coefs_gpu.shape[0] >= 3:
            up, center, down = coefs_gpu[:-2, :], coefs_gpu[1:-1, :], coefs_gpu[2:, :]
            if max_var:
                yy, xx = cp.where((center > up) & (center > down))
                pmaxc = [[int(x), int(y + 1)] for y, x in zip(cp.asnumpy(yy), cp.asnumpy(xx))]
            if min_var:
                yy, xx = cp.where((center < up) & (center < down))
                pminc = [[int(x), int(y + 1)] for y, x in zip(cp.asnumpy(yy), cp.asnumpy(xx))]
        return pmaxr, pmaxc, pminr, pminc

    @staticmethod
    def _filtered_gpu(coefs_gpu, row_var, col_var, max_var, min_var, *, find_peaks, kwargs):
        import cupy as cp

        pmaxr, pmaxc, pminr, pminc = [], [], [], []
        if row_var and (max_var or min_var):
            for y in range(coefs_gpu.shape[0]):
                signal = coefs_gpu[y]
                if max_var:
                    peaks, _ = find_peaks(signal, **kwargs)
                    pmaxr.extend([int(x), int(y)] for x in cp.asnumpy(peaks))
                if min_var:
                    peaks, _ = find_peaks(-signal, **kwargs)
                    pminr.extend([int(x), int(y)] for x in cp.asnumpy(peaks))

        if col_var and (max_var or min_var):
            for x in range(coefs_gpu.shape[1]):
                signal = coefs_gpu[:, x]
                if max_var:
                    peaks, _ = find_peaks(signal, **kwargs)
                    pmaxc.extend([int(x), int(y)] for y in cp.asnumpy(peaks))
                if min_var:
                    peaks, _ = find_peaks(-signal, **kwargs)
                    pminc.extend([int(x), int(y)] for y in cp.asnumpy(peaks))
        return pmaxr, pmaxc, pminr, pminc
