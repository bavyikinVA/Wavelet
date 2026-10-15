from compute.backend_policy import GPU_FALLBACK_ERRORS
from compute.extremes.interpol import get_row_envelopes, get_column_envelopes
from compute.numerics import COMPUTE_DTYPE_NAME


class Interpolator:
    """Backend-invariant sparse PCHIP-envelope extrema extraction.

    GPU mode uses a vectorized whole-map implementation: support grouping,
    exact-value plateau compression and sparse PCHIP-extrema extraction are
    performed on CUDA without a Python loop over rows/columns.  Only the final
    compact Nx2 coordinates are copied to host memory.
    """

    def __init__(self, gpu_backend):
        self.gpu_backend = gpu_backend
        self._gpu_processor = None

    def _gpu(self, coefs, max_points, min_points, direction):
        if self._gpu_processor is None:
            from compute.extremes.gpu_envelopes import GPUEnvelopeProcessor
            self._gpu_processor = GPUEnvelopeProcessor()
        if direction == 'row':
            return self._gpu_processor.get_row_envelopes_gpu(
                coefs, max_points, min_points
            )
        if direction == 'col':
            return self._gpu_processor.get_col_envelopes_gpu(
                coefs, max_points, min_points
            )
        raise ValueError("direction должен быть 'row' или 'col'")

    def get_envelopes(self, coefs, max_points, min_points, direction='row',
                      *, protocol=None, stage=None):
        stage = stage or f"envelopes:{direction}"
        use_gpu = bool(getattr(self.gpu_backend, "use_gpu", False))

        if use_gpu:
            try:
                result = self._gpu(coefs, max_points, min_points, direction)
                if protocol is not None:
                    protocol.record_stage(
                        stage,
                        actual_backend="gpu",
                        dtype=COMPUTE_DTYPE_NAME,
                        fallback=False,
                        details={
                            "algorithm": "pchip-grouped-sparse-gpu-vectorized-v4",
                            "sparse_evaluation": True,
                            "backend_policy": "gpu-vectorized-primary",
                        },
                    )
                return result
            except GPU_FALLBACK_ERRORS as exc:
                if protocol is not None:
                    protocol.record_fallback(
                        stage,
                        reason=exc.__class__.__name__,
                        message=str(exc),
                    )
                result = self._cpu(coefs, max_points, min_points, direction)
                if protocol is not None:
                    protocol.record_stage(
                        stage,
                        actual_backend="cpu",
                        dtype=COMPUTE_DTYPE_NAME,
                        fallback=True,
                        details={
                            "algorithm": "pchip-grouped-sparse-v3",
                            "backend_policy": "gpu-infrastructure-fallback",
                        },
                    )
                return result

        result = self._cpu(coefs, max_points, min_points, direction)
        if protocol is not None:
            protocol.record_stage(
                stage,
                actual_backend="cpu",
                dtype=COMPUTE_DTYPE_NAME,
                fallback=False,
                details={
                    "algorithm": "pchip-grouped-sparse-v3",
                    "sparse_evaluation": True,
                    "backend_policy": "cpu-requested",
                },
            )
        return result

    @staticmethod
    def _cpu(coefs, max_points, min_points, direction):
        if direction == 'row':
            return get_row_envelopes(coefs, max_points, min_points)
        if direction == 'col':
            return get_column_envelopes(coefs, max_points, min_points)
        raise ValueError("direction должен быть 'row' или 'col'")
