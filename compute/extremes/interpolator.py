from compute.backend_policy import GPU_FALLBACK_ERRORS
from compute.extremes.interpol import get_row_envelopes, get_column_envelopes
from compute.numerics import COMPUTE_DTYPE_NAME


class Interpolator:
    """Backend-invariant sparse PCHIP-envelope extrema extraction."""

    def __init__(self, gpu_backend):
        self.gpu_backend = gpu_backend
        self._gpu_processor = None

    def get_envelopes(self, coefs, max_points, min_points, direction='row',
                      *, protocol=None, stage=None):
        stage = stage or f"envelopes:{direction}"
        requested_gpu = bool(getattr(self.gpu_backend, "use_gpu", False))
        fallback = False
        if requested_gpu:
            try:
                from compute.extremes.gpu_envelopes import GPUEnvelopeProcessor
                if self._gpu_processor is None:
                    self._gpu_processor = GPUEnvelopeProcessor()
                if direction == 'row':
                    result = self._gpu_processor.get_row_envelopes_gpu(coefs, max_points, min_points)
                elif direction == 'col':
                    result = self._gpu_processor.get_col_envelopes_gpu(coefs, max_points, min_points)
                else:
                    raise ValueError("direction должен быть 'row' или 'col'")
                actual_backend = "gpu"
            except GPU_FALLBACK_ERRORS as exc:
                strict = bool(getattr(self.gpu_backend, "strict_backend", False)) or bool(
                    protocol is not None and protocol.strict_backend
                )
                if strict:
                    raise
                if protocol is not None:
                    protocol.record_fallback(stage, reason=exc.__class__.__name__, message=str(exc))
                result = self._cpu(coefs, max_points, min_points, direction)
                actual_backend = "cpu"
                fallback = True
        else:
            result = self._cpu(coefs, max_points, min_points, direction)
            actual_backend = "cpu"

        if protocol is not None:
            protocol.record_stage(
                stage, actual_backend=actual_backend, dtype=COMPUTE_DTYPE_NAME,
                fallback=fallback,
                details={
                    "algorithm": "pchip-grouped-sparse-v3",
                    "sparse_evaluation": True,
                    "backend_policy": "cpu-gpu-mathematical-parity",
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
