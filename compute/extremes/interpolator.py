from compute.extremes.interpol import get_row_envelopes, get_column_envelopes
from compute.backend_policy import classify_gpu_exception
from compute.numerics import COMPUTE_DTYPE_NAME


class Interpolator:
    def __init__(self, gpu_backend):
        self.gpu_backend = gpu_backend
        self.gpu_processor = None
        if self.gpu_backend.use_gpu:
            try:
                from compute.extremes.gpu_envelopes import GPUEnvelopeProcessor
                self.gpu_processor = GPUEnvelopeProcessor()
            except Exception as exc:
                mapped = classify_gpu_exception(exc)
                if mapped is None:
                    raise
                if self.gpu_backend.strict_backend:
                    raise mapped from exc
                self.gpu_processor = None

    def get_envelopes(self, coefs, max_points, min_points, direction='row',
                      *, protocol=None, stage=None):
        stage = stage or f"envelopes:{direction}"
        use_gpu = (
            self.gpu_backend.use_gpu
            and self.gpu_processor is not None
            and len(max_points) + len(min_points) >= 2000
        )

        if not use_gpu:
            result = self._cpu(coefs, max_points, min_points, direction)
            if protocol is not None:
                protocol.record_stage(
                    stage, actual_backend="cpu", dtype=COMPUTE_DTYPE_NAME
                )
            return result

        try:
            import cupy as cp
            coefs_gpu = cp.asarray(coefs, dtype=cp.float32)
            if direction == 'row':
                result = self.gpu_processor.get_row_envelopes_gpu(
                    coefs_gpu, max_points, min_points
                )
            else:
                result = self.gpu_processor.get_col_envelopes_gpu(
                    coefs_gpu, max_points, min_points
                )
            if protocol is not None:
                protocol.record_stage(
                    stage, actual_backend="gpu", dtype=COMPUTE_DTYPE_NAME
                )
            return result
        except Exception as exc:
            mapped = classify_gpu_exception(exc)
            if mapped is None:
                raise
            if self.gpu_backend.strict_backend or (
                protocol is not None and protocol.strict_backend
            ):
                raise mapped from exc
            if protocol is not None:
                protocol.record_fallback(
                    stage,
                    reason=mapped.__class__.__name__,
                    message=str(mapped),
                )
            result = self._cpu(coefs, max_points, min_points, direction)
            if protocol is not None:
                protocol.record_stage(
                    stage,
                    actual_backend="cpu",
                    dtype=COMPUTE_DTYPE_NAME,
                    fallback=True,
                    details={"fallback_reason": mapped.__class__.__name__},
                )
            return result

    @staticmethod
    def _cpu(coefs, max_points, min_points, direction):
        if direction == 'row':
            return get_row_envelopes(coefs, max_points, min_points)
        return get_column_envelopes(coefs, max_points, min_points)
