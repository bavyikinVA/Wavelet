from compute.extremes.interpol import get_row_envelopes, get_column_envelopes
from compute.numerics import COMPUTE_DTYPE_NAME


class Interpolator:
    """Grouped CPU PCHIP envelopes.

    PCHIP is intentionally executed by the CPU implementation even when Morlet
    CWT runs on the GPU.  The production path is sparse/grouped and does not
    allocate dense HxW envelope matrices, which makes it suitable for the
    project's 2500x2000 upper image size and avoids CPU/GPU algorithm drift.
    """

    def __init__(self, gpu_backend):
        self.gpu_backend = gpu_backend

    def get_envelopes(self, coefs, max_points, min_points, direction='row',
                      *, protocol=None, stage=None):
        stage = stage or f"envelopes:{direction}"
        result = self._cpu(coefs, max_points, min_points, direction)
        if protocol is not None:
            protocol.record_stage(
                stage,
                actual_backend="cpu",
                dtype=COMPUTE_DTYPE_NAME,
                details={
                    "algorithm": "pchip-grouped-v2",
                    "sparse_evaluation": True,
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
