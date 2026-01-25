from compute.extremes.gpu_envelopes import GPUEnvelopeProcessor
import cupy as cp
from compute.extremes.interpol import get_row_envelopes, get_column_envelopes


class Interpolator:
    def __init__(self, gpu_backend):
        self.gpu_backend = gpu_backend
        if self.gpu_backend.use_gpu:
            self.gpu_processor = GPUEnvelopeProcessor()

    def get_envelopes(self, coefs, max_points, min_points, direction='row'):
        if not self.gpu_backend.use_gpu or len(max_points) + len(min_points) < 2000:
            # Для малого количества точек используем CPU
            if direction == 'row':
                return get_row_envelopes(coefs, max_points, min_points)
            else:
                return get_column_envelopes(coefs, max_points, min_points)
        else: # gpu
            coefs_gpu = cp.asarray(coefs)
            if direction == 'row':
                return self.gpu_processor.get_row_envelopes_gpu(coefs_gpu, max_points, min_points)
            else:
                return self.gpu_processor.get_col_envelopes_gpu(coefs_gpu, max_points, min_points)