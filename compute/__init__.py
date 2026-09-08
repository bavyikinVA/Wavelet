from .backend import ComputeBackend
from compute.wavelets.gpu_processor import GPUWaveletProcessor
from compute.wavelets.cpu_wavelet import morlet_wavelet_with_padding
from compute.extremes.extremes_finder import ExtremesFinder
from compute.extremes import interpolator

__all__ = ['ComputeBackend', 'GPUWaveletProcessor', 'morlet_wavelet_with_padding', 'ExtremesFinder', 'gpu_envelopes', 'interpolator']


def __getattr__(name):
    if name == 'gpu_envelopes':
        from importlib import import_module
        module = import_module('compute.extremes.gpu_envelopes')
        globals()[name] = module
        return module
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
