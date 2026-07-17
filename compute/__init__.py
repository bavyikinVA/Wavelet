from .backend import ComputeBackend
from compute.wavelets.gpu_processor import GPUWaveletProcessor
from compute.wavelets.cpu_wavelet import morlet_wavelet_with_padding
from compute.extremes.extremes_finder import ExtremesFinder
from compute.extremes import gpu_envelopes
from compute.extremes import interpolator

__all__ = ['ComputeBackend', 'GPUWaveletProcessor', 'morlet_wavelet_with_padding', 'ExtremesFinder', 'gpu_envelopes', 'interpolator']