"""Public compute API without importing unused numerical backends."""

__all__ = ['ComputeBackend', 'GPUWaveletProcessor', 'morlet_wavelet_with_padding', 'ExtremesFinder', 'gpu_envelopes', 'interpolator']


def __getattr__(name):
    from importlib import import_module
    exports = {
        'ComputeBackend': ('compute.backend', 'ComputeBackend'),
        'GPUWaveletProcessor': ('compute.wavelets.gpu_processor', 'GPUWaveletProcessor'),
        'morlet_wavelet_with_padding': ('compute.wavelets.cpu_wavelet', 'morlet_wavelet_with_padding'),
        'ExtremesFinder': ('compute.extremes.extremes_finder', 'ExtremesFinder'),
        'interpolator': ('compute.extremes.interpolator', None),
        'gpu_envelopes': ('compute.extremes.gpu_envelopes', None),
    }
    if name in exports:
        module_name, attribute = exports[name]
        module = import_module(module_name)
        value = getattr(module, attribute) if attribute else module
        globals()[name] = value
        return value
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
