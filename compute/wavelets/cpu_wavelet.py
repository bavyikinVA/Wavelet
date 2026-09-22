import numpy as np
from compute.numerics import COMPUTE_DTYPE
from numba import jit, prange


# оригинальный код расчета НВП Морле для CPU
"""
@jit(nopython=True)
def morlet_wavelet_single_scale(data, scale, j):
    w0 = np.float32(0.0)
    for k in range(len(data)):
        t = (k - j) / scale
        w0 += data[k] * 0.75 * np.exp(-(t * t) / 2) * np.cos(2 * np.pi * t)
    return w0 / np.sqrt(scale)

@jit(nopython=True, parallel=True)
def morlet_wavelet(data, scales):
    data = np.asarray(data, dtype=COMPUTE_DTYPE)
    scales = np.asarray(scales, dtype=COMPUTE_DTYPE)
    coef = np.zeros((len(scales), len(data)), dtype=COMPUTE_DTYPE)
    for i in prange(len(scales)):
        for j in range(len(data)):
            coef[i, j] = morlet_wavelet_single_scale(data, scales[i], j)
    return coef
"""

# расчет НВП с симметричным отражением для строк и столбцов
@jit(nopython=True)
def morlet_wavelet_single_scale_with_padding(data, scale, j, pad_width):
    w0 = np.float32(0.0)
    data_len = len(data)

    for k in range(-pad_width, data_len + pad_width):
        # Symmetric padding repeats edge samples, even beyond one reflection.
        reflected = k % (2 * data_len)
        actual_k = reflected if reflected < data_len else 2 * data_len - 1 - reflected

        t = (k - j) / scale
        w0 += data[actual_k] * 0.75 * np.exp(-(t * t) / 2) * np.cos(2 * np.pi * t)

    return w0 / np.sqrt(scale)


@jit(nopython=True, parallel=True)
def morlet_wavelet_with_padding(data, scales):
    if len(data) == 0 or len(scales) == 0:
        raise ValueError("Сигнал и список масштабов не должны быть пустыми")
    for scale in scales:
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Все масштабы должны быть конечными числами больше нуля")
    coef = np.zeros((len(scales), len(data)), dtype=np.float32)

    for i in prange(len(scales)):
        # Each scale has the same support on CPU and GPU, independent of peers.
        pad_width = int(7 * scales[i]) // 2 + 1
        for j in range(len(data)):
            coef[i, j] = morlet_wavelet_single_scale_with_padding(
                data, scales[i], j, pad_width
            )
    return coef

# ---------------------------------------------------------------------------
# Flat CPU batch backend
# ---------------------------------------------------------------------------
# Instead of combining Python multiprocessing with a second Numba parallel
# region, parallelize one flat signal×scale work grid inside a single process.
# This avoids Windows spawn/IPC overhead while preserving the exact Morlet
# kernel used by morlet_wavelet_with_padding().
@jit(nopython=True, parallel=True)
def _morlet_wavelet_batch_flat_kernel(data, scales):
    n_signals = data.shape[0]
    signal_len = data.shape[1]
    n_scales = len(scales)
    result = np.empty((n_signals, n_scales, signal_len), dtype=np.float32)

    for flat_index in prange(n_signals * n_scales):
        signal_index = flat_index // n_scales
        scale_index = flat_index - signal_index * n_scales
        scale = scales[scale_index]
        pad_width = int(7 * scale) // 2 + 1
        for j in range(signal_len):
            result[signal_index, scale_index, j] = morlet_wavelet_single_scale_with_padding(
                data[signal_index], scale, j, pad_width
            )
    return result


def morlet_wavelet_batch_flat(data, scales):
    """Compute a batch of 1-D Morlet transforms in one Numba parallel region.

    Input shape:  ``[signals, samples]``
    Output shape: ``[signals, scales, samples]``
    """
    data = np.asarray(data, dtype=COMPUTE_DTYPE)
    scales = np.asarray(scales, dtype=COMPUTE_DTYPE)
    if data.ndim != 2:
        raise ValueError("Пакет сигналов должен быть двумерным")
    if data.shape[0] == 0 or data.shape[1] == 0 or len(scales) == 0:
        raise ValueError("Сигналы и список масштабов не должны быть пустыми")
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("Все масштабы должны быть конечными числами больше нуля")
    return np.asarray(_morlet_wavelet_batch_flat_kernel(data, scales), dtype=COMPUTE_DTYPE)


def recommended_cpu_threads():
    """Return a conservative default matching physical cores on SMT systems."""
    import os
    explicit = os.environ.get("WAVELETS_NUMBA_THREADS")
    if explicit:
        value = int(explicit)
        if value < 1:
            raise ValueError("WAVELETS_NUMBA_THREADS должен быть положительным")
        return value
    try:
        import psutil  # optional
        physical = psutil.cpu_count(logical=False)
        if physical:
            return max(1, int(physical))
    except Exception:
        pass
    logical = int(os.cpu_count() or 1)
    # Most desktop CPUs expose two logical threads per physical core. This is
    # intentionally conservative; users can override it with the env variable.
    return max(1, logical // 2 if logical > 1 else 1)


def morlet_wavelet_batch_flat_threaded(data, scales, threads=None):
    """Flat batch CWT using a bounded Numba thread team."""
    import numba
    target_threads = int(threads or recommended_cpu_threads())
    previous = numba.get_num_threads()
    try:
        numba.set_num_threads(max(1, target_threads))
        return morlet_wavelet_batch_flat(data, scales)
    finally:
        numba.set_num_threads(previous)
