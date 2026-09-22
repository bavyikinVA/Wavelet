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
