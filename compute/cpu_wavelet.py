import numpy as np
from numba import jit, prange


# оригинальный код расчета НВП Морле для CPU
"""
@jit(nopython=True)
def morlet_wavelet_single_scale(data, scale, j):
    w0 = 0.0
    for k in range(len(data)):
        t = (k - j) / scale
        w0 += data[k] * 0.75 * np.exp(-(t * t) / 2) * np.cos(2 * np.pi * t)
    return w0 / np.sqrt(scale)

@jit(nopython=True, parallel=True)
def morlet_wavelet(data, scales):
    coef = np.zeros((len(scales), len(data)))
    for i in prange(len(scales)):
        for j in range(len(data)):
            coef[i, j] = morlet_wavelet_single_scale(data, scales[i], j)
    return coef
"""

# расчет НВП с симметричным отражением для строк и столбцов
@jit(nopython=True)
def morlet_wavelet_single_scale_with_padding(data, scale, j, pad_width):
    """
    Вычисление одного коэффициента вейвлет-преобразования с симметричным отражением
    """
    w0 = 0.0
    data_len = len(data)

    for k in range(-pad_width, data_len + pad_width):
        # определяем индекс с учетом симметричного отражения
        if k < 0:
            # отражение слева
            actual_k = -k - 1
        elif k >= data_len:
            # отражение справа
            actual_k = 2 * data_len - k - 1
        else:
            # внутренняя часть
            actual_k = k

        t = (k - j) / scale
        w0 += data[actual_k] * 0.75 * np.exp(-(t * t) / 2) * np.cos(2 * np.pi * t)

    return w0 / np.sqrt(scale)


@jit(nopython=True, parallel=True)
def morlet_wavelet_with_padding(data, scales):
    """
    Вейвлет-преобразование Морле со встроенным симметричным отражением
    """
    max_scale = max(scales)
    pad_width = int(7 * max_scale) // 2 + 1

    coef = np.zeros((len(scales), len(data)))

    for i in prange(len(scales)):
        for j in range(len(data)):
            coef[i, j] = morlet_wavelet_single_scale_with_padding(
                data, scales[i], j, pad_width
            )
    return coef