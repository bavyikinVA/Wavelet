import numpy as np
from numba import jit, prange

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


@jit(nopython=True)
def apply_gaussian_edge_filter(channel_data, scales, wavelet_lengths):
    """
    Применяет гауссов фильтр только к краевым зонам, сохраняя внутреннюю часть неизменной

    Параметры:
    channel_data - массив вейвлет-коэффициентов (scales, rows, cols)
    scales - список масштабов
    wavelet_lengths - список длин вейвлетов для каждого масштаба
    """
    num_scales = len(scales)
    rows, cols = channel_data.shape[1], channel_data.shape[2]

    for scale_idx in range(num_scales):
        edge_size = wavelet_lengths[scale_idx]

        # Создаем гауссов фильтр (только возрастающую часть)
        x = np.linspace(-3, 0, edge_size)
        gaussian = np.exp(-(x ** 2) / 2)
        gaussian = (gaussian - gaussian[0]) / (gaussian[-1] - gaussian[0])  # Нормализуем от 0 до 1

        # Применяем фильтр к началу каждой строки
        for row in range(rows):
            for col in range(edge_size):
                channel_data[scale_idx, row, col] *= gaussian[col]

        # Применяем фильтр к концу каждой строки (зеркально)
        for row in range(rows):
            for col in range(cols - edge_size, cols):
                idx = cols - col - 1
                channel_data[scale_idx, row, col] *= gaussian[idx]

    return channel_data