"""
import numpy as np
import math
from multiprocessing import Pool
from numba import jit
import matplotlib.pyplot as plt

@jit(nopython=True)
def morlet_wavelet_single_scale(data, data_size, scale, j):
    w0 = 0
    for k in range(data_size):
        t = (k - j) / scale
        w0 += data[k] * 0.75 * math.exp(-(t * t) / 2) * math.cos(2 * math.pi * t)
    return w0 / math.sqrt(scale)

def morlet_wavelet(data, data_size, scales, weight_size):
    coef = np.zeros((weight_size, data_size))
    for i in range(weight_size):
        for j in range(data_size):
            coef[i][j] = morlet_wavelet_single_scale(data, data_size, scales[i], j)
    return coef

def process_row(args):
    row_data, scales, scales_size = args
    return morlet_wavelet(row_data, len(row_data), scales, scales_size)

def process_channel(data, scales):
    rows = data.shape[0]
    cols = data.shape[1]
    scales_size = len(scales)
    result = np.zeros((rows, scales_size, cols))

    print("start_morlet")

    with Pool() as pool:
        args = [(data[i], scales, scales_size) for i in range(rows)]
        results = pool.map(process_row, args)

    for i, res in enumerate(results):
        result[i] = res

    return result

# Пример использования
if __name__ == "__main__":
    # Пример матрицы пикселей изображения
    image_data = np.random.rand(100, 100)  # Замените это на вашу матрицу пикселей
    scales = np.linspace(1, 10, 10)  # Пример масштабов

    # Вывод исходного изображения
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image_data, cmap='gray')
    plt.colorbar()

    # Выполнение преобразований
    result = process_channel(image_data, scales)

    # Вывод преобразованного изображения (первый масштаб)
    plt.subplot(1, 2, 2)
    plt.title("Transformed Image (First Scale)")
    plt.imshow(result[:, 0, :], cmap='gray')
    plt.colorbar()

    plt.show()
"""