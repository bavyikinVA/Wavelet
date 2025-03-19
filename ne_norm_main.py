"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from multiprocessing import Pool, freeze_support
from numba import jit
import math

# Ваш код для вейвлет-преобразования
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

def create_array_with_one_in_middle(size, value):
    array = np.zeros(size)
    middle_index = size // 2
    array[middle_index] = value
    return array

def process_row(args):
    i, size, scales, scales_size = args
    array = create_array_with_one_in_middle(size, i + 1)
    return morlet_wavelet(array, size, scales, scales_size)

def process_channel(data, scales):
    rows = data.shape[0]
    cols = data.shape[1]
    scales_size = len(scales)
    result = np.zeros((rows, scales_size, cols))

    with Pool() as pool:
        args = [(i, cols, scales, scales_size) for i in range(rows)]
        results = pool.map(process_row, args)

    for i, res in enumerate(results):
        result[i] = res

    return result

# Функция для загрузки изображения и вычисления матриц цветовых каналов
def process_image(image_path, scales):
    # Загрузка изображения
    image = Image.open(image_path)
    image_data = np.array(image)

    # Разделение на цветовые каналы
    if image_data.ndim == 3:
        red_channel = image_data[:, :, 0]
        green_channel = image_data[:, :, 1]
        blue_channel = image_data[:, :, 2]
    else:
        raise ValueError("Изображение должно быть цветным (RGB)")

    # Обработка каждого канала
    red_result = process_channel(red_channel, scales)
    green_result = process_channel(green_channel, scales)
    blue_result = process_channel(blue_channel, scales)

    return red_result, green_result, blue_result

if __name__ == '__main__':
    freeze_support()  # Это необходимо для Windows

    # Пример использования
    image_path = r'C:\Users\bavyk\Downloads\delta_vert.png'  # Путь к вашему изображению
    scales = np.arange(1, 11)  # Пример масштабов от 1 до 10

    red_result, green_result, blue_result = process_image(image_path, scales)

    # Визуализация результатов (пример)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(red_result.sum(axis=1), cmap='hot')
    axs[0].set_title('Red Channel')
    axs[1].imshow(green_result.sum(axis=1), cmap='hot')
    axs[1].set_title('Green Channel')
    axs[2].imshow(blue_result.sum(axis=1), cmap='hot')
    axs[2].set_title('Blue Channel')
    plt.show()
"""