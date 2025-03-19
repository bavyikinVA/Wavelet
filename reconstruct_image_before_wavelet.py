import numpy as np
import cv2

# Загрузка матрицы из файла
with open(r'C:\Users\bavyk\Downloads\Вейвлет_преобразования_03_10_2024_14_34_56\Масштаб_55.0\Рассчет_вейвлетов_Масштаб_55.0_Цветовой_канал_Синий.txt', 'r') as file:
    matrix2d = np.loadtxt(file, delimiter=',', dtype=int)

# Найти минимальный элемент в матрице
min_limit = np.min(matrix2d)

# Сместить все значения в матрице, чтобы минимальный элемент стал 0
matrix2d = matrix2d + abs(min_limit)

# Ограничить значения в матрице до диапазона 0-255
matrix2d = np.clip(matrix2d, 0, 255).astype(np.uint8)

# Создание изображения из трех одинаковых каналов (в данном случае, синий канал)
image = cv2.merge([matrix2d, matrix2d, matrix2d])

# Отображение изображения
cv2.imshow('Reconstructed Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
