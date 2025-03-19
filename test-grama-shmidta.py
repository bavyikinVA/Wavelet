import numpy as np
from Gram_Shmidt import change_channels
import cv2
import matplotlib.pyplot as plt
import ctypes

# r - g - b
v_blue = np.array([54, 28, 99], dtype=np.double)  # blue
v_pink = np.array([150, 122, 147], dtype=np.double)  # pink
image_path = 'C:/Users/bavyk/Downloads/shum.png'
image = cv2.imread(image_path)
b, g, r = cv2.split(image)
data = np.array([r, g, b], dtype=np.double)

print(data)
data_next = np.array(change_channels(v_blue, v_pink, data), dtype=np.double)

print(data_next)

r_next = data_next[0]
g_next = data_next[1]
b_next = data_next[2]
'''
with open("blue_0_str_before.txt", 'w') as f1:
    for element in b_next[0]:
        f1.write(element + ' ')

with open("green_0_str_before.txt", 'w') as f2:
    for element in g_next[0]:
        f1.write(element + ' ')

with open("red_0_str_before.txt", 'w') as f3:
    for element in r_next[0]:
        f1.write(element + ' ')
'''
'''
# Создаем фигуру и осевой объект 3D
plt.figure()

array_b_0 = np.array(b[0])
array_g_0 = np.array(g[0])
array_r_0 = np.array(r[0])

# Создаем три массива длиной 500
array1 = np.array(b_next[0])
array2 = np.array(g_next[0])
array3 = np.array(r_next[0])

# Создаем список точек на основе элементов массивов
points1 = np.vstack((array_b_0, array_g_0, array_r_0)).T
points2 = np.vstack((array1, array2, array3)).T

# Рисуем точки и раскрашиваем их в указанные цвета
plt.plot(array_b_0, c='blue')  # голубые
plt.plot(array_g_0, c='green')  # зеленые
plt.plot(array_r_0, c='red')  # красные

plt.plot(array1, c='cyan')  # синие
plt.plot(array2, c='yellow')  # желтые
plt.plot(array3, c='purple')  # фиолетовые

# Показываем график
plt.show()


# image_next = cv2.merge([b_next, g_next, r_next])

# cv2.imshow('img', image_next)
# cv2.waitKey(0)
'''
lib = ctypes.CDLL("./dll_wavelets.dll")
lib.morlet_wavelet.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double),
                               ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]

# delta = np.zeros(501)
# delta[250] = 1
delta_column = np.zeros((501, 1), dtype=np.float64)
delta_column[250] = 1
np.savetxt('delta_col.txt', delta_column, fmt='%.3f')
# delta = np.array([delta, delta, delta], dtype=np.double)
# delta_end = np.array([delta, delta, delta], dtype=np.double)
# delta_next = np.array(change_channels(v_blue, v_pink, delta), dtype=np.double)
data_channel = np.array(delta_column, dtype=np.double)  # матрица канала
'''
for row in data_channel:
    mean = np.mean(row, dtype=np.double)
    row -= mean
'''
rows = len(delta_column)  # data_channel.shape)[0]
cols = 1  # len(delta)  # data_channel.shape)[1]
scales = np.array([10, 100], dtype=np.double)  # масштабы вейвлета (формат - лист)
num_scale = len(scales)
result_shape = (num_scale, rows, cols)
result = np.empty(result_shape, dtype=np.double)

lib.morlet_wavelet(num_scale, rows, cols, data_channel.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                   scales.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                   result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

# np.savetxt('r_after_wavelets.csv', result[0], delimiter=',', fmt='%d')
# print(result[0])

# plt.figure()
fig, ax = plt.subplots(1, 2)
length = cols
ax[0].plot(range(length), result[0][0], c='red')
ax[0].set_title('График 1')

# Отображаем второй график на правом подграфике
ax[1].plot(range(length), result[1][0], c='green')
ax[1].set_title('График 2')
# plt.plot(range(length), result[0][0], c='red')
# plt.plot(range(length), result[1][0], c='green')
# plt.plot(length, b[250], c='yellow')
plt.show()
