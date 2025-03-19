import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Создаем фигуру и осевой объект 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Создаем три массива длиной 500
array1 = np.random.rand(500)
array2 = np.random.rand(500)
array3 = np.random.rand(500)

# Создаем список точек на основе элементов массивов
points = np.vstack((array1, array2, array3)).T

# Рисуем точки
ax.scatter(points[:, 0], points[:, 1], points[:, 2])

# Настраиваем оси
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Показываем график
plt.show()


'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Создаем фигуру и осевой объект 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Координаты начала и конца векторов
vector1 = np.array([0.46474079, 0.24097671, 0.85202478])
vector2 = np.array([0.49152521,  0.73016599, -0.47461625])
vector3 = np.array([-0.73649098,  0.63936519,  0.22089179])

# Начало векторов
start = np.zeros(3)
vector_blue = np.array([54, 28, 99])
vector_pink = np.array([150, 122, 147])

# Рисуем векторы с помощью стрелок
ax.quiver(start[0], start[1], start[2], vector_pink[0], vector_pink[1], vector_pink[2], color='black')
ax.quiver(start[0], start[1], start[2], vector_blue[0], vector_blue[1], vector_blue[2], color='black')

# Рисуем векторы с помощью стрелок
ax.quiver(start[0], start[1], start[2], vector1[0], vector1[1], vector1[2], color='red')
ax.quiver(start[0], start[1], start[2], vector2[0], vector2[1], vector2[2], color='red')
ax.quiver(start[0], start[1], start[2], vector3[0], vector3[1], vector3[2], color='red')

# Настраиваем оси
ax.set_xlabel('R')
ax.set_ylabel('G')
ax.set_zlabel('B')
plt.grid()
# Показываем график
plt.show()
'''