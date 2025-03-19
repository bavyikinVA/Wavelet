import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def nalozhenie(image_path, file_path, folder_path, channel):
    # Загружаем изображение
    img = mpimg.imread(image_path)

    # Создаем фигуру и осевой объект
    fig, ax = plt.subplots()

    # Отображаем изображение
    ax.imshow(img)

    # Загружаем координаты точек из файла
    points = np.loadtxt(file_path)
    
    # Накладываем точки на изображение
    ax.scatter(points[:, 0], points[:, 1], c='r', marker='.', s=1)

    save_path = folder_path + f'Наложение_точек_на_изображение_{channel}.png'

    plt.savefig(save_path)
    plt.close()
    # Показываем график
    # plt.show()
'''
def nalozhenie(image_path, file_path, folder_path, channel):
    # Загружаем изображение
    img = mpimg.imread(image_path)
    height, width = img.shape[:2]

    # Создаем фигуру и осевой объект
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))  # Устанавливаем размеры фигуры соответственно исходному изображению
    ax.set_aspect('equal')  # Устанавливаем одинаковое соотношение сторон

    # Отображаем изображение
    ax.imshow(img)

    # Загружаем координаты точек из файла
    points = np.loadtxt(file_path)

    # Накладываем точки на изображение
    ax.scatter(points[:, 0], points[:, 1], c='r', marker='.', s=1)

    save_path = os.path.join(folder_path, f'Наложение_точек_на_изображение_{channel}.png')

    plt.savefig(save_path)
    plt.close()
    # Показываем график
    # plt.show()
'''