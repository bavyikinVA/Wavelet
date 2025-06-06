import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import os
import cv2


def find_k_nearest_neighbors(points, k):
    """Находит k ближайших соседей для каждой точки"""
    if len(points) < 2:
        return {}

    if len(points) <= k:
        k = len(points) - 1

    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(points)
    distances, indices = nbrs.kneighbors(points)

    neighbors_dict = {}
    for i in range(len(points)):
        neighbors_dict[i] = {
            'indices': indices[i][1:],  # исключаем саму точку
            'distances': distances[i][1:]  # расстояния до соседей
        }

    return neighbors_dict


def draw_knn_graph(points, neighbors_dict, scale, channel, extreme_type, filename, k, original_image):
    """Граф связей с k ближайшими соседями"""
    if len(points) == 0:
        return

    plt.figure(figsize=(7, 8))
    ax = plt.gca()

    # Если есть исходное изображение, отображаем его как фон
    if original_image is not None:
        plt.imshow(original_image)
        ax.invert_yaxis()  # Инвертируем ось Y для правильного отображения координат изображения
    else:
        ax.invert_yaxis()

    ax.invert_yaxis()

    # рисуем все соединения
    for i, neighbors in neighbors_dict.items():
        for neighbor_idx in neighbors['indices']:
            plt.plot([points[i, 0], points[neighbor_idx, 0]],
                     [points[i, 1], points[neighbor_idx, 1]],
                     'b-', linewidth=0.7, alpha=0.3)

    # рисуем точки
    plt.scatter(points[:, 0], points[:, 1], c='gray', s=10, alpha=0.5, label='Точки экстремумов')

    colors = ['Красный', 'Зеленый', 'Синий']
    plt.title(f'Граф связей {k}-ближайших соседей\nМасштаб: {scale}, Канал: {colors[channel]}, Тип: {extreme_type}')
    plt.xlabel('X координата (пиксели)')
    plt.ylabel('Y координата (пиксели)')
    plt.legend(loc='upper right', framealpha=0.5)

    dpi = 150 if len(points) > 100 else 300
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close()


def process_extremes_with_knn(extreme_dict, scale_folder_path, k, original_image):
    """Обрабатывает экстремумы для одного набора параметров
        extremes_dict =  {'type_data': type_data, -> 01 Str/Tr
                    'channel': channel, -> (012)
                    'scale': scale, -> int
                    'max_by_row': pmaxr, ->[]
                    'max_by_column': pmaxc, ->[]
                    'min_by_row': pminr, ->[]
                    'min_by_column': pminc} ->[]
        """

    color_names = ['Red', 'Green', 'Blue']
    type_names = {0: 'Str', 1: 'Tr'}

    # Получаем параметры из словаря
    type_data = extreme_dict['type_data']
    channel = extreme_dict['channel']
    scale = extreme_dict['scale']

    for extreme_type in ['max_by_row', 'max_by_column', 'min_by_row', 'min_by_column']:
        points = np.array(extreme_dict[extreme_type])
        if len(points) < 2:
            continue

        # Находим k-ближайших соседей
        neighbors_dict = find_k_nearest_neighbors(points, k)

        # Формируем уникальные имена файлов
        graph_filename = (
            f"KNN_{type_names[type_data]}_Graph_Scale_{scale}_"
            f"Channel_{color_names[channel]}_{extreme_type}.png"
        )

        info_filename = (
            f"KNN_{type_names[type_data]}_Info_Scale_{scale}_"
            f"Channel_{color_names[channel]}_{extreme_type}.txt"
        )

        # Рисуем и сохраняем граф
        draw_knn_graph(
            points, neighbors_dict,
            scale, channel, extreme_type, os.path.join(scale_folder_path, graph_filename), k, original_image)


        # Сохраняем информацию о соседях
        save_knn_info(os.path.join(scale_folder_path, info_filename), points, neighbors_dict,
            scale, channel, extreme_type, k)

def save_knn_info(filename, points, neighbors_dict, scale, channel, extreme_type, k):
    """Сохраняет информацию о k-ближайших соседях в текстовый файл"""
    colors = ['Red', 'Green', 'Blue']


    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Масштаб: {scale}\n")
        f.write(f"Канал: {colors[channel]}\n")
        f.write(f"Тип экстремумов: {extreme_type}\n")
        f.write(f"Количество соседей (k): {k}\n")
        f.write(f"Всего точек: {len(points)}\n\n")

        f.write("Индекс точки, X координата, Y координата, Индексы соседей, Расстояния до соседей\n")

        for i in neighbors_dict:
            neighbors = ', '.join(map(str, neighbors_dict[i]['indices']))
            dists = ', '.join([f"{d:.2f}" for d in neighbors_dict[i]['distances']])
            f.write(f"{i} {points[i][0]} {points[i][1]} {neighbors} {dists}\n")