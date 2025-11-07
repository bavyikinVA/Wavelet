import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import os
import gc


def find_k_nearest_neighbors(points, k, progress_callback=None, log_callback=None):
    """Поиск k-ближайших соседей"""
    if log_callback:
        log_callback(f"Поиск {k}-ближайших соседей для {len(points)} точек")

    if len(points) < 2:
        if log_callback:
            log_callback("Недостаточно точек для поиска соседей (требуется минимум 2)")
        return {}

    if len(points) <= k:
        k = len(points) - 1
        if log_callback:
            log_callback(f"Количество соседей уменьшено до {k} (всего точек: {len(points)})")

    if progress_callback:
        progress_callback(0.1, "Инициализация алгоритма NearestNeighbors...")

    try:
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='kd_tree').fit(points)
        if progress_callback:
            progress_callback(0.4, "Вычисление расстояний между точками...")

        distances, indices = nbrs.kneighbors(points)

        if progress_callback:
            progress_callback(0.8, "Формирование словаря соседей...")

        neighbors_dict = {}
        for i in range(len(points)):
            neighbors_dict[i] = {
                'indices': indices[i][1:],  # исключаем саму точку
                'distances': distances[i][1:]  # расстояния до соседей
            }

        #if log_callback:
            #avg_neighbors = sum(len(v['indices']) for v in neighbors_dict.values()) / len(neighbors_dict)
            #log_callback(
            #    f"✅ Найдены соседи для {len(neighbors_dict)} точек, в среднем {avg_neighbors:.1f} соседей на точку")

        if progress_callback:
            progress_callback(1.0, "Поиск соседей завершен")

        return neighbors_dict

    except Exception as e:
        if log_callback:
            log_callback(f"Ошибка при поиске соседей: {e}")
        return {}


def draw_knn_graph(points, neighbors_dict, scale, channel, extreme_type, filename, k, original_image,
                   progress_callback=None, log_callback=None):
    """Отрисовка графа KNN"""
    if len(points) == 0:
        if log_callback:
            log_callback("Нет точек для отрисовки графа")
        return

    if log_callback:
        log_callback(f"Начало отрисовки графа KNN: {len(points)} точек")

    try:
        if progress_callback:
            progress_callback(0.1, "Подготовка данных для визуализации...")

        plt.figure(figsize=(7, 8))
        ax = plt.gca()

        # отображаем исходное изображение как фон
        if original_image is not None:
            if progress_callback:
                progress_callback(0.3, "Добавление фонового изображения...")
            plt.imshow(original_image)
            ax.invert_yaxis()
        else:
            ax.invert_yaxis()
        ax.invert_yaxis()
        if progress_callback:
            progress_callback(0.5, "Отрисовка соединений между точками...")

        # рисуем все соединения
        connections_count = 0
        for i, neighbors in neighbors_dict.items():
            for neighbor_idx in neighbors['indices']:
                plt.plot([points[i, 0], points[neighbor_idx, 0]],
                         [points[i, 1], points[neighbor_idx, 1]],
                         'b-', linewidth=1, alpha=0.7)
                connections_count += 1

        if progress_callback:
            progress_callback(0.7, "Отрисовка точек экстремумов...")

        # рисуем точки
        plt.scatter(points[:, 0], points[:, 1], c='red', s=1, alpha=1)

        colors = ['Красный', 'Зеленый', 'Синий']
        plt.title(f'Граф связей {k}-ближайших соседей\nМасштаб: {scale}, Канал: {colors[channel]}, Тип: {extreme_type}')
        plt.xlabel('X (пиксели)')
        plt.ylabel('Y (пиксели)')

        if progress_callback:
            progress_callback(0.9, "Сохранение графика...")

        dpi = 200
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        plt.close()

        if log_callback:
            log_callback(f"График KNN сохранён: {filename}, соединений: {connections_count}")

        if progress_callback:
            progress_callback(1.0, "График сохранен")

        gc.collect()

    except Exception as e:
        if log_callback:
            log_callback(f"Ошибка при попытке отрисовки графа KNN: {e}")


def save_knn_info(filename, points, neighbors_dict, scale, channel, extreme_type, k,
                  progress_callback=None, log_callback=None):
    """Сохранение информации о KNN"""
    colors = ['Red', 'Green', 'Blue']

    if log_callback:
        log_callback(f"Сохранение информации KNN в файл: {filename}")

    try:
        if progress_callback:
            progress_callback(0.3, "Формирование заголовка файла...")

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Масштаб: {scale}\n")
            f.write(f"Канал: {colors[channel]}\n")
            f.write(f"Тип экстремумов: {extreme_type}\n")
            f.write(f"Количество соседей (k): {k}\n")
            f.write(f"Всего точек: {len(points)}\n\n")

            if progress_callback:
                progress_callback(0.6, "Запись данных о точках и соседях...")

            f.write("Индекс точки, X координата, Y координата, Индексы соседей, Расстояния до соседей\n")

            total_connections = 0
            for i in neighbors_dict:
                neighbors = ', '.join(map(str, neighbors_dict[i]['indices']))
                dists = ', '.join([f"{d:.2f}" for d in neighbors_dict[i]['distances']])
                f.write(f"{i} {points[i][0]} {points[i][1]} {neighbors} {dists}\n")
                total_connections += len(neighbors_dict[i]['indices'])

            if progress_callback:
                progress_callback(0.9, "Завершение записи файла...")

        if log_callback:
            log_callback(f"Файл KNN сохранён: {filename}, всего соединений: {total_connections}")

        if progress_callback:
            progress_callback(1.0, "Файл сохранен")

    except Exception as e:
        if log_callback:
            log_callback(f"Ошибка при сохранении файла KNN: {e}")


def process_extremes_with_knn(extreme_dict, scale_folder_path, k, original_image, print_text_var, print_image_var,
                              progress_callback=None, log_callback=None):
    """
    Обработка экстремумов с KNN

    extremes_dict =  {'type_data': type_data, -> 01 Str/Tr
                    'channel': channel, -> (012)
                    'scale': scale, -> int
                    'max_by_row': pmaxr, ->[]
                    'max_by_column': pmaxc, ->[]
                    'min_by_row': pminr, ->[]
                    'min_by_column': pminc} ->[]
    """

    if log_callback:
        log_callback(f"Начало обработки KNN для масштаба {extreme_dict['scale']}")

    color_names = ['Red', 'Green', 'Blue']
    type_names = {0: 'Str', 1: 'Tr'}

    # параметры из словаря
    type_data = extreme_dict['type_data']
    channel = extreme_dict['channel']
    scale = extreme_dict['scale']

    extreme_types = ['max_by_row', 'max_by_column', 'min_by_row', 'min_by_column']
    total_extreme_types = len(extreme_types)
    processed_types = 0

    for extreme_type in extreme_types:
        processed_types += 1
        points = np.array(extreme_dict[extreme_type])

        if log_callback:
            log_callback(f"Обработка {extreme_type}: {len(points)} точек")

        if len(points) < 2:
            if log_callback:
                log_callback(f"Пропуск {extreme_type}: недостаточно точек ({len(points)})")
            continue

        # Прогресс для текущего типа экстремумов
        current_progress = processed_types / total_extreme_types

        def update_progress(stage_progress, message):
            """Функция для обновления прогресса"""
            if progress_callback:
                overall_progress = (current_progress - 1 / total_extreme_types) + (stage_progress / total_extreme_types)
                progress_callback(overall_progress, f"KNN {extreme_type}: {message}")

        # k-ближайших соседей
        neighbors_dict = find_k_nearest_neighbors(
            points, k,
            progress_callback=lambda p, m: update_progress(p * 0.4, m),
            log_callback=log_callback
        )

        if not neighbors_dict:
            continue

        # формируем имена файлов
        graph_filename = (
            f"KNN_{type_names[type_data]}_Graph_Scale_{scale}_"
            f"Channel_{color_names[channel]}_{extreme_type}.png"
        )

        info_filename = (
            f"KNN_{type_names[type_data]}_Info_Scale_{scale}_"
            f"Channel_{color_names[channel]}_{extreme_type}.txt"
        )

        if print_image_var:
            draw_knn_graph(
                points, neighbors_dict,
                scale, channel, extreme_type,
                os.path.join(scale_folder_path, graph_filename),
                k, original_image,
                progress_callback=lambda p, m: update_progress(0.4 + p * 0.3, m),
                log_callback=log_callback
            )

        if print_text_var:
            save_knn_info(
                os.path.join(scale_folder_path, info_filename),
                points, neighbors_dict,
                scale, channel, extreme_type, k,
                progress_callback=lambda p, m: update_progress(0.7 + p * 0.3, m),
                log_callback=log_callback
            )

    if log_callback:
        log_callback(f"Завершена обработка KNN для масштаба {scale}")