import os
import warnings
from compute.knn.knn_cpu import process_extremes_with_knn
from compute.extremes.extremes_finder import ExtremesFinder
from compute.extremes.interpolator import Interpolator


def setup_cuda_environment():
    """Настройка окружения CUDA"""
    warnings.filterwarnings("ignore", message="CUDA path could not be detected")

    # Автоматический поиск пути CUDA
    cuda_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5",
        os.environ.get('CUDA_PATH'),
        os.environ.get('CUDA_HOME'),
    ]

    for path in cuda_paths:
        if path and os.path.exists(path):
            os.environ['CUDA_PATH'] = path
            # Добавляем в PATH для доступа к библиотекам
            cuda_bin = os.path.join(path, 'bin')
            cuda_lib = os.path.join(path, 'lib', 'x64')
            if os.path.exists(cuda_bin) and cuda_bin not in os.environ['PATH']:
                os.environ['PATH'] = cuda_bin + os.pathsep + os.environ['PATH']
            if os.path.exists(cuda_lib) and cuda_lib not in os.environ['PATH']:
                os.environ['PATH'] = cuda_lib + os.pathsep + os.environ['PATH']
            print(f"Настроен путь CUDA: {path}")
            return path

    print("CUDA путь не найден")
    return None

cuda_path = setup_cuda_environment()

# настройки для подавления предупреждений
os.environ['CUPY_CUDA_DISABLE_CUBIN_CACHE'] = '1'
os.environ['CUPY_CACHE_DIR'] = os.path.join(os.path.expanduser('~'), '.cupy', 'cache')


import threading
import time
import tkinter as tk
import traceback
from multiprocessing import Pool, freeze_support
from tkinter import messagebox as mb

import customtkinter as ctk
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

from Gram_Shmidt import change_channels
from compute.processing_task import ProcessingTask
from image_cropper_app import run_cropper
from pipette import run_pipette
from utils.gui import TkinterApp, ScrollableFrame, CollapsibleFrame
from utils.progress_manager import ProgressManager
from compute.wavelets.cpu_wavelet import morlet_wavelet_with_padding


def process_row_static(args_):
    row_data, scales_ = args_
    return morlet_wavelet_with_padding(row_data, scales_)

def process_column_static(args_):
    col_idx, column_data, scales_ = args_
    from compute.wavelets.cpu_wavelet import morlet_wavelet_with_padding
    return col_idx, morlet_wavelet_with_padding(column_data, scales_)


class ImageProcessor:
    def __init__(self, progress_manager: ProgressManager):
        self.progress = progress_manager
        self.tasks = []
        self.current_task_index = -1
        self.root_folder_path = ""

        # Инициализация бэкендов
        from compute.backend import ComputeBackend
        from compute.wavelets.gpu_processor import GPUWaveletProcessor
        self.backend = ComputeBackend()
        self.gpu_processor = GPUWaveletProcessor()

        # Инициализация KNN процессора
        from compute.knn.knn_cpu import get_knn_processor
        self.knn_processor = get_knn_processor(self.backend.use_gpu)

        # Логирование
        backend_info = self.backend.get_backend_info()
        self.progress.log_info(f"Вычислительный бэкенд: {backend_info['device_name']}")
        self.progress.log_info(f"KNN бэкенд: {'GPU' if self.knn_processor.is_gpu_available() else 'CPU'}")

    def clear_gpu_memory(self):
        """Очистка GPU памяти"""
        if hasattr(self, 'backend'):
            self.backend.clear_gpu_cache()
        if hasattr(self, 'gpu_processor'):
            self.gpu_processor.clear_cache()

    def add_task(self, task):
        task.task_id = len(self.tasks) + 1
        task.task_name = f"Задача {task.task_id}"
        self.tasks.append(task)

        # Создаём корневую папку для первой задачи
        if len(self.tasks) == 1:
            current_time = time.strftime("%d_%m_%Y_%H_%M")
            root_folder_name = f"ВП_{current_time}"
            self.create_downloads_folder(root_folder_name)
        return task.task_id

    def remove_task(self, task_id):
        self.tasks = [task for task in self.tasks if task.task_id != task_id]
        for i, task in enumerate(self.tasks):
            task.task_id = i + 1
            task.task_name = f"Задача {task.task_id}"

    def get_current_task(self):
        if 0 <= self.current_task_index < len(self.tasks):
            return self.tasks[self.current_task_index]
        return None

    def set_current_task(self, task_id):
        for i, task in enumerate(self.tasks):
            if task.task_id == task_id:
                self.current_task_index = i
                return True
        return False

    def load_image_for_task(self, task, master_window=None):
        self.progress.log_info("Загрузка изображения...")
        image_path = run_cropper(master_window)
        if image_path:
            task.image_path = image_path
            self.progress.log_info(f'Изображение загружено: {image_path}')
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Не удалось загрузить изображение: {image_path}")
            task.original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            b, g, r = cv2.split(image)
            task.data = [r, g, b]
            if not all(isinstance(ch, np.ndarray) for ch in task.data):
                raise ValueError("Один или несколько каналов изображения не являются массивами NumPy")
            task.data_copy = [channel.copy() for channel in task.data]
            self.progress.log_info("Изображение успешно обработано")
            return True
        else:
            self.progress.log_error('Ошибка загрузки/обработки изображения')
            return False

    def pipette_channel_for_task(self, task):
        self.progress.log_info("Запуск инструмента 'Пипетка'...")
        task.color1, task.color2 = run_pipette(master=None, image_path=task.image_path)
        self.progress.log_info("Цвета успешно выбраны")

    def create_downloads_folder(self, folder_name):
        downloads_path = os.path.join(os.path.expanduser('~'), 'Downloads')
        self.root_folder_path = os.path.join(downloads_path, folder_name)
        try:
            os.makedirs(self.root_folder_path, exist_ok=True)
            self.progress.log_info(f"Создана корневая папка для результатов: {self.root_folder_path}")
        except Exception as e:
            self.progress.log_error(f"Ошибка при создании корневой папки: {e}")
            raise

    def create_task_folder(self, task):
        """Создает папку для задачи и сохраняет путь в задаче"""
        if not self.root_folder_path:
            raise ValueError("Корневая папка не инициализирована")

        # Если папка уже создана для этой задачи, возвращаем существующий путь
        if hasattr(task, 'task_folder_path') and task.task_folder_path:
            return task.task_folder_path

        task_start_time = time.strftime("%d_%m_%Y_%H_%M")
        task_folder_name = f"{task.task_name} {task_start_time}"
        task.task_folder_path = os.path.join(self.root_folder_path, task_folder_name)
        try:
            os.makedirs(task.task_folder_path, exist_ok=True)
            self.progress.log_info(f"Создана папка для задачи {task.task_name}: {task.task_folder_path}")
            return task.task_folder_path
        except Exception as e:
            self.progress.log_error(f"Ошибка при создании папки задачи: {e}")
            raise

    def create_scale_folder(self, scale, task_folder=None):
        if task_folder is None:
            task = self.get_current_task()
            if not task or not hasattr(task, 'task_folder_path') or not task.task_folder_path:
                raise ValueError("Папка задачи не инициализирована")
            task_folder = task.task_folder_path

        scale_folder_path = os.path.join(task_folder, f"Scale_{scale}")
        try:
            os.makedirs(scale_folder_path, exist_ok=True)
            return scale_folder_path
        except Exception as e:
            self.progress.log_error(f"Ошибка при создании папки масштаба: {e}")
            raise

    def save_orig_channels_txt(self, task, print_channels_txt):
        """Сохранение исходных каналов для задачи"""
        self.progress.log_info("Сохранение исходных каналов...")
        if print_channels_txt:
            colors = ['Red', 'Green', 'Blue']
            task_folder = self.create_task_folder(task)  # Создаём папку для задачи
            for channel in range(len(task.data_copy)):
                filename = f"Исходный_цветовой_канал_{colors[channel]}.txt"
                array_2d = task.data_copy[channel]
                file_path = os.path.join(task_folder, filename)
                np.savetxt(file_path, array_2d, fmt='%d', delimiter=",")
                self.progress.log_info(f"Сохранен файл: {file_path}")

    def gram_shmidt_transform_for_task(self, task):
        self.progress.log_info("Применение преобразования Грамма-Шмидта...")
        task.data = change_channels(task.color1, task.color2, task.data)
        self.progress.log_info("Преобразование Грамма-Шмидта завершено")

    def load_scales_for_task(self, task, start, end, step):
        task.scales = np.arange(start=start, stop=end + 1, step=step)
        task.num_scale = task.scales.shape[0]
        self.progress.log_info(f"Загружены масштабы: {len(task.scales)} значений от {start} до {end}")

    def load_scales_from_file_for_task(self, task, filename):
        self.progress.log_info(f"Загрузка масштабов из файла: {filename}")
        task.scales = np.array([])
        with open(filename, 'r') as file_of_scales:
            for line in file_of_scales:
                numbers = [np.double(x) for x in line.split()]
                task.scales = np.append(task.scales, numbers)
        task.scales = np.array(task.scales)
        task.num_scale = len(task.scales)
        self.progress.log_info(f"Загружено {task.num_scale} масштабов")

    def toggle_gpu_backend(self):
        """Переключение между CPU и GPU"""
        success = self.backend.toggle_backend()
        backend_info = self.backend.get_backend_info()
        if success:
            self.progress.log_info(f"Переключено на: {backend_info['device_name']}")
        return success, backend_info

    def get_backend_info(self):
        """Получение информации о бэкенде"""
        return self.backend.get_backend_info()

    def process_channel(self, data, scales):
        """
        Обработка канала с встроенным симметричным отражением на CPU
        """
        rows = data.shape[0]
        cols = data.shape[1]
        scales_size = len(scales)
        result = np.zeros((rows, scales_size, cols))

        self.progress.log_info(f"CPU обработка {rows} строк...")

        with Pool() as pool:
            args = [(data[i], scales) for i in range(rows)]
            results = pool.map(process_row_static, args)

        for i, res in enumerate(results):
            result[i] = res

        return result

    def process_channel_columns(self, data, scales):
        """
        Параллельная обработка столбцов с использованием multiprocessing на CPU
        """
        cols = data.shape[1]
        scales_size = len(scales)
        rows = data.shape[0]

        result_3d = np.zeros((scales_size, cols, rows))

        self.progress.log_info(f"🔹 CPU обработка {cols} столбцов...")

        # Подготавливаем аргументы для каждого столбца
        args = [(col_idx, data[:, col_idx], scales) for col_idx in range(cols)]

        # Обрабатываем столбцы параллельно
        with Pool() as pool:
            results = pool.map(process_column_static, args)

        # Собираем результаты
        for col_idx, column_result in results:
            result_3d[:, col_idx, :] = column_result

        return np.transpose(result_3d, (0, 2, 1))

    def process_channel_batch_gpu(self, data_channel, scales):
        """Батчевая обработка ВСЕГО канала на GPU (используется для обоих backend)"""
        if not self.backend.use_gpu:
            return self.process_channel(data_channel, scales)  # CPU fallback

        try:
            rows, cols = data_channel.shape
            self.progress.log_info(f"Подготовка батча для GPU: {rows} строк × {cols} колонок")

            result = self.gpu_processor.morlet_wavelet_batch(data_channel, scales)  # [rows, scales, cols]
            return result

        except Exception as e:
            self.progress.log_error(f"❌ GPU батчевая обработка не удалась: {e}. Возврат к CPU.")
            return self.process_channel(data_channel, scales)

    def wavelets(self, task, type_data, data_3_channel):
        """Версия для GPU с прогресс-баром"""
        t_compute_wavelet_start = time.time()

        backend_info = "GPU" if self.backend.use_gpu else "CPU"
        direction = "построчно" if type_data == 0 else "по столбцам"

        self.progress.log_info(f"Начало вейвлет-преобразования ({backend_info}, {direction})")

        num_channels = 3
        num_rows = task.data[0].shape[0]
        num_cols = task.data[0].shape[1]

        self.progress.log_info(f"Масштабы: {len(task.scales)}, Строки: {num_rows}, Столбцы: {num_cols}")
        self.progress.log_info(
            f"Общий объем: {num_channels} канала × {num_rows if type_data == 0 else num_cols} направлений")

        total_operations = num_channels
        current_operation = 0

        for channel in range(num_channels):
            channel_name = ['Красный', 'Зеленый', 'Синий'][channel]

            # Обновляем прогресс для каждого канала
            progress = current_operation / total_operations
            self.progress.update_progress(
                progress,
                f"Подготовка канала {channel_name}..."
            )

            data_channel = task.data[channel].astype(np.float64)

            # Вычитание среднего
            insert_filename = "rows" if type_data == 0 else "cols"
            file_mean_path = os.path.join(task.task_folder_path, f'mean_to_{insert_filename}_by_channel_{channel}.txt')

            with open(file_mean_path, 'w') as file:
                for i in range(data_channel.shape[0] if type_data == 0 else data_channel.shape[1]):
                    if type_data == 0:
                        row = data_channel[i]
                        mean = np.mean(row)
                        file.write(str(mean) + "\n")
                        data_channel[i] -= mean
                    else:
                        col = data_channel[:, i]
                        mean = np.mean(col)
                        file.write(str(mean) + "\n")
                        data_channel[:, i] -= mean

            # GPU обработка
            if self.backend.use_gpu:
                self.progress.update_progress(
                    progress + 0.1 / total_operations,
                    f"GPU обработка канала {channel_name}..."
                )

                try:
                    if type_data == 0:
                        self.progress.log_info(f"GPU обработка {num_rows} строк канала {channel_name}...")
                        data_channel_after = self.gpu_processor.morlet_wavelet_batch(data_channel, task.scales)
                        data_channel_after_transposed = np.transpose(data_channel_after, (1, 0, 2))
                    else:
                        self.progress.log_info(f"GPU обработка {num_cols} столбцов канала {channel_name}...")
                        data_channel_transposed = data_channel.T
                        data_channel_after = self.gpu_processor.morlet_wavelet_batch(data_channel_transposed,
                                                                                     task.scales)
                        data_channel_after_transposed = np.transpose(data_channel_after, (1, 2, 0))

                    data_3_channel[channel] = data_channel_after_transposed

                except Exception as e:
                    self.progress.log_error(f"Ошибка GPU: {e}. Переход на CPU...")
                    # Fallback to CPU
                    if type_data == 0:
                        data_channel_after = self.process_channel(data_channel, task.scales)
                        data_channel_after_transposed = np.transpose(data_channel_after, (1, 0, 2))
                    else:
                        data_channel_after = self.process_channel_columns(data_channel, task.scales)
                        data_channel_after_transposed = data_channel_after

                    data_3_channel[channel] = data_channel_after_transposed
            else:
                if type_data == 0:
                    data_channel_after = self.process_channel(data_channel, task.scales)
                    data_channel_after_transposed = np.transpose(data_channel_after, (1, 0, 2))
                else:
                    data_channel_after = self.process_channel_columns(data_channel, task.scales)
                    data_channel_after_transposed = data_channel_after

                data_3_channel[channel] = data_channel_after_transposed
            current_operation += 1
            self.progress.update_progress(
                current_operation / total_operations,
                f"Завершен канал {channel_name}"
            )

        elapsed_time = time.time() - t_compute_wavelet_start
        self.progress.log_info(f"Вейвлет-преобразование завершено за {elapsed_time:.2f} секунд")

        return data_3_channel


    def compute_wavelets(self, task, info_out):
        self.progress.update_progress(0.1, "Подготовка данных для вейвлет-преобразования...")
        # (3, scales, rows, cols)
        data_3_channels = np.zeros((3, task.num_scale, task.data[0].shape[0], task.data[0].shape[1]))
        data_3_channels = self.wavelets(task, 0, data_3_channels)
        task.result.append(data_3_channels)
        self.save_print_wavelets(task, 0, info_out)

        self.progress.update_progress(0.6, "Обработка транспонированных данных...")
        # (3, scales, cols, rows)
        data_3_channels_tr = np.zeros((3, task.num_scale, task.data[0].shape[0], task.data[0].shape[1]))
        data_3_channels_tr = self.wavelets(task, 1, data_3_channels_tr)
        data_3_channels_tr = np.transpose(data_3_channels_tr, (0, 1, 2, 3))
        task.result.append(data_3_channels_tr)
        self.save_print_wavelets(task, 1, info_out)

        self.progress.update_progress(1.0, "Вейвлет-преобразование завершено")

    def save_print_wavelets(self, task, type_data, info_out):
        colors = ['Красный', 'Зелёный', 'Синий']
        type_matrix_str = "построчно" if type_data == 0 else "по_столбцам"

        total_scales = task.num_scale * 3
        current_scale = 0

        if not task:
            self.progress.log_error("Задача не найдена при сохранении вейвлетов")
            return

        task_folder = self.create_task_folder(task)

        for channel in range(3):
            for scale in range(task.num_scale):
                scale_folder_path = self.create_scale_folder(task.scales[scale], task_folder)
                array_2d = task.result[type_data][channel][scale]

                current_scale += 1
                progress = 0.1 + (current_scale / total_scales) * 0.9
                self.progress.update_progress(
                    progress,
                    f"Сохранение результатов: {colors[channel]}, масштаб {task.scales[scale]}")

                if info_out == 0 or info_out == 10:
                    filename = f"Расчет_вейвлетов_{type_matrix_str}_Масштаб_{task.scales[scale]}_{colors[channel]}.txt"
                    file_path = os.path.join(scale_folder_path, filename)
                    np.savetxt(file_path, array_2d, fmt='%.3f', delimiter=",")
                    self.progress.log_debug(f"Сохранен текстовый файл: {file_path}")

                if info_out == 0 or info_out == 1:
                    fig, ax = plt.subplots(figsize=(6, 5))

                    im = ax.imshow(array_2d, cmap='viridis', interpolation='nearest')

                    ax.set_title(f'Wavelets: Scale={task.scales[scale]}, Channel={colors[channel]}')
                    fig.colorbar(im, ax=ax)
                    fig.savefig(
                        os.path.join(
                            scale_folder_path,
                            f'График_расчетов_В_П_{type_matrix_str}_Масштаб_{task.scales[scale]}_{colors[channel]}.png'),
                        dpi=300)

                    plt.close(fig)

    @staticmethod
    def find_extremes(coefs, row_var, col_var, max_var, min_var):
        points_max_by_row = []
        points_min_by_row = []
        points_max_by_column = []
        points_min_by_column = []

        # Экстремумы построчно
        if row_var and (max_var or min_var):
            left = coefs[:, :-2]
            center = coefs[:, 1:-1]
            right = coefs[:, 2:]

            if max_var:
                max_mask = (center > left) & (center > right)
                max_coords = np.where(max_mask)
                points_max_by_row = [[x + 1, y] for y, x in zip(max_coords[0], max_coords[1])]

            if min_var:
                min_mask = (center < left) & (center < right)
                min_coords = np.where(min_mask)
                points_min_by_row = [[x + 1, y] for y, x in zip(min_coords[0], min_coords[1])]

        # Экстремумы по столбцам
        if col_var and (max_var or min_var):
            up = coefs[:-2, :]
            center = coefs[1:-1, :]
            down = coefs[2:, :]

            if max_var:
                max_mask = (center > up) & (center > down)
                max_coords = np.where(max_mask)
                points_max_by_column = [[x, y + 1] for y, x in zip(max_coords[0], max_coords[1])]

            if min_var:
                min_mask = (center < up) & (center < down)
                min_coords = np.where(min_mask)
                points_min_by_column = [[x, y + 1] for y, x in zip(min_coords[0], min_coords[1])]

        return coefs, points_max_by_row, points_max_by_column, points_min_by_row, points_min_by_column

    def compute_points(self, task, row_var, col_var, max_var, min_var,
                       knn_var, knn_bool_text_var, knn_bool_image_var,
                       print_text_var, print_graphic, pipette_state):
        """
        Поиск экстремумов после вейвлет-преобразования.

        Реализовано:
        - используется только построчная матрица Str, потому что анализ идёт вдоль строк изображения;
        - для гистограммы берутся только максимумы верхней огибающей;
        - сохраняются обычные и сжатые гистограммы по блокам масштабов;
        - дополнительно рассчитывается межстрочная синхронизация максимумов верхней огибающей;
        - межстрочная синхронизация считается по бинарным событиям вдоль оси X;
        - масштабы внутри блока объединяются через OR;
        - для каждой пары строк считается метрика Jaccard/Dice/Phi.
        """

        self.progress.update_progress(0.1, "Начало поиска экстремумов...")
        self.progress.log_info("Запущена функция подсчета экстремумов")

        use_gpu_knn = self.backend.use_gpu and self.knn_processor.is_gpu_available()

        if knn_var:
            if use_gpu_knn:
                self.progress.log_info("Использование GPU для KNN вычислений")
            else:
                self.progress.log_info("Использование CPU для KNN вычислений")

        extremes = []

        # Для этой задачи нужен только Str: построчное вейвлет-преобразование.
        channels_count = 1 if pipette_state == 'normal' else 3
        total_operations = channels_count * task.num_scale
        current_operation = 0

        type_data = 0
        direction = "построчно"
        type_matrix_str = "Str"

        channels_to_process = [0] if pipette_state == 'normal' else range(3)
        colors = ['Красный', 'Зелёный', 'Синий']
        channel_codes = ['red', 'green', 'blue']

        # Размеры блоков масштабов для сжатых гистограмм и межстрочной синхронизации.
        # Можно задать в task.scale_block_sizes, например [5].
        scale_block_sizes = getattr(task, 'scale_block_sizes', [5])
        if isinstance(scale_block_sizes, int):
            scale_block_sizes = [scale_block_sizes]

        # Настройки межстрочной синхронизации.
        # row_sync_stride=25 означает сравнивать строки 0, 25, 50, ...
        # tolerance=1 расширяет событие по X на ±1 пиксель, чтобы учитывать небольшой сдвиг.
        row_sync_stride = int(getattr(task, 'row_sync_stride', 1))
        row_sync_tolerance = int(getattr(task, 'row_sync_tolerance', 1))
        row_sync_metrics = getattr(task, 'row_sync_metrics', ['jaccard'])
        if isinstance(row_sync_metrics, str):
            row_sync_metrics = [row_sync_metrics]

        for channel in channels_to_process:
            channel_name = colors[channel]
            channel_code = channel_codes[channel]

            row_hist_scales = []
            row_hist_max_counts = []
            row_index_for_histogram = None

            # Для межстрочной синхронизации сохраняем точки максимумов верхней огибающей
            # отдельно для каждого масштаба.
            upper_points_by_scale = {}

            for scale in range(task.num_scale):
                current_operation += 1

                progress = 0.1 + (current_operation / max(total_operations, 1)) * 0.8
                self.progress.update_progress(
                    progress,
                    f"Поиск экстремумов: {channel_name}, масштаб {task.scales[scale]}, {direction}"
                )

                # Получение коэффициентов вейвлета Str: [channel][scale] -> 2D матрица.
                coefs_2d = task.result[type_data][channel][scale]
                coefs_2d = np.round(coefs_2d, decimals=3)

                # Для построения верхней огибающей нужны максимумы по строкам.
                # Остальные направления оставлены для совместимости с текущим сохранением/KNN.
                if self.backend.use_gpu:
                    coefs_2d, pmaxr, pmaxc, pminr, pminc = ExtremesFinder.find_extremes_gpu(
                        coefs_2d,
                        row_var=row_var,
                        col_var=col_var,
                        max_var=max_var,
                        min_var=min_var
                    )
                else:
                    coefs_2d, pmaxr, pmaxc, pminr, pminc = self.find_extremes(
                        coefs=coefs_2d,
                        row_var=row_var,
                        col_var=col_var,
                        max_var=max_var,
                        min_var=min_var
                    )

                self.progress.log_info(
                    f"Масштаб {task.scales[scale]}, {channel_name}, {direction}: "
                    f"макс_строк={len(pmaxr)}, мин_строк={len(pminr)}, "
                    f"макс_столб={len(pmaxc)}, мин_столб={len(pminc)}"
                )

                interpolator = Interpolator(self.backend)

                upper_max_row_points, lower_min_row_points = interpolator.get_envelopes(
                    coefs_2d,
                    pmaxr,
                    pminr,
                    direction='row'
                )

                upper_max_col_points, lower_min_col_points = interpolator.get_envelopes(
                    coefs_2d,
                    pmaxc,
                    pminc,
                    direction='col'
                )

                # Валидация точек.
                if not isinstance(upper_max_row_points, (list, np.ndarray)) or len(upper_max_row_points) == 0:
                    upper_max_row_points = []
                if not isinstance(lower_min_row_points, (list, np.ndarray)) or len(lower_min_row_points) == 0:
                    lower_min_row_points = []
                if not isinstance(upper_max_col_points, (list, np.ndarray)) or len(upper_max_col_points) == 0:
                    upper_max_col_points = []
                if not isinstance(lower_min_col_points, (list, np.ndarray)) or len(lower_min_col_points) == 0:
                    lower_min_col_points = []

                scale_value = task.scales[scale]
                upper_points_by_scale[scale_value] = upper_max_row_points

                # Гистограмма: считаем только максимумы верхней огибающей на средней строке изображения.
                if row_index_for_histogram is None:
                    row_index_for_histogram = coefs_2d.shape[0] // 2

                upper_count_on_row = self.count_points_on_row(
                    upper_max_row_points,
                    row_index_for_histogram
                )

                row_hist_scales.append(scale_value)
                row_hist_max_counts.append(upper_count_on_row)

                self.progress.log_info(
                    f"Масштаб {scale_value}, {channel_name}, "
                    f"строка y={row_index_for_histogram}: "
                    f"максимумов верхней огибающей={upper_count_on_row}"
                )

                # Подготовка данных для стандартного сохранения точек экстремумов.
                extremes_to_process = []
                titles = []

                if max_var:
                    if row_var:
                        extremes_to_process.append(upper_max_row_points)
                        titles.append(
                            f"{type_matrix_str}_Точки_максимума_по_строкам_масштаб_{scale_value}_{channel_name}"
                        )
                    if col_var:
                        extremes_to_process.append(upper_max_col_points)
                        titles.append(
                            f"{type_matrix_str}_Точки_максимума_по_cтолбцам_масштаб_{scale_value}_{channel_name}"
                        )

                if min_var:
                    if row_var:
                        extremes_to_process.append(lower_min_row_points)
                        titles.append(
                            f"{type_matrix_str}_Точки_минимума_по_строкам_масштаб_{scale_value}_{channel_name}"
                        )
                    if col_var:
                        extremes_to_process.append(lower_min_col_points)
                        titles.append(
                            f"{type_matrix_str}_Точки_минимума_по_cтолбцам_масштаб_{scale_value}_{channel_name}"
                        )

                scale_folder = self.find_scale_folder(task, scale_value)

                for i, p in enumerate(extremes_to_process):
                    if len(p) > 0:
                        if print_text_var:
                            self.save_extremes_to_file(scale_folder, titles[i], p)
                            self.progress.log_debug(f"Сохранен текстовый файл: {titles[i]}.txt")

                        if print_graphic:
                            self.save_extremes_graphic(
                                scale_folder,
                                titles[i],
                                p,
                                coefs_2d=coefs_2d
                            )
                            self.progress.log_debug(
                                f"Сохранен график экстремумов огибающей: {titles[i]}.png"
                            )

                # Подготовка данных для KNN. Оставлено для совместимости с текущей архитектурой.
                knn_extremes = {
                    'type_data': type_data,
                    'channel': channel,
                    'scale': scale_value,
                    'max_by_row': upper_max_row_points if (row_var and max_var) else [],
                    'max_by_column': upper_max_col_points if (col_var and max_var) else [],
                    'min_by_row': lower_min_row_points if (row_var and min_var) else [],
                    'min_by_column': lower_min_col_points if (col_var and min_var) else []
                }
                extremes.append(knn_extremes)

                if knn_bool_text_var or knn_bool_image_var:
                    self.progress.update_progress(0.85, "Обработка KNN...")
                    process_extremes_with_knn(
                        knn_extremes,
                        scale_folder,
                        knn_var,
                        task.original_image,
                        knn_bool_text_var,
                        knn_bool_image_var,
                        use_gpu=use_gpu_knn,
                        progress_callback=self.progress.update_progress,
                        log_callback=self.progress.log_info
                    )

            # После всех масштабов текущего канала сохраняем итоговую гистограмму.
            if len(row_hist_scales) > 0:
                histogram_dir = os.path.join(
                    task.task_folder_path,
                    "Гистограммы_экстремумов_огибающих"
                )

                histogram_file_base = (
                    f"hist_upper_envelope_maxima_row_y_{row_index_for_histogram}_{channel_code}"
                )

                saved_png_path, saved_csv_path = self.save_upper_envelope_maxima_row_histogram(
                    histogram_dir,
                    histogram_file_base,
                    row_hist_scales,
                    row_hist_max_counts,
                    row_index_for_histogram,
                    channel_name=channel_name
                )

                if saved_png_path:
                    self.progress.log_info(
                        f"Гистограмма максимумов верхней огибающей сохранена: {saved_png_path}"
                    )
                else:
                    self.progress.log_error(
                        f"Не удалось сохранить PNG-гистограмму максимумов верхней огибающей для канала {channel_name}"
                    )

                if saved_csv_path:
                    self.progress.log_info(
                        f"CSV с данными гистограммы сохранён: {saved_csv_path}"
                    )

                # Сжатые гистограммы по блокам масштабов.
                for block_size in scale_block_sizes:
                    try:
                        block_size = int(block_size)
                    except (TypeError, ValueError):
                        self.progress.log_error(
                            f"Некорректный размер блока масштабов: {block_size}"
                        )
                        continue

                    if block_size <= 1:
                        continue

                    block_labels, block_counts, block_ranges = self.compress_scale_counts_to_blocks(
                        row_hist_scales,
                        row_hist_max_counts,
                        block_size
                    )

                    block_file_base = (
                        f"hist_upper_envelope_maxima_row_y_{row_index_for_histogram}_"
                        f"{channel_code}_blocks_{block_size}"
                    )

                    block_png_path, block_csv_path = self.save_upper_envelope_maxima_block_histogram(
                        histogram_dir,
                        block_file_base,
                        block_labels,
                        block_counts,
                        block_ranges,
                        row_index_for_histogram,
                        block_size,
                        channel_name=channel_name
                    )

                    if block_png_path:
                        self.progress.log_info(
                            f"Сжатая гистограмма по блокам масштабов сохранена: {block_png_path}"
                        )
                    else:
                        self.progress.log_error(
                            f"Не удалось сохранить сжатую гистограмму для канала {channel_name}, "
                            f"размер блока={block_size}"
                        )

                    if block_csv_path:
                        self.progress.log_info(
                            f"CSV с данными сжатой гистограммы сохранён: {block_csv_path}"
                        )

                # Межстрочная синхронизация.
                # Сравниваем бинарные ряды разных строк по X внутри блоков масштабов.
                if len(upper_points_by_scale) > 0:
                    image_height, image_width = task.original_image.shape[:2]

                    safe_stride = max(1, row_sync_stride)
                    rows_to_analyze = list(range(0, image_height, safe_stride))
                    middle_row = image_height // 2
                    if middle_row not in rows_to_analyze:
                        rows_to_analyze.append(middle_row)
                        rows_to_analyze = sorted(rows_to_analyze)

                    sync_dir = os.path.join(
                        task.task_folder_path,
                        "Синхронизация_строк"
                    )

                    sync_title = f"row_sync_upper_envelope_maxima_{channel_code}"

                    for block_size in scale_block_sizes:
                        try:
                            block_size = int(block_size)
                        except (TypeError, ValueError):
                            continue

                        if block_size <= 0:
                            continue

                        for metric_name in row_sync_metrics:
                            self.calculate_row_synchronization_for_scale_block(
                                path=sync_dir,
                                title=sync_title,
                                points_by_scale=upper_points_by_scale,
                                scales=row_hist_scales,
                                rows_to_analyze=rows_to_analyze,
                                image_width=image_width,
                                block_size=block_size,
                                tolerance=row_sync_tolerance,
                                metric_name=metric_name,
                                channel_name=channel_name
                            )

        self.progress.update_progress(0.95, "Завершение поиска экстремумов...")

        total_points = sum(
            len(extreme['max_by_row']) + len(extreme['max_by_column']) +
            len(extreme['min_by_row']) + len(extreme['min_by_column'])
            for extreme in extremes
        )

        self.progress.log_info(f"Всего найдено точек экстремумов: {total_points}")
        self.progress.update_progress(1.0, "Поиск экстремумов завершен")
        return extremes

    @staticmethod
    def save_extremes_to_file(path, title, local_points):
        """Сохранение точек экстремумов в файл"""
        if not local_points:
            print(f"Нет точек для сохранения в {title}")
            return

        file_path = os.path.join(path, f"{title}.txt")
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                for point in local_points:
                    file.write(f"{point[0]}, {point[1]}\n")
            print(f"Файл сохранён: {file_path}")
        except Exception as e:
            print(f"Ошибка при сохранении файла {file_path}: {str(e)}")

    # @staticmethod
    # def save_extremes_graphic(path, title, points_local, original_img_shape):
    #     """Сохранение графиков точек экстремумов"""
    #     if not points_local:
    #         print(f"Нет точек для отображения: {title}")
    #         return

    #     try:
    #         plt.figure(figsize=(10, 10))

    #         data = np.array(points_local)
    #         x = data[:, 0]
    #         y = data[:, 1]

    #         # оси с сохранением пропорций
    #         ax = plt.gca()

    #         if original_img_shape is not None:
    #             height, width = original_img_shape[:2]
    #             ax.set_xlim(0, width)
    #             ax.set_ylim(height, 0)  # инвертируем ось Y
    #             ax.set_aspect('equal')  # фиксируем соотношение сторон 1:1

    #         # рисуем точки
    #         plt.scatter(x, y, s=1, alpha=0.6)
    #         plt.title(title)

    #         plt.grid(True)
    #         plt.xlabel('X (пиксели)')
    #         plt.ylabel('Y (пиксели)')

    #         filename = os.path.join(path, f"{title}.png")
    #         plt.savefig(filename, bbox_inches='tight', dpi=96)
    #         plt.close()
    #         print(f"График сохранён: {filename}")

    #     except Exception as e:
    #         print(f"Ошибка при сохранении графика {title}: {str(e)}")
    @staticmethod
    def save_extremes_graphic(path, title, points_local, coefs_2d=None, original_img_shape=None):
        """
        Сохранение изображения с точками экстремумов огибающих.

        Если передана матрица coefs_2d, точки накладываются на результат
        вейвлет-преобразования. Если coefs_2d не передана, рисуется только
        поле с точками.
        """
        if points_local is None or len(points_local) == 0:
            print(f"Нет точек для отображения: {title}")
            return

        try:
            points = np.asarray(points_local)

            if points.ndim != 2 or points.shape[1] != 2:
                print(f"Некорректный формат точек для {title}: {points.shape}")
                return

            x = points[:, 0]
            y = points[:, 1]

            fig, ax = plt.subplots(figsize=(10, 10))

            if coefs_2d is not None:
                background = np.asarray(coefs_2d)

                # Для контрастной визуализации ограничиваем выбросы
                vmin, vmax = np.percentile(background, [1, 99])

                ax.imshow(
                    background,
                    cmap='gray',
                    interpolation='nearest',
                    vmin=vmin,
                    vmax=vmax
                )

                height, width = background.shape[:2]
                ax.set_xlim(0, width)
                ax.set_ylim(height, 0)
            else:
                if original_img_shape is not None:
                    height, width = original_img_shape[:2]
                    ax.set_xlim(0, width)
                    ax.set_ylim(height, 0)

            ax.scatter(
                x,
                y,
                s=6,
                c='red',
                marker='o',
                alpha=0.85,
                linewidths=0
            )

            ax.set_title(title)
            ax.set_xlabel('X, пиксели')
            ax.set_ylabel('Y, пиксели')
            ax.set_aspect('equal')
            ax.grid(False)

            filename = os.path.join(path, f"{title}.png")
            plt.savefig(filename, bbox_inches='tight', dpi=150)
            plt.close(fig)

            print(f"График экстремумов огибающей сохранён: {filename}")

        except Exception as e:
            print(f"Ошибка при сохранении графика {title}: {str(e)}")

    @staticmethod
    def count_points_on_row(points, row_index):
        """
        Считает количество точек экстремумов, лежащих на заданной строке изображения.

        points: список точек в формате [(x, y), ...]
        row_index: номер строки y
        """
        if points is None or len(points) == 0:
            return 0

        points = np.asarray(points)

        if points.ndim != 2 or points.shape[1] != 2:
            return 0

        y = points[:, 1]

        return int(np.sum(y == row_index))

    @staticmethod
    def save_upper_envelope_maxima_row_histogram(
            path,
            file_base_name,
            scales,
            max_counts,
            row_index,
            channel_name=None
    ):
        """
        Строит и сохраняет гистограмму распределения количества максимумов
        верхней огибающей на выбранной строке изображения в зависимости от масштаба.

        Возвращает:
            (png_path, csv_path)
        """
        png_path = None
        csv_path = None

        try:
            if scales is None or len(scales) == 0:
                print(f"Нет данных для построения гистограммы: {file_base_name}")
                return None, None

            os.makedirs(path, exist_ok=True)

            scales = np.asarray(scales)
            max_counts = np.asarray(max_counts)

            if len(scales) != len(max_counts):
                raise ValueError(
                    f"Размеры scales и max_counts не совпадают: "
                    f"{len(scales)} != {len(max_counts)}"
                )

            # Дополнительно сохраняем численные данные, чтобы можно было проверить график.
            csv_path = os.path.join(path, f"{file_base_name}.csv")
            with open(csv_path, 'w', encoding='utf-8') as file:
                file.write("scale,upper_envelope_maxima_count,row_index\n")
                for scale, count in zip(scales, max_counts):
                    file.write(f"{scale},{int(count)},{row_index}\n")

            x = np.arange(len(scales))

            fig, ax = plt.subplots(figsize=(14, 6))

            ax.bar(
                x,
                max_counts,
                label='Максимумы верхней огибающей'
            )

            title_channel = f"Канал: {channel_name}" if channel_name else ""
            ax.set_title(
                "Распределение максимумов верхней огибающей по масштабам\n"
                f"{title_channel}, строка изображения y = {row_index}"
            )
            ax.set_xlabel("Масштаб вейвлет-преобразования")
            ax.set_ylabel("Количество максимумов верхней огибающей")

            ax.set_xticks(x)
            ax.set_xticklabels([str(s) for s in scales], rotation=90)

            ax.grid(axis='y', alpha=0.3)
            ax.legend()

            png_path = os.path.join(path, f"{file_base_name}.png")
            fig.savefig(png_path, bbox_inches='tight', dpi=150)
            plt.close(fig)

            # Проверяем, что файл реально появился на диске.
            if not os.path.exists(png_path):
                raise FileNotFoundError(f"PNG-файл не был создан: {png_path}")

            if os.path.getsize(png_path) == 0:
                raise IOError(f"PNG-файл создан, но он пустой: {png_path}")

            return png_path, csv_path

        except Exception as e:
            print(f"Ошибка при построении гистограммы {file_base_name}: {str(e)}")
            try:
                plt.close('all')
            except Exception:
                pass
            return None, csv_path

    @staticmethod
    def compress_scale_counts_to_blocks(scales, counts, block_size):
        """
        Сжимает последовательность масштабов в блоки фиксированного размера.

        Пример:
            scales = [5, 6, 7, 8, 9, 10]
            counts = [25, 18, 15, 12, 15, 10]
            block_size = 3

        Результат:
            block_labels = ['5-7', '8-10']
            block_counts = [58, 37]
            block_ranges = [(5, 7, [5, 6, 7]), (8, 10, [8, 9, 10])]

        Значение блока — сумма количества максимумов верхней огибающей
        на масштабах, входящих в этот блок.
        """
        if scales is None or counts is None:
            return [], [], []

        scales = list(scales)
        counts = [int(c) for c in counts]

        if len(scales) != len(counts):
            raise ValueError(
                f"Размеры scales и counts не совпадают: {len(scales)} != {len(counts)}"
            )

        if block_size <= 0:
            raise ValueError(f"Размер блока должен быть положительным, получено: {block_size}")

        block_labels = []
        block_counts = []
        block_ranges = []

        for start_idx in range(0, len(scales), block_size):
            end_idx = min(start_idx + block_size, len(scales))

            block_scales = scales[start_idx:end_idx]
            block_values = counts[start_idx:end_idx]

            if not block_scales:
                continue

            scale_start = block_scales[0]
            scale_end = block_scales[-1]
            block_sum = int(np.sum(block_values))

            if scale_start == scale_end:
                label = str(scale_start)
            else:
                label = f"{scale_start}-{scale_end}"

            block_labels.append(label)
            block_counts.append(block_sum)
            block_ranges.append((scale_start, scale_end, block_scales))

        return block_labels, block_counts, block_ranges

    @staticmethod
    def save_upper_envelope_maxima_block_histogram(
            path,
            file_base_name,
            block_labels,
            block_counts,
            block_ranges,
            row_index,
            block_size,
            channel_name=None
    ):
        """
        Сохраняет сжатую гистограмму по блокам масштабов.

        Каждый столбец соответствует диапазону масштабов.
        Высота столбца — сумма количества максимумов верхней огибающей
        по всем масштабам внутри блока.

        Возвращает:
            (png_path, csv_path)
        """
        png_path = None
        csv_path = None

        try:
            if block_labels is None or len(block_labels) == 0:
                print(f"Нет данных для построения сжатой гистограммы: {file_base_name}")
                return None, None

            os.makedirs(path, exist_ok=True)

            block_counts = [int(c) for c in block_counts]

            if len(block_labels) != len(block_counts):
                raise ValueError(
                    f"Размеры block_labels и block_counts не совпадают: "
                    f"{len(block_labels)} != {len(block_counts)}"
                )

            csv_path = os.path.join(path, f"{file_base_name}.csv")
            with open(csv_path, 'w', encoding='utf-8') as file:
                file.write(
                    "block_label,block_start_scale,block_end_scale,"
                    "scales_in_block,upper_envelope_maxima_sum,row_index,block_size\n"
                )

                for label, count, block_range in zip(block_labels, block_counts, block_ranges):
                    scale_start, scale_end, block_scales = block_range
                    scales_as_text = "|".join(str(s) for s in block_scales)
                    file.write(
                        f"{label},{scale_start},{scale_end},"
                        f"{scales_as_text},{int(count)},{row_index},{block_size}\n"
                    )

            x = np.arange(len(block_labels))

            fig, ax = plt.subplots(figsize=(14, 6))

            ax.bar(
                x,
                block_counts,
                label=f'Сумма максимумов в блоке, размер блока = {block_size}'
            )

            title_channel = f"Канал: {channel_name}" if channel_name else ""
            ax.set_title(
                "Сжатое распределение максимумов верхней огибающей по блокам масштабов\n"
                f"{title_channel}, строка изображения y = {row_index}, "
                f"размер блока = {block_size}"
            )
            ax.set_xlabel("Блок масштабов")
            ax.set_ylabel("Суммарное количество максимумов верхней огибающей")

            ax.set_xticks(x)
            ax.set_xticklabels(block_labels, rotation=90)

            ax.grid(axis='y', alpha=0.3)
            ax.legend()

            png_path = os.path.join(path, f"{file_base_name}.png")
            fig.savefig(png_path, bbox_inches='tight', dpi=150)
            plt.close(fig)

            if not os.path.exists(png_path):
                raise FileNotFoundError(f"PNG-файл не был создан: {png_path}")

            if os.path.getsize(png_path) == 0:
                raise IOError(f"PNG-файл создан, но он пустой: {png_path}")

            return png_path, csv_path

        except Exception as e:
            print(f"Ошибка при построении сжатой гистограммы {file_base_name}: {str(e)}")
            try:
                plt.close('all')
            except Exception:
                pass
            return None, csv_path


    @staticmethod
    def build_binary_row_from_points(points, row_index, width, tolerance=0):
        """
        Строит бинарный ряд длины width для одной строки изображения.

        1 означает, что в позиции x есть максимум верхней огибающей.
        tolerance расширяет событие по X на несколько пикселей влево/вправо.
        """
        binary = np.zeros(width, dtype=np.uint8)

        if points is None or len(points) == 0:
            return binary

        points = np.asarray(points)

        if points.ndim != 2 or points.shape[1] != 2:
            return binary

        row_mask = points[:, 1].astype(int) == int(row_index)
        x_values = points[row_mask][:, 0].astype(int)

        for x in x_values:
            if x < 0 or x >= width:
                continue
            left = max(0, x - tolerance)
            right = min(width, x + tolerance + 1)
            binary[left:right] = 1

        return binary

    @staticmethod
    def calculate_binary_sync_metrics(binary_a, binary_b):
        """
        Считает метрики синхронизации между двумя бинарными рядами.
        """
        binary_a = np.asarray(binary_a).astype(bool)
        binary_b = np.asarray(binary_b).astype(bool)

        if binary_a.shape != binary_b.shape:
            raise ValueError(
                f"Размеры бинарных рядов не совпадают: {binary_a.shape} != {binary_b.shape}"
            )

        n11 = int(np.sum(binary_a & binary_b))
        n10 = int(np.sum(binary_a & ~binary_b))
        n01 = int(np.sum(~binary_a & binary_b))
        n00 = int(np.sum(~binary_a & ~binary_b))

        denom_jaccard = n11 + n10 + n01
        jaccard = n11 / denom_jaccard if denom_jaccard > 0 else 0.0

        denom_dice = 2 * n11 + n10 + n01
        dice = (2 * n11) / denom_dice if denom_dice > 0 else 0.0

        denom_phi = np.sqrt(
            (n11 + n10) *
            (n01 + n00) *
            (n11 + n01) *
            (n10 + n00)
        )
        phi = ((n11 * n00) - (n10 * n01)) / denom_phi if denom_phi > 0 else 0.0

        return {
            "n11": n11,
            "n10": n10,
            "n01": n01,
            "n00": n00,
            "jaccard": float(jaccard),
            "dice": float(dice),
            "phi": float(phi)
        }

    @staticmethod
    def save_row_sync_heatmap(path, file_base_name, sync_matrix, rows, metric_name, block_label, channel_name=None):
        """
        Сохраняет heatmap синхронизации между строками изображения.
        """
        try:
            os.makedirs(path, exist_ok=True)

            sync_matrix = np.asarray(sync_matrix, dtype=float)
            rows = list(rows)

            fig, ax = plt.subplots(figsize=(10, 8))

            if metric_name == "phi":
                vmin, vmax = -1, 1
                cmap = "coolwarm"
            else:
                vmin, vmax = 0, 1
                cmap = "viridis"

            im = ax.imshow(
                sync_matrix,
                cmap=cmap,
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax
            )

            title_channel = f"Канал: {channel_name}" if channel_name else ""
            ax.set_title(
                f"Межстрочная синхронизация максимумов верхней огибающей\n"
                f"{title_channel}, блок масштабов {block_label}, метрика: {metric_name}"
            )
            ax.set_xlabel("Строка изображения")
            ax.set_ylabel("Строка изображения")

            max_ticks = 20
            tick_step = max(1, len(rows) // max_ticks)

            tick_positions = np.arange(0, len(rows), tick_step)
            tick_labels = [str(rows[i]) for i in tick_positions]

            ax.set_xticks(tick_positions)
            ax.set_yticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=90)
            ax.set_yticklabels(tick_labels)

            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(metric_name)

            png_path = os.path.join(path, f"{file_base_name}.png")
            fig.savefig(png_path, bbox_inches="tight", dpi=150)
            plt.close(fig)

            if not os.path.exists(png_path):
                raise FileNotFoundError(f"PNG-файл не был создан: {png_path}")
            if os.path.getsize(png_path) == 0:
                raise IOError(f"PNG-файл создан, но он пустой: {png_path}")

            return png_path

        except Exception as e:
            print(f"Ошибка при сохранении heatmap синхронизации строк {file_base_name}: {e}")
            try:
                plt.close('all')
            except Exception:
                pass
            return None

    @staticmethod
    def save_row_sync_matrix_csv(path, file_base_name, sync_matrix, rows):
        """
        Сохраняет матрицу синхронизации строк в CSV.
        """
        try:
            os.makedirs(path, exist_ok=True)
            csv_path = os.path.join(path, f"{file_base_name}.csv")

            sync_matrix = np.asarray(sync_matrix, dtype=float)
            rows = list(rows)

            with open(csv_path, "w", encoding="utf-8") as file:
                file.write("row," + ",".join(str(r) for r in rows) + "\n")
                for row_value, matrix_row in zip(rows, sync_matrix):
                    values = ",".join(f"{float(v):.6f}" for v in matrix_row)
                    file.write(f"{row_value},{values}\n")

            return csv_path

        except Exception as e:
            print(f"Ошибка при сохранении CSV матрицы синхронизации строк {file_base_name}: {e}")
            return None

    @staticmethod
    def save_row_sync_pair_metrics_csv(path, file_base_name, pair_rows):
        """
        Сохраняет подробную таблицу метрик по всем парам строк.
        """
        try:
            os.makedirs(path, exist_ok=True)
            csv_path = os.path.join(path, f"{file_base_name}_pairs.csv")

            with open(csv_path, "w", encoding="utf-8") as file:
                file.write(
                    "row_a,row_b,n11,n10,n01,n00,jaccard,dice,phi\n"
                )
                for item in pair_rows:
                    file.write(
                        f"{item['row_a']},{item['row_b']},"
                        f"{item['n11']},{item['n10']},{item['n01']},{item['n00']},"
                        f"{item['jaccard']:.6f},{item['dice']:.6f},{item['phi']:.6f}\n"
                    )

            return csv_path

        except Exception as e:
            print(f"Ошибка при сохранении подробной CSV таблицы пар строк {file_base_name}: {e}")
            return None

    def calculate_row_synchronization_for_scale_block(
            self,
            path,
            title,
            points_by_scale,
            scales,
            rows_to_analyze,
            image_width,
            block_size=5,
            tolerance=1,
            metric_name="jaccard",
            channel_name=None
    ):
        """
        Рассчитывает межстрочную синхронизацию максимумов верхней огибающей.

        Для каждого блока масштабов:
        1. Для каждой строки строится бинарный ряд по X.
        2. События внутри блока масштабов объединяются через OR.
        3. Для каждой пары строк строится таблица сопряжённости n11/n10/n01/n00.
        4. Сохраняется heatmap выбранной метрики и CSV-таблицы.
        """
        try:
            os.makedirs(path, exist_ok=True)

            scales = list(scales)
            rows_to_analyze = sorted(set(int(r) for r in rows_to_analyze))
            metric_name = str(metric_name).lower()

            if metric_name not in {"jaccard", "dice", "phi"}:
                self.progress.log_error(
                    f"Неизвестная метрика синхронизации: {metric_name}. Использую jaccard."
                )
                metric_name = "jaccard"

            if len(scales) == 0 or len(rows_to_analyze) == 0:
                self.progress.log_error("Нет данных для расчёта межстрочной синхронизации")
                return []

            results = []

            for block_start in range(0, len(scales), block_size):
                block_scales = scales[block_start:block_start + block_size]
                if len(block_scales) == 0:
                    continue

                block_label = f"{block_scales[0]}-{block_scales[-1]}" if block_scales[0] != block_scales[-1] else str(block_scales[0])

                # Для каждой строки строим бинарный ряд по X, объединяя масштабы блока через OR.
                row_binary_events = {}

                for row in rows_to_analyze:
                    block_binary = np.zeros(image_width, dtype=np.uint8)

                    for scale_value in block_scales:
                        points = points_by_scale.get(scale_value, [])
                        row_binary = self.build_binary_row_from_points(
                            points,
                            row_index=row,
                            width=image_width,
                            tolerance=tolerance
                        )
                        block_binary = np.logical_or(block_binary, row_binary).astype(np.uint8)

                    row_binary_events[row] = block_binary

                n_rows = len(rows_to_analyze)
                sync_matrix = np.zeros((n_rows, n_rows), dtype=float)
                pair_rows = []

                for i, row_a in enumerate(rows_to_analyze):
                    for j, row_b in enumerate(rows_to_analyze):
                        metrics = self.calculate_binary_sync_metrics(
                            row_binary_events[row_a],
                            row_binary_events[row_b]
                        )

                        # Диагональ принудительно равна 1 для читаемой heatmap.
                        value = 1.0 if i == j else metrics.get(metric_name, 0.0)
                        sync_matrix[i, j] = value

                        pair_rows.append({
                            "row_a": row_a,
                            "row_b": row_b,
                            **metrics
                        })

                safe_block_label = str(block_label).replace('-', '_')
                file_base_name = (
                    f"{title}_block_{safe_block_label}_size_{block_size}_{metric_name}"
                )

                heatmap_path = self.save_row_sync_heatmap(
                    path,
                    file_base_name,
                    sync_matrix,
                    rows_to_analyze,
                    metric_name,
                    block_label,
                    channel_name=channel_name
                )

                matrix_csv_path = self.save_row_sync_matrix_csv(
                    path,
                    file_base_name,
                    sync_matrix,
                    rows_to_analyze
                )

                pairs_csv_path = self.save_row_sync_pair_metrics_csv(
                    path,
                    file_base_name,
                    pair_rows
                )

                if heatmap_path:
                    self.progress.log_info(f"Heatmap межстрочной синхронизации сохранён: {heatmap_path}")
                else:
                    self.progress.log_error(
                        f"Не удалось сохранить heatmap межстрочной синхронизации: {file_base_name}"
                    )

                if matrix_csv_path:
                    self.progress.log_info(f"CSV матрицы межстрочной синхронизации сохранён: {matrix_csv_path}")

                if pairs_csv_path:
                    self.progress.log_info(f"CSV парных метрик межстрочной синхронизации сохранён: {pairs_csv_path}")

                results.append({
                    "block_label": block_label,
                    "block_scales": block_scales,
                    "metric_name": metric_name,
                    "rows": rows_to_analyze,
                    "matrix": sync_matrix,
                    "heatmap_path": heatmap_path,
                    "matrix_csv_path": matrix_csv_path,
                    "pairs_csv_path": pairs_csv_path
                })

            return results

        except Exception as e:
            self.progress.log_error(f"Ошибка при расчёте межстрочной синхронизации: {e}")
            return []

    def find_scale_folder(self, task, scale):
        """Находит папку масштаба для текущей задачи"""
        if not task or not hasattr(task, 'task_folder_path') or not task.task_folder_path:
            self.progress.log_error("Папка задачи не инициализирована")
            return None

        scale_folder_name = f"Scale_{scale}"
        scale_folder_path = os.path.join(task.task_folder_path, scale_folder_name)
        if os.path.exists(scale_folder_path) and os.path.isdir(scale_folder_path):
            return scale_folder_path + "\\"
        else:
            print(f"Directory {scale_folder_path} is not found")
            return None

    def compute_for_task(self, task, wp_var1, wp_var2, print_channels_txt_var, row_var, col_var, max_var, min_var,
                         p_ex_var1, p_ex_var2, knn_bool_text_var, knn_bool_image_var, pipette_state):
        """Выполнение вычислений для конкретной задачи"""
        try:
            task_folder = self.create_task_folder(task)
            self.progress.log_info(f"Создана папка для {task.task_name}: {task_folder}")

            # Сохранение исходных каналов - 5%
            self.progress.update_progress(0.05, "Сохранение исходных каналов...")
            self.save_orig_channels_txt(task, print_channels_txt_var)

            task.result = []
            info_out = self._get_output_type(wp_var1.get(), wp_var2.get())

            # Вейвлет-преобразование - 60%
            self.progress.update_progress(0.1, "Начало вейвлет-преобразования...")
            self.compute_wavelets(task, info_out)

            # Поиск экстремумов - 30%
            self.progress.update_progress(0.7, "Начало поиска экстремумов...")
            if p_ex_var1.get() or p_ex_var2.get():
                self.compute_points(task, row_var.get(), col_var.get(), max_var.get(), min_var.get(),
                                    task.k_neighbors, knn_bool_text_var.get(), knn_bool_image_var.get(),
                                    p_ex_var1.get(), p_ex_var2.get(), pipette_state)

            # Завершение - 5%
            self.progress.update_progress(1.0, f"Вычисления для {task.task_name} завершены успешно")
            self.progress.log_info(f"Вычисления для {task.task_name} завершены успешно")

        except Exception as e:
            import traceback
            error_msg = f"Ошибка при обработке задачи {task.task_name}: {str(e)}\n{traceback.format_exc()}"
            self.progress.log_error(error_msg)
            raise
        finally:
            # Очищаем временные данные в задаче после вычислений
            task.result = []
            # self.clear_gpu_cache()

    @staticmethod
    def _get_output_type(wp_var1, wp_var2):
        """Определяет тип вывода на основе настроек"""
        if wp_var1 and wp_var2:
            return 0  # Оба формата
        elif wp_var1 and not wp_var2:
            return 1  # Только изображения
        elif not wp_var1 and wp_var2:
            return 10  # Только текстовые файлы
        else:
            return 11  # Ничего


class App(TkinterApp):
    def __init__(self):
        super().__init__()
        self._compute_thread = None
        self.title("Wavelets - Professional Edition")
        self.resizable(True, True)
        self.geometry(f"{self.winfo_screenwidth()}x{self.winfo_screenheight() - 40}+0+0")

        # настройка темы
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # инициализация всех атрибутов UI
        self._initialize_ui_variables()

        # создаем главный контейнер с тремя панелями
        self.main_container = ctk.CTkFrame(self)
        self.main_container.pack(fill="both", expand=True, padx=5, pady=5)
        self.main_container.grid_columnconfigure(0, weight=1)  # Левая панель
        self.main_container.grid_columnconfigure(1, weight=1)  # Центральная панель
        self.main_container.grid_columnconfigure(2, weight=1)  # Правая панель
        self.main_container.grid_rowconfigure(0, weight=1)

        # панель управления задачами
        self.tasks_panel = self._create_tasks_panel()
        self.tasks_panel.grid(row=0, column=0, sticky="nsew", padx=2)
        # настройки ввода
        self.left_panel = self._create_left_panel()
        self.left_panel.grid(row=0, column=1, sticky="nsew", padx=(0, 2))
        # настройки вывода
        self.right_panel = self._create_right_panel()
        self.right_panel.grid(row=0, column=2, sticky="nsew", padx=(2, 0))

        self.progress_manager = ProgressManager(self)

        # менеджер изображений
        self.image_processor = ImageProcessor(self.progress_manager)

        # текущая задача
        self.current_task = None
        self.knn_text_var.trace('w', self.update_knn_for_current_task)

        # Переменная для GPU/CPU переключения
        self.use_gpu_var = tk.BooleanVar(value=True)

        # Обновляем UI после инициализации image_processor
        self.after(100, self._update_gpu_section)

        # ждем создания всех виджетов и затем максимизируем
        self.after(100, self._maximize_properly)

    def _update_gpu_section(self):
        """Обновление секции GPU после инициализации image_processor"""
        if hasattr(self, 'compute_settings_section') and self.compute_settings_section:
            # Очищаем секцию
            for widget in self.compute_settings_section.content.winfo_children():
                widget.destroy()

            # Пересоздаем содержимое
            self._setup_compute_settings_section()

    def _maximize_properly(self):
        """Правильная максимизация после создания всех виджетов"""
        # Обновляем геометрию
        self.update_idletasks()

        # Максимизируем
        if self.tk.call('tk', 'windowingsystem') == 'win32':
            self.state('zoomed')
        else:
            self.attributes('-zoomed', True)

        # Принудительное обновление layout
        self.update()
        self.after(200, self._final_adjustment)

    def _final_adjustment(self):
        """Финальная корректировка размеров"""
        self.update_idletasks()

        # Принудительно обновляем все области с перемоткой
        for child in self.winfo_children():
            if hasattr(child, 'update_scrollbar'):
                child.update_scrollbar()

        self.update()

    def _create_main_container(self):
        """Создание главного контейнера с четким разделением на верхнюю и нижнюю части"""
        # Главный контейнер - используем вертикальное расположение pack
        self.main_container = ctk.CTkFrame(self)
        self.main_container.pack(fill="both", expand=True, padx=5, pady=5)

        # Верхняя часть - 3 панели (будет занимать 85% пространства)
        self.panels_frame = ctk.CTkFrame(self.main_container)
        self.panels_frame.pack(fill="both", expand=True, pady=(0, 5))

        # Нижняя часть - прогресс (фиксированная высота, 15% пространства)
        # ProgressManager автоматически создаст свой frame и упакует его

    def _create_ui(self):
        """Создание всех UI элементов с четким разделением"""
        # ==================== ВЕРХНЯЯ ЧАСТЬ - 3 ПАНЕЛИ ====================
        self._create_top_panels()

        # ==================== НИЖНЯЯ ЧАСТЬ - ПРОГРЕСС ====================
        self._setup_bottom_progress()

    def _create_top_panels(self):
        """Создание 3 панелей в верхней части"""
        # Настройка сетки для 3 панелей - равномерное распределение
        self.panels_frame.grid_columnconfigure(0, weight=1)  # Левая панель - задачи
        self.panels_frame.grid_columnconfigure(1, weight=1)  # Центральная панель - ввод
        self.panels_frame.grid_columnconfigure(2, weight=1)  # Правая панель - вывод
        self.panels_frame.grid_rowconfigure(0, weight=1)  # Одна строка

        # 1. Левая панель - управление задачами
        self.tasks_panel = self._create_tasks_panel()
        self.tasks_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 2))

        # 2. Центральная панель - настройки ввода
        self.left_panel = self._create_left_panel()
        self.left_panel.grid(row=0, column=1, sticky="nsew", padx=2)

        # 3. Правая панель - настройки вывода
        self.right_panel = self._create_right_panel()
        self.right_panel.grid(row=0, column=2, sticky="nsew", padx=(2, 0))

    def _setup_bottom_progress(self):
        """Настройка нижней панели прогресса"""
        # Переупаковываем фрейм прогресса в нижнюю часть главного контейнера.
        # Устанавливаем фиксированную высоту и запрещаем изменение размера
        self.progress_manager.frame.configure(height=180)  # Фиксированная высота
        self.progress_manager.frame.pack_propagate(False)  # Запрещаем авто-изменение размера

        # Упаковываем вниз главного контейнера
        self.progress_manager.frame.pack(side="bottom", fill="x", padx=0, pady=0)

    def _initialize_ui_variables(self):
        """Инициализация всех переменных UI"""
        self.row_var = tk.BooleanVar(value=True)
        self.col_var = tk.BooleanVar(value=True)
        self.max_var = tk.BooleanVar(value=True)
        self.min_var = tk.BooleanVar(value=True)
        self.wp_var1 = tk.BooleanVar(value=True)
        self.wp_var2 = tk.BooleanVar(value=False)
        self.p_ex_var1 = tk.BooleanVar(value=False)
        self.p_ex_var2 = tk.BooleanVar(value=True)
        self.knn_bool_text_var = tk.BooleanVar(value=False)
        self.knn_bool_image_var = tk.BooleanVar(value=False)
        self.print_channels_txt_var = tk.BooleanVar(value=False)

        self.data = tk.StringVar()
        self.knn_text_var = tk.StringVar(value="0")

        # Widget references
        self.load_button = None
        self.print_load_image = None
        self.pipette_button = None
        self.gram_shmidt_button = None
        self.entry_start = None
        self.entry_end = None
        self.entry_step = None
        self.button_save_scales = None
        self.label_custom_scale = None
        self.button_load_scales_file = None
        self.entry_near_point = None
        self.app_start_button = None
        self.task_widgets = []
        self.compute_settings_section = None
        self.gpu_checkbox = None

    def _create_tasks_panel(self):
        """Создание панели управления задачами"""
        panel = ctk.CTkFrame(self.main_container)

        # Заголовок панели
        header = ctk.CTkLabel(
            panel,
            text="📋 Управление задачами",
            font=ctk.CTkFont(size=16, weight="bold"),
            anchor="w"  # Выравнивание текста слева
        )
        header.pack(fill="x", padx=10, pady=(10, 15))

        # Кнопка добавления задачи
        self.add_task_btn = ctk.CTkButton(
            panel,
            text="➕ Добавить задачу",
            command=self.add_new_task,
            height=40,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#28a745",
            hover_color="#218838"
        )
        self.add_task_btn.pack(fill="x", padx=10, pady=(0, 10))

        # Фрейм для списка задач с прокруткой
        tasks_scrollable = ScrollableFrame(panel)
        tasks_scrollable.pack(fill="both", expand=True, padx=10, pady=5)

        # Заголовок списка задач
        tasks_label = ctk.CTkLabel(
            tasks_scrollable.scrollable_frame,
            text="Список задач:",
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w"
        )
        tasks_label.pack(fill="x", padx=10, pady=(10, 5))

        # Контейнер для задач
        self.tasks_container = ctk.CTkFrame(tasks_scrollable.scrollable_frame, fg_color="transparent")
        self.tasks_container.pack(fill="x", padx=10, pady=5, expand=True)

        # Статус задач
        self.tasks_status_label = ctk.CTkLabel(
            tasks_scrollable.scrollable_frame,
            text="Задачи не добавлены",
            font=ctk.CTkFont(size=11),
            text_color="gray",
            anchor="w"
        )
        self.tasks_status_label.pack(fill="x", padx=10, pady=(5, 10))

        return panel

    def _create_left_panel(self):
        """Создание левой панели с настройками ввода"""
        panel = ctk.CTkFrame(self.main_container)

        # Используем ScrollableFrame для прокрутки
        scrollable_panel = ScrollableFrame(panel)
        scrollable_panel.pack(fill="both", expand=True)

        # Заголовок панели
        header = ctk.CTkLabel(
            scrollable_panel.scrollable_frame,
            text="Настройки ввода и обработки",
            font=ctk.CTkFont(size=16, weight="bold"),
            anchor="w"
        )
        header.pack(fill="x", padx=10, pady=(10, 15))

        # Секция загрузки изображения
        self.load_section = CollapsibleFrame(scrollable_panel.scrollable_frame, title="📁 Загрузка изображения")
        self.load_section.pack(fill="x", padx=5, pady=2)
        self._setup_load_section()

        # Секция работы с каналами
        self.channel_section = CollapsibleFrame(scrollable_panel.scrollable_frame, title="🎨 Работа с каналами")
        self.channel_section.pack(fill="x", padx=5, pady=2)
        self._setup_channel_section()

        # Секция масштабов
        self.scales_section = CollapsibleFrame(scrollable_panel.scrollable_frame, title="📏 Настройка масштабов")
        self.scales_section.pack(fill="x", padx=5, pady=2)
        self._setup_scales_section()

        # Секция точек экстремумов
        self.extremes_section = CollapsibleFrame(scrollable_panel.scrollable_frame, title="📊 Точки экстремумов")
        self.extremes_section.pack(fill="x", padx=5, pady=2)
        self._setup_extremes_section()

        return panel

    def _create_right_panel(self):
        """Создание правой панели с настройками вывода"""
        panel = ctk.CTkFrame(self.main_container)

        # Используем ScrollableFrame для прокрутки
        scrollable_panel = ScrollableFrame(panel)
        scrollable_panel.pack(fill="both", expand=True)

        # Заголовок панели
        header = ctk.CTkLabel(
            scrollable_panel.scrollable_frame,
            text="Настройки вывода и вычислений",
            font=ctk.CTkFont(size=16, weight="bold"),
            anchor="w"
        )
        header.pack(fill="x", padx=10, pady=(10, 15))

        self.compute_settings_section = CollapsibleFrame(scrollable_panel.scrollable_frame,
                                                         title="Настройки вычислений")
        self.compute_settings_section.pack(fill="x", padx=5, pady=2)
        temp_label = ctk.CTkLabel(
            self.compute_settings_section.content,
            text="Загрузка настроек...",
            font=ctk.CTkFont(size=11),
            text_color="gray")
        temp_label.pack(pady=10)

        self.wavelet_section = CollapsibleFrame(scrollable_panel.scrollable_frame, title="Вейвлет-преобразование")
        self.wavelet_section.pack(fill="x", padx=5, pady=2)
        self._setup_wavelet_section()

        # Секция точек экстремумов (вывод)
        self.output_extremes_section = CollapsibleFrame(scrollable_panel.scrollable_frame,
                                                        title="Вывод точек экстремумов")
        self.output_extremes_section.pack(fill="x", padx=5, pady=2)
        self._setup_output_extremes_section()

        # Секция K-ближайших соседей
        self.knn_section = CollapsibleFrame(scrollable_panel.scrollable_frame, title="K-ближайшие соседи")
        self.knn_section.pack(fill="x", padx=5, pady=2)
        self._setup_knn_section()

        # Секция промежуточных вычислений
        self.intermediate_section = CollapsibleFrame(scrollable_panel.scrollable_frame,
                                                     title="Промежуточные вычисления")
        self.intermediate_section.pack(fill="x", padx=5, pady=2)
        self._setup_intermediate_section()

        # Кнопка вычислений
        self.compute_section = ctk.CTkFrame(scrollable_panel.scrollable_frame, fg_color="transparent")
        self.compute_section.pack(fill="x", padx=5, pady=20)
        self._setup_compute_section()

        return panel

    def _setup_compute_settings_section(self):
        """Настройка секции параметров вычислений"""
        # Очищаем секцию
        for widget in self.compute_settings_section.content.winfo_children():
            widget.destroy()

        # Проверяем, что image_processor инициализирован
        if not hasattr(self, 'image_processor') or self.image_processor is None:
            error_label = ctk.CTkLabel(
                self.compute_settings_section.content,
                text="Ошибка инициализации процессора",
                font=ctk.CTkFont(size=11),
                text_color="red"
            )
            error_label.pack(pady=10)
            return

        # Информация о бэкенде
        backend_info = self.image_processor.get_backend_info()
        info_text = f"Устройство: {backend_info['device_name']}"
        if backend_info['use_gpu'] and backend_info['gpu_memory'] != "N/A":
            info_text += f" | Память: {backend_info['gpu_memory']}"

        info_label = ctk.CTkLabel(
            self.compute_settings_section.content,
            text=info_text,
            font=ctk.CTkFont(size=11),
            text_color="lightblue",
            anchor="w"
        )
        info_label.pack(fill="x", pady=(0, 10))

        # Переключатель CPU/GPU
        gpu_available = self.image_processor.backend.is_gpu_available()

        self.gpu_checkbox = ctk.CTkCheckBox(
            self.compute_settings_section.content,
            text="Использовать GPU (ускорение)",
            variable=self.use_gpu_var,
            command=self.toggle_gpu_backend,
            state="normal" if gpu_available else "disabled"
        )
        # Устанавливаем начальное значение
        self.use_gpu_var.set(self.image_processor.backend.use_gpu)
        self.gpu_checkbox.pack(fill="x", pady=5)

        if not gpu_available:
            warning_label = ctk.CTkLabel(
                self.compute_settings_section.content,
                text="GPU не обнаружен. Установите CuPy для CUDA поддержки.",
                font=ctk.CTkFont(size=10),
                text_color="orange",
                anchor="w"
            )
            warning_label.pack(fill="x", pady=(5, 0))
        else:
            status_label = ctk.CTkLabel(
                self.compute_settings_section.content,
                text="GPU доступен для вычислений",
                font=ctk.CTkFont(size=10),
                text_color="green",
                anchor="w"
            )
            status_label.pack(fill="x", pady=(5, 0))

            # Выбор режима точности GPU
            accuracy_frame = ctk.CTkFrame(self.compute_settings_section.content, fg_color="transparent")
            accuracy_frame.pack(fill="x", pady=5)

            accuracy_label = ctk.CTkLabel(
                accuracy_frame,
                text="Режим точности GPU:",
                font=ctk.CTkFont(size=11, weight="bold"),
                anchor="w"
            )
            accuracy_label.pack(fill="x")

            self.accuracy_var = tk.StringVar(value="balanced")

            accuracy_options = [
                ("Высокая точность", "high"),
                ("Сбалансированный", "balanced"),
                ("Высокая скорость", "fast")
            ]

            for text, mode in accuracy_options:
                rb = ctk.CTkRadioButton(
                    accuracy_frame,
                    text=text,
                    variable=self.accuracy_var,
                    value=mode,
                    command=self.on_accuracy_mode_changed
                )
                rb.pack(anchor="w", pady=2)

    def on_accuracy_mode_changed(self):
        """Обработчик изменения режима точности"""
        if hasattr(self, 'image_processor') and self.image_processor.gpu_processor:
            self.image_processor.gpu_processor.set_accuracy_mode(self.accuracy_var.get())
            self.progress_manager.log_info(f"Режим точности GPU изменен на: {self.accuracy_var.get()}")


    def toggle_gpu_backend(self):
        """Переключение между CPU и GPU"""
        success, backend_info = self.image_processor.toggle_gpu_backend()

        # Обновляем UI
        if success:
            new_state = "GPU" if backend_info['use_gpu'] else "CPU"
            self.progress_manager.log_info(f"Переключено на вычисления на {new_state}")

            # Обновляем информацию в UI
            self._update_compute_settings_display()


    def _update_compute_settings_display(self):
        """Обновление отображения настроек вычислений"""
        if not hasattr(self, 'image_processor') or self.image_processor is None:
            return

        knn_gpu_available = self.image_processor.knn_processor.is_gpu_available()

        # Обновляем информацию о KNN
        knn_info = f"KNN: {'GPU' if knn_gpu_available else 'CPU'}"
        if hasattr(self, 'knn_status_label'):
            self.knn_status_label.configure(
                text=knn_info,
                text_color="lightgreen" if knn_gpu_available else "orange"
            )

    def _setup_load_section(self):
        """Настройка секции загрузки изображения"""
        self.load_button = ctk.CTkButton(
            self.load_section.content,
            text="Загрузить и обрезать изображение",
            command=self.load_image_callback,
            height=40,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#2b5b84",
            hover_color="#1e4160"
        )
        self.load_section.add_widget(self.load_button, pady=5)

        self.print_load_image = ctk.CTkLabel(
            self.load_section.content,
            text="Изображение не загружено",
            font=ctk.CTkFont(size=11),
            text_color="gray",
            anchor="w",
            wraplength=0  # Отключаем перенос текста
        )
        self.load_section.add_widget(self.print_load_image, pady=(0, 5))

    def _setup_channel_section(self):
        """Настройка секции работы с каналами"""
        # Пипетка
        pipette_frame = ctk.CTkFrame(self.channel_section.content, fg_color="transparent")
        self.channel_section.add_widget(pipette_frame, pady=2)

        pipette_label = ctk.CTkLabel(
            pipette_frame,
            text="Выбор цветовых каналов:",
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w"
        )
        pipette_label.pack(fill="x")

        self.pipette_button = ctk.CTkButton(
            pipette_frame,
            text="Активировать пипетку",
            command=self.pipette_channel,
            height=35,
            font=ctk.CTkFont(size=12)
        )
        pipette_frame.pack(fill="x")
        self.pipette_button.pack(fill="x", pady=(5, 0))

        # Преобразование Грамма-Шмидта
        gram_frame = ctk.CTkFrame(self.channel_section.content, fg_color="transparent")
        self.channel_section.add_widget(gram_frame, pady=(10, 2))

        gram_label = ctk.CTkLabel(
            gram_frame,
            text="Преобразование каналов:",
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w"
        )
        gram_label.pack(fill="x")

        self.gram_shmidt_button = ctk.CTkButton(
            gram_frame,
            text="Применить Грамма-Шмидта",
            command=self.gramm_shmidt_transform,
            height=35,
            font=ctk.CTkFont(size=12)
        )
        gram_frame.pack(fill="x")
        self.gram_shmidt_button.pack(fill="x", pady=(5, 0))

    def _setup_scales_section(self):
        """Настройка секции масштабов"""
        # Поля ввода
        input_frame = ctk.CTkFrame(self.scales_section.content, fg_color="transparent")
        self.scales_section.add_widget(input_frame, pady=2)

        # От
        start_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        start_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(start_frame, text="От:", width=40, anchor="w").pack(side="left")
        self.entry_start = ctk.CTkEntry(start_frame, placeholder_text="1")
        self.entry_start.pack(side="left", fill="x", expand=True)

        # До
        end_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        end_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(end_frame, text="До:", width=40, anchor="w").pack(side="left")
        self.entry_end = ctk.CTkEntry(end_frame, placeholder_text="10")
        self.entry_end.pack(side="left", fill="x", expand=True)

        # Шаг
        step_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        step_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(step_frame, text="Шаг:", width=40, anchor="w").pack(side="left")
        self.entry_step = ctk.CTkEntry(step_frame, placeholder_text="1")
        self.entry_step.pack(side="left", fill="x", expand=True)

        # Кнопки
        button_frame = ctk.CTkFrame(self.scales_section.content, fg_color="transparent")
        self.scales_section.add_widget(button_frame, pady=(10, 2))

        self.button_save_scales = ctk.CTkButton(
            button_frame,
            text="Сохранить значения",
            command=self.load_scales,
            height=35
        )
        self.button_save_scales.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.button_load_scales_file = ctk.CTkButton(
            button_frame,
            text="Загрузить из файла",
            command=self.load_scales_from_file,
            height=35
        )
        self.button_load_scales_file.pack(side="left", fill="x", expand=True, padx=(5, 0))

        # Отображение загруженных масштабов
        self.label_custom_scale = ctk.CTkLabel(
            self.scales_section.content,
            text="",
            font=ctk.CTkFont(size=10),
            text_color="gray",
            anchor="w",
            wraplength=0
        )
        self.scales_section.add_widget(self.label_custom_scale, pady=(5, 0))

    def _setup_extremes_section(self):
        """Настройка секции точек экстремумов"""
        # Направления поиска
        direction_frame = ctk.CTkFrame(self.extremes_section.content, fg_color="transparent")
        self.extremes_section.add_widget(direction_frame, pady=2)

        direction_label = ctk.CTkLabel(
            direction_frame,
            text="Направления поиска:",
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w"
        )
        direction_label.pack(fill="x")

        directions_subframe = ctk.CTkFrame(direction_frame, fg_color="transparent")
        directions_subframe.pack(fill="x", pady=5)

        self.row_checkbox = ctk.CTkCheckBox(
            directions_subframe,
            text="По строкам",
            variable=self.row_var
        )
        self.row_checkbox.pack(side="left", padx=(0, 10))

        self.col_checkbox = ctk.CTkCheckBox(
            directions_subframe,
            text="По столбцам",
            variable=self.col_var
        )
        self.col_checkbox.pack(side="left")

        # Типы экстремумов
        type_frame = ctk.CTkFrame(self.extremes_section.content, fg_color="transparent")
        self.extremes_section.add_widget(type_frame, pady=2)

        type_label = ctk.CTkLabel(
            type_frame,
            text="Типы экстремумов:",
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w"
        )
        type_label.pack(fill="x")

        types_subframe = ctk.CTkFrame(type_frame, fg_color="transparent")
        types_subframe.pack(fill="x", pady=5)

        self.max_checkbox = ctk.CTkCheckBox(
            types_subframe,
            text="Максимумы",
            variable=self.max_var
        )
        self.max_checkbox.pack(side="left", padx=(0, 10))

        self.min_checkbox = ctk.CTkCheckBox(
            types_subframe,
            text="Минимумы",
            variable=self.min_var
        )
        self.min_checkbox.pack(side="left")

        # K-ближайшие соседи
        knn_frame = ctk.CTkFrame(self.extremes_section.content, fg_color="transparent")
        self.extremes_section.add_widget(knn_frame, pady=(10, 2))

        knn_label = ctk.CTkLabel(
            knn_frame,
            text="Количество ближайших точек:",
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w"
        )
        knn_label.pack(fill="x")

        self.entry_near_point = ctk.CTkEntry(
            knn_frame,
            textvariable=self.knn_text_var,
            placeholder_text="5"
        )
        self.entry_near_point.pack(fill="x", pady=(5, 0))
        self.entry_near_point.bind("<Button-1>", self.on_entry_click)

    def add_new_task(self):
        try:
            # Создаем новую задачу
            new_task = ProcessingTask()
            task_id = self.image_processor.add_task(new_task)
            # устанавливаем как текущую
            self.current_task = new_task
            self.image_processor.set_current_task(task_id)
            # обновляем интерфейс окна
            self._update_ui_for_current_task()
            self._update_tasks_display()

            self.progress_manager.log_info(f"Добавлена новая задача #{task_id}")

        except Exception as e:
            self.progress_manager.log_error(f"Ошибка при добавлении задачи: {e}")
            mb.showerror("Ошибка", f"Не удалось добавить задачу: {e}")

    def _update_tasks_display(self):
        """Обновление отображения списка задач"""
        # Очищаем контейнер задач
        for widget in self.task_widgets:
            try:
                widget.destroy()
            except Exception as e:
                print(f"Ошибка при удалении виджета задачи: {e}")
        self.task_widgets.clear()

        # Обновляем статус
        task_count = len(self.image_processor.tasks)
        if task_count == 0:
            self.tasks_status_label.configure(
                text="Задачи не добавлены",
                text_color="gray"
            )
        else:
            status_text = f"Всего задач: {task_count}"
            active_tasks = sum(1 for task in self.image_processor.tasks
                               if task.image_path and len(task.scales) > 0)
            if active_tasks > 0:
                status_text += f" | Готовы к вычислениям: {active_tasks}"

            self.tasks_status_label.configure(
                text=status_text,
                text_color="white"
            )

        # Создаем виджеты для каждой задачи
        for task in self.image_processor.tasks:
            self._create_task_widget(task)

    def _create_task_widget(self, task):
        """Создание виджета для отображения задачи"""
        # Основной фрейм задачи
        task_frame = ctk.CTkFrame(
            self.tasks_container,
            border_width=1,
            border_color="#3a3a3a",
            corner_radius=8
        )
        task_frame.pack(fill="x", padx=5, pady=3)
        self.task_widgets.append(task_frame)

        # Верхняя часть - заголовок и статус
        header_frame = ctk.CTkFrame(task_frame, fg_color="transparent")
        header_frame.pack(fill="x", padx=12, pady=(10, 5))

        # Заголовок задачи с номером и статусом
        title_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        title_frame.pack(fill="x")

        # Номер задачи и название
        task_title = ctk.CTkLabel(
            title_frame,
            text=f"{task.task_name}",
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w"
        )
        task_title.pack(side="left", fill="x", expand=True)

        # Статус задачи
        status_text = self._get_task_status(task)
        status_color = self._get_status_color(status_text)
        task_status = ctk.CTkLabel(
            title_frame,
            text=status_text,
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=status_color
        )
        task_status.pack(side="right", padx=(5, 0))

        # Основная информация о задаче
        info_frame = ctk.CTkFrame(task_frame, fg_color="transparent")
        info_frame.pack(fill="x", padx=12, pady=(0, 8))

        # Первая строка информации
        row1_frame = ctk.CTkFrame(info_frame, fg_color="transparent")
        row1_frame.pack(fill="x", pady=2)

        # Информация об изображении
        if task.image_path:
            image_info = f"{os.path.basename(task.image_path)}"
            if hasattr(task, 'original_image') and task.original_image is not None:
                height, width = task.original_image.shape[:2]
                image_info += f" ({width}×{height}px)"
        else:
            image_info = "Изображение не загружено"

        image_label = ctk.CTkLabel(
            row1_frame,
            text=image_info,
            font=ctk.CTkFont(size=11),
            anchor="w"
        )
        image_label.pack(side="left", fill="x", expand=True)

        # Вторая строка информации
        row2_frame = ctk.CTkFrame(info_frame, fg_color="transparent")
        row2_frame.pack(fill="x", pady=2)

        # Информация о масштабах
        scales_info = self._get_scales_info(task)
        scales_label = ctk.CTkLabel(
            row2_frame,
            text=f"{scales_info}",
            font=ctk.CTkFont(size=11),
            anchor="w"
        )
        scales_label.pack(side="left", fill="x", expand=True)

        # Третья строка информации
        row3_frame = ctk.CTkFrame(info_frame, fg_color="transparent")
        row3_frame.pack(fill="x", pady=2)

        # Информация о настройках обработки
        processing_info = self._get_processing_info(task)
        processing_label = ctk.CTkLabel(
            row3_frame,
            text=f"{processing_info}",
            font=ctk.CTkFont(size=11),
            anchor="w"
        )
        processing_label.pack(side="left", fill="x", expand=True)

        # Кнопки управления задачей
        button_frame = ctk.CTkFrame(task_frame, fg_color="transparent")
        button_frame.pack(fill="x", padx=12, pady=(5, 10))

        def make_task_active():
            self.current_task = task
            self.image_processor.set_current_task(task.task_id)
            self._update_ui_for_current_task()
            self._update_tasks_display()

        def remove_task():
            self.image_processor.remove_task(task.task_id)
            if self.current_task and self.current_task.task_id == task.task_id:
                self.current_task = None
            self._update_tasks_display()
            self._update_ui_for_current_task()

        # Кнопка активации
        activate_btn = ctk.CTkButton(
            button_frame,
            text="Активировать",
            command=make_task_active,
            width=90,
            height=28,
            font=ctk.CTkFont(size=11),
            fg_color="#2b5b84",
            hover_color="#1e4160"
        )
        activate_btn.pack(side="left", padx=(0, 8))

        # Кнопка удаления
        remove_btn = ctk.CTkButton(
            button_frame,
            text="Удалить",
            command=remove_task,
            width=70,
            height=28,
            font=ctk.CTkFont(size=11),
            fg_color="#dc3545",
            hover_color="#c82333"
        )
        remove_btn.pack(side="left")

        # Подсветка текущей активной задачи
        if self.current_task and self.current_task.task_id == task.task_id:
            task_frame.configure(border_color="#007bff", border_width=2)

    @staticmethod
    def _get_task_status(task):
        """Получить текстовый статус задачи"""
        if not task.image_path:
            return "Ожидает изображение"
        elif len(task.scales) == 0:
            return "Ожидает масштабы"
        elif task.color1 is not None and task.color2 is not None:
            return "Готова к вычислениям"
        else:
            return "Готова к настройке"

    @staticmethod
    def _get_status_color(status):
        """Получить цвет статуса"""
        status_colors = {
            "Ожидает изображение": "#ffc107",  # желтый
            "Ожидает масштабы": "#ffc107",  # желтый
            "Готова к настройке": "#17a2b8",  # голубой
            "Готова к вычислениям": "#28a745"  # зеленый
        }
        return status_colors.get(status, "#6c757d")  # серый по умолчанию

    @staticmethod
    def _get_scales_info(task):
        """Получить информацию о масштабах в читаемом формате"""
        if len(task.scales) == 0:
            return "Масштабы не заданы"

        if len(task.scales) <= 5:
            # Для небольшого количества показываем все масштабы
            scales_str = ", ".join(map(str, task.scales))
            return f"Масштабы: {scales_str}. Кол-во: {len(task.scales)}"
        else:
            # Для большого количества показываем диапазон
            min_scale = min(task.scales)
            max_scale = max(task.scales)
            return f"Масштабы: от {min_scale} до {max_scale}. Кол-во: {len(task.scales)}"

    def _get_processing_info(self, task):
        """Получить информацию о настройках обработки"""
        info_parts = [f"KNN: {task.k_neighbors}"]

        # Информация о цветовых каналах
        if task.color1 is not None and task.color2 is not None:
            info_parts.append("Пипетка настроена")

            # Проверяем, применено ли преобразование Грамма-Шмидта
            # если кнопка Грамма-Шмидта отключена, значит преобразование применено
            if hasattr(self, 'gram_shmidt_button') and self.gram_shmidt_button.cget('state') == 'disabled':
                info_parts.append("Грамм-Шмидт применен")

        return " • ".join(info_parts) if info_parts else "Базовые настройки"

    def _update_ui_for_current_task(self):
        """Обновление UI в соответствии с текущей задачей"""
        if self.current_task:
            # Обновляем информацию о загруженном изображении
            if self.current_task.image_path:
                self.load_button.configure(
                    text="Изображение загружено",
                    fg_color="#28a745",
                    hover_color="#218838"
                )
                text = f"Файл: {os.path.basename(self.current_task.image_path)}"
                self.print_load_image.configure(text=text, text_color="white")
            else:
                self.load_button.configure(
                    text="Загрузить и обрезать изображение",
                    fg_color="#2b5b84",
                    hover_color="#1e4160"
                )
                self.print_load_image.configure(text="Изображение не загружено", text_color="gray")

            # Проверяем наличие цветов для пипетки
            has_colors = (self.current_task.color1 is not None and
                          self.current_task.color2 is not None and
                          isinstance(self.current_task.color1, np.ndarray) and
                          isinstance(self.current_task.color2, np.ndarray) and
                          self.current_task.color1.size > 0 and
                          self.current_task.color2.size > 0)

            if has_colors:
                self.pipette_button.configure(
                    text="Пипетка активирована",
                    state='disabled',
                    fg_color="#6c757d",
                    hover_color="#5a6268"
                )
                # Активируем кнопку Грамма-Шмидта если есть цвета
                self.gram_shmidt_button.configure(
                    text="Применить Грамма-Шмидта",  # Всегда сбрасываем текст
                    state='normal',
                    fg_color="#2b5b84",
                    hover_color="#1e4160"
                )
            else:
                self.pipette_button.configure(
                    text="Активировать пипетку",
                    state='normal',
                    fg_color="#2b5b84",
                    hover_color="#1e4160"
                )
                # Деактивируем кнопку Грамма-Шмидта если нет цветов
                self.gram_shmidt_button.configure(
                    text="Применить Грамма-Шмидта",  # Всегда сбрасываем текст
                    state='disabled',
                    fg_color="#6c757d",
                    hover_color="#5a6268"
                )

            # Обновляем KNN
            self.knn_text_var.set(str(self.current_task.k_neighbors))

            # Сбрасываем состояние кнопок масштабов для новой задачи
            if len(self.current_task.scales) == 0:
                self.button_save_scales.configure(
                    text="Сохранить значения",
                    fg_color="#2b5b84",
                    hover_color="#1e4160",
                    state='normal'
                )
                self.button_load_scales_file.configure(
                    text="Загрузить из файла",
                    fg_color="#2b5b84",
                    hover_color="#1e4160",
                    state='normal'
                )
                self.label_custom_scale.configure(text="", text_color="gray")

                # Включаем поля ввода
                self.entry_start.configure(state='normal')
                self.entry_end.configure(state='normal')
                self.entry_step.configure(state='normal')
            else:
                # Если масштабы уже загружены, показываем это
                if len(self.current_task.scales) <= 5:
                    scales_text = f"Масштабы: {', '.join(map(str, self.current_task.scales))}"
                else:
                    scales_text = f"Загружено {len(self.current_task.scales)} масштабов"
                self.label_custom_scale.configure(text=scales_text, text_color="white")

        else:
            # Сбрасываем UI если нет активной задачи
            self.load_button.configure(
                text="Загрузить и обрезать изображение",
                fg_color="#2b5b84",
                hover_color="#1e4160"
            )
            self.print_load_image.configure(text="Изображение не загружено", text_color="gray")
            self.pipette_button.configure(
                text="Активировать пипетку",
                state='normal',
                fg_color="#2b5b84",
                hover_color="#1e4160"
            )
            self.gram_shmidt_button.configure(
                text="Применить Грамма-Шмидта",
                state='disabled',
                fg_color="#6c757d",
                hover_color="#5a6268"
            )
            self.button_save_scales.configure(
                text="Сохранить значения",
                fg_color="#2b5b84",
                hover_color="#1e4160",
                state='disabled'
            )
            self.button_load_scales_file.configure(
                text="Загрузить из файла",
                fg_color="#2b5b84",
                hover_color="#1e4160",
                state='disabled'
            )
            self.label_custom_scale.configure(text="", text_color="gray")

        # Всегда обновляем отображение задач для подсветки активной
        self._update_tasks_display()

    # Обновляем методы загрузки изображения и работы с каналами для работы с текущей задачей
    def load_image_callback(self):
        """Обработчик загрузки изображения для текущей задачи"""
        if not self.current_task:
            mb.showwarning("Внимание", "Сначала создайте задачу")
            return

        try:
            if self.image_processor.load_image_for_task(self.current_task, self):
                self._update_ui_for_current_task()
                self._update_tasks_display()
            else:
                self.load_button.configure(
                    text="Ошибка загрузки",
                    fg_color="#dc3545",
                    hover_color="#c82333"
                )
                self.print_load_image.configure(text="Ошибка загрузки изображения", text_color="red")
        except Exception as e:
            self.progress_manager.log_error(f"Ошибка при загрузке изображения: {e}")
            mb.showerror("Ошибка", f"Не удалось загрузить изображение: {e}")

    def pipette_channel(self):
        """Обработчик выбора пипетки для текущей задачи"""
        if not self.current_task:
            mb.showwarning("Внимание", "Сначала создайте задачу")
            return

        if not self.current_task.image_path:
            mb.showwarning("Внимание", "Сначала загрузите изображение")
            return

        self.image_processor.pipette_channel_for_task(self.current_task)
        self._update_ui_for_current_task()
        self._update_tasks_display()

    def gramm_shmidt_transform(self):
        """Обработчик преобразования Грамма-Шмидта для текущей задачи"""
        if not self.current_task:
            mb.showwarning("Внимание", "Сначала создайте задачу")
            return

        if (self.current_task.color1 is None or
                self.current_task.color2 is None or
                not isinstance(self.current_task.color1, np.ndarray) or
                not isinstance(self.current_task.color2, np.ndarray) or
                self.current_task.color1.size == 0 or
                self.current_task.color2.size == 0):
            mb.showwarning("Внимание", "Сначала выберите цвета пипеткой")
            return

        self.image_processor.gram_shmidt_transform_for_task(self.current_task)
        self.gram_shmidt_button.configure(
            text="Преобразование применено",
            state='disabled',
            fg_color="#6c757d",
            hover_color="#5a6268"
        )

        # Обновляем отображение задач, чтобы показать новый статус
        self._update_tasks_display()

    def load_scales(self):
        """Загрузка масштабов для текущей задачи"""
        if not self.current_task:
            mb.showwarning("Внимание", "Сначала создайте задачу")
            return
        try:
            start = int(self.entry_start.get() or "1")
            end = int(self.entry_end.get() or "10")
            step = int(self.entry_step.get() or "1")
            self.image_processor.load_scales_for_task(self.current_task, start, end, step)
            self.button_save_scales.configure(
                text="Масштабы сохранены",
                fg_color="#28a745",
                hover_color="#218838"
            )
            self.button_load_scales_file.configure(state='disabled')

            scale_info = f"Масштабы: {start}-{end} (шаг {step})"
            self.label_custom_scale.configure(text=scale_info, text_color="white")

            self._update_tasks_display()

        except ValueError as e:
            mb.showerror("Ошибка", f"Пожалуйста, введите корректные числовые значения: {e}")

    def load_scales_from_file(self):
        """Загрузка масштабов из файла для текущей задачи"""
        if not self.current_task:
            mb.showwarning("Внимание", "Сначала создайте задачу")
            return

        # Временно блокируем поля ввода
        self.entry_start.configure(state='disabled')
        self.entry_step.configure(state='disabled')
        self.entry_end.configure(state='disabled')

        filetypes = (
            ('Text files', '*.txt'),
            ('All files', '*.*')
        )
        filename = tk.filedialog.askopenfilename(
            title='Выберите файл с масштабами',
            initialdir='/',
            filetypes=filetypes
        )

        if filename:
            self.image_processor.load_scales_from_file_for_task(self.current_task, filename)

            if self.current_task.num_scale <= 10:
                scales_text = f"Масштабы: {', '.join(map(str, self.current_task.scales))}"
            else:
                scales_text = f"Загружено {self.current_task.num_scale} масштабов"

            self.label_custom_scale.configure(text=scales_text, text_color="white")

            self.button_load_scales_file.configure(
                text="Файл загружен",
                fg_color="#28a745",
                hover_color="#218838"
            )
            self.button_save_scales.configure(
                text="Сохранить значения",
                fg_color="#6c757d",
                hover_color="#5a6268",
                state='disabled'
            )

            self._update_tasks_display()
        else:
            # Если файл не выбран, разблокируем поля ввода
            self.entry_start.configure(state='normal')
            self.entry_step.configure(state='normal')
            self.entry_end.configure(state='normal')

    def on_entry_click(self, event=None):
        """Обработчик клика по полю ввода KNN"""
        if self.entry_near_point.get() == "5":
            self.entry_near_point.delete(0, ctk.END)

    def update_knn_for_current_task(self, *args):
        """Обновление KNN для текущей задачи при изменении поля"""
        if self.current_task and self.knn_text_var.get().isdigit():
            self.current_task.k_neighbors = int(self.knn_text_var.get())
            self._update_tasks_display()

    def _setup_wavelet_section(self):
        """Настройка секции вейвлет-преобразования"""
        info_label = ctk.CTkLabel(
            self.wavelet_section.content,
            text="Формат вывода результатов вейвлет-преобразования:",
            font=ctk.CTkFont(size=12),
            anchor="w",
            wraplength=0
        )
        self.wavelet_section.add_widget(info_label, pady=(0, 10))

        self.wp1_checkbox = ctk.CTkCheckBox(
            self.wavelet_section.content,
            text="Вывести изображением",
            variable=self.wp_var1
        )
        self.wavelet_section.add_widget(self.wp1_checkbox, fill="x")

        self.wp2_checkbox = ctk.CTkCheckBox(
            self.wavelet_section.content,
            text="Вывести текстовым файлом",
            variable=self.wp_var2
        )
        self.wavelet_section.add_widget(self.wp2_checkbox, fill="x")

    def _setup_output_extremes_section(self):
        """Настройка секции вывода точек экстремумов"""
        info_label = ctk.CTkLabel(
            self.output_extremes_section.content,
            text="Формат вывода точек экстремумов:",
            font=ctk.CTkFont(size=12),
            anchor="w",
            wraplength=0
        )
        self.output_extremes_section.add_widget(info_label, pady=(0, 10))

        self.p_ex2_checkbox = ctk.CTkCheckBox(
            self.output_extremes_section.content,
            text="Вывести изображением",
            variable=self.p_ex_var2
        )
        self.output_extremes_section.add_widget(self.p_ex2_checkbox, fill="x")
        self.p_ex1_checkbox = ctk.CTkCheckBox(
            self.output_extremes_section.content,
            text="Вывести текстовым файлом",
            variable=self.p_ex_var1
        )
        self.output_extremes_section.add_widget(self.p_ex1_checkbox, fill="x")

    def _setup_knn_section(self):
        """Настройка секции K-ближайших соседей"""
        info_label = ctk.CTkLabel(
            self.knn_section.content,
            text="Формат вывода K-ближайших соседей:",
            font=ctk.CTkFont(size=12),
            anchor="w",
            wraplength=0
        )
        self.knn_section.add_widget(info_label, pady=(0, 10))

        self.knn_text_checkbox = ctk.CTkCheckBox(
            self.knn_section.content,
            text="Вывести текстовым файлом",
            variable=self.knn_bool_text_var
        )
        self.knn_section.add_widget(self.knn_text_checkbox, fill="x")

        self.knn_image_checkbox = ctk.CTkCheckBox(
            self.knn_section.content,
            text="Вывести изображением",
            variable=self.knn_bool_image_var
        )
        self.knn_section.add_widget(self.knn_image_checkbox, fill="x")

    def _setup_intermediate_section(self):
        """Настройка секции промежуточных вычислений"""
        info_label = ctk.CTkLabel(
            self.intermediate_section.content,
            text="Дополнительные выходные данные:",
            font=ctk.CTkFont(size=12),
            anchor="w",
            wraplength=0
        )
        self.intermediate_section.add_widget(info_label, pady=(0, 10))

        self.print_channels_txt_checkbox = ctk.CTkCheckBox(
            self.intermediate_section.content,
            text="Исходные матрицы RGB",
            variable=self.print_channels_txt_var
        )
        self.intermediate_section.add_widget(self.print_channels_txt_checkbox, fill="x")

    def _setup_compute_section(self):
        """Настройка секции вычислений"""
        self.app_start_button = ctk.CTkButton(
            self.compute_section,
            text="Начать вычисления",
            command=self.safe_compute,
            height=50,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#28a745",
            hover_color="#218838",
            border_width=2,
            border_color="#1e7e34"
        )
        self.compute_section.pack(fill="x")
        self.app_start_button.pack(fill="x", pady=10)

    def safe_compute(self):
        """Безопасный запуск вычислений в отдельном потоке"""
        # Инициализируем _compute_thread если он None
        if self._compute_thread is None:
            self._compute_thread = threading.Thread()

        if self._compute_thread.is_alive():
            mb.showwarning("Внимание", "Вычисления уже выполняются")
            return

        # Блокируем UI на время вычислений
        self._disable_ui_during_compute(True)

        self._compute_thread = threading.Thread(target=self._compute_wrapper)
        self._compute_thread.daemon = True
        self._compute_thread.start()

    def _compute_wrapper(self):
        """Обертка для безопасного выполнения в потоке"""
        try:
            self.compute()
        except Exception as e:
            error_msg = f"Ошибка вычислений: {str(e)}\n{traceback.format_exc()}"
            self.progress_manager.log_error(error_msg)
            self.after_safe(0, lambda msg=error_msg: mb.showerror("Ошибка", msg))  # Фиксируем error_msg
        finally:
            try:
                self.after_safe(0, lambda: self._disable_ui_during_compute(False))
            except Exception as e:
                self.progress_manager.log_error(f"Ошибка при отключении UI: {str(e)}")

    def _disable_ui_during_compute(self, disable: bool):
        """Блокировка/разблокировка UI во время вычислений"""
        state = "disabled" if disable else "normal"

        widgets_to_disable = [
            self.load_button, self.pipette_button, self.gram_shmidt_button,
            self.button_save_scales, self.button_load_scales_file,
            self.app_start_button, self.add_task_btn]

        for widget in widgets_to_disable:
            try:
                widget.configure(state=state)
            except Exception as e:
                self.progress_manager.log_error(str(e))

    def compute(self): # Основная функция вычислений для всех задач
        if not self.image_processor.tasks:
            mb.showwarning("Внимание", "Нет задач для обработки")
            return

        # Проверяем, что все задачи готовы к обработке
        for i, task in enumerate(self.image_processor.tasks):
            if not task.image_path:
                mb.showerror("Ошибка", f"Задача {i + 1}: не загружено изображение")
                return
            if len(task.scales) == 0:
                mb.showerror("Ошибка", f"Задача {i + 1}: не заданы масштабы")
                return

        try:
            timer = time.time()
            total_tasks = len(self.image_processor.tasks)
            current_task_num = 0

            for task in self.image_processor.tasks:
                current_task_num += 1
                self.progress_manager.log_info(f"Обработка задачи {current_task_num}/{total_tasks}: {task.task_name}")

                # Выполняем вычисления для задачи
                self.image_processor.compute_for_task(
                    task,
                    self.wp_var1, self.wp_var2, self.print_channels_txt_var,
                    self.row_var, self.col_var,
                    self.max_var, self.min_var,
                    self.p_ex_var1, self.p_ex_var2,
                    self.knn_bool_text_var, self.knn_bool_image_var,
                    self.pipette_button.cget('state')
                )

                # Обновляем прогресс
                progress = current_task_num / total_tasks
                self.progress_manager.update_progress(
                    progress,
                    f"Обработано {current_task_num}/{total_tasks} задач"
                )

            elapsed_time = time.time() - timer
            self.after_safe(0, lambda: self.show_success_message(elapsed_time, total_tasks))

        except Exception as e:
            self.progress_manager.log_error(f"Критическая ошибка: {str(e)}")
            self.after_safe(0, lambda: mb.showerror("Ошибка", f"Произошла ошибка при вычислениях: {str(e)}"))

    def show_success_message(self, elapsed_time: float, total_tasks: int):
        msg_box = ctk.CTkToplevel(self)
        msg_box.title("Вычисления завершены")
        msg_box.resizable(False, False)
        msg_box.transient(self)
        msg_box.grab_set()

        def on_closing():
            try:
                msg_box.grab_release()
                msg_box.destroy()
            except tk.TclError as e:
                self.progress_manager.log_error(f"Ошибка закрытия окна: {str(e)}")

        msg_box.protocol("WM_DELETE_WINDOW", on_closing)

        # Центрируем окно
        msg_box.update_idletasks()
        width = 450
        height = 280  # Уменьшили высоту
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        msg_box.geometry(f"{width}x{height}+{x}+{y}")
        success_label = ctk.CTkLabel(
            msg_box,
            text="Вычисления выполнены успешно!",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        success_label.pack(pady=(20, 10))

        tasks_label = ctk.CTkLabel(
            msg_box,
            text=f"Обработано задач: {total_tasks}",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        tasks_label.pack(pady=(0, 5))

        time_label = ctk.CTkLabel(
            msg_box,
            text=f"Затраченное время: {format_time(elapsed_time)}",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        time_label.pack(pady=(0, 25))

        # Кнопки размещаем прямо в окне, по центру в одну линию
        buttons_frame = ctk.CTkFrame(msg_box, fg_color="transparent")
        buttons_frame.pack(pady=(10, 20))

        def open_folder_action(): # открыть папку с результатами
            try:
                if hasattr(self.image_processor, 'root_folder_path') and os.path.exists(
                        self.image_processor.root_folder_path):
                    os.startfile(self.image_processor.root_folder_path)
            except Exception as e:
                self.progress_manager.log_error(f"Не удалось открыть папку: {e}")

        def close_action():
            on_closing()
            self.safe_destroy()

        open_folder_btn = ctk.CTkButton(
            buttons_frame,
            text="Открыть папку с результатами",
            command=open_folder_action,
            width=180,
            height=35,
            font=ctk.CTkFont(size=12)
        )
        open_folder_btn.pack(side="left", padx=(0, 15))

        close_btn = ctk.CTkButton(
            buttons_frame,
            text="Закрыть",
            command=close_action,
            width=100,
            height=35,
            font=ctk.CTkFont(size=12)
        )
        close_btn.pack(side="left")

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours} ч {minutes:02d} м {seconds:02d} с"


if __name__ == '__main__':
    freeze_support()  # for multiprocess

    def main():
        try:
            app = App()
            app.mainloop()
        except Exception as e:
            print(f"Critical error: {e}")
            import traceback
            traceback.print_exc()


    main()