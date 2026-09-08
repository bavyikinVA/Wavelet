import os
import sys
import warnings
from uuid import uuid4

# Подавляем известное предупреждение экспериментального интерфейса CuPy до
# импорта модулей, которые могут косвенно загрузить cupyx.jit.
warnings.filterwarnings(
    "ignore",
    message=r"cupyx\.jit\.rawkernel is experimental.*",
    category=FutureWarning,
    module=r"cupyx\.jit\._interface"
)

from compute.knn.knn_cpu import process_extremes_with_knn
from compute.extremes.extremes_finder import ExtremesFinder
from compute.extremes.interpolator import Interpolator


def setup_cuda_environment():
    """Настройка окружения CUDA"""
    warnings.filterwarnings("ignore", message="CUDA path could not be detected")
    # cupyx.jit.rawkernel используется зависимостями CuPy и пока помечен как
    # experimental. Предупреждение не влияет на расчёты и засоряет консоль
    # ещё до появления главного окна.
    warnings.filterwarnings(
        "ignore",
        message=r"cupyx\.jit\.rawkernel is experimental.*",
        category=FutureWarning,
        module=r"cupyx\.jit\._interface"
    )

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
from tkinter import filedialog
from tkinter import messagebox as mb

import customtkinter as ctk
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image

from Gram_Shmidt import change_channels
from history.source_image import save_run_image
from history.result_catalog import save_coefficient_preview
from compute.validation import validate_scales
from compute.processing_task import ProcessingTask
from image_cropper_app import run_cropper
from pipette import run_pipette
from utils.gui import TkinterApp, ScrollableFrame, CollapsibleFrame
from utils.progress_manager import ProgressManager
from utils.theme import AppTheme
from utils.icon_button import IconButton
from utils.animation import animate_visibility
from compute.wavelets.cpu_wavelet import morlet_wavelet_with_padding
from ml.clustering import ClusteringError, run_clustering
from history import (
    RunHistoryStore, feature_checkpoint_available,
    restore_feature_checkpoint, save_feature_checkpoint,
)


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
        self.backend = ComputeBackend()
        # Используем один экземпляр GPU-процессора во всём приложении.
        self.gpu_processor = self.backend.gpu_processor

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
        if getattr(self, 'gpu_processor', None) is not None:
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
            self.progress.log_info(f'Изображение загружено: {image_path}')
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Не удалось загрузить изображение: {image_path}")
            original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            b, g, r = cv2.split(image)
            source_channels = [r, g, b]
            if not all(isinstance(ch, np.ndarray) for ch in source_channels):
                raise ValueError("Один или несколько каналов изображения не являются массивами NumPy")

            # Источник заменяется атомарно только после успешного чтения.
            # Последующий выбор 1D/2D изменяет лишь настройки анализа задачи.
            task.invalidate_source_results()
            task.image_path = image_path
            task.original_image = original_image
            task.data = source_channels
            task.data_copy = [channel.copy() for channel in source_channels]
            task.gram_schmidt_applied = False
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

        task_start_time = time.strftime("%d_%m_%Y_%H_%M_%S")
        milliseconds = int(time.time() * 1000) % 1000
        task_folder_name = f"{task.task_name} {task_start_time}_{milliseconds:03d}_{uuid4().hex[:8]}"
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
        task.gram_schmidt_applied = True
        self.progress.log_info("Преобразование Грамма-Шмидта завершено")

    def load_scales_for_task(self, task, start, end, step):
        if step <= 0 or start <= 0 or end < start:
            raise ValueError("Начало и шаг должны быть положительными, конец — не меньше начала")
        task.scales = validate_scales(np.arange(start=start, stop=end + 1, step=step))
        task.num_scale = task.scales.shape[0]
        self.progress.log_info(f"Загружены масштабы: {len(task.scales)} значений от {start} до {end}")

    def load_scales_from_file_for_task(self, task, filename):
        self.progress.log_info(f"Загрузка масштабов из файла: {filename}")
        values = []
        with open(filename, 'r') as file_of_scales:
            for line in file_of_scales:
                numbers = [np.double(x) for x in line.split()]
                values.extend(numbers)
        task.scales = validate_scales(values)
        task.num_scale = len(task.scales)
        self.progress.log_info(f"Загружено {task.num_scale} масштабов")

    def set_gpu_enabled(self, enabled):
        """Включить или отключить GPU для всех поддерживаемых вычислений."""
        success = self.backend.set_use_gpu(enabled)
        self.knn_processor.toggle_gpu(self.backend.use_gpu)
        backend_info = self.backend.get_backend_info()
        self.progress.log_info(f"Вычислительное устройство: {backend_info['device_name']}")
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

        self.progress.log_info(f"CPU обработка {cols} столбцов...")

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
            self.progress.log_error(f"GPU батчевая обработка не удалась: {e}. Возврат к CPU.")
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

            # Центрирование всегда участвует в расчёте, но его служебные
            # значения сохраняются только по явному выбору пользователя.
            insert_filename = "rows" if type_data == 0 else "cols"
            means = []
            count = (
                data_channel.shape[0] if type_data == 0
                else data_channel.shape[1]
            )
            for i in range(count):
                if type_data == 0:
                    mean = np.mean(data_channel[i])
                    data_channel[i] -= mean
                else:
                    mean = np.mean(data_channel[:, i])
                    data_channel[:, i] -= mean
                means.append(float(mean))

            if task.save_centering_means:
                file_mean_path = os.path.join(
                    task.task_folder_path,
                    f'mean_to_{insert_filename}_by_channel_{channel}.txt'
                )
                np.savetxt(file_mean_path, np.asarray(means), fmt="%.10g")

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
        if task.process_rows:
            data_3_channels = np.zeros(
                (3, task.num_scale, task.data[0].shape[0], task.data[0].shape[1])
            )
            task.result[0] = self.wavelets(task, 0, data_3_channels)
            self.save_print_wavelets(task, 0, info_out)

        if task.process_columns:
            self.progress.update_progress(0.6, "Обработка столбцов...")
            data_3_channels_tr = np.zeros(
                (3, task.num_scale, task.data[0].shape[0], task.data[0].shape[1])
            )
            task.result[1] = self.wavelets(task, 1, data_3_channels_tr)
            self.save_print_wavelets(task, 1, info_out)

        self.progress.update_progress(1.0, "Вейвлет-преобразование завершено")

    def compute_wavelets_2d(self, task):
        """Выполнить отдельное ориентационное 2D-преобразование Морле."""
        from compute.wavelets.morlet_2d import (
            cwt2d_morlet_cpu,
            cwt2d_morlet_gpu,
        )

        task_folder = self.create_task_folder(task)
        total = 3 * len(task.scales) * len(task.orientations)
        completed = 0
        use_gpu = self.backend.use_gpu and self.backend.is_gpu_available()
        channel_names = ["Красный", "Зелёный", "Синий"]

        for channel_index, channel_name in enumerate(channel_names):
            channel = task.data[channel_index].astype(np.float32)
            for scale in task.scales:
                scale_folder = self.create_scale_folder(scale, task_folder)
                output_folder = os.path.join(scale_folder, "2D_Morlet")
                os.makedirs(output_folder, exist_ok=True)

                for angle in task.orientations:
                    completed += 1
                    self.progress.update_progress(
                        completed / max(total, 1),
                        (f"2D Morlet: {channel_name}, масштаб {scale}, "
                         f"ориентация {angle:g}°")
                    )

                    if use_gpu:
                        try:
                            coefficients = cwt2d_morlet_gpu(
                                channel, [scale], [angle],
                                task.morlet_omega0,
                                task.morlet_anisotropy
                            )[0, 0]
                        except Exception as error:
                            self.progress.log_error(
                                f"Ошибка 2D Morlet на GPU: {error}. Переход на CPU."
                            )
                            use_gpu = False
                            coefficients = cwt2d_morlet_cpu(
                                channel, [scale], [angle],
                                task.morlet_omega0,
                                task.morlet_anisotropy
                            )[0, 0]
                    else:
                        coefficients = cwt2d_morlet_cpu(
                            channel, [scale], [angle],
                            task.morlet_omega0,
                            task.morlet_anisotropy
                        )[0, 0]

                    file_stem = (
                        f"2D_Morlet_{channel_name}_scale_{scale}_"
                        f"angle_{angle:g}"
                    )
                    magnitude = np.abs(coefficients)
                    save_coefficient_preview(task_folder, file_stem, coefficients)
                    phase = np.angle(coefficients)

                    if task.output_wavelet_numpy:
                        np.save(
                            os.path.join(output_folder, file_stem + "_complex.npy"),
                            coefficients.astype(np.complex64, copy=False)
                        )

                    if task.output_wavelet_text:
                        np.savetxt(
                            os.path.join(output_folder, file_stem + "_magnitude.csv"),
                            magnitude, fmt="%.6g", delimiter=","
                        )
                        np.savetxt(
                            os.path.join(output_folder, file_stem + "_phase.csv"),
                            phase, fmt="%.6g", delimiter=","
                        )

                    if task.output_wavelet_image:
                        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
                        magnitude_plot = axes[0].imshow(magnitude, cmap="viridis")
                        axes[0].set_title("Модуль коэффициентов")
                        axes[0].axis("off")
                        fig.colorbar(magnitude_plot, ax=axes[0], fraction=0.046)

                        phase_plot = axes[1].imshow(
                            phase, cmap="twilight", vmin=-np.pi, vmax=np.pi
                        )
                        axes[1].set_title("Фаза")
                        axes[1].axis("off")
                        fig.colorbar(phase_plot, ax=axes[1], fraction=0.046)
                        fig.suptitle(
                            f"{channel_name}: масштаб {scale}, ориентация {angle:g}°"
                        )
                        fig.tight_layout()
                        fig.savefig(
                            os.path.join(output_folder, file_stem + ".png"),
                            dpi=220,
                            bbox_inches="tight"
                        )
                        plt.close(fig)

        self.progress.update_progress(1.0, "2D-преобразование Морле завершено")

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
                save_coefficient_preview(task.task_folder_path,
                    f"1D_{type_matrix_str}_{colors[channel]}_масштаб_{task.scales[scale]}", array_2d)

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

                if task.output_wavelet_numpy:
                    np.save(
                        os.path.join(
                            scale_folder_path,
                            f'Wavelet_{type_matrix_str}_Scale_{task.scales[scale]}_{colors[channel]}.npy'
                        ),
                        array_2d
                    )

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
                       print_text_var, print_graphic, pipette_state,
                       pipeline_plan=None):
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

        plan = pipeline_plan or task.resolve_pipeline()
        max_var = bool(max_var or plan.maxima_required)

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
                # Keep computational precision: rounding can erase strict peaks.
                # Formatting belongs to export, not extrema/envelope detection.

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

                upper_max_row_points = []
                lower_min_row_points = []
                upper_max_col_points = []
                lower_min_col_points = []
                if plan.envelopes:
                    interpolator = Interpolator(self.backend)
                    upper_max_row_points, lower_min_row_points = interpolator.get_envelopes(
                        coefs_2d, pmaxr, pminr, direction='row'
                    )
                    upper_max_col_points, lower_min_col_points = interpolator.get_envelopes(
                        coefs_2d, pmaxc, pminc, direction='col'
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
                if plan.synchronization:
                    upper_points_by_scale[scale_value] = upper_max_row_points

                # Гистограмма: считаем только максимумы верхней огибающей на средней строке изображения.
                if plan.statistics or plan.synchronization:
                    if row_index_for_histogram is None:
                        row_index_for_histogram = coefs_2d.shape[0] // 2
                    row_hist_scales.append(scale_value)

                if plan.statistics:
                    upper_count_on_row = self.count_points_on_row(
                        upper_max_row_points,
                        row_index_for_histogram
                    )
                    row_hist_max_counts.append(upper_count_on_row)
                    self.progress.log_info(
                        f"Масштаб {scale_value}, {channel_name}, "
                        f"строка y={row_index_for_histogram}: "
                        f"максимумов верхней огибающей={upper_count_on_row}"
                    )

                # Сырые экстремумы и точки огибающих являются разными
                # артефактами и сохраняются независимо.
                raw_extremes_to_process = []
                raw_titles = []

                if max_var:
                    if row_var:
                        raw_extremes_to_process.append(pmaxr)
                        raw_titles.append(
                            f"{type_matrix_str}_Сырые_максимумы_по_строкам_масштаб_{scale_value}_{channel_name}"
                        )
                    if col_var:
                        raw_extremes_to_process.append(pmaxc)
                        raw_titles.append(
                            f"{type_matrix_str}_Сырые_максимумы_по_столбцам_масштаб_{scale_value}_{channel_name}"
                        )

                if min_var:
                    if row_var:
                        raw_extremes_to_process.append(pminr)
                        raw_titles.append(
                            f"{type_matrix_str}_Сырые_минимумы_по_строкам_масштаб_{scale_value}_{channel_name}"
                        )
                    if col_var:
                        raw_extremes_to_process.append(pminc)
                        raw_titles.append(
                            f"{type_matrix_str}_Сырые_минимумы_по_столбцам_масштаб_{scale_value}_{channel_name}"
                        )

                scale_folder = self.find_scale_folder(task, scale_value)

                for i, p in enumerate(raw_extremes_to_process):
                    if len(p) > 0:
                        if plan.extrema and task.output_extremes_text:
                            self.save_extremes_to_file(scale_folder, raw_titles[i], p)
                            self.progress.log_debug(f"Сохранен текстовый файл: {raw_titles[i]}.txt")

                        if plan.extrema and task.output_extremes_image:
                            self.save_extremes_graphic(
                                scale_folder, raw_titles[i], p, coefs_2d=coefs_2d
                            )

                envelope_points = []
                envelope_titles = []
                if plan.envelopes:
                    if max_var and row_var:
                        envelope_points.append(upper_max_row_points)
                        envelope_titles.append(
                            f"{type_matrix_str}_Верхняя_огибающая_по_строкам_масштаб_{scale_value}_{channel_name}"
                        )
                    if max_var and col_var:
                        envelope_points.append(upper_max_col_points)
                        envelope_titles.append(
                            f"{type_matrix_str}_Верхняя_огибающая_по_столбцам_масштаб_{scale_value}_{channel_name}"
                        )
                    if min_var and row_var:
                        envelope_points.append(lower_min_row_points)
                        envelope_titles.append(
                            f"{type_matrix_str}_Нижняя_огибающая_по_строкам_масштаб_{scale_value}_{channel_name}"
                        )
                    if min_var and col_var:
                        envelope_points.append(lower_min_col_points)
                        envelope_titles.append(
                            f"{type_matrix_str}_Нижняя_огибающая_по_столбцам_масштаб_{scale_value}_{channel_name}"
                        )

                for title, points in zip(envelope_titles, envelope_points):
                    if len(points) == 0:
                        continue
                    if task.output_envelopes_text:
                        self.save_extremes_to_file(scale_folder, title, points)
                    if task.output_envelopes_image:
                        self.save_extremes_graphic(
                            scale_folder, title, points, coefs_2d=coefs_2d
                        )

                # Подготовка данных для KNN. Оставлено для совместимости с текущей архитектурой.
                knn_extremes = {
                    'type_data': type_data,
                    'channel': channel,
                    'scale': scale_value,
                    'max_by_row': (
                        upper_max_row_points if plan.envelopes else pmaxr
                    ) if (row_var and max_var) else [],
                    'max_by_column': (
                        upper_max_col_points if plan.envelopes else pmaxc
                    ) if (col_var and max_var) else [],
                    'min_by_row': (
                        lower_min_row_points if plan.envelopes else pminr
                    ) if (row_var and min_var) else [],
                    'min_by_column': (
                        lower_min_col_points if plan.envelopes else pminc
                    ) if (col_var and min_var) else []
                }
                extremes.append(knn_extremes)

                if plan.knn:
                    self.progress.update_progress(0.85, "Обработка KNN...")
                    knn_result = process_extremes_with_knn(
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
                    task.knn_results[(channel_code, scale_value)] = knn_result

            # После всех масштабов текущего канала сохраняем итоговую гистограмму.
            if len(row_hist_scales) > 0:
                histogram_dir = os.path.join(
                    task.task_folder_path,
                    "Гистограммы_экстремумов_огибающих"
                )

                histogram_file_base = (
                    f"hist_upper_envelope_maxima_row_y_{row_index_for_histogram}_{channel_code}"
                )

                if plan.statistics:
                    task.statistics_results[channel_code] = {
                        "row_index": row_index_for_histogram,
                        "scales": list(row_hist_scales),
                        "upper_envelope_maxima_counts": list(
                            row_hist_max_counts
                        ),
                        "scale_blocks": {},
                    }

                saved_png_path, saved_csv_path = self.save_upper_envelope_maxima_row_histogram(
                    histogram_dir,
                    histogram_file_base,
                    row_hist_scales,
                    row_hist_max_counts,
                    row_index_for_histogram,
                    channel_name=channel_name,
                    save_png=(plan.statistics
                              and task.statistics_output_image),
                    save_csv=(plan.statistics
                              and task.statistics_output_csv)
                )

                if saved_png_path:
                    self.progress.log_info(
                        f"Гистограмма максимумов верхней огибающей сохранена: {saved_png_path}"
                    )
                elif plan.statistics and task.statistics_output_image:
                    self.progress.log_error(
                        f"Не удалось сохранить PNG-гистограмму максимумов верхней огибающей для канала {channel_name}"
                    )

                if saved_csv_path:
                    self.progress.log_info(
                        f"CSV с данными гистограммы сохранён: {saved_csv_path}"
                    )

                # Сжатые гистограммы по блокам масштабов.
                for block_size in (
                        scale_block_sizes if plan.statistics else []):
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
                    task.statistics_results[channel_code]["scale_blocks"][
                        block_size
                    ] = {
                        "labels": list(block_labels),
                        "counts": list(block_counts),
                        "ranges": list(block_ranges),
                    }

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
                        channel_name=channel_name,
                        save_png=task.statistics_output_image,
                        save_csv=task.statistics_output_csv
                    )

                    if block_png_path:
                        self.progress.log_info(
                            f"Сжатая гистограмма по блокам масштабов сохранена: {block_png_path}"
                        )
                    elif task.statistics_output_image:
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
                if (plan.synchronization
                        and len(upper_points_by_scale) > 0):
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
                            sync_results = self.calculate_row_synchronization_for_scale_block(
                                path=sync_dir,
                                title=sync_title,
                                points_by_scale=upper_points_by_scale,
                                scales=row_hist_scales,
                                rows_to_analyze=rows_to_analyze,
                                image_width=image_width,
                                block_size=block_size,
                                tolerance=row_sync_tolerance,
                                metric_name=metric_name,
                                channel_name=channel_name,
                                save_heatmap=task.synchronization_output_heatmap,
                                save_matrix_csv=(
                                    task.synchronization_output_matrix_csv
                                ),
                                save_pairs_csv=(
                                    task.synchronization_output_pairs_csv
                                )
                            )
                            result_key = (
                                channel_code, block_size, metric_name
                            )
                            task.synchronization_results[result_key] = (
                                sync_results
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
        if local_points is None or len(local_points) == 0:
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
            native_shape = (np.asarray(coefs_2d).shape[:2] if coefs_2d is not None
                            else original_img_shape[:2] if original_img_shape is not None else None)
            if native_shape is not None:
                np.savez_compressed(os.path.join(path, f"{title}.npz"),
                                    points=points, shape=native_shape)
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
            channel_name=None,
            save_png=True,
            save_csv=True
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
            if not save_png and not save_csv:
                return None, None
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
            if save_csv:
                csv_path = os.path.join(path, f"{file_base_name}.csv")
                with open(csv_path, 'w', encoding='utf-8') as file:
                    file.write("scale,upper_envelope_maxima_count,row_index\n")
                    for scale, count in zip(scales, max_counts):
                        file.write(f"{scale},{int(count)},{row_index}\n")

            if not save_png:
                return None, csv_path

            x = np.arange(len(scales))
            np.savez_compressed(os.path.join(path, f"{file_base_name}.npz"),
                                series=np.column_stack((scales, max_counts)),
                                labels=['Масштаб', 'Количество максимумов'])

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
            channel_name=None,
            save_png=True,
            save_csv=True
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
            if not save_png and not save_csv:
                return None, None
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

            if save_csv:
                csv_path = os.path.join(path, f"{file_base_name}.csv")
                with open(csv_path, 'w', encoding='utf-8') as file:
                    file.write(
                        "block_label,block_start_scale,block_end_scale,"
                        "scales_in_block,upper_envelope_maxima_sum,row_index,block_size\n"
                    )
                    for label, count, block_range in zip(
                            block_labels, block_counts, block_ranges):
                        scale_start, scale_end, block_scales = block_range
                        scales_as_text = "|".join(str(s) for s in block_scales)
                        file.write(
                            f"{label},{scale_start},{scale_end},"
                            f"{scales_as_text},{int(count)},{row_index},{block_size}\n"
                        )

            if not save_png:
                return None, csv_path

            x = np.arange(len(block_labels))
            np.savez_compressed(os.path.join(path, f"{file_base_name}.npz"),
                                series=np.column_stack((x, block_counts)),
                                labels=['Блок', 'Количество максимумов'], ticks=block_labels)

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
            np.savez_compressed(os.path.join(path, f"{file_base_name}.npz"), matrix=sync_matrix)

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
            channel_name=None,
            save_heatmap=True,
            save_matrix_csv=True,
            save_pairs_csv=True
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

                heatmap_path = None
                if save_heatmap:
                    heatmap_path = self.save_row_sync_heatmap(
                        path, file_base_name, sync_matrix, rows_to_analyze,
                        metric_name, block_label, channel_name=channel_name
                    )

                matrix_csv_path = None
                if save_matrix_csv:
                    matrix_csv_path = self.save_row_sync_matrix_csv(
                        path, file_base_name, sync_matrix, rows_to_analyze
                    )

                pairs_csv_path = None
                if save_pairs_csv:
                    pairs_csv_path = self.save_row_sync_pair_metrics_csv(
                        path, file_base_name, pair_rows
                    )

                if heatmap_path:
                    self.progress.log_info(f"Heatmap межстрочной синхронизации сохранён: {heatmap_path}")
                elif save_heatmap:
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

    def compute_for_task(self, task):
        """Выполнение вычислений для конкретной задачи"""
        try:
            pipette_state = 'disabled' if task.has_colors_selected() else 'normal'
            self.progress.begin_stage(f"{task.task_name} · Подготовка данных")
            task_folder = self.create_task_folder(task)
            self.progress.log_info(f"Создана папка для {task.task_name}: {task_folder}")

            # Сохранение исходных каналов - 5%
            self.progress.update_progress(0.05, "Сохранение исходных каналов...")
            self.save_orig_channels_txt(task, task.save_source_channels)

            task.result = {}
            pipeline_plan = task.resolve_pipeline()
            task.knn_results = {}
            task.statistics_results = {}
            task.synchronization_results = {}
            task.ml_result = None
            self.progress.log_info(
                "План расчёта: "
                f"экстремумы={pipeline_plan.extrema}, "
                f"огибающие={pipeline_plan.envelopes}, "
                f"KNN={pipeline_plan.knn}, "
                f"статистики={pipeline_plan.statistics}, "
                f"синхронизации={pipeline_plan.synchronization}"
            )
            info_out = self._get_output_type(
                task.output_wavelet_image,
                task.output_wavelet_text
            )

            wavelet_stage = "2D Morlet" if task.analysis_mode == "2d" else "Вейвлет-преобразование"
            self.progress.begin_stage(f"{task.task_name} · {wavelet_stage}")
            self.progress.update_progress(0.0, "Начало вейвлет-преобразования...")
            if task.analysis_mode == "2d":
                self.compute_wavelets_2d(task)
            else:
                self.compute_wavelets(task, info_out)

                # Текущий конвейер огибающих относится к построчному 1D-анализу.
                self.progress.update_progress(0.7, "Начало поиска экстремумов...")
                if pipeline_plan.point_pipeline_required:
                    selected = ["экстремумы"]
                    if pipeline_plan.envelopes:
                        selected.append("огибающие")
                    if pipeline_plan.knn:
                        selected.append("KNN")
                    if pipeline_plan.statistics:
                        selected.append("статистики")
                    if pipeline_plan.synchronization:
                        selected.append("синхронизации")
                    self.progress.begin_stage(
                        f"{task.task_name} · Анализ точек: {', '.join(selected)}"
                    )
                    self.compute_points(
                        task,
                        task.process_rows,
                        task.process_columns,
                        task.find_maxima,
                        task.find_minima,
                        task.k_neighbors,
                        task.output_knn_text,
                        task.output_knn_image,
                        task.output_extremes_text,
                        task.output_extremes_image,
                        pipette_state,
                        pipeline_plan=pipeline_plan
                    )

            # Для ML-сценария кластеризация выполняется уровнем приложения
            # сразу после подготовки KNN-признаков.
            completion = 1.0
            message = (
                f"Основной анализ для {task.task_name} завершён; запуск ML..."
                if pipeline_plan.ml else
                f"Вычисления для {task.task_name} завершены успешно"
            )
            self.progress.update_progress(completion, message)
            self.progress.log_info(message)

        except Exception as e:
            import traceback
            error_msg = f"Ошибка при обработке задачи {task.task_name}: {str(e)}\n{traceback.format_exc()}"
            self.progress.log_error(error_msg)
            raise
        finally:
            # Очищаем временные данные в задаче после вычислений
            task.result = {}
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


class WorkspaceTabs(ctk.CTkFrame):
    """Верхняя навигация по рабочим страницам в стиле desktop-приложений."""

    def __init__(self, master, tab_names):
        super().__init__(master, fg_color="transparent", corner_radius=0)
        self._pages = {}
        self._buttons = {}
        self._underlines = {}
        self._current_name = None

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.navigation = ctk.CTkFrame(
            self,
            height=AppTheme.NAV_HEIGHT,
            corner_radius=0,
            fg_color=AppTheme.NAV_BACKGROUND
        )
        self.navigation.grid(row=0, column=0, sticky="ew")
        self.navigation.grid_propagate(False)
        self.navigation.pack_propagate(False)

        self.tabs_strip = ctk.CTkFrame(
            self.navigation,
            fg_color=AppTheme.NAV_BACKGROUND,
            corner_radius=0
        )
        self.tabs_strip.pack(side="left", fill="y", padx=18)

        self.page_container = ctk.CTkFrame(
            self, fg_color="transparent", corner_radius=0
        )
        self.page_container.grid(row=1, column=0, sticky="nsew")
        self.page_container.grid_columnconfigure(0, weight=1)
        self.page_container.grid_rowconfigure(0, weight=1)
        self.page_container.grid_propagate(False)

        for name in tab_names:
            self.add(name)

    def add(self, name):
        item = ctk.CTkFrame(
            self.tabs_strip, fg_color="transparent", corner_radius=0
        )
        item.pack(side="left", fill="y")

        button = ctk.CTkButton(
            item,
            text=name,
            command=lambda tab_name=name: self.set(tab_name),
            width=max(88, 12 * len(name)),
            height=AppTheme.NAV_HEIGHT - 4,
            corner_radius=0,
            fg_color="transparent",
            hover_color=AppTheme.NAV_HOVER,
            text_color=AppTheme.NAV_TEXT_INACTIVE,
            border_width=0,
            font=ctk.CTkFont(size=14),
        )
        button.pack(fill="both", expand=True)

        underline = ctk.CTkFrame(
            item,
            height=3,
            corner_radius=2,
            fg_color="transparent"
        )
        page = ctk.CTkFrame(
            self.page_container, fg_color="transparent", corner_radius=0
        )
        self._buttons[name] = button
        self._underlines[name] = underline
        self._pages[name] = page
        page.grid(row=0, column=0, sticky="nsew")
        if self._current_name is None:
            self.set(name)
        else:
            self._pages[self._current_name].tkraise()
        return page

    def tab(self, name):
        return self._pages[name]

    def set(self, name):
        if name not in self._pages:
            return
        self._pages[name].tkraise()
        self._current_name = name
        for tab_name, button in self._buttons.items():
            active = tab_name == name
            button.configure(
                fg_color="transparent",
                text_color=(
                    AppTheme.NAV_TEXT_ACTIVE if active
                    else AppTheme.NAV_TEXT_INACTIVE
                )
            )
            underline = self._underlines[tab_name]
            if active:
                underline.configure(fg_color=AppTheme.NAV_ACTIVE)
                underline.place(
                    relx=0.08, rely=1.0, relwidth=0.84, y=-4
                )
            else:
                underline.place_forget()

    def set_enabled(self, enabled):
        state = "normal" if enabled else "disabled"
        for button in self._buttons.values():
            button.configure(state=state)


class App(TkinterApp):
    ANALYSIS_MODE_LABELS = {
        "1D-анализ": "1d",
        "2D-анализ": "2d",
    }
    ANALYSIS_MODE_NAMES = {
        "1d": "1D-анализ",
        "2d": "2D-анализ",
    }

    def __init__(self):
        super().__init__()
        self._compute_thread = None
        self._loading_task_settings = False
        self.run_history = RunHistoryStore()
        self.title("Wavelet Analysis")
        self.resizable(True, True)
        self.geometry(f"{self.winfo_screenwidth()}x{self.winfo_screenheight() - 40}+0+0")

        # настройка темы
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # инициализация всех атрибутов UI
        self._initialize_ui_variables()

        self._tasks_visible = True
        self._progress_visible = True
        self.layout_controls = ctk.CTkFrame(self, height=30, fg_color='transparent')
        self.layout_controls.pack(fill='x', padx=8, pady=2)
        self.tasks_toggle = IconButton(self.layout_controls, 'sidebar', 'Панель задач', self._toggle_tasks_panel)
        self.tasks_toggle.pack(side='left', padx=3)
        self.progress_toggle = IconButton(self.layout_controls, 'bottom', 'Панель выполнения', self._toggle_progress_panel)
        self.progress_toggle.pack(side='left', padx=3)
        IconButton(self.layout_controls, 'focus', 'Фокус на рабочей области', self._toggle_focus_layout).pack(side='left', padx=3)

        # Постоянная панель задач и единая рабочая область со вкладками.
        self.main_container = ctk.CTkFrame(self)
        self.main_container.pack(
            fill="both", expand=True,
            padx=AppTheme.WINDOW_PADDING,
            pady=(2, AppTheme.SECTION_GAP)
        )
        self.main_container.grid_columnconfigure(
            0, weight=0, minsize=AppTheme.SIDEBAR_WIDTH
        )
        self.main_container.grid_columnconfigure(1, weight=1, minsize=0)
        self.main_container.grid_rowconfigure(0, weight=1)

        # панель управления задачами
        self.tasks_panel = self._create_tasks_panel()
        self.tasks_panel.grid(
            row=0, column=0, sticky="nsew",
            padx=(0, AppTheme.PANEL_GAP // 2)
        )
        self.tasks_panel.configure(width=AppTheme.SIDEBAR_WIDTH)
        self.tasks_panel.pack_propagate(False)
        self.sidebar_grip = ctk.CTkFrame(self.tasks_panel, width=6, corner_radius=0,
                                        fg_color=AppTheme.BORDER, cursor='sb_h_double_arrow')
        self.sidebar_grip.place(relx=1, rely=0, relheight=1, anchor='ne')
        self.sidebar_grip.bind('<B1-Motion>', self._resize_tasks_panel)
        self.workspace_frame = ctk.CTkFrame(self.main_container)
        self.workspace_frame.grid(
            row=0, column=1, sticky="nsew",
            padx=(AppTheme.PANEL_GAP // 2, 0)
        )
        self.workspace_frame.grid_columnconfigure(0, weight=1)
        self.workspace_frame.grid_rowconfigure(0, weight=1)
        self.workspace_frame.grid_propagate(False)

        self._create_workspace_tabs()

        self.progress_manager = ProgressManager(self)

        # менеджер изображений
        self.image_processor = ImageProcessor(self.progress_manager)

        # текущая задача
        self.current_task = None
        self.knn_text_var.trace('w', self.update_knn_for_current_task)
        for variable in (
                self.row_var, self.col_var, self.max_var, self.min_var,
                self.wp_var1, self.wp_var2, self.p_ex_var1, self.p_ex_var2,
                self.wavelet_numpy_var,
                self.envelope_text_var, self.envelope_image_var,
                self.knn_bool_text_var, self.knn_bool_image_var,
                self.calculate_extrema_var, self.calculate_envelopes_var,
                self.calculate_knn_var,
                self.print_channels_txt_var, self.centering_means_var,
                self.orientations_var,
                self.morlet_omega0_var, self.morlet_anisotropy_var,
                self.calculate_statistics_var, self.statistics_image_var,
                self.statistics_csv_var, self.calculate_sync_var,
                self.sync_heatmap_var, self.sync_matrix_csv_var,
                self.sync_pairs_csv_var, self.scale_block_sizes_var,
                self.sync_stride_var, self.sync_tolerance_var,
                self.sync_metric_var):
            variable.trace_add('write', self._store_settings_for_current_task)
        for variable in (
                self.ml_algorithm_var, self.ml_point_filter_var,
                self.ml_feature_set_var, self.ml_standardize_var,
                self.ml_n_clusters_var, self.ml_random_state_var,
                self.ml_eps_var, self.ml_min_samples_var):
            variable.trace_add('write', self._store_ml_settings_for_current_task)

        # Переменная для GPU/CPU переключения
        self.use_gpu_var = tk.BooleanVar(value=True)

        # Обновляем UI после инициализации image_processor
        self.after(100, self._update_gpu_section)
        self.after_idle(self._update_action_availability)

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
        # Максимизируем
        if self.tk.call('tk', 'windowingsystem') == 'win32':
            self.state('zoomed')
        else:
            self.attributes('-zoomed', True)

        # Let the normal event loop settle geometry; never nest update() here.
        self.after_safe(200, self._final_adjustment)

    def _final_adjustment(self):
        """Финальная корректировка размеров"""
        # Принудительно обновляем все области с перемоткой
        for child in self.winfo_children():
            if hasattr(child, 'update_scrollbar'):
                child.update_scrollbar()

    def _initialize_ui_variables(self):
        """Инициализация всех переменных UI"""
        self.analysis_mode_var = tk.StringVar(value="1D-анализ")
        self.pipeline_preset_var = tk.StringVar(value="Только вейвлет")
        self.calculate_extrema_var = tk.BooleanVar(value=False)
        self.calculate_envelopes_var = tk.BooleanVar(value=False)
        self.calculate_knn_var = tk.BooleanVar(value=False)
        self.row_var = tk.BooleanVar(value=True)
        self.col_var = tk.BooleanVar(value=True)
        self.max_var = tk.BooleanVar(value=True)
        self.min_var = tk.BooleanVar(value=True)
        self.wp_var1 = tk.BooleanVar(value=True)
        self.wp_var2 = tk.BooleanVar(value=False)
        self.wavelet_numpy_var = tk.BooleanVar(value=False)
        self.p_ex_var1 = tk.BooleanVar(value=False)
        self.p_ex_var2 = tk.BooleanVar(value=False)
        self.envelope_text_var = tk.BooleanVar(value=False)
        self.envelope_image_var = tk.BooleanVar(value=True)
        self.knn_bool_text_var = tk.BooleanVar(value=False)
        self.knn_bool_image_var = tk.BooleanVar(value=False)
        self.print_channels_txt_var = tk.BooleanVar(value=False)
        self.centering_means_var = tk.BooleanVar(value=False)
        self.calculate_statistics_var = tk.BooleanVar(value=False)
        self.statistics_image_var = tk.BooleanVar(value=True)
        self.statistics_csv_var = tk.BooleanVar(value=True)
        self.calculate_sync_var = tk.BooleanVar(value=False)
        self.sync_heatmap_var = tk.BooleanVar(value=True)
        self.sync_matrix_csv_var = tk.BooleanVar(value=True)
        self.sync_pairs_csv_var = tk.BooleanVar(value=True)
        self.scale_block_sizes_var = tk.StringVar(value="5")
        self.sync_stride_var = tk.StringVar(value="1")
        self.sync_tolerance_var = tk.StringVar(value="1")
        self.sync_metric_var = tk.StringVar(value="jaccard")

        self.data = tk.StringVar()
        self.knn_text_var = tk.StringVar(value="0")
        self.orientations_var = tk.StringVar(value="0, 45, 90, 135")
        self.morlet_omega0_var = tk.StringVar(value="6.0")
        self.morlet_anisotropy_var = tk.StringVar(value="1.0")

        # ML-настройки. Набор полей алгоритма перестраивается динамически.
        self.ml_algorithm_var = tk.StringVar(value="K-means")
        self.ml_point_filter_var = tk.StringVar(value="Все точки KNN")
        self.ml_feature_set_var = tk.StringVar(value="Геометрия KNN")
        self.ml_standardize_var = tk.BooleanVar(value=True)
        self.ml_n_clusters_var = tk.StringVar(value="5")
        self.ml_random_state_var = tk.StringVar(value="42")
        self.ml_eps_var = tk.StringVar(value="0.8")
        self.ml_min_samples_var = tk.StringVar(value="5")
        self._ml_thread = None
        self._ml_preview_image = None

        # Journal filters are interface state, not analysis parameters.
        self.history_search_var = tk.StringVar(value="")
        self.history_status_var = tk.StringVar(value="Все статусы")

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
        self.gpu_switch = None
        self.analysis_mode_selector = None
        self.analysis_mode_description = None
        self.analysis_source_label = None
        self.two_d_section = None
        self.pipeline_section = None
        self.pipeline_preset_selector = None
        self.calculate_extrema_switch = None
        self.calculate_envelopes_switch = None
        self.calculate_knn_switch = None
        self.pipeline_dependency_label = None
        self.pipeline_chain_label = None
        self.extremes_hint_label = None
        self.statistics_section = None
        self.calculate_statistics_switch = None
        self.calculate_sync_switch = None
        self.statistics_output_widgets = []
        self.synchronization_output_widgets = []
        self.statistics_parameter_widgets = []
        self.synchronization_parameter_widgets = []
        self.extremes_output_widgets = []
        self.envelopes_output_widgets = []
        self.knn_output_widgets = []

    def _create_tasks_panel(self):
        """Создание панели управления задачами"""
        panel = ctk.CTkFrame(self.main_container)

        # Заголовок панели
        header = ctk.CTkLabel(
            panel,
            text="Задачи",
            font=AppTheme.panel_title_font(),
            anchor="w"  # Выравнивание текста слева
        )
        header.pack(fill="x", padx=10, pady=(10, 15))

        # Кнопка добавления задачи
        self.add_task_btn = ctk.CTkButton(
            panel,
            text="Добавить задачу",
            command=self.add_new_task,
            width=190,
            height=AppTheme.BUTTON_HEIGHT,
            font=AppTheme.section_title_font(),
            fg_color=AppTheme.PRIMARY,
            hover_color=AppTheme.PRIMARY_HOVER
        )
        self.add_task_btn.pack(
            anchor="w", padx=10, pady=(0, 10),
            ipadx=12
        )

        # Фрейм для списка задач с прокруткой
        tasks_scrollable = ScrollableFrame(panel)
        tasks_scrollable.pack(fill="both", expand=True, padx=10, pady=5)

        # Заголовок списка задач
        tasks_label = ctk.CTkLabel(
            tasks_scrollable.scrollable_frame,
            text="Список задач:",
            font=AppTheme.section_title_font(),
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
            font=AppTheme.body_font(),
            text_color=AppTheme.MUTED,
            anchor="w"
        )
        self.tasks_status_label.pack(fill="x", padx=10, pady=(5, 10))

        return panel

    def _create_workspace_tabs(self):
        """Создать рабочие вкладки без потери контекста активной задачи."""
        self.workspace_tabs = WorkspaceTabs(
            self.workspace_frame,
            ("Данные", "Анализ", "Настройка вывода", "Результаты", "ML", "Предыдущие запуски")
        )
        self.workspace_tabs.grid(
            row=0, column=0, sticky="nsew"
        )

        self.data_panel = self._create_data_tab(
            self.workspace_tabs.tab("Данные")
        )
        self.analysis_panel = self._create_analysis_tab(
            self.workspace_tabs.tab("Анализ")
        )
        self.right_panel = self._create_right_panel(
            parent=self.workspace_tabs.tab("Настройка вывода"),
            include_compute=True
        )
        self.right_panel.pack(fill="both", expand=True)
        self.ml_panel = self._create_ml_tab(self.workspace_tabs.tab("ML"))
        from utils.result_viewer import ResultsPanel
        self.results_panel = ResultsPanel(self.workspace_tabs.tab("Результаты"))
        self.results_panel.pack(fill="both", expand=True)
        self.history_panel = self._create_history_tab(
            self.workspace_tabs.tab("Предыдущие запуски")
        )
        self.workspace_tabs.set("Данные")

    def _create_history_tab(self, parent):
        """Create a researcher-friendly view of persistent previous runs."""
        panel = ctk.CTkFrame(parent, fg_color="transparent")
        panel.pack(fill="both", expand=True)
        header = ctk.CTkFrame(panel, fg_color="transparent")
        header.pack(fill="x", padx=12, pady=(12, 6))
        ctk.CTkLabel(
            header, text="Предыдущие запуски",
            font=AppTheme.panel_title_font(), anchor="w"
        ).pack(side="left", fill="x", expand=True)
        ctk.CTkButton(
            header, text="Обновить", width=110,
            height=AppTheme.SMALL_BUTTON_HEIGHT,
            command=self._refresh_history_tab
        ).pack(side="right")
        ctk.CTkLabel(
            panel,
            text=("Здесь сохраняются настройки и результаты исследований. "
                  "Любой запуск можно повторить как новую задачу."),
            font=AppTheme.caption_font(), text_color=AppTheme.TEXT_SECONDARY,
            anchor="w", justify="left"
        ).pack(fill="x", padx=12, pady=(0, 8))
        filters = ctk.CTkFrame(panel, fg_color="transparent")
        filters.pack(fill="x", padx=12, pady=(0, 8))
        search_entry = ctk.CTkEntry(
            filters, textvariable=self.history_search_var,
            placeholder_text="Поиск по изображению, сценарию или задаче",
            width=360
        )
        search_entry.pack(side="left", fill="x", expand=True)
        search_entry.bind("<Return>", lambda _event: self._refresh_history_tab())
        ctk.CTkOptionMenu(
            filters, variable=self.history_status_var,
            values=["Все статусы", "Завершённые", "С ошибкой"],
            command=lambda _value: self._refresh_history_tab(), width=145
        ).pack(side="left", padx=(8, 0))
        ctk.CTkButton(
            filters, text="Найти", width=80,
            height=AppTheme.COMPACT_CONTROL_HEIGHT,
            command=self._refresh_history_tab
        ).pack(side="left", padx=(8, 0))
        self.history_scrollable = ScrollableFrame(panel)
        self.history_scrollable.pack(fill="both", expand=True, padx=6, pady=(0, 8))
        self.history_cards = []
        self._refresh_history_tab()
        return panel

    def _refresh_history_tab(self):
        if not hasattr(self, "history_scrollable"):
            return
        for card in self.history_cards:
            card.destroy()
        self.history_cards.clear()
        runs = self.run_history.list_runs(limit=200)
        query = self.history_search_var.get().strip().casefold()
        status_filter = self.history_status_var.get()
        filtered_runs = []
        for run in runs:
            if status_filter == "Завершённые" and run["status"] != "completed":
                continue
            if status_filter == "С ошибкой" and run["status"] == "completed":
                continue
            searchable = " ".join((
                run.get("task_name", ""), run.get("image_path", ""),
                run.get("pipeline_preset", ""), run.get("ml_algorithm", ""),
                run.get("researcher_note", "")
            )).casefold()
            if query and query not in searchable:
                continue
            filtered_runs.append(run)
        runs = filtered_runs
        parent = self.history_scrollable.scrollable_frame
        if not runs:
            empty = ctk.CTkLabel(
                parent, text="Завершённых запусков пока нет",
                text_color=AppTheme.TEXT_SECONDARY, font=AppTheme.body_font()
            )
            empty.pack(anchor="w", padx=12, pady=16)
            self.history_cards.append(empty)
            return
        for run in runs:
            card = ctk.CTkFrame(
                parent, border_width=1, border_color=AppTheme.BORDER,
                corner_radius=8
            )
            card.pack(fill="x", padx=6, pady=4)
            self.history_cards.append(card)
            status = "Завершён" if run["status"] == "completed" else "Ошибка"
            status_color = AppTheme.SUCCESS if run["status"] == "completed" else AppTheme.DANGER
            title = ctk.CTkFrame(card, fg_color="transparent")
            title.pack(fill="x", padx=12, pady=(9, 2))
            image_name = os.path.basename(run["image_path"]) or "Источник не указан"
            ctk.CTkLabel(
                title, text=f"Эксперимент №{run['id']} · {run['created_at'].replace('T', ' ')}",
                font=AppTheme.section_title_font(), anchor="w"
            ).pack(side="left", fill="x", expand=True)
            ctk.CTkLabel(
                title, text=status, text_color=status_color,
                font=AppTheme.caption_font()
            ).pack(side="right")
            try:
                snapshot = self.run_history.get_settings(run["id"])
                details = self._format_history_snapshot(snapshot)
            except Exception:
                details = {
                    "wavelet": "Morlet",
                    "scales": "данные недоступны",
                    "stages": run["pipeline_preset"],
                    "nuances": run["analysis_mode"].upper(),
                }
            ctk.CTkLabel(
                card,
                text=f"{image_name} · {run['task_name']} · {format_time(run['duration_seconds'])}",
                font=AppTheme.body_font(), anchor="w"
            ).pack(fill="x", padx=12, pady=(1, 3))
            detail_text = (
                f"Вейвлет: {details['wavelet']}\n"
                f"Масштабы: {details['scales']}\n"
                f"Сценарий: {run['pipeline_preset']}\n"
                f"Выполнено: {details['stages']}\n"
                f"Нюансы: {details['nuances']}"
            )
            ctk.CTkLabel(
                card, text=detail_text, font=AppTheme.caption_font(),
                text_color=AppTheme.TEXT_SECONDARY, anchor="w",
                justify="left", wraplength=850
            ).pack(fill="x", padx=12, pady=1)
            if feature_checkpoint_available(run.get("output_path", "")):
                ctk.CTkLabel(
                    card,
                    text="● Контрольная точка признаков доступна для продолжения",
                    font=AppTheme.caption_font(), text_color=AppTheme.SUCCESS,
                    anchor="w"
                ).pack(fill="x", padx=12, pady=(3, 1))
            if run["ml_summary"]:
                ctk.CTkLabel(
                    card, text=run["ml_summary"], font=AppTheme.caption_font(),
                    text_color=AppTheme.TEXT_SECONDARY, anchor="w"
                ).pack(fill="x", padx=12, pady=1)
            if run["error_message"]:
                ctk.CTkLabel(
                    card, text=run["error_message"], font=AppTheme.caption_font(),
                    text_color=AppTheme.DANGER, anchor="w", justify="left",
                    wraplength=760
                ).pack(fill="x", padx=12, pady=1)
            note_row = ctk.CTkFrame(card, fg_color="transparent")
            note_row.pack(fill="x", padx=12, pady=(5, 1))
            note_var = tk.StringVar(value=run.get("researcher_note", ""))
            ctk.CTkEntry(
                note_row, textvariable=note_var,
                placeholder_text="Заметка исследователя", height=28
            ).pack(side="left", fill="x", expand=True)
            ctk.CTkButton(
                note_row, text="Сохранить заметку", width=135,
                height=AppTheme.COMPACT_CONTROL_HEIGHT,
                command=lambda run_id=run["id"], var=note_var:
                    self._save_history_note(run_id, var.get())
            ).pack(side="left", padx=(8, 0))
            actions = ctk.CTkFrame(card, fg_color="transparent")
            actions.pack(fill="x", padx=12, pady=(5, 9))
            ctk.CTkButton(
                actions, text="Результаты", width=110,
                command=lambda folder=run['output_path']: self._show_saved_results(folder)
            ).pack(side="left", padx=(0, 6))
            ctk.CTkButton(
                actions, text="Повторить с этими настройками", width=220,
                height=AppTheme.COMPACT_CONTROL_HEIGHT,
                command=lambda run_id=run["id"]: self._restore_history_run(run_id)
            ).pack(side="left")
            if run["output_path"]:
                ctk.CTkButton(
                    actions, text="Открыть результаты", width=150,
                    height=AppTheme.COMPACT_CONTROL_HEIGHT,
                    command=lambda path=run["output_path"]: self._open_result_path(path)
                ).pack(side="left", padx=(8, 0))

    @staticmethod
    def _format_history_snapshot(snapshot):
        task = ProcessingTask()
        task.apply_settings_snapshot(snapshot)
        mode = "1D" if task.analysis_mode == "1d" else "2D"
        wavelet = f"Morlet · {mode}"
        if task.analysis_mode == "2d":
            wavelet += f" · ω₀={task.morlet_omega0:g} · γ={task.morlet_anisotropy:g}"
        return {
            "wavelet": wavelet,
            "scales": task.scales_summary(),
            "stages": task.executed_stage_summary(),
            "nuances": task.nuances_summary() or "без дополнительных особенностей",
        }

    def _save_history_note(self, run_id, note):
        try:
            self.run_history.update_note(run_id, note)
            self.progress_manager.log_info(f"Заметка эксперимента №{run_id} сохранена")
        except Exception as error:
            self.progress_manager.log_error(
                f"Не удалось сохранить заметку эксперимента №{run_id}: {error}"
            )

    def _open_result_path(self, path):
        if not path or not os.path.exists(path):
            mb.showwarning("Результаты", "Папка результатов больше не найдена")
            return
        try:
            if os.name == "nt":
                os.startfile(path)
            elif sys.platform == "darwin":
                import subprocess
                subprocess.Popen(["open", path])
            else:
                import subprocess
                subprocess.Popen(["xdg-open", path])
        except Exception as error:
            mb.showerror("Результаты", f"Не удалось открыть папку: {error}")

    def _restore_history_run(self, run_id):
        try:
            snapshot = self.run_history.get_settings(run_id)
            run_record = self.run_history.get_run(run_id)
            task = ProcessingTask()
            task.apply_settings_snapshot(snapshot)
            source_path = task.image_path
            if source_path and os.path.isfile(source_path):
                image = cv2.imread(source_path)
                if image is not None:
                    task.original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    b, g, r = cv2.split(image)
                    task.data = [r, g, b]
                    task.data_copy = [channel.copy() for channel in task.data]
                    if (snapshot.get("gram_schmidt_applied") and
                            task.color1 is not None and task.color2 is not None):
                        task.data = change_channels(task.color1, task.color2, task.data)
                else:
                    task.image_path = ""
            else:
                task.image_path = ""
            restored_checkpoints = []
            output_path = run_record.get("output_path", "")
            if output_path and os.path.isdir(output_path):
                try:
                    restored_checkpoints = restore_feature_checkpoint(
                        task, output_path
                    )
                except Exception as checkpoint_error:
                    self.progress_manager.log_error(
                        f"Контрольная точка не восстановлена: {checkpoint_error}"
                    )
            # A restored experiment is always a new variant. Reused data stays
            # in memory, while new ML files go to a fresh result directory.
            task.task_folder_path = ""
            task_id = self.image_processor.add_task(task)
            self.image_processor.set_current_task(task_id)
            self.current_task = task
            self._update_ui_for_current_task()
            self._update_tasks_display()
            # Как и при обычной загрузке, переход между этапами остаётся
            # осознанным действием исследователя.
            self.workspace_tabs.set("Данные")
            if restored_checkpoints:
                self.progress_manager.log_info(
                    "Восстановлена контрольная точка: "
                    + ", ".join(restored_checkpoints)
                )
                self._update_ml_tab_state()
            if not task.image_path:
                mb.showwarning(
                    "Источник не найден",
                    "Настройки восстановлены, но исходное изображение было перемещено "
                    "или удалено. Загрузите его заново."
                )
        except Exception as error:
            mb.showerror("Предыдущие запуски", f"Не удалось восстановить настройки: {error}")

    @staticmethod
    def _add_tab_heading(parent, title, description):
        ctk.CTkLabel(
            parent,
            text=title,
            font=AppTheme.panel_title_font(),
            anchor="w"
        ).pack(fill="x", padx=10, pady=(10, 4))
        if description:
            ctk.CTkLabel(
                parent,
                text=description,
                font=AppTheme.caption_font(),
                text_color=AppTheme.TEXT_SECONDARY,
                anchor="w",
                justify="left",
                wraplength=760
            ).pack(fill="x", padx=10, pady=(0, 12))

    def _create_data_tab(self, parent):
        panel = ctk.CTkFrame(parent, fg_color="transparent")
        panel.pack(fill="both", expand=True)
        scrollable = ScrollableFrame(panel)
        scrollable.pack(fill="both", expand=True)
        content = scrollable.scrollable_frame
        self._add_tab_heading(
            content,
            "Подготовка данных",
            "Загрузите изображение, выполните обрезку и при необходимости настройте цветовые каналы."
        )

        self.load_section = CollapsibleFrame(content, title="Загрузка и обрезка")
        self.load_section.pack(fill="x", padx=5, pady=2)
        self._setup_load_section()

        self.channel_section = CollapsibleFrame(content, title="Цветовые каналы")
        self.channel_section.pack(fill="x", padx=5, pady=2)
        self._setup_channel_section()
        self.data_next_button = ctk.CTkButton(
            content,
            text="Далее: параметры анализа",
            command=lambda: self.workspace_tabs.set("Анализ"),
            width=AppTheme.ACTION_BUTTON_WIDTH,
            height=AppTheme.BUTTON_HEIGHT,
            state="disabled"
        )
        self.data_next_button.pack(anchor="e", padx=12, pady=(14, 10))
        return panel

    def _create_analysis_tab(self, parent):
        panel = ctk.CTkFrame(parent, fg_color="transparent")
        panel.pack(fill="both", expand=True)
        scrollable = ScrollableFrame(panel)
        scrollable.pack(fill="both", expand=True)
        content = scrollable.scrollable_frame
        self._add_tab_heading(
            content,
            "Параметры анализа",
            None
        )

        self._setup_analysis_mode_section(content)

        self.scales_section = CollapsibleFrame(content, title="Масштабы")
        self.scales_section.pack(fill="x", padx=5, pady=2)
        self._setup_scales_section()

        self.two_d_section = CollapsibleFrame(content, title="Параметры 2D Morlet")
        self._setup_2d_section()

        self.extremes_section = CollapsibleFrame(
            content, title="Экстремумы и параметры KNN"
        )
        self.extremes_section.pack(fill="x", padx=5, pady=2)
        self._setup_extremes_section()
        self.analysis_next_button = ctk.CTkButton(
            content,
            text="Далее: настройка вывода",
            command=lambda: self.workspace_tabs.set("Настройка вывода"),
            width=AppTheme.ACTION_BUTTON_WIDTH,
            height=AppTheme.BUTTON_HEIGHT,
            state="disabled"
        )
        self.analysis_next_button.pack(anchor="e", padx=12, pady=(14, 10))
        return panel

    def _create_ml_tab(self, parent):
        panel = ctk.CTkFrame(parent, fg_color="transparent")
        panel.pack(fill="both", expand=True)
        scrollable = ScrollableFrame(panel)
        scrollable.pack(fill="both", expand=True)
        content = scrollable.scrollable_frame
        self._add_tab_heading(
            content,
            "Машинное обучение",
            "Кластеризация точек по координатам, расстояниям и направлениям KNN."
        )

        source_frame = ctk.CTkFrame(content)
        source_frame.pack(fill="x", padx=10, pady=(4, 8))
        ctk.CTkLabel(
            source_frame,
            text="Источник данных",
            font=AppTheme.section_title_font(),
            anchor="w"
        ).pack(fill="x", padx=12, pady=(10, 4))
        self.ml_source_label = ctk.CTkLabel(
            source_frame,
            text="Активная задача не выбрана",
            font=AppTheme.body_font(),
            text_color=AppTheme.TEXT_SECONDARY,
            anchor="w",
            justify="left"
        )
        self.ml_source_label.pack(fill="x", padx=12, pady=(0, 10))

        settings_frame = ctk.CTkFrame(content)
        settings_frame.pack(fill="x", padx=10, pady=8)
        ctk.CTkLabel(
            settings_frame,
            text="Настройки кластеризации",
            font=AppTheme.section_title_font(),
            anchor="w"
        ).pack(fill="x", padx=12, pady=(10, 6))

        common_grid = ctk.CTkFrame(settings_frame, fg_color="transparent")
        common_grid.pack(fill="x", padx=12, pady=(0, 8))
        common_grid.grid_columnconfigure(1, weight=1)
        labels_and_widgets = [
            ("Алгоритм", ctk.CTkOptionMenu(
                common_grid, values=["K-means", "DBSCAN"],
                variable=self.ml_algorithm_var,
                command=self._on_ml_algorithm_changed, width=220)),
            ("Тип точек", ctk.CTkOptionMenu(
                common_grid,
                values=["Все точки KNN", "Максимумы по строкам",
                        "Минимумы по строкам", "Максимумы по столбцам",
                        "Минимумы по столбцам"],
                variable=self.ml_point_filter_var, width=260)),
            ("Набор признаков", ctk.CTkOptionMenu(
                common_grid,
                values=["Геометрия KNN", "Координаты + KNN"],
                variable=self.ml_feature_set_var, width=260)),
        ]
        for row, (label_text, widget) in enumerate(labels_and_widgets):
            ctk.CTkLabel(
                common_grid, text=label_text, font=AppTheme.body_font(),
                anchor="w"
            ).grid(row=row, column=0, sticky="w", padx=(0, 14), pady=4)
            widget.grid(row=row, column=1, sticky="w", pady=4)

        self.ml_standardize_checkbox = ctk.CTkCheckBox(
            settings_frame,
            text="Привести признаки к единому масштабу",
            variable=self.ml_standardize_var,
        )
        self.ml_standardize_checkbox.pack(fill="x", padx=12, pady=(2, 8))

        self.ml_algorithm_parameters = ctk.CTkFrame(
            settings_frame, fg_color="transparent"
        )
        self.ml_algorithm_parameters.pack(fill="x", padx=12, pady=(0, 8))
        self.ml_algorithm_help_label = ctk.CTkLabel(
            settings_frame, text="", font=AppTheme.caption_font(),
            text_color=AppTheme.TEXT_SECONDARY, anchor="w", justify="left",
            wraplength=760
        )
        self.ml_algorithm_help_label.pack(fill="x", padx=12, pady=(0, 8))

        self.ml_run_button = ctk.CTkButton(
            settings_frame,
            text="Выполнить кластеризацию",
            command=self._start_ml_clustering,
            width=230,
            height=AppTheme.BUTTON_HEIGHT,
            state="disabled",
        )
        self.ml_run_button.pack(anchor="e", padx=12, pady=(2, 12))

        results_frame = ctk.CTkFrame(content)
        results_frame.pack(fill="x", padx=10, pady=(8, 12))
        ctk.CTkLabel(
            results_frame, text="Результат", font=AppTheme.section_title_font(),
            anchor="w"
        ).pack(fill="x", padx=12, pady=(10, 4))
        self.ml_result_label = ctk.CTkLabel(
            results_frame,
            text="Кластеризация ещё не выполнялась",
            font=AppTheme.caption_font(),
            text_color=AppTheme.TEXT_SECONDARY,
            anchor="w",
            justify="left",
            wraplength=760
        )
        self.ml_result_label.pack(fill="x", padx=12, pady=(2, 8))

        preview_controls = ctk.CTkFrame(results_frame, fg_color="transparent")
        preview_controls.pack(fill="x", padx=12, pady=(2, 6))
        self.ml_preview_selector = ctk.CTkOptionMenu(
            preview_controls,
            values=["Кластеры на изображении", "Кластеры в признаках"],
            command=lambda _value: self._refresh_ml_preview(),
            width=250,
            state="disabled",
        )
        self.ml_preview_selector.pack(side="left")
        self.ml_open_folder_button = ctk.CTkButton(
            preview_controls, text="Открыть папку", width=140,
            height=AppTheme.SMALL_BUTTON_HEIGHT,
            command=self._open_ml_result_folder, state="disabled"
        )
        self.ml_open_folder_button.pack(side="right")

        self.ml_preview_label = ctk.CTkLabel(
            results_frame, text="", fg_color=AppTheme.NAV_BACKGROUND,
            corner_radius=6
        )
        self.ml_preview_label.pack(fill="x", padx=12, pady=(4, 12))
        self._on_ml_algorithm_changed(self.ml_algorithm_var.get())
        return panel

    def _update_ml_tab_state(self):
        """Обновить источник, настройки и последний результат активной задачи."""
        if not hasattr(self, "ml_source_label"):
            return
        task = getattr(self, "current_task", None)
        if task is None:
            self.ml_source_label.configure(
                text="Активная задача не выбрана",
                text_color=AppTheme.TEXT_SECONDARY
            )
            self.ml_run_button.configure(state="disabled")
            self._show_ml_result(None)
            return

        has_knn = bool(getattr(task, "knn_results", None))
        source_status = (
            "KNN-признаки готовы к кластеризации" if has_knn else
            "KNN-признаки ещё не рассчитаны. На вкладке «Настройка вывода» "
            "выберите сценарий «Подготовка данных для ML» и запустите расчёт."
        )
        self.ml_source_label.configure(
            text=f"{task.task_name}: {source_status}",
            text_color=AppTheme.TEXT_ON_DARK if has_knn else AppTheme.TEXT_SECONDARY
        )
        self.ml_run_button.configure(state="normal" if has_knn else "disabled")

        self._loading_task_settings = True
        self.ml_algorithm_var.set(
            "K-means" if task.ml_algorithm == "kmeans" else "DBSCAN"
        )
        point_names = {
            "all": "Все точки KNN", "max_by_row": "Максимумы по строкам",
            "min_by_row": "Минимумы по строкам",
            "max_by_column": "Максимумы по столбцам",
            "min_by_column": "Минимумы по столбцам",
        }
        self.ml_point_filter_var.set(point_names.get(task.ml_point_filter, "Все точки KNN"))
        self.ml_feature_set_var.set(
            "Геометрия KNN" if task.ml_feature_set == "knn" else "Координаты + KNN"
        )
        self.ml_standardize_var.set(task.ml_standardize)
        self.ml_n_clusters_var.set(str(task.ml_n_clusters))
        self.ml_random_state_var.set(str(task.ml_random_state))
        self.ml_eps_var.set(f"{task.ml_eps:g}")
        self.ml_min_samples_var.set(str(task.ml_min_samples))
        self._loading_task_settings = False
        self._on_ml_algorithm_changed(self.ml_algorithm_var.get())
        self._show_ml_result(task.ml_result)

    def _on_ml_algorithm_changed(self, _value=None):
        """Показать только параметры выбранного алгоритма."""
        frame = getattr(self, "ml_algorithm_parameters", None)
        if frame is None:
            return
        for child in frame.winfo_children():
            child.destroy()
        frame.grid_columnconfigure(1, weight=1)
        if self.ml_algorithm_var.get() == "K-means":
            fields = [
                ("Количество кластеров", self.ml_n_clusters_var),
                ("Начальное состояние", self.ml_random_state_var),
            ]
            help_text = (
                "K-means делит все точки на заранее заданное число групп. "
                "Шумовые точки отдельно не выделяются."
            )
        else:
            fields = [
                ("Радиус соседства ε", self.ml_eps_var),
                ("Минимум точек", self.ml_min_samples_var),
            ]
            help_text = (
                "DBSCAN сам определяет число плотных групп и помечает "
                "одиночные точки как шум."
            )
        for row, (label, variable) in enumerate(fields):
            ctk.CTkLabel(
                frame, text=label, font=AppTheme.body_font(), anchor="w"
            ).grid(row=row, column=0, sticky="w", padx=(0, 14), pady=4)
            ctk.CTkEntry(frame, textvariable=variable, width=160).grid(
                row=row, column=1, sticky="w", pady=4
            )
        if getattr(self, "ml_algorithm_help_label", None) is not None:
            self.ml_algorithm_help_label.configure(text=help_text)

    def _store_ml_settings(self, task):
        point_filters = {
            "Все точки KNN": "all", "Максимумы по строкам": "max_by_row",
            "Минимумы по строкам": "min_by_row",
            "Максимумы по столбцам": "max_by_column",
            "Минимумы по столбцам": "min_by_column",
        }
        task.ml_algorithm = "kmeans" if self.ml_algorithm_var.get() == "K-means" else "dbscan"
        task.ml_point_filter = point_filters[self.ml_point_filter_var.get()]
        task.ml_feature_set = (
            "knn" if self.ml_feature_set_var.get() == "Геометрия KNN"
            else "coordinates_knn"
        )
        task.ml_standardize = bool(self.ml_standardize_var.get())
        task.ml_n_clusters = int(self.ml_n_clusters_var.get())
        task.ml_random_state = int(self.ml_random_state_var.get())
        task.ml_eps = float(self.ml_eps_var.get().replace(",", "."))
        task.ml_min_samples = int(self.ml_min_samples_var.get())

    def _store_ml_settings_for_current_task(self, *_args):
        task = getattr(self, "current_task", None)
        if task is None or self._loading_task_settings:
            return
        try:
            self._store_ml_settings(task)
            task.ml_result = None
            if getattr(self, "ml_result_label", None) is not None:
                self._show_ml_result(None)
        except (ValueError, KeyError):
            # Во время ввода значение может быть временно пустым. Проверка с
            # понятным сообщением выполняется при запуске кластеризации.
            pass

    def _start_ml_clustering(self):
        task = self.current_task
        if task is None or not task.knn_results:
            mb.showwarning("ML", "Сначала рассчитайте KNN-признаки для активной задачи")
            return
        if self._ml_thread is not None and self._ml_thread.is_alive():
            mb.showwarning("ML", "Кластеризация уже выполняется")
            return
        try:
            self._store_ml_settings(task)
        except (ValueError, KeyError):
            mb.showerror("ML", "Проверьте числовые параметры алгоритма")
            return
        try:
            task.task_folder_path = ""
            self.image_processor.create_task_folder(task)
        except Exception as error:
            mb.showerror("ML", f"Не удалось создать папку результатов: {error}")
            return
        self.ml_run_button.configure(state="disabled", text="Выполняется...")
        self.ml_result_label.configure(
            text="Подготовка признаков и кластеризация...",
            text_color=AppTheme.TEXT_SECONDARY,
        )
        self.progress_manager.begin_run(
            f"{task.task_name} · повторная ML-кластеризация",
            [f"{task.task_name} · ML-кластеризация"]
        )
        self.progress_manager.begin_stage(f"{task.task_name} · ML-кластеризация")
        self._ml_thread = threading.Thread(
            target=self._ml_clustering_worker, args=(task,), daemon=True
        )
        self._ml_thread.start()

    def _ml_clustering_worker(self, task):
        started = time.time()
        snapshot = task.settings_snapshot()
        try:
            save_run_image(task)
            snapshot = task.settings_snapshot()
            result = run_clustering(
                knn_results=task.knn_results,
                algorithm=task.ml_algorithm,
                output_root=task.task_folder_path,
                image=task.original_image,
                point_filter=task.ml_point_filter,
                feature_set=task.ml_feature_set,
                standardize=task.ml_standardize,
                n_clusters=task.ml_n_clusters,
                random_state=task.ml_random_state,
                eps=task.ml_eps,
                min_samples=task.ml_min_samples,
            )
            task.ml_result = result
            save_feature_checkpoint(task)
            self.progress_manager.save_run_log(
                task.task_folder_path,
                header=(
                    f"Изображение: {os.path.basename(task.image_path)}\n"
                    f"Повторная ML-кластеризация: {task.ml_algorithm.upper()}\n"
                    f"Масштабы: {task.scales_summary()}"
                ),
            )
            self.run_history.add_run(
                task=task,
                status="completed",
                duration_seconds=time.time() - started,
                settings=snapshot,
                ml_summary=self._history_ml_summary(task),
                ml_algorithm=task.ml_algorithm,
            )
            self.after_safe(0, self._refresh_history_tab)
            self.after_safe(0, lambda: self._finish_ml_clustering(task, result))
        except (ClusteringError, ValueError) as error:
            self.run_history.add_run(
                task=task, status="failed",
                duration_seconds=time.time() - started,
                settings=snapshot, error_message=str(error),
                ml_algorithm=task.ml_algorithm
            )
            self.after_safe(0, self._refresh_history_tab)
            self.after_safe(0, lambda e=str(error): self._fail_ml_clustering(e))
        except Exception as error:
            details = f"Ошибка ML: {error}\n{traceback.format_exc()}"
            self.progress_manager.log_error(details)
            self.run_history.add_run(
                task=task, status="failed",
                duration_seconds=time.time() - started,
                settings=snapshot, error_message=str(error),
                ml_algorithm=task.ml_algorithm
            )
            self.after_safe(0, self._refresh_history_tab)
            self.after_safe(0, lambda e=str(error): self._fail_ml_clustering(e))

    def _finish_ml_clustering(self, task, result):
        self._show_saved_results(task.task_folder_path)
        if self.current_task is task:
            self.ml_run_button.configure(state="normal", text="Выполнить кластеризацию")
            self._show_ml_result(result)
        self.progress_manager.log_info(
            f"ML-кластеризация завершена: {result['output_dir']}"
        )
        self.progress_manager.finish_run(
            "ML-кластеризация завершена",
            result_callback=lambda: self._show_saved_results(task.task_folder_path),
            folder_callback=lambda: self._open_result_path(result["output_dir"]),
            history_callback=lambda: self.workspace_tabs.set("Предыдущие запуски"),
        )

    def _fail_ml_clustering(self, message):
        self.ml_run_button.configure(state="normal", text="Выполнить кластеризацию")
        self.ml_result_label.configure(text=message, text_color=AppTheme.DANGER)
        self.progress_manager.fail_run(f"Ошибка ML: {message}")

    @staticmethod
    def _format_metric(value):
        return "не рассчитывается" if value is None else f"{value:.4f}"

    def _show_ml_result(self, result):
        if not result:
            self.ml_result_label.configure(
                text="Кластеризация ещё не выполнялась",
                text_color=AppTheme.TEXT_SECONDARY,
            )
            self.ml_preview_selector.configure(state="disabled")
            self.ml_open_folder_button.configure(state="disabled")
            self.ml_preview_label.configure(image=None, text="")
            self._ml_preview_image = None
            return
        metrics = result["metrics"]
        counts = ", ".join(
            (f"шум: {count}" if label == -1 else f"кластер {label + 1}: {count}")
            for label, count in result["counts"].items()
        )
        self.ml_result_label.configure(
            text=(
                f"Алгоритм: {result['algorithm_name']}\n"
                f"Кластеров: {metrics['cluster_count']}; шумовых точек: {metrics['noise_count']}\n"
                f"Распределение: {counts}\n"
                f"Silhouette: {self._format_metric(metrics['silhouette'])}; "
                f"Davies–Bouldin: {self._format_metric(metrics['davies_bouldin'])}; "
                f"Calinski–Harabasz: {self._format_metric(metrics['calinski_harabasz'])}\n"
                "Silhouette: ближе к 1 — лучше; Davies–Bouldin: меньше — лучше; "
                "Calinski–Harabasz: больше — лучше."
            ),
            text_color=AppTheme.TEXT_ON_DARK,
        )
        self.ml_preview_selector.configure(state="normal")
        self.ml_open_folder_button.configure(state="normal")
        self._refresh_ml_preview()

    def _refresh_ml_preview(self):
        task = self.current_task
        result = task.ml_result if task is not None else None
        if not result:
            return
        path = (
            result["image_plot_path"]
            if self.ml_preview_selector.get() == "Кластеры на изображении"
            else result["feature_plot_path"]
        )
        try:
            image = Image.open(path)
            image.thumbnail((760, 430), Image.Resampling.LANCZOS)
            self._ml_preview_image = ctk.CTkImage(
                light_image=image.copy(), dark_image=image.copy(), size=image.size
            )
            self.ml_preview_label.configure(image=self._ml_preview_image, text="")
        except Exception as error:
            self.ml_preview_label.configure(image=None, text=f"Не удалось показать изображение: {error}")

    def _open_ml_result_folder(self):
        task = self.current_task
        result = task.ml_result if task is not None else None
        if result and os.path.isdir(result["output_dir"]):
            os.startfile(result["output_dir"])

    def _update_navigation_buttons(self):
        """Блокировать переход вперёд, пока обязательный этап не завершён."""
        task = getattr(self, "current_task", None)
        data_ready = bool(task is not None and task.image_path)
        analysis_ready = bool(
            data_ready
            and len(task.scales) > 0
            and (
                (task.analysis_mode == "1d" and (
                    task.process_rows or task.process_columns
                ))
                or (task.analysis_mode == "2d" and task.orientations)
            )
        )
        if getattr(self, "data_next_button", None) is not None:
            self.data_next_button.configure(
                state="normal" if data_ready else "disabled"
            )
        if getattr(self, "analysis_next_button", None) is not None:
            self.analysis_next_button.configure(
                state="normal" if analysis_ready else "disabled"
            )

    def _setup_2d_section(self):
        """Настройки ориентационного двумерного вейвлета Морле."""
        ctk.CTkLabel(
            self.two_d_section.content,
            text="Ориентации, градусы (через запятую)",
            font=AppTheme.body_font(),
            anchor="w"
        ).pack(fill="x", pady=(0, 4))
        ctk.CTkEntry(
            self.two_d_section.content,
            textvariable=self.orientations_var,
            placeholder_text="0, 45, 90, 135"
        ).pack(fill="x", pady=(0, 8))

        ctk.CTkLabel(
            self.two_d_section.content,
            text="Центральная частота ω₀",
            font=AppTheme.body_font(),
            anchor="w"
        ).pack(fill="x", pady=(0, 4))
        ctk.CTkEntry(
            self.two_d_section.content,
            textvariable=self.morlet_omega0_var,
            placeholder_text="6.0"
        ).pack(fill="x", pady=(0, 8))

        ctk.CTkLabel(
            self.two_d_section.content,
            text="Анизотропность",
            font=AppTheme.body_font(),
            anchor="w"
        ).pack(fill="x", pady=(0, 4))
        ctk.CTkEntry(
            self.two_d_section.content,
            textvariable=self.morlet_anisotropy_var,
            placeholder_text="1.0"
        ).pack(fill="x")

        ctk.CTkLabel(
            self.two_d_section.content,
            text=("Ориентация задаёт направление чувствительности фильтра; "
                  "ω₀ управляет частотой несущей, а анизотропность — "
                  "вытянутостью вейвлета."),
            font=AppTheme.caption_font(),
            text_color=AppTheme.TEXT_SECONDARY,
            anchor="w",
            justify="left",
            wraplength=420
        ).pack(fill="x", pady=(8, 0))

    def _setup_analysis_mode_section(self, parent):
        """Создать верхнеуровневый переключатель режима активной задачи."""
        mode_frame = ctk.CTkFrame(parent, corner_radius=8)
        mode_frame.pack(fill="x", padx=5, pady=(0, 8))

        ctk.CTkLabel(
            mode_frame,
            text="Режим вейвлет-анализа",
            font=AppTheme.section_title_font(),
            anchor="w"
        ).pack(fill="x", padx=12, pady=(10, 6))

        self.analysis_source_label = ctk.CTkLabel(
            mode_frame,
            text="Источник: сначала загрузите изображение",
            font=AppTheme.body_font(),
            text_color=AppTheme.TEXT_SECONDARY,
            anchor="w"
        )
        self.analysis_source_label.pack(fill="x", padx=12, pady=(0, 6))

        self.analysis_mode_selector = ctk.CTkSegmentedButton(
            mode_frame,
            values=list(self.ANALYSIS_MODE_LABELS.keys()),
            variable=self.analysis_mode_var,
            command=self.on_analysis_mode_changed,
            height=AppTheme.BUTTON_HEIGHT,
            state="disabled"
        )
        self.analysis_mode_selector.pack(fill="x", padx=12, pady=(0, 8))
        self.analysis_mode_selector.set("1D-анализ")

        self.analysis_mode_description = ctk.CTkLabel(
            mode_frame,
            text="Создайте или активируйте задачу, чтобы выбрать режим.",
            font=AppTheme.caption_font(),
            text_color=AppTheme.MUTED,
            anchor="w",
            justify="left",
            wraplength=430
        )
        self.analysis_mode_description.pack(fill="x", padx=12, pady=(0, 10))

    def on_analysis_mode_changed(self, selected_label):
        """Сохранить выбранный режим в активной задаче и обновить интерфейс."""
        if not self.current_task:
            self.analysis_mode_selector.set("1D-анализ")
            return

        if not self.current_task.image_path:
            current_label = self.ANALYSIS_MODE_NAMES[
                self.current_task.analysis_mode
            ]
            self.analysis_mode_var.set(current_label)
            self.analysis_mode_selector.set(current_label)
            return

        selected_mode = self.ANALYSIS_MODE_LABELS[selected_label]
        if self.current_task.analysis_mode == selected_mode:
            return

        self.current_task.set_analysis_mode(selected_mode)
        self.progress_manager.log_info(
            f"{self.current_task.task_name}: выбран режим {selected_label}"
        )
        self._apply_analysis_mode_ui()
        self._update_pipeline_controls_state()
        self._update_action_availability()
        self._update_tasks_display()

    def _apply_analysis_mode_ui(self):
        """Показать только настройки, корректные для режима активной задачи."""
        if not self.current_task:
            self.analysis_mode_var.set("1D-анализ")
            self.analysis_mode_selector.set("1D-анализ")
            self.analysis_mode_selector.configure(state="disabled")
            self.analysis_mode_description.configure(
                text="Создайте или активируйте задачу, чтобы выбрать режим.",
                text_color=AppTheme.MUTED
            )
            self.analysis_source_label.configure(
                text="Источник: сначала загрузите изображение",
                text_color=AppTheme.TEXT_SECONDARY
            )
            if not self.extremes_section.winfo_manager():
                self.extremes_section.pack(
                    fill="x", padx=5, pady=2,
                    before=self.analysis_next_button
                )
            self.two_d_section.pack_forget()
            self.knn_section.pack(fill="x", padx=5, pady=2, before=self.intermediate_section)
            self.statistics_section.pack(
                fill="x", padx=5, pady=2, before=self.knn_section
            )
            self.output_extremes_section.pack(
                fill="x", padx=5, pady=2, before=self.statistics_section
            )
            self.app_start_button.configure(
                text="Запустить",
                state="normal",
                fg_color=AppTheme.PRIMARY,
                hover_color=AppTheme.PRIMARY_HOVER
            )
            return

        mode = self.current_task.analysis_mode
        mode_label = self.ANALYSIS_MODE_NAMES[mode]
        self.analysis_mode_var.set(mode_label)
        self.analysis_mode_selector.set(mode_label)
        source_loaded = bool(self.current_task.image_path)
        self.analysis_mode_selector.configure(
            state="normal" if source_loaded else "disabled"
        )
        self.analysis_source_label.configure(
            text=(
                f"Источник: {os.path.basename(self.current_task.image_path)}"
                if source_loaded else
                "Источник: сначала загрузите изображение"
            ),
            text_color=(
                AppTheme.TEXT_ON_DARK if source_loaded
                else AppTheme.TEXT_SECONDARY
            )
        )

        if mode == "1d":
            self.analysis_mode_description.configure(
                text=("Одномерное преобразование Морле по строкам и/или "
                      "столбцам изображения."),
                text_color=AppTheme.TEXT_SECONDARY
            )
            if not self.extremes_section.winfo_manager():
                self.extremes_section.pack(
                    fill="x", padx=5, pady=2,
                    before=self.analysis_next_button
                )
            self.two_d_section.pack_forget()
            if not self.knn_section.winfo_manager():
                self.knn_section.pack(fill="x", padx=5, pady=2, before=self.intermediate_section)
            if not self.statistics_section.winfo_manager():
                self.statistics_section.pack(
                    fill="x", padx=5, pady=2, before=self.knn_section
                )
            if not self.output_extremes_section.winfo_manager():
                self.output_extremes_section.pack(
                    fill="x", padx=5, pady=2, before=self.statistics_section
                )
            self.app_start_button.configure(
                text="Запустить",
                state="normal",
                fg_color=AppTheme.PRIMARY,
                hover_color=AppTheme.PRIMARY_HOVER
            )
        else:
            self.analysis_mode_description.configure(
                text=("Двумерное комплексное преобразование Морле по масштабам "
                      "и ориентациям. Результат содержит модуль и фазу."),
                text_color=AppTheme.TEXT_SECONDARY
            )
            self.extremes_section.pack_forget()
            if not self.two_d_section.winfo_manager():
                self.two_d_section.pack(
                    fill="x", padx=5, pady=2,
                    before=self.analysis_next_button
                )
            self.output_extremes_section.pack_forget()
            self.statistics_section.pack_forget()
            self.knn_section.pack_forget()
            self.app_start_button.configure(
                text="Недоступно: настройте параметры 2D",
                state="disabled",
                fg_color=AppTheme.DISABLED_DARK,
                hover_color=AppTheme.DISABLED_DARK
            )

    def _create_right_panel(self, parent=None, include_compute=True):
        """Создание правой панели с настройками вывода"""
        panel = ctk.CTkFrame(parent or self.main_container, fg_color="transparent")

        # Используем ScrollableFrame для прокрутки
        scrollable_panel = ScrollableFrame(panel)
        scrollable_panel.pack(fill="both", expand=True)

        # Заголовок панели
        header = ctk.CTkLabel(
            scrollable_panel.scrollable_frame,
            text="Настройки вывода и вычислений",
            font=AppTheme.panel_title_font(),
            anchor="w"
        )
        header.pack(fill="x", padx=10, pady=(10, 15))

        self.compute_settings_section = CollapsibleFrame(scrollable_panel.scrollable_frame,
                                                         title="Настройки вычислений")
        self.compute_settings_section.pack(fill="x", padx=5, pady=2)
        temp_label = ctk.CTkLabel(
            self.compute_settings_section.content,
            text="Загрузка настроек...",
            font=AppTheme.body_font(),
            text_color=AppTheme.MUTED)
        temp_label.pack(pady=10)

        self.pipeline_section = CollapsibleFrame(
            scrollable_panel.scrollable_frame,
            title="Сценарий вычислений"
        )
        self.pipeline_section.pack(fill="x", padx=5, pady=2)
        self._setup_pipeline_section()

        self.wavelet_section = CollapsibleFrame(scrollable_panel.scrollable_frame, title="Вейвлет-преобразование")
        self.wavelet_section.pack(fill="x", padx=5, pady=2)
        self._setup_wavelet_section()

        # Секция точек экстремумов (вывод)
        self.output_extremes_section = CollapsibleFrame(scrollable_panel.scrollable_frame,
                                                        title="Вывод точек экстремумов")
        self.output_extremes_section.pack(fill="x", padx=5, pady=2)
        self._setup_output_extremes_section()
        self.output_extremes_section.toggle()

        self.statistics_section = CollapsibleFrame(
            scrollable_panel.scrollable_frame,
            title="Статистики и синхронизации"
        )
        self.statistics_section.pack(fill="x", padx=5, pady=2)
        self._setup_statistics_section()
        self.statistics_section.toggle()

        # Секция K-ближайших соседей
        self.knn_section = CollapsibleFrame(scrollable_panel.scrollable_frame, title="K-ближайшие соседи")
        self.knn_section.pack(fill="x", padx=5, pady=2)
        self._setup_knn_section()
        self.knn_section.toggle()

        # Секция промежуточных вычислений
        self.intermediate_section = CollapsibleFrame(scrollable_panel.scrollable_frame,
                                                     title="Промежуточные вычисления")
        self.intermediate_section.pack(fill="x", padx=5, pady=2)
        self._setup_intermediate_section()
        self.intermediate_section.toggle()

        if include_compute:
            self.compute_section = ctk.CTkFrame(
                scrollable_panel.scrollable_frame, fg_color="transparent"
            )
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
                font=AppTheme.body_font(),
                text_color=AppTheme.DANGER
            )
            error_label.pack(pady=10)
            return

        # Информация о доступном и активном вычислительном устройстве.
        backend_info = self.image_processor.get_backend_info()
        gpu_available = backend_info['gpu_available']

        status_label = ctk.CTkLabel(
            self.compute_settings_section.content,
            text="GPU доступен" if gpu_available else "GPU не обнаружен",
            font=AppTheme.body_font(),
            text_color=AppTheme.GPU_AVAILABLE if gpu_available else AppTheme.WARNING,
            anchor="w"
        )
        status_label.pack(fill="x", pady=(0, 4))

        if gpu_available:
            gpu_details = backend_info['gpu_device_name']
            if backend_info['gpu_memory'] != "N/A":
                gpu_details += f" | Память: {backend_info['gpu_memory']}"
        else:
            gpu_details = "Вычисления будут выполняться на CPU"

        details_label = ctk.CTkLabel(
            self.compute_settings_section.content,
            text=gpu_details,
            font=AppTheme.caption_font(),
            text_color=AppTheme.TEXT_SECONDARY,
            anchor="w"
        )
        details_label.pack(fill="x", pady=(0, 10))

        self.gpu_switch = ctk.CTkSwitch(
            self.compute_settings_section.content,
            text="Использовать GPU",
            variable=self.use_gpu_var,
            command=self.toggle_gpu_backend,
            state="normal" if gpu_available else "disabled"
        )
        self.use_gpu_var.set(self.image_processor.backend.use_gpu)
        self.gpu_switch.pack(fill="x", pady=(2, 8))

    def toggle_gpu_backend(self):
        """Применить состояние переключателя CPU/GPU."""
        requested_state = self.use_gpu_var.get()
        enabled, backend_info = self.image_processor.set_gpu_enabled(requested_state)
        self.use_gpu_var.set(enabled)
        active_device = "GPU" if enabled else "CPU"
        self.progress_manager.log_info(f"Для вычислений выбран {active_device}")
        self._update_compute_settings_display()
        self.after(0, self._update_gpu_section)


    def _update_compute_settings_display(self):
        """Обновление отображения настроек вычислений"""
        if not hasattr(self, 'image_processor') or self.image_processor is None:
            return

        knn_gpu_available = (
            self.image_processor.backend.use_gpu and
            self.image_processor.knn_processor.is_gpu_available()
        )

        # Обновляем информацию о KNN
        knn_info = f"KNN: {'GPU' if knn_gpu_available else 'CPU'}"
        if hasattr(self, 'knn_status_label'):
            self.knn_status_label.configure(
                text=knn_info,
                text_color=AppTheme.GPU_AVAILABLE if knn_gpu_available else AppTheme.WARNING
            )

    def _setup_load_section(self):
        """Настройка секции загрузки изображения"""
        self.load_button = ctk.CTkButton(
            self.load_section.content,
            text="Загрузить и обрезать изображение",
            command=self.load_image_callback,
            width=AppTheme.ACTION_BUTTON_WIDTH,
            height=AppTheme.BUTTON_HEIGHT,
            font=AppTheme.section_title_font(),
            fg_color=AppTheme.PRIMARY,
            hover_color=AppTheme.PRIMARY_HOVER
        )
        self.load_section.add_widget(
            self.load_button, fill="none", anchor="w", pady=5
        )

        self.print_load_image = ctk.CTkLabel(
            self.load_section.content,
            text="Изображение не загружено",
            font=AppTheme.body_font(),
            text_color=AppTheme.MUTED,
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
            font=AppTheme.body_font(),
            anchor="w"
        )
        pipette_label.pack(fill="x")

        self.pipette_button = ctk.CTkButton(
            pipette_frame,
            text="Активировать пипетку",
            command=self.pipette_channel,
            width=AppTheme.ACTION_BUTTON_WIDTH,
            height=AppTheme.BUTTON_HEIGHT,
            font=AppTheme.body_font()
        )
        pipette_frame.pack(fill="x")
        self.pipette_button.pack(anchor="w", pady=(5, 0))

        # Преобразование Грамма-Шмидта
        gram_frame = ctk.CTkFrame(self.channel_section.content, fg_color="transparent")
        self.channel_section.add_widget(gram_frame, pady=(10, 2))

        gram_label = ctk.CTkLabel(
            gram_frame,
            text="Преобразование каналов:",
            font=AppTheme.body_font(),
            anchor="w"
        )
        gram_label.pack(fill="x")

        self.gram_shmidt_button = ctk.CTkButton(
            gram_frame,
            text="Преобразование Грамма-Шмидта",
            command=self.gramm_shmidt_transform,
            width=AppTheme.ACTION_BUTTON_WIDTH,
            height=AppTheme.BUTTON_HEIGHT,
            font=AppTheme.body_font()
        )
        gram_frame.pack(fill="x")
        self.gram_shmidt_button.pack(anchor="w", pady=(5, 0))

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
            height=AppTheme.BUTTON_HEIGHT
        )
        self.button_save_scales.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.button_load_scales_file = ctk.CTkButton(
            button_frame,
            text="Загрузить из файла",
            command=self.load_scales_from_file,
            height=AppTheme.BUTTON_HEIGHT
        )
        self.button_load_scales_file.pack(side="left", fill="x", expand=True, padx=(5, 0))

        # Отображение загруженных масштабов
        self.label_custom_scale = ctk.CTkLabel(
            self.scales_section.content,
            text="",
            font=AppTheme.caption_font(),
            text_color=AppTheme.MUTED,
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
            text="Направления 1D-преобразования:",
            font=AppTheme.body_font(),
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

        self.extremes_hint_label = ctk.CTkLabel(
            direction_frame,
            text="Огибающие и синхронизация рассчитываются для построчного результата.",
            font=AppTheme.caption_font(),
            text_color=AppTheme.TEXT_SECONDARY,
            anchor="w",
            justify="left",
            wraplength=400
        )
        self.extremes_hint_label.pack(fill="x", pady=(2, 6))

        # Типы экстремумов
        type_frame = ctk.CTkFrame(self.extremes_section.content, fg_color="transparent")
        self.extremes_section.add_widget(type_frame, pady=2)

        type_label = ctk.CTkLabel(
            type_frame,
            text="Типы экстремумов:",
            font=AppTheme.body_font(),
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
            font=AppTheme.body_font(),
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
            if hasattr(self, "workspace_tabs"):
                self.workspace_tabs.set("Данные")

            self.progress_manager.log_info(f"Добавлена новая задача #{task_id}")

        except Exception as e:
            self.progress_manager.log_error(f"Ошибка при добавлении задачи: {e}")
            mb.showerror("Ошибка", f"Не удалось добавить задачу: {e}")

    def _update_tasks_display(self):
        """Обновление отображения списка задач"""
        if not hasattr(self, '_task_cards'):
            self._task_cards = {}
        active_ids = {id(task) for task in self.image_processor.tasks}
        for identifier in list(self._task_cards):
            if identifier not in active_ids:
                card = self._task_cards.pop(identifier)
                self.task_widgets.remove(card['frame'])
                card['frame'].destroy()

        # Обновляем статус
        task_count = len(self.image_processor.tasks)
        if task_count == 0:
            self.tasks_status_label.configure(
                text="Задачи не добавлены",
                text_color=AppTheme.MUTED
            )
        else:
            status_text = f"Всего задач: {task_count}"
            active_tasks = sum(
                1 for task in self.image_processor.tasks
                if (task.image_path and len(task.scales) > 0 and
                    (task.analysis_mode == "2d" or task.process_rows or
                     task.process_columns))
            )
            if active_tasks > 0:
                status_text += f" | Готовы к вычислениям: {active_tasks}"

            self.tasks_status_label.configure(
                text=status_text,
                text_color=AppTheme.TEXT_ON_DARK
            )

        # Создаем виджеты для каждой задачи
        for task in self.image_processor.tasks:
            if id(task) not in self._task_cards:
                self._create_task_widget(task)
            card = self._task_cards[id(task)]
            status = self._get_task_status(task)
            image_info = os.path.basename(task.image_path) if task.image_path else 'Изображение не загружено'
            if task.image_path and task.original_image is not None:
                h, w = task.original_image.shape[:2]
                image_info += f' ({w}×{h}px)'
            card['title'].configure(text=task.task_name)
            card['mode'].configure(text='1D' if task.analysis_mode == '1d' else '2D',
                                   fg_color=AppTheme.PRIMARY if task.analysis_mode == '1d' else AppTheme.MODE_2D)
            card['status'].configure(text=status, text_color=self._get_status_color(status))
            card['image'].configure(text=image_info)
            card['scales'].configure(text=self._get_scales_info(task))
            card['lines'].configure(text='\n'.join(self._get_processing_lines(task)))
            selected = self.current_task is task
            card['frame'].configure(border_color=AppTheme.ACTIVE_BORDER if selected else AppTheme.BORDER,
                                    border_width=2 if selected else 1)

    def _create_task_widget(self, task):
        """Создание виджета для отображения задачи"""
        # Основной фрейм задачи
        task_frame = ctk.CTkFrame(
            self.tasks_container,
            border_width=1,
            border_color=AppTheme.BORDER,
            corner_radius=8
        )
        task_frame.pack(fill="x", padx=5, pady=2)
        self.task_widgets.append(task_frame)

        # Верхняя часть - заголовок и статус
        header_frame = ctk.CTkFrame(task_frame, fg_color="transparent")
        header_frame.pack(fill="x", padx=10, pady=(7, 3))

        # Заголовок задачи с номером и статусом
        title_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        title_frame.pack(fill="x")

        # Номер задачи и название
        task_title = ctk.CTkLabel(
            title_frame,
            text=f"{task.task_name}",
            font=AppTheme.section_title_font(),
            anchor="w"
        )
        task_title.pack(side="left", fill="x", expand=True)

        mode_badge = ctk.CTkLabel(
            title_frame,
            text="1D" if task.analysis_mode == "1d" else "2D",
            width=30,
            height=AppTheme.BADGE_HEIGHT,
            corner_radius=5,
            fg_color=AppTheme.PRIMARY if task.analysis_mode == "1d" else AppTheme.MODE_2D,
            font=AppTheme.caption_font()
        )
        mode_badge.pack(side="left", padx=(6, 4))

        # Статус задачи
        status_text = self._get_task_status(task)
        status_color = self._get_status_color(status_text)
        task_status = ctk.CTkLabel(
            title_frame,
            text=status_text,
            font=AppTheme.caption_font(),
            text_color=status_color
        )
        task_status.pack(side="right", padx=(5, 0))

        # Основная информация о задаче
        info_frame = ctk.CTkFrame(task_frame, fg_color="transparent")
        info_frame.pack(fill="x", padx=10, pady=(0, 3))

        # Первая строка информации
        row1_frame = ctk.CTkFrame(info_frame, fg_color="transparent")
        row1_frame.pack(fill="x")

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
            font=AppTheme.body_font(),
            height=18,
            anchor="w"
        )
        image_label.pack(side="left", fill="x", expand=True)

        # Вторая строка информации
        row2_frame = ctk.CTkFrame(info_frame, fg_color="transparent")
        row2_frame.pack(fill="x")

        # Информация о масштабах
        scales_info = self._get_scales_info(task)
        scales_label = ctk.CTkLabel(
            row2_frame,
            text=f"{scales_info}",
            font=AppTheme.body_font(),
            height=18,
            anchor="w"
        )
        scales_label.pack(side="left", fill="x", expand=True)

        # Каждая настройка выводится отдельной строкой, чтобы текст не обрезался.
        for line in ['\n'.join(self._get_processing_lines(task))]:
            line_label = ctk.CTkLabel(
                info_frame,
                text=line,
                font=AppTheme.body_font(),
                height=18,
                anchor="w",
                justify="left"
            )
            line_label.pack(fill="x")

        # Кнопки управления задачей
        button_frame = ctk.CTkFrame(task_frame, fg_color="transparent")
        button_frame.pack(fill="x", padx=10, pady=(3, 7))

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
            width=84,
            height=AppTheme.COMPACT_CONTROL_HEIGHT,
            font=AppTheme.body_font(),
            fg_color=AppTheme.PRIMARY,
            hover_color=AppTheme.PRIMARY_HOVER
        )
        activate_btn.pack(side="left", padx=(0, 8))

        # Кнопка удаления
        remove_btn = ctk.CTkButton(
            button_frame,
            text="Удалить",
            command=remove_task,
            width=64,
            height=AppTheme.COMPACT_CONTROL_HEIGHT,
            font=AppTheme.body_font(),
            fg_color=AppTheme.DANGER,
            hover_color=AppTheme.DANGER_HOVER
        )
        remove_btn.pack(side="left")

        # Подсветка текущей активной задачи
        if self.current_task and self.current_task.task_id == task.task_id:
            task_frame.configure(border_color=AppTheme.ACTIVE_BORDER, border_width=2)
        self._task_cards[id(task)] = dict(frame=task_frame, title=task_title, mode=mode_badge,
                                         status=task_status, image=image_label, scales=scales_label,
                                         lines=line_label)

    @staticmethod
    def _get_task_status(task):
        """Получить текстовый статус задачи"""
        if not task.image_path:
            return "Ожидает изображение"
        elif len(task.scales) == 0:
            return "Ожидает масштабы"
        elif task.analysis_mode == "1d" and not (
                task.process_rows or task.process_columns):
            return "Выберите направление"
        return "Готова к вычислениям"

    @staticmethod
    def _get_status_color(status):
        """Получить цвет статуса"""
        status_colors = {
            "Ожидает изображение": AppTheme.WARNING,
            "Ожидает масштабы": AppTheme.WARNING,
            "Выберите направление": AppTheme.WARNING,
            "Готова к вычислениям": AppTheme.SUCCESS
        }
        return status_colors.get(status, AppTheme.DISABLED)

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

    def _get_processing_lines(self, task):
        """Получить отдельные строки состояния для карточки задачи."""
        mode_name = self.ANALYSIS_MODE_NAMES[task.analysis_mode]
        lines = []

        if task.analysis_mode == "1d":
            directions = []
            if task.process_rows:
                directions.append("строки")
            if task.process_columns:
                directions.append("столбцы")
            lines.append(
                f"{mode_name}. Направления: " +
                (", ".join(directions) if directions else "не выбраны")
            )
            plan = task.resolve_pipeline()
            enabled_stages = ["вейвлет"]
            if plan.extrema:
                enabled_stages.append("экстремумы")
            if plan.envelopes:
                enabled_stages.append("огибающие")
            if plan.knn:
                enabled_stages.append("KNN")
            if plan.statistics:
                enabled_stages.append("статистики")
            if plan.synchronization:
                enabled_stages.append("синхронизации")
            if plan.ml:
                enabled_stages.append("ML-кластеризация")
            lines.append("Этапы: " + ", ".join(enabled_stages))
            lines.append(
                f"KNN: {task.k_neighbors} соседей"
                if plan.knn else "KNN: выключен"
            )
        else:
            lines.append(f"{mode_name}. Ориентаций: {len(task.orientations)}")
            lines.append("KNN: не используется в режиме 2D")

        if task.has_colors_selected():
            color1 = [int(value) for value in task.color1[:3]]
            color2 = [int(value) for value in task.color2[:3]]
            lines.append("Пипетка: настроена")
            lines.append(
                f"Цвет 1 — R: {color1[0]}, G: {color1[1]}, B: {color1[2]}"
            )
            lines.append(
                f"Цвет 2 — R: {color2[0]}, G: {color2[1]}, B: {color2[2]}"
            )
        else:
            lines.append("Пипетка: не настроена")

        if task.is_gram_schmidt_applied():
            lines.append("Преобразование Грамма-Шмидта: применено")

        return lines

    def _update_ui_for_current_task(self):
        """Обновление UI в соответствии с текущей задачей"""
        if hasattr(self, 'results_panel'):
            self.results_panel.load_folder(self.current_task.task_folder_path if self.current_task else '')
        if self.current_task:
            self.analysis_mode_selector.configure(state="normal")
            self._loading_task_settings = True
            self.row_var.set(self.current_task.process_rows)
            self.col_var.set(self.current_task.process_columns)
            self.max_var.set(self.current_task.find_maxima)
            self.min_var.set(self.current_task.find_minima)
            self.pipeline_preset_var.set(self.current_task.pipeline_preset)
            self.calculate_extrema_var.set(
                self.current_task.calculate_extrema
            )
            self.calculate_envelopes_var.set(
                self.current_task.calculate_envelopes
            )
            self.calculate_knn_var.set(self.current_task.calculate_knn)
            self.wp_var1.set(self.current_task.output_wavelet_image)
            self.wp_var2.set(self.current_task.output_wavelet_text)
            self.wavelet_numpy_var.set(
                self.current_task.output_wavelet_numpy
            )
            self.p_ex_var1.set(self.current_task.output_extremes_text)
            self.p_ex_var2.set(self.current_task.output_extremes_image)
            self.envelope_text_var.set(
                self.current_task.output_envelopes_text
            )
            self.envelope_image_var.set(
                self.current_task.output_envelopes_image
            )
            self.knn_bool_text_var.set(self.current_task.output_knn_text)
            self.knn_bool_image_var.set(self.current_task.output_knn_image)
            self.print_channels_txt_var.set(self.current_task.save_source_channels)
            self.centering_means_var.set(
                self.current_task.save_centering_means
            )
            self.calculate_statistics_var.set(
                self.current_task.calculate_statistics
            )
            self.statistics_image_var.set(
                self.current_task.statistics_output_image
            )
            self.statistics_csv_var.set(
                self.current_task.statistics_output_csv
            )
            self.calculate_sync_var.set(
                self.current_task.calculate_synchronization
            )
            self.sync_heatmap_var.set(
                self.current_task.synchronization_output_heatmap
            )
            self.sync_matrix_csv_var.set(
                self.current_task.synchronization_output_matrix_csv
            )
            self.sync_pairs_csv_var.set(
                self.current_task.synchronization_output_pairs_csv
            )
            self.scale_block_sizes_var.set(
                ", ".join(
                    str(value)
                    for value in self.current_task.scale_block_sizes
                )
            )
            self.sync_stride_var.set(str(self.current_task.row_sync_stride))
            self.sync_tolerance_var.set(
                str(self.current_task.row_sync_tolerance)
            )
            self.sync_metric_var.set(
                self.current_task.row_sync_metrics[0]
                if self.current_task.row_sync_metrics else "jaccard"
            )
            self.orientations_var.set(
                ", ".join(f"{value:g}" for value in self.current_task.orientations)
            )
            self.morlet_omega0_var.set(f"{self.current_task.morlet_omega0:g}")
            self.morlet_anisotropy_var.set(f"{self.current_task.morlet_anisotropy:g}")
            self._loading_task_settings = False
            # Обновляем информацию о загруженном изображении
            if self.current_task.image_path:
                self.load_button.configure(
                    text="Изображение загружено",
                    fg_color=AppTheme.SUCCESS,
                    hover_color=AppTheme.SUCCESS_HOVER
                )
                text = f"Файл: {os.path.basename(self.current_task.image_path)}"
                self.print_load_image.configure(text=text, text_color=AppTheme.TEXT_ON_DARK)
            else:
                self.load_button.configure(
                    text="Загрузить и обрезать изображение",
                    fg_color=AppTheme.PRIMARY,
                    hover_color=AppTheme.PRIMARY_HOVER
                )
                self.print_load_image.configure(text="Изображение не загружено", text_color=AppTheme.MUTED)

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
                    fg_color=AppTheme.DISABLED,
                    hover_color=AppTheme.DISABLED_HOVER
                )
                # Активируем кнопку Грамма-Шмидта если есть цвета
                self.gram_shmidt_button.configure(
                    text="Преобразование Грамма-Шмидта",  # Всегда сбрасываем текст
                    state='normal',
                    fg_color=AppTheme.PRIMARY,
                    hover_color=AppTheme.PRIMARY_HOVER
                )
            else:
                self.pipette_button.configure(
                    text="Активировать пипетку",
                    state='normal',
                    fg_color=AppTheme.PRIMARY,
                    hover_color=AppTheme.PRIMARY_HOVER
                )
                # Деактивируем кнопку Грамма-Шмидта если нет цветов
                self.gram_shmidt_button.configure(
                    text="Преобразование Грамма-Шмидта",  # Всегда сбрасываем текст
                    state='disabled',
                    fg_color=AppTheme.DISABLED,
                    hover_color=AppTheme.DISABLED_HOVER
                )

            # Обновляем KNN
            self.knn_text_var.set(str(self.current_task.k_neighbors))

            # Сбрасываем состояние кнопок масштабов для новой задачи
            if len(self.current_task.scales) == 0:
                self.button_save_scales.configure(
                    text="Сохранить значения",
                    fg_color=AppTheme.PRIMARY,
                    hover_color=AppTheme.PRIMARY_HOVER,
                    state='normal'
                )
                self.button_load_scales_file.configure(
                    text="Загрузить из файла",
                    fg_color=AppTheme.PRIMARY,
                    hover_color=AppTheme.PRIMARY_HOVER,
                    state='normal'
                )
                self.label_custom_scale.configure(text="", text_color=AppTheme.MUTED)

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
                self.label_custom_scale.configure(text=scales_text, text_color=AppTheme.TEXT_ON_DARK)

        else:
            # Сбрасываем UI если нет активной задачи
            self.load_button.configure(
                text="Загрузить и обрезать изображение",
                fg_color=AppTheme.PRIMARY,
                hover_color=AppTheme.PRIMARY_HOVER
            )
            self.print_load_image.configure(text="Изображение не загружено", text_color=AppTheme.MUTED)
            self.pipette_button.configure(
                text="Активировать пипетку",
                state='normal',
                fg_color=AppTheme.PRIMARY,
                hover_color=AppTheme.PRIMARY_HOVER
            )
            self.gram_shmidt_button.configure(
                text="Преобразование Грамма-Шмидта",
                state='disabled',
                fg_color=AppTheme.DISABLED,
                hover_color=AppTheme.DISABLED_HOVER
            )
            self.button_save_scales.configure(
                text="Сохранить значения",
                fg_color=AppTheme.PRIMARY,
                hover_color=AppTheme.PRIMARY_HOVER,
                state='disabled'
            )
            self.button_load_scales_file.configure(
                text="Загрузить из файла",
                fg_color=AppTheme.PRIMARY,
                hover_color=AppTheme.PRIMARY_HOVER,
                state='disabled'
            )
            self.label_custom_scale.configure(text="", text_color=AppTheme.MUTED)

        self._apply_analysis_mode_ui()
        self._update_pipeline_controls_state()
        self._update_statistics_controls_state()
        self._update_action_availability()
        self._update_ml_tab_state()

        # Всегда обновляем отображение задач для подсветки активной
        self._update_tasks_display()

    def _update_action_availability(self):
        """Применить естественную последовательность обязательных шагов."""
        task = self.current_task
        self._update_navigation_buttons()
        if task is None:
            self._set_row_analysis_controls_enabled(False)
            self.analysis_mode_selector.configure(state="disabled")
            self.load_button.configure(state="disabled")
            self.pipette_button.configure(state="disabled")
            self.gram_shmidt_button.configure(state="disabled")
            self.button_save_scales.configure(state="disabled")
            self.button_load_scales_file.configure(state="disabled")
            self.app_start_button.configure(state="disabled")
            self._set_workflow_status(
                "Шаг 1 из 3: создайте задачу и загрузите изображение",
                AppTheme.TEXT_SECONDARY
            )
            return

        self._set_row_analysis_controls_enabled(
            task.analysis_mode == "1d" and task.process_rows
        )
        self.load_button.configure(state="normal")
        if not task.image_path:
            self.analysis_mode_selector.configure(state="disabled")
            self.pipette_button.configure(state="disabled")
            self.gram_shmidt_button.configure(state="disabled")
            self.button_save_scales.configure(state="disabled")
            self.button_load_scales_file.configure(state="disabled")
            self.app_start_button.configure(state="disabled")
            self._set_workflow_status(
                "Шаг 1 из 3: загрузите изображение",
                AppTheme.WARNING
            )
            return

        self.analysis_mode_selector.configure(state="normal")
        if not task.has_colors_selected():
            self.pipette_button.configure(state="normal")
        self.button_save_scales.configure(state="normal")
        self.button_load_scales_file.configure(state="normal")

        if len(task.scales) == 0:
            self.app_start_button.configure(state="disabled")
            self._set_workflow_status(
                "Шаг 2 из 3: настройте режим анализа и масштабы",
                AppTheme.WARNING
            )
            return

        if task.analysis_mode == "1d" and not (
                task.process_rows or task.process_columns):
            self.app_start_button.configure(state="disabled")
            self._set_workflow_status(
                "Шаг 2 из 3: выберите строки и/или столбцы для 1D-анализа",
                AppTheme.WARNING
            )
            return

        self.app_start_button.configure(
            state="normal",
            text="Запустить",
            fg_color=AppTheme.PRIMARY,
            hover_color=AppTheme.PRIMARY_HOVER
        )
        self._set_workflow_status(
            "Шаг 3 из 3: настройте форматы вывода и запустите вычисления",
            AppTheme.TEXT_ON_DARK
        )

    def _set_workflow_status(self, text, color):
        """Показать текущий этап в общей строке состояния журнала."""
        progress_manager = getattr(self, "progress_manager", None)
        if progress_manager is not None:
            progress_manager.set_status(text, color)

    def _set_row_analysis_controls_enabled(self, enabled):
        """Огибающие и KNN доступны только для построчного результата 1D."""
        state = "normal" if enabled else "disabled"
        for widget in (
                self.max_checkbox, self.min_checkbox, self.entry_near_point,
        ):
            if widget is not None:
                widget.configure(state=state)

        if self.extremes_hint_label is not None:
            self.extremes_hint_label.configure(
                text=("Огибающие и синхронизация доступны для построчного результата."
                      if enabled else
                      "Недоступно: включите 1D-преобразование по строкам."),
                text_color=(AppTheme.TEXT_SECONDARY if enabled else AppTheme.WARNING)
            )
        self._update_pipeline_controls_state()

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
                    fg_color=AppTheme.DANGER,
                    hover_color=AppTheme.DANGER_HOVER
                )
                self.print_load_image.configure(text="Ошибка загрузки изображения", text_color=AppTheme.DANGER)
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

        try:
            self.image_processor.gram_shmidt_transform_for_task(self.current_task)
        except ValueError as error:
            self.progress_manager.log_error(str(error))
            mb.showwarning("Не удалось преобразовать каналы", str(error))
            return
        self.gram_shmidt_button.configure(
            text="Преобразование применено",
            state='disabled',
            fg_color=AppTheme.DISABLED,
            hover_color=AppTheme.DISABLED_HOVER
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
                fg_color=AppTheme.SUCCESS,
                hover_color=AppTheme.SUCCESS_HOVER
            )
            self.button_load_scales_file.configure(state='disabled')

            scale_info = f"Масштабы: {start}-{end} (шаг {step})"
            self.label_custom_scale.configure(text=scale_info, text_color=AppTheme.TEXT_ON_DARK)

            self._update_tasks_display()
            self._update_action_availability()

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
            try:
                self.image_processor.load_scales_from_file_for_task(self.current_task, filename)
            except (ValueError, OSError) as error:
                mb.showerror("Масштабы", str(error))
                return

            if self.current_task.num_scale <= 10:
                scales_text = f"Масштабы: {', '.join(map(str, self.current_task.scales))}"
            else:
                scales_text = f"Загружено {self.current_task.num_scale} масштабов"

            self.label_custom_scale.configure(text=scales_text, text_color=AppTheme.TEXT_ON_DARK)

            self.button_load_scales_file.configure(
                text="Файл загружен",
                fg_color=AppTheme.SUCCESS,
                hover_color=AppTheme.SUCCESS_HOVER
            )
            self.button_save_scales.configure(
                text="Сохранить значения",
                fg_color=AppTheme.DISABLED,
                hover_color=AppTheme.DISABLED_HOVER,
                state='disabled'
            )

            self._update_tasks_display()
            self._update_action_availability()
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

    def _store_settings_for_current_task(self, *args):
        """Сохранить значения элементов управления в активной задаче."""
        task = getattr(self, 'current_task', None)
        if task is None or self._loading_task_settings:
            return

        task.process_rows = bool(self.row_var.get())
        task.process_columns = bool(self.col_var.get())
        task.find_maxima = bool(self.max_var.get())
        task.find_minima = bool(self.min_var.get())
        task.calculate_extrema = bool(self.calculate_extrema_var.get())
        task.calculate_envelopes = bool(self.calculate_envelopes_var.get())
        task.calculate_knn = bool(self.calculate_knn_var.get())
        task.output_wavelet_image = bool(self.wp_var1.get())
        task.output_wavelet_text = bool(self.wp_var2.get())
        task.output_wavelet_numpy = bool(self.wavelet_numpy_var.get())
        task.output_extremes_text = bool(self.p_ex_var1.get())
        task.output_extremes_image = bool(self.p_ex_var2.get())
        task.output_envelopes_text = bool(self.envelope_text_var.get())
        task.output_envelopes_image = bool(self.envelope_image_var.get())
        task.output_knn_text = bool(self.knn_bool_text_var.get())
        task.output_knn_image = bool(self.knn_bool_image_var.get())
        task.save_source_channels = bool(self.print_channels_txt_var.get())
        task.save_centering_means = bool(self.centering_means_var.get())
        task.calculate_statistics = bool(self.calculate_statistics_var.get())
        task.statistics_output_image = bool(self.statistics_image_var.get())
        task.statistics_output_csv = bool(self.statistics_csv_var.get())
        task.calculate_synchronization = bool(self.calculate_sync_var.get())
        task.synchronization_output_heatmap = bool(self.sync_heatmap_var.get())
        task.synchronization_output_matrix_csv = bool(
            self.sync_matrix_csv_var.get()
        )
        task.synchronization_output_pairs_csv = bool(
            self.sync_pairs_csv_var.get()
        )

        try:
            block_sizes = [
                int(value.strip())
                for value in self.scale_block_sizes_var.get().split(',')
                if value.strip()
            ]
            if block_sizes and all(value > 0 for value in block_sizes):
                task.scale_block_sizes = block_sizes
        except ValueError:
            pass

        try:
            stride = int(self.sync_stride_var.get())
            if stride > 0:
                task.row_sync_stride = stride
        except ValueError:
            pass

        try:
            tolerance = int(self.sync_tolerance_var.get())
            if tolerance >= 0:
                task.row_sync_tolerance = tolerance
        except ValueError:
            pass

        if self.sync_metric_var.get() in {"jaccard", "dice", "phi"}:
            task.row_sync_metrics = [self.sync_metric_var.get()]

        try:
            orientations = [
                float(value.strip())
                for value in self.orientations_var.get().split(',')
                if value.strip()
            ]
            if orientations:
                task.orientations = orientations
        except ValueError:
            pass

        try:
            omega0 = float(self.morlet_omega0_var.get())
            if omega0 > 0:
                task.morlet_omega0 = omega0
        except ValueError:
            pass

        try:
            anisotropy = float(self.morlet_anisotropy_var.get())
            if anisotropy > 0:
                task.morlet_anisotropy = anisotropy
        except ValueError:
            pass

        if self.app_start_button is not None:
            self.after_idle(self._update_pipeline_controls_state)
            self.after_idle(self._update_action_availability)

    def _setup_pipeline_section(self):
        """Единая точка управления этапами вычислительного конвейера."""
        ctk.CTkLabel(
            self.pipeline_section.content,
            text="Готовый сценарий",
            font=AppTheme.body_font(),
            anchor="w"
        ).pack(fill="x", padx=5, pady=(0, 4))

        preset_values = [
            "Пользовательский",
            *self.current_pipeline_preset_names(),
        ]
        self.pipeline_preset_selector = ctk.CTkOptionMenu(
            self.pipeline_section.content,
            values=preset_values,
            variable=self.pipeline_preset_var,
            command=self.on_pipeline_preset_changed,
            width=AppTheme.DROPDOWN_WIDTH
        )
        self.pipeline_preset_selector.pack(
            anchor="w", padx=5, pady=(0, 10)
        )

        self.calculate_extrema_switch = ctk.CTkSwitch(
            self.pipeline_section.content,
            text="Искать экстремумы",
            variable=self.calculate_extrema_var,
            command=self._on_pipeline_controls_changed
        )
        self.calculate_extrema_switch.pack(fill="x", padx=5, pady=4)

        self.calculate_envelopes_switch = ctk.CTkSwitch(
            self.pipeline_section.content,
            text="Строить огибающие",
            variable=self.calculate_envelopes_var,
            command=self._on_pipeline_controls_changed
        )
        self.calculate_envelopes_switch.pack(fill="x", padx=5, pady=4)

        self.calculate_knn_switch = ctk.CTkSwitch(
            self.pipeline_section.content,
            text="Выполнять KNN и расчёт углов",
            variable=self.calculate_knn_var,
            command=self._on_pipeline_controls_changed
        )
        self.calculate_knn_switch.pack(fill="x", padx=5, pady=4)

        ctk.CTkLabel(
            self.pipeline_section.content,
            text="Фактическая цепочка",
            font=AppTheme.section_title_font(), anchor="w"
        ).pack(fill="x", padx=5, pady=(12, 3))
        self.pipeline_chain_label = ctk.CTkLabel(
            self.pipeline_section.content,
            text="Вейвлеты",
            font=AppTheme.body_font(),
            text_color=AppTheme.INFO,
            anchor="w", justify="left", wraplength=400
        )
        self.pipeline_chain_label.pack(fill="x", padx=5, pady=(0, 3))

        self.pipeline_dependency_label = ctk.CTkLabel(
            self.pipeline_section.content,
            text="Вейвлет-преобразование выполняется всегда.",
            font=AppTheme.caption_font(),
            text_color=AppTheme.TEXT_SECONDARY,
            anchor="w",
            justify="left",
            wraplength=400
        )
        self.pipeline_dependency_label.pack(fill="x", padx=5, pady=(8, 0))
        self.after_idle(self._update_pipeline_controls_state)

    @staticmethod
    def current_pipeline_preset_names():
        return list(ProcessingTask.PIPELINE_PRESETS.keys())

    def on_pipeline_preset_changed(self, preset_name):
        """Применить пресет к активной задаче без изменения форматов файлов."""
        if self.current_task is None:
            self.pipeline_preset_var.set("Пользовательский")
            return
        self.current_task.apply_pipeline_preset(preset_name)
        self._update_ui_for_current_task()

    def _on_pipeline_controls_changed(self):
        """Перевести сценарий в пользовательский после ручного изменения."""
        if self.current_task is not None and not self._loading_task_settings:
            self.current_task.pipeline_preset = "Пользовательский"
            self.pipeline_preset_var.set("Пользовательский")
        self._store_settings_for_current_task()
        self._update_pipeline_controls_state()

    def _update_pipeline_controls_state(self):
        """Отобразить фактический план и состояния зависимых форматов."""
        task = getattr(self, "current_task", None)
        if task is None:
            for widget in (
                    self.pipeline_preset_selector,
                    self.calculate_extrema_switch,
                    self.calculate_envelopes_switch,
                    self.calculate_knn_switch):
                if widget is not None:
                    widget.configure(state="disabled")
            if self.pipeline_dependency_label is not None:
                self.pipeline_dependency_label.configure(
                    text="Создайте или активируйте задачу."
                )
            if self.pipeline_chain_label is not None:
                self.pipeline_chain_label.configure(text="Нет активной задачи")
            return

        is_1d = task.analysis_mode == "1d"
        if self.pipeline_preset_selector is not None:
            self.pipeline_preset_selector.configure(
                state="normal" if is_1d else "disabled"
            )
        row_available = is_1d and task.process_rows
        stage_state = "normal" if row_available else "disabled"
        for widget in (
                self.calculate_extrema_switch,
                self.calculate_envelopes_switch,
                self.calculate_knn_switch):
            if widget is not None:
                widget.configure(state=stage_state)

        plan = task.resolve_pipeline()
        if self.pipeline_chain_label is not None:
            self.pipeline_chain_label.configure(
                text=task.executed_stage_summary()
            )
        reasons = plan.automatic_reasons()
        if not is_1d:
            dependency_text = (
                "Для 2D сейчас выполняется отдельный базовый этап Morlet. "
                "Последующие 2D-этапы будут добавлены отдельно."
            )
        elif not task.process_rows:
            dependency_text = (
                "Экстремумы и последующие этапы требуют включённого "
                "построчного 1D-преобразования."
            )
        elif reasons or (plan.maxima_required and not task.find_maxima):
            stage_names = {
                "extrema": "экстремумы",
                "envelopes": "огибающие",
            }
            reason_items = [
                f"{stage_names.get(name, name)} — {reason}"
                for name, reason in reasons.items()
            ]
            if plan.maxima_required and not task.find_maxima:
                reason_items.append(
                    "максимумы — требуются выбранными этапами анализа"
                )
            dependency_text = "Автоматически: " + "; ".join(reason_items)
            if plan.ml:
                dependency_text += ". Затем автоматически выполняется ML-кластеризация."
        elif plan.ml:
            dependency_text = (
                "После KNN автоматически выполняется выбранная ML-кластеризация."
            )
        else:
            dependency_text = "Вейвлет-преобразование выполняется всегда."
        if self.pipeline_dependency_label is not None:
            self.pipeline_dependency_label.configure(text=dependency_text)

        for widget in self.extremes_output_widgets:
            widget.configure(state="normal" if plan.extrema else "disabled")
        for widget in self.envelopes_output_widgets:
            widget.configure(state="normal" if plan.envelopes else "disabled")
        for widget in self.knn_output_widgets:
            widget.configure(state="normal" if plan.knn else "disabled")

        point_state = "normal" if plan.extrema else "disabled"
        for widget in (self.max_checkbox, self.min_checkbox):
            if widget is not None:
                widget.configure(state=point_state)
        if self.entry_near_point is not None:
            self.entry_near_point.configure(
                state="normal" if plan.knn else "disabled"
            )
        if self.extremes_hint_label is not None and row_available:
            self.extremes_hint_label.configure(
                text=(
                    "Выберите типы экстремумов для активного конвейера."
                    if plan.extrema else
                    "Включите экстремумы или один из зависимых этапов."
                ),
                text_color=AppTheme.TEXT_SECONDARY
            )

        self._update_statistics_controls_state()

    def _setup_wavelet_section(self):
        """Настройка секции вейвлет-преобразования"""
        info_label = ctk.CTkLabel(
            self.wavelet_section.content,
            text="Формат вывода результатов вейвлет-преобразования:",
            font=AppTheme.body_font(),
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

        self.wavelet_numpy_checkbox = ctk.CTkCheckBox(
            self.wavelet_section.content,
            text="Сохранить числовой массив NPY",
            variable=self.wavelet_numpy_var
        )
        self.wavelet_section.add_widget(self.wavelet_numpy_checkbox, fill="x")


    def _setup_output_extremes_section(self):
        """Настройка секции вывода точек экстремумов"""
        info_label = ctk.CTkLabel(
            self.output_extremes_section.content,
            text="Сырые локальные экстремумы:",
            font=AppTheme.body_font(),
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
        self.extremes_output_widgets = [
            self.p_ex1_checkbox, self.p_ex2_checkbox
        ]

        envelope_label = ctk.CTkLabel(
            self.output_extremes_section.content,
            text="Точки верхней и нижней огибающих:",
            font=AppTheme.body_font(),
            anchor="w"
        )
        self.output_extremes_section.add_widget(
            envelope_label, pady=(12, 6)
        )
        self.envelope_image_checkbox = ctk.CTkCheckBox(
            self.output_extremes_section.content,
            text="Сохранить огибающие PNG",
            variable=self.envelope_image_var
        )
        self.output_extremes_section.add_widget(
            self.envelope_image_checkbox, fill="x"
        )
        self.envelope_text_checkbox = ctk.CTkCheckBox(
            self.output_extremes_section.content,
            text="Сохранить огибающие TXT",
            variable=self.envelope_text_var
        )
        self.output_extremes_section.add_widget(
            self.envelope_text_checkbox, fill="x"
        )
        self.envelopes_output_widgets = [
            self.envelope_image_checkbox, self.envelope_text_checkbox
        ]

    def _setup_statistics_section(self):
        """Настройки расчёта и сохранения статистик и синхронизаций."""
        self.calculate_statistics_switch = ctk.CTkSwitch(
            self.statistics_section.content,
            text="Считать статистики экстремумов",
            variable=self.calculate_statistics_var,
            command=self._on_statistics_stage_changed
        )
        self.statistics_section.add_widget(
            self.calculate_statistics_switch, pady=(0, 6)
        )

        block_size_label = ctk.CTkLabel(
            self.statistics_section.content,
            text="Размеры блоков масштабов (через запятую)",
            font=AppTheme.caption_font(),
            anchor="w"
        )
        self.statistics_section.add_widget(block_size_label, pady=(2, 2))
        self.scale_block_sizes_entry = ctk.CTkEntry(
            self.statistics_section.content,
            textvariable=self.scale_block_sizes_var,
            placeholder_text="5"
        )
        self.statistics_section.add_widget(
            self.scale_block_sizes_entry, pady=(0, 6)
        )
        self.statistics_parameter_widgets = [
            self.scale_block_sizes_entry
        ]

        statistics_image = ctk.CTkCheckBox(
            self.statistics_section.content,
            text="Гистограммы PNG",
            variable=self.statistics_image_var
        )
        self.statistics_section.add_widget(statistics_image)
        statistics_csv = ctk.CTkCheckBox(
            self.statistics_section.content,
            text="Таблицы статистик CSV",
            variable=self.statistics_csv_var
        )
        self.statistics_section.add_widget(statistics_csv, pady=(2, 10))
        self.statistics_output_widgets = [statistics_image, statistics_csv]

        self.calculate_sync_switch = ctk.CTkSwitch(
            self.statistics_section.content,
            text="Считать межстрочные синхронизации",
            variable=self.calculate_sync_var,
            command=self._on_statistics_stage_changed
        )
        self.statistics_section.add_widget(
            self.calculate_sync_switch, pady=(4, 6)
        )

        sync_stride_label = ctk.CTkLabel(
            self.statistics_section.content,
            text="Шаг выбора строк",
            font=AppTheme.caption_font(),
            anchor="w"
        )
        self.statistics_section.add_widget(sync_stride_label, pady=(2, 2))
        self.sync_stride_entry = ctk.CTkEntry(
            self.statistics_section.content,
            textvariable=self.sync_stride_var,
            placeholder_text="1"
        )
        self.statistics_section.add_widget(self.sync_stride_entry, pady=(0, 4))

        sync_tolerance_label = ctk.CTkLabel(
            self.statistics_section.content,
            text="Допустимое смещение по X, пиксели",
            font=AppTheme.caption_font(),
            anchor="w"
        )
        self.statistics_section.add_widget(sync_tolerance_label, pady=(2, 2))
        self.sync_tolerance_entry = ctk.CTkEntry(
            self.statistics_section.content,
            textvariable=self.sync_tolerance_var,
            placeholder_text="1"
        )
        self.statistics_section.add_widget(
            self.sync_tolerance_entry, pady=(0, 4)
        )

        sync_metric_label = ctk.CTkLabel(
            self.statistics_section.content,
            text="Метрика синхронизации",
            font=AppTheme.caption_font(),
            anchor="w"
        )
        self.statistics_section.add_widget(sync_metric_label, pady=(2, 2))
        self.sync_metric_selector = ctk.CTkOptionMenu(
            self.statistics_section.content,
            values=["jaccard", "dice", "phi"],
            variable=self.sync_metric_var,
            width=180
        )
        self.statistics_section.add_widget(
            self.sync_metric_selector,
            fill="none", anchor="w", pady=(0, 6)
        )
        self.synchronization_parameter_widgets = [
            self.sync_stride_entry,
            self.sync_tolerance_entry,
            self.sync_metric_selector,
        ]

        sync_heatmap = ctk.CTkCheckBox(
            self.statistics_section.content,
            text="Heatmap синхронизаций PNG",
            variable=self.sync_heatmap_var
        )
        self.statistics_section.add_widget(sync_heatmap)
        sync_matrix = ctk.CTkCheckBox(
            self.statistics_section.content,
            text="Матрица синхронизаций CSV",
            variable=self.sync_matrix_csv_var
        )
        self.statistics_section.add_widget(sync_matrix)
        sync_pairs = ctk.CTkCheckBox(
            self.statistics_section.content,
            text="Метрики пар строк CSV",
            variable=self.sync_pairs_csv_var
        )
        self.statistics_section.add_widget(sync_pairs)
        self.synchronization_output_widgets = [
            sync_heatmap, sync_matrix, sync_pairs
        ]

        ctk.CTkLabel(
            self.statistics_section.content,
            text=("Синхронизации используют максимумы верхней огибающей "
                  "построчного 1D-преобразования."),
            font=AppTheme.caption_font(),
            text_color=AppTheme.TEXT_SECONDARY,
            anchor="w",
            justify="left",
            wraplength=400
        ).pack(fill="x", padx=5, pady=(8, 0))
        self._update_statistics_controls_state()

    def _on_statistics_stage_changed(self):
        if self.current_task is not None and not self._loading_task_settings:
            self.current_task.pipeline_preset = "Пользовательский"
            self.pipeline_preset_var.set("Пользовательский")
        self._store_settings_for_current_task()
        self._update_pipeline_controls_state()

    def _update_statistics_controls_state(self):
        """Блокировать форматы вывода, когда соответствующий расчёт отключён."""
        task = getattr(self, "current_task", None)
        row_analysis_available = bool(
            task is not None
            and task.analysis_mode == "1d"
            and task.process_rows
        )
        switch_state = "normal" if row_analysis_available else "disabled"
        if self.calculate_statistics_switch is not None:
            self.calculate_statistics_switch.configure(state=switch_state)
        if self.calculate_sync_switch is not None:
            self.calculate_sync_switch.configure(state=switch_state)

        plan = task.resolve_pipeline() if task is not None else None
        statistics_state = "normal" if (
            row_analysis_available and plan.statistics
        ) else "disabled"
        sync_state = "normal" if (
            row_analysis_available and plan.synchronization
        ) else "disabled"
        block_state = "normal" if (
            row_analysis_available
            and (plan.statistics or plan.synchronization)
        ) else "disabled"
        for widget in self.statistics_output_widgets:
            widget.configure(state=statistics_state)
        for widget in self.statistics_parameter_widgets:
            widget.configure(state=block_state)
        for widget in self.synchronization_output_widgets:
            widget.configure(state=sync_state)
        for widget in self.synchronization_parameter_widgets:
            widget.configure(state=sync_state)

    def _setup_knn_section(self):
        """Настройка секции K-ближайших соседей"""
        info_label = ctk.CTkLabel(
            self.knn_section.content,
            text="Формат вывода K-ближайших соседей:",
            font=AppTheme.body_font(),
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
        self.knn_output_widgets = [
            self.knn_text_checkbox, self.knn_image_checkbox
        ]

    def _setup_intermediate_section(self):
        """Настройка секции промежуточных вычислений"""
        info_label = ctk.CTkLabel(
            self.intermediate_section.content,
            text="Дополнительные выходные данные:",
            font=AppTheme.body_font(),
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

        self.centering_means_checkbox = ctk.CTkCheckBox(
            self.intermediate_section.content,
            text="Средние значения центрирования TXT",
            variable=self.centering_means_var
        )
        self.intermediate_section.add_widget(
            self.centering_means_checkbox, fill="x"
        )

    def _setup_compute_section(self):
        """Настройка секции вычислений"""
        self.compute_section.grid_columnconfigure(0, weight=1)

        self.app_start_button = ctk.CTkButton(
            self.compute_section,
            text="Запустить",
            command=self.safe_compute,
            width=220,
            height=AppTheme.PRIMARY_BUTTON_HEIGHT,
            font=AppTheme.panel_title_font(),
            fg_color=AppTheme.PRIMARY,
            hover_color=AppTheme.PRIMARY_HOVER
        )
        self.app_start_button.grid(
            row=0, column=0, sticky="e",
            padx=AppTheme.CONTENT_PADDING,
            pady=AppTheme.CONTENT_PADDING
        )

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
            self.progress_manager.fail_run(f"Вычисления остановлены: {str(e)}")
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
            self.app_start_button, self.add_task_btn, self.analysis_mode_selector,
            self.gpu_switch]

        for widget in widgets_to_disable:
            try:
                widget.configure(state=state)
            except Exception as e:
                self.progress_manager.log_error(str(e))

        self.workspace_tabs.set_enabled(not disable)

        if not disable:
            self._apply_analysis_mode_ui()
            self._update_action_availability()

    def _run_integrated_ml(self, task):
        """Run configured clustering as the final stage of an ML preset."""
        plan = task.resolve_pipeline()
        if not plan.ml:
            return None
        if not task.knn_results:
            raise ClusteringError(
                "ML-этап не получил KNN-признаки. Проверьте направления анализа "
                "и параметры экстремумов."
            )
        self.progress_manager.begin_stage(f"{task.task_name} · ML-кластеризация")
        self.progress_manager.update_progress(0.0, "ML: подготовка признаков...")
        result = run_clustering(
            knn_results=task.knn_results,
            algorithm=task.ml_algorithm,
            output_root=task.task_folder_path,
            image=task.original_image,
            point_filter=task.ml_point_filter,
            feature_set=task.ml_feature_set,
            standardize=task.ml_standardize,
            n_clusters=task.ml_n_clusters,
            random_state=task.ml_random_state,
            eps=task.ml_eps,
            min_samples=task.ml_min_samples,
        )
        task.ml_result = result
        self.progress_manager.log_info(
            f"ML-кластеризация встроенного конвейера завершена: {result['output_dir']}"
        )
        self.progress_manager.update_progress(1.0, "ML-кластеризация завершена")
        return result

    def _history_ml_summary(self, task):
        result = task.ml_result
        if not result:
            return ""
        metrics = result.get("metrics", {})
        return (
            f"Кластеров: {metrics.get('cluster_count', 0)}; "
            f"шумовых точек: {metrics.get('noise_count', 0)}"
        )

    def compute(self): # Основная функция вычислений для всех задач
        if not self.image_processor.tasks:
            mb.showwarning("Внимание", "Нет задач для обработки")
            return

        # Проверяем, что все задачи готовы к обработке
        for i, task in enumerate(self.image_processor.tasks):
            try:
                validate_scales(task.scales)
            except ValueError as error:
                mb.showerror("Масштабы", f"Задача {i + 1}: {error}")
                return
            if not task.image_path:
                mb.showerror("Ошибка", f"Задача {i + 1}: не загружено изображение")
                return
            if len(task.scales) == 0:
                mb.showerror("Ошибка", f"Задача {i + 1}: не заданы масштабы")
                return
            if task.analysis_mode == "1d" and not (
                    task.process_rows or task.process_columns):
                mb.showerror(
                    "Ошибка",
                    f"Задача {i + 1}: не выбрано направление 1D-анализа"
                )
                return
            if task.analysis_mode == "2d" and not task.orientations:
                mb.showerror(
                    "Ошибка",
                    f"Задача {i + 1}: не заданы ориентации 2D Morlet"
                )
                return

        try:
            timer = time.time()
            total_tasks = len(self.image_processor.tasks)
            current_task_num = 0
            execution_stages = []
            for planned_task in self.image_processor.tasks:
                execution_stages.extend(
                    f"{planned_task.task_name} · {stage}"
                    for stage in planned_task.execution_stage_names()
                )
            self.progress_manager.begin_run(
                f"Исследование · задач: {total_tasks}", execution_stages
            )

            for task in self.image_processor.tasks:
                current_task_num += 1
                self.progress_manager.log_info(f"Обработка задачи {current_task_num}/{total_tasks}: {task.task_name}")

                task_started = time.time()
                snapshot = task.settings_snapshot()
                try:
                    # Каждый запуск получает собственный каталог, поэтому
                    # исследования с разными параметрами не смешиваются.
                    task.task_folder_path = ""
                    self.image_processor.create_task_folder(task)
                    save_run_image(task)
                    snapshot = task.settings_snapshot()
                    # Основной анализ и, для ML-сценария, кластеризация образуют
                    # один воспроизводимый исследовательский запуск.
                    self.image_processor.compute_for_task(task)
                    self._run_integrated_ml(task)
                    self.progress_manager.begin_stage(
                        f"{task.task_name} · Сохранение результатов"
                    )
                    log_header = (
                        f"Изображение: {os.path.basename(task.image_path)}\n"
                        f"Вейвлет: Morlet ({task.analysis_mode.upper()})\n"
                        f"Масштабы: {task.scales_summary()}\n"
                        f"Сценарий: {task.pipeline_preset}\n"
                        f"Этапы: {task.executed_stage_summary()}\n"
                        f"Нюансы: {task.nuances_summary()}"
                    )
                    self.progress_manager.save_run_log(
                        task.task_folder_path, header=log_header
                    )
                    checkpoint = save_feature_checkpoint(task)
                    if checkpoint:
                        self.progress_manager.log_info(
                            f"Контрольная точка признаков сохранена: {checkpoint}"
                        )
                    self.run_history.add_run(
                        task=task,
                        status="completed",
                        duration_seconds=time.time() - task_started,
                        settings=snapshot,
                        ml_summary=self._history_ml_summary(task),
                    )
                    self.progress_manager.update_progress(
                        1.0, f"Результаты {task.task_name} сохранены"
                    )
                except Exception as error:
                    self.run_history.add_run(
                        task=task,
                        status="failed",
                        duration_seconds=time.time() - task_started,
                        settings=snapshot,
                        error_message=str(error),
                    )
                    raise

            elapsed_time = time.time() - timer
            self.after_safe(0, self._refresh_history_tab)
            self.after_safe(0, lambda: self.show_success_message(elapsed_time, total_tasks))

        except Exception as e:
            self.progress_manager.log_error(f"Критическая ошибка: {str(e)}")
            self.progress_manager.fail_run(f"Ошибка вычислений: {str(e)}")
            self.after_safe(0, self._refresh_history_tab)

    def _toggle_tasks_panel(self):
        self._apply_panel_layout(not self._tasks_visible, self._progress_visible)

    def _apply_panel_layout(self, tasks_visible, progress_visible):
        """Apply geometry together before repainting either toggle control."""
        self._tasks_visible = tasks_visible
        self._progress_visible = progress_visible
        self.main_container.grid_columnconfigure(0, minsize=0)
        animate_visibility(self.tasks_panel, tasks_visible, self.tasks_panel.grid,
                           self.tasks_panel.grid_remove, axis='width')
        frame = self.progress_manager.frame
        animate_visibility(frame, progress_visible,
                           lambda: frame.pack(fill='x', padx=20, pady=(4, 10)), frame.pack_forget)
        self.tasks_toggle.configure(fg_color=AppTheme.NAV_HOVER if self._tasks_visible else 'transparent')
        self.progress_toggle.configure(fg_color=AppTheme.NAV_HOVER if self._progress_visible else 'transparent')

    def _resize_tasks_panel(self, event):
        width = max(240, min(event.x_root - self.main_container.winfo_rootx(),
                             max(240, self.main_container.winfo_width() - 600)))
        # CTk widget dimensions use logical units; pointer coordinates are pixels.
        width = width / self.tasks_panel._get_widget_scaling()
        self.tasks_panel.configure(width=width)
        self.main_container.grid_columnconfigure(0, minsize=0)

    def _toggle_progress_panel(self):
        self._apply_panel_layout(self._tasks_visible, not self._progress_visible)

    def _toggle_focus_layout(self):
        hide = self._tasks_visible or self._progress_visible
        self._apply_panel_layout(not hide, not hide)

    def _open_result_viewer(self, task=None):
        task = task or self.current_task
        self._show_saved_results(task.task_folder_path if task else '')

    def _show_saved_results(self, folder):
        self.results_panel.load_folder(folder, force=True)
        self.workspace_tabs.set("Результаты")

    def show_success_message(self, elapsed_time: float, total_tasks: int):
        self._update_ml_tab_state()
        def show_results_action():
            self._open_result_viewer()
        self._open_result_viewer()
        self.progress_manager.finish_run(
            f"Завершено: {total_tasks} задач · {format_time(elapsed_time)}",
            result_callback=show_results_action,
            folder_callback=lambda: self._open_result_path(
                self.image_processor.root_folder_path
            ),
            history_callback=lambda: self.workspace_tabs.set(
                "Предыдущие запуски"
            ),
        )

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
