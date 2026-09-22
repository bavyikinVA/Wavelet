import os
import sys
import warnings
import platform
from uuid import uuid4
from dataclasses import replace


warnings.filterwarnings(
    "ignore",
    message=r"cupyx\.jit\.rawkernel is experimental.*",
    category=FutureWarning,
    module=r"cupyx\.jit\._interface"
)

from compute.extremes.extremes_finder import ExtremesFinder


def setup_cuda_environment():
    """Настройка окружения CUDA"""
    warnings.filterwarnings("ignore", message="CUDA path could not be detected")
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

# настройки для подавления предупреждений
os.environ['CUPY_CUDA_DISABLE_CUBIN_CACHE'] = '1'
os.environ['CUPY_CACHE_DIR'] = os.path.join(os.path.expanduser('~'), '.cupy', 'cache')


import threading
import time
import tkinter as tk
import traceback
from datetime import datetime
from multiprocessing import Pool, freeze_support
from tkinter import messagebox as mb

import customtkinter as ctk
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
from PIL import Image

from Gram_Shmidt import change_channels
from history.source_image import save_run_image
from history.result_catalog import save_coefficient_preview
from history.pipeline_resume import (
    cwt_signature,
    load_complete_cwt,
    load_point_cache,
    save_point_cache,
)
from history.pipeline_state import (
    stage_statuses, minimal_execution_plan, mark_stage_ready,
    format_stage_status, format_execution_plan,
)
from result_naming import (
    CHANNEL_CODES,
    centering_mean_name,
    cwt1d_stem,
    cwt2d_stem,
    point_stem,
    run_root_folder_name,
    scale_folder_name,
    source_channel_name,
    statistics_stem,
    synchronization_prefix,
    task_run_folder_name,
)
from compute.validation import validate_scales
from compute.numerics import COMPUTE_DTYPE, COMPUTE_DTYPE_NAME
from compute.backend_policy import GPU_FALLBACK_ERRORS, GPUUnavailableError, classify_gpu_exception
from compute.processing_task import ProcessingTask
from compute.channel_analysis import (
    apply_detection_defaults,
    clear_prepared_channels,
    prepare_task_channels,
    prepared_channel_meta,
    write_channel_manifest,
)
from image_cropper_app import run_cropper
from pipette import run_pipette
from utils.gui import TkinterApp, ScrollableFrame, CollapsibleFrame
from utils.tooltip import HoverTooltip
from utils.progress_manager import ProgressManager
from utils.theme import AppTheme
from utils.icon_button import IconButton
from utils.animation import animate_visibility
from compute.run_control import RunCancelled
from compute.memory_estimate import estimate_label
from ml.errors import ClusteringError
from ml.ui_state import ml_controls_editable
from history import (
    RunHistoryStore, feature_checkpoint_available,
    restore_feature_checkpoint, save_feature_checkpoint,
)


def process_row_static(args_):
    from compute.wavelets.cpu_wavelet import morlet_wavelet_with_padding
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
        self.backend = ComputeBackend(lazy=True)
        self._knn_processor = None

    @property
    def gpu_processor(self):
        return self.backend.gpu_processor

    @property
    def knn_processor(self):
        from compute.knn.knn_cpu import get_knn_processor
        self._knn_processor = get_knn_processor(self.backend.use_gpu)
        return self._knn_processor

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
            root_folder_name = run_root_folder_name(datetime.now())
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
        result = run_cropper(master_window)
        if result is not None:
            effective_image, image_path = result
            self.progress.log_info(f'Изображение загружено: {image_path}')
            original_image = np.array(effective_image, copy=True)
            if original_image.ndim == 2:
                original_image = np.repeat(original_image[:, :, None], 3, axis=2)
            elif original_image.ndim == 3 and original_image.shape[2] >= 3:
                original_image = original_image[:, :, :3].copy()
            else:
                raise ValueError("Неподдерживаемый формат изображения")

            source_channels = [original_image[:, :, i].copy() for i in range(3)]
            if not all(isinstance(ch, np.ndarray) for ch in source_channels):
                raise ValueError("Один или несколько каналов изображения не являются массивами NumPy")

            # Источник заменяется атомарно только после успешного чтения.
            task.invalidate_source_results()
            task.image_path = image_path
            task.original_image = original_image
            task.data = source_channels
            task.data_copy = [channel.copy() for channel in source_channels]
            from compute.channel_analysis import apply_detection_defaults as _apply_detection_defaults
            detected, similarity = _apply_detection_defaults(task, original_image)
            self.progress.log_info(
                "Тип изображения: "
                f"{'grayscale' if detected == 'grayscale' else 'цветное'} "
                f"(совпадение RGB: {similarity:.2%}); "
                f"режим по умолчанию: {task.channel_summary()}"
            )
            self.progress.log_info("Изображение успешно обработано")
            return True
        else:
            self.progress.log_info('Загрузка изображения отменена')
            return False

    def pipette_channel_for_task(self, task, master_window=None):
        self.progress.log_info("Запуск инструмента 'Пипетка'...")
        color1, color2 = run_pipette(master=master_window, image=task.original_image)
        if color1 is None or color2 is None:
            self.progress.log_info('Выбор цветов отменён')
            return False
        # Пипетка теперь задаёт именно базис Gram–Schmidt, поэтому сразу
        # отбрасываем нулевые/коллинеарные цветовые векторы. Полное
        # изображение при этой проверке не преобразуется.
        change_channels(color1, color2, [[[0.0]], [[0.0]], [[0.0]]])
        task.color1, task.color2 = color1, color2
        if task.channel_representation == "gram_schmidt":
            task.invalidate_analysis_results()
        self.progress.log_info("Цветовой базис Gram–Schmidt успешно выбран")
        return True

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

        task_folder_name = task_run_folder_name(
            task.task_id, datetime.now(), uuid4().hex[:8]
        )
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

        scale_folder_path = os.path.join(
            task_folder, "scales", scale_folder_name(scale)
        )
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
            task_folder = self.create_task_folder(task)  # Создаём папку для задачи
            source_folder = os.path.join(task_folder, "source")
            os.makedirs(source_folder, exist_ok=True)
            for channel in range(len(task.data_copy)):
                filename = source_channel_name(channel) + ".txt"
                array_2d = task.data_copy[channel]
                file_path = os.path.join(source_folder, filename)
                np.savetxt(file_path, array_2d, fmt='%d', delimiter=",")
                self.progress.log_info(f"Сохранен файл: {file_path}")

    def gram_shmidt_transform_for_task(self, task):
        """Выбрать GS-представление, не изменяя исходные RGB-данные."""
        color1 = getattr(task, "color1", None)
        color2 = getattr(task, "color2", None)
        if color1 is None or color2 is None:
            raise ValueError("Сначала выберите два цвета пипеткой")
        # Проверяем корректность базиса на крошечном служебном массиве.
        # Полное изображение преобразуется только непосредственно перед CWT.
        change_channels(color1, color2, [[[0.0]], [[0.0]], [[0.0]]])
        if hasattr(task, "set_channel_analysis"):
            task.set_channel_analysis("gram_schmidt")
        else:
            task.gram_schmidt_applied = True
        self.progress.log_info(
            "Для анализа выбрано представление Gram–Schmidt (GS1/GS2/GS3)"
        )

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
        if self._knn_processor is not None:
            self._knn_processor.use_gpu = self.backend.use_gpu
        backend_info = self.backend.get_backend_info()
        self.progress.log_info(f"Вычислительное устройство: {backend_info['device_name']}")
        return success, backend_info

    def get_backend_info(self):
        """Получение информации о бэкенде"""
        return self.backend.get_backend_info()

    def _cpu_flat_chunk_size(self):
        try:
            return max(1, int(os.environ.get("WAVELETS_CPU_CHUNK_SIZE", "128")))
        except ValueError as exc:
            raise ValueError("WAVELETS_CPU_CHUNK_SIZE должен быть положительным целым") from exc

    def process_channel(self, data, scales):
        """CPU CWT for rows; numba-flat is default, legacy Pool is opt-in."""
        rows = data.shape[0]
        cols = data.shape[1]
        scales_size = len(scales)
        self.progress.log_info(
            f"CPU обработка {rows} строк (engine={self.backend.cpu_engine})..."
        )

        if self.backend.cpu_engine == "legacy-pool":
            result = np.zeros((rows, scales_size, cols), dtype=COMPUTE_DTYPE)
            with Pool() as pool:
                args = [(data[i], scales) for i in range(rows)]
                pending = pool.map_async(process_row_static, args)
                while not pending.ready():
                    self.progress.run_control.check()
                    pending.wait(.1)
                self.progress.run_control.check()
                results = pending.get()
            for i, res in enumerate(results):
                result[i] = res
            return result

        # Chunking preserves cancellation responsiveness without reintroducing
        # process-spawn overhead. Each chunk is one flat Numba parallel region.
        result = np.empty((rows, scales_size, cols), dtype=COMPUTE_DTYPE)
        chunk_size = self._cpu_flat_chunk_size()
        for start in range(0, rows, chunk_size):
            self.progress.run_control.check()
            stop = min(rows, start + chunk_size)
            result[start:stop] = self.backend.compute_cpu_batch(data[start:stop], scales)
        self.progress.run_control.check()
        return result

    def process_channel_columns(self, data, scales):
        """CPU CWT for columns with the same engine as row processing."""
        cols = data.shape[1]
        rows = data.shape[0]
        scales_size = len(scales)
        self.progress.log_info(
            f"CPU обработка {cols} столбцов (engine={self.backend.cpu_engine})..."
        )

        if self.backend.cpu_engine == "legacy-pool":
            result_3d = np.zeros((scales_size, cols, rows), dtype=COMPUTE_DTYPE)
            args = [(col_idx, data[:, col_idx], scales) for col_idx in range(cols)]
            with Pool() as pool:
                pending = pool.map_async(process_column_static, args)
                while not pending.ready():
                    self.progress.run_control.check()
                    pending.wait(.1)
                self.progress.run_control.check()
                results = pending.get()
            for col_idx, column_result in results:
                result_3d[:, col_idx, :] = column_result
            return np.transpose(result_3d, (0, 2, 1))

        transposed = np.ascontiguousarray(data.T, dtype=COMPUTE_DTYPE)
        result = np.empty((cols, scales_size, rows), dtype=COMPUTE_DTYPE)
        chunk_size = self._cpu_flat_chunk_size()
        for start in range(0, cols, chunk_size):
            self.progress.run_control.check()
            stop = min(cols, start + chunk_size)
            result[start:stop] = self.backend.compute_cpu_batch(
                transposed[start:stop], scales
            )
        self.progress.run_control.check()
        return np.transpose(result, (1, 2, 0))

    def process_channel_batch_gpu(self, data_channel, scales):
        """Батчевая обработка ВСЕГО канала на GPU (используется для обоих backend)"""
        if not self.backend.use_gpu:
            return self.process_channel(data_channel, scales)  # CPU fallback

        try:
            rows, cols = data_channel.shape
            self.progress.log_info(f"Подготовка батча для GPU: {rows} строк × {cols} колонок")

            result = self.gpu_processor.morlet_wavelet_batch(data_channel, scales)  # [rows, scales, cols]
            return result

        except GPU_FALLBACK_ERRORS as e:
            if self.backend.strict_backend:
                raise
            self.progress.log_error(
                f"GPU батчевая обработка недоступна ({e.__class__.__name__}): {e}. Возврат к CPU."
            )
            return self.process_channel(data_channel, scales)

    def wavelets(self, task, type_data, data_3_channel):
        """Версия для GPU с прогресс-баром"""
        t_compute_wavelet_start = time.time()

        backend_info = "GPU" if self.backend.use_gpu else "CPU"
        direction = "построчно" if type_data == 0 else "по столбцам"

        self.progress.log_info(f"Начало вейвлет-преобразования ({backend_info}, {direction})")

        if not task.analysis_data:
            prepare_task_channels(task)
        channel_meta = prepared_channel_meta(task)
        num_channels = len(task.analysis_data)
        num_rows = task.analysis_data[0].shape[0]
        num_cols = task.analysis_data[0].shape[1]

        self.progress.log_info(f"Масштабы: {len(task.scales)}, Строки: {num_rows}, Столбцы: {num_cols}")
        self.progress.log_info(
            f"Общий объем: {num_channels} канала × {num_rows if type_data == 0 else num_cols} направлений")

        total_operations = num_channels
        current_operation = 0

        for channel in range(num_channels):
            _channel_key, channel_code, channel_name = channel_meta[channel]

            # Обновляем прогресс для каждого канала
            progress = current_operation / total_operations
            self.progress.update_progress(
                progress,
                f"Подготовка канала {channel_name}..."
            )

            data_channel = np.asarray(task.analysis_data[channel], dtype=COMPUTE_DTYPE)

            # Центрирование всегда участвует в расчёте, но его служебные
            # значения сохраняются только по явному выбору пользователя.
            cwt_axis = "row" if type_data == 0 else "col"
            protocol_stage = f"wavelet:{cwt_axis}:{channel_code}"
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
                metadata_folder = os.path.join(
                    task.task_folder_path, "metadata"
                )
                os.makedirs(metadata_folder, exist_ok=True)
                file_mean_path = os.path.join(
                    metadata_folder,
                    centering_mean_name(cwt_axis, channel_code) + ".txt"
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

                    data_3_channel[channel] = np.asarray(
                        data_channel_after_transposed, dtype=COMPUTE_DTYPE
                    )
                    task.execution_protocol.record_stage(
                        protocol_stage, actual_backend="gpu", dtype=COMPUTE_DTYPE_NAME
                    )

                except GPU_FALLBACK_ERRORS as e:
                    if task.execution_protocol.strict_backend or self.backend.strict_backend:
                        raise
                    self.progress.log_error(
                        f"Ошибка инфраструктуры GPU ({e.__class__.__name__}): {e}. Переход на CPU..."
                    )
                    task.execution_protocol.record_fallback(
                        protocol_stage, reason=e.__class__.__name__, message=str(e)
                    )
                    # Fallback to CPU
                    if type_data == 0:
                        data_channel_after = self.process_channel(data_channel, task.scales)
                        data_channel_after_transposed = np.transpose(data_channel_after, (1, 0, 2))
                    else:
                        data_channel_after = self.process_channel_columns(data_channel, task.scales)
                        data_channel_after_transposed = data_channel_after

                    data_3_channel[channel] = np.asarray(
                        data_channel_after_transposed, dtype=COMPUTE_DTYPE
                    )
                    task.execution_protocol.record_stage(
                        protocol_stage, actual_backend="cpu", dtype=COMPUTE_DTYPE_NAME,
                        fallback=True,
                        details={
                            "fallback_reason": e.__class__.__name__,
                            "cpu_engine": self.backend.cpu_engine,
                        },
                    )
            else:
                if type_data == 0:
                    data_channel_after = self.process_channel(data_channel, task.scales)
                    data_channel_after_transposed = np.transpose(data_channel_after, (1, 0, 2))
                else:
                    data_channel_after = self.process_channel_columns(data_channel, task.scales)
                    data_channel_after_transposed = data_channel_after

                data_3_channel[channel] = np.asarray(
                    data_channel_after_transposed, dtype=COMPUTE_DTYPE
                )
                task.execution_protocol.record_stage(
                    protocol_stage, actual_backend="cpu", dtype=COMPUTE_DTYPE_NAME,
                    details={"cpu_engine": self.backend.cpu_engine},
                )
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
            if not task.analysis_data:
                prepare_task_channels(task)
            channel_count = len(task.analysis_data)
            height, width = task.analysis_data[0].shape
            data_3_channels = np.zeros(
                (channel_count, task.num_scale, height, width), dtype=COMPUTE_DTYPE
            )
            task.result[0] = self.wavelets(task, 0, data_3_channels)
            self.save_print_wavelets(task, 0, info_out)

        if task.process_columns:
            self.progress.update_progress(0.6, "Обработка столбцов...")
            if not task.analysis_data:
                prepare_task_channels(task)
            channel_count = len(task.analysis_data)
            height, width = task.analysis_data[0].shape
            data_3_channels_tr = np.zeros(
                (channel_count, task.num_scale, height, width), dtype=COMPUTE_DTYPE
            )
            task.result[1] = self.wavelets(task, 1, data_3_channels_tr)
            self.save_print_wavelets(task, 1, info_out)

        self.progress.update_progress(1.0, "Вейвлет-преобразование завершено")

    def compute_wavelets_2d(self, task):
        """Выполнить отдельное ориентационное 2D-преобразование Морле."""
        import matplotlib.pyplot as plt
        from compute.wavelets.morlet_2d import (
            cwt2d_morlet_cpu,
            cwt2d_morlet_gpu,
        )

        task_folder = self.create_task_folder(task)
        if not task.analysis_data:
            prepare_task_channels(task)
        channel_meta = prepared_channel_meta(task)
        total = len(channel_meta) * len(task.scales) * len(task.orientations)
        completed = 0
        use_gpu = self.backend.use_gpu and self.backend.is_gpu_available()

        for channel_index, (_channel_key, channel_code, channel_name) in enumerate(channel_meta):
            channel = task.analysis_data[channel_index].astype(np.float32)
            for scale in task.scales:
                scale_folder = self.create_scale_folder(scale, task_folder)
                output_folder = os.path.join(scale_folder, "wavelets_2d")
                os.makedirs(output_folder, exist_ok=True)

                for angle in task.orientations:
                    completed += 1
                    self.progress.update_progress(
                        completed / max(total, 1),
                        (f"2D Morlet: {channel_name}, масштаб {scale}, "
                         f"ориентация {angle:g}°")
                    )

                    stage_name = (
                        f"wavelet_2d:{channel_code}:{float(scale):g}:{float(angle):g}"
                    )
                    if use_gpu:
                        try:
                            coefficients = cwt2d_morlet_gpu(
                                channel, [scale], [angle],
                                task.morlet_omega0,
                                task.morlet_anisotropy
                            )[0, 0]
                            task.execution_protocol.record_stage(
                                stage_name, actual_backend="gpu", dtype="complex64"
                            )
                        except Exception as error:
                            mapped = classify_gpu_exception(error)
                            if mapped is None:
                                raise
                            if task.execution_protocol.strict_backend or self.backend.strict_backend:
                                raise mapped from error
                            self.progress.log_error(
                                f"Ошибка 2D Morlet на GPU ({mapped.__class__.__name__}): "
                                f"{mapped}. Переход на CPU."
                            )
                            task.execution_protocol.record_fallback(
                                stage_name, reason=mapped.__class__.__name__,
                                message=str(mapped),
                            )
                            use_gpu = False
                            coefficients = cwt2d_morlet_cpu(
                                channel, [scale], [angle],
                                task.morlet_omega0,
                                task.morlet_anisotropy
                            )[0, 0]
                            task.execution_protocol.record_stage(
                                stage_name, actual_backend="cpu", dtype="complex64",
                                fallback=True,
                                details={"fallback_reason": mapped.__class__.__name__},
                            )
                    else:
                        coefficients = cwt2d_morlet_cpu(
                            channel, [scale], [angle],
                            task.morlet_omega0,
                            task.morlet_anisotropy
                        )[0, 0]
                        task.execution_protocol.record_stage(
                            stage_name, actual_backend="cpu", dtype="complex64"
                        )

                    file_stem = cwt2d_stem(channel_code, scale, angle)
                    magnitude = np.abs(coefficients)
                    save_coefficient_preview(task_folder, file_stem, coefficients)
                    phase = np.angle(coefficients)

                    if task.output_wavelet_numpy:
                        np.save(
                            os.path.join(output_folder, file_stem + "_complex.npy"),
                            coefficients
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
        import matplotlib.pyplot as plt
        channel_meta = prepared_channel_meta(task)
        colors = [item[2] for item in channel_meta]
        channel_codes = [item[1] for item in channel_meta]
        cwt_axis = "row" if type_data == 0 else "col"

        total_scales = task.num_scale * len(channel_meta)
        current_scale = 0

        if not task:
            self.progress.log_error("Задача не найдена при сохранении вейвлетов")
            return

        task_folder = self.create_task_folder(task)

        for channel in range(len(channel_meta)):
            for scale in range(task.num_scale):
                scale_folder_path = self.create_scale_folder(task.scales[scale], task_folder)
                array_2d = task.result[type_data][channel][scale]
                file_stem = cwt1d_stem(
                    cwt_axis, channel_codes[channel], task.scales[scale]
                )
                save_coefficient_preview(
                    task.task_folder_path, file_stem, array_2d
                )

                current_scale += 1
                progress = 0.1 + (current_scale / total_scales) * 0.9
                self.progress.update_progress(
                    progress,
                    f"Сохранение результатов: {colors[channel]}, масштаб {task.scales[scale]}")

                if info_out == 0 or info_out == 10:
                    filename = file_stem + ".txt"
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
                            file_stem + ".png"),
                        dpi=300)

                    plt.close(fig)

                if task.output_wavelet_numpy:
                    np.save(
                        os.path.join(
                            scale_folder_path,
                            file_stem + ".npy"
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
                       print_text_var, print_graphic, pipeline_plan=None):
        """Analyse both feature axes independently inside every selected CWT.

        ``cwt_axis`` identifies how coefficients were produced. ``feature_axis``
        identifies whether extrema/envelopes are traced along rows or columns.
        Keeping both dimensions gives four unambiguous combinations instead of
        silently treating the extrema axis as the CWT source direction.
        """
        plan = pipeline_plan or task.resolve_pipeline()
        protocol = getattr(task, "execution_protocol", None)
        max_var = bool(max_var or plan.maxima_required)
        # Local import keeps this method independently testable when its code
        # object is rebound with a minimal globals dictionary.
        from history.pipeline_resume import (
            cwt_signature as _resume_cwt_signature,
            load_point_cache as _load_point_cache,
            save_point_cache as _save_point_cache,
        )
        from history.pipeline_state import stage_statuses as _stage_statuses
        resume_cache_folder = getattr(task, "task_folder_path", "")
        resume_points_allowed = bool(
            resume_cache_folder
            and getattr(task, "reuse_existing_results", True)
            and getattr(task, "last_completed_cwt_signature", None)
            == _resume_cwt_signature(task)
        )
        # Stage-level validity is stricter than CWT validity. Changing extrema
        # settings must invalidate extrema/envelopes/KNN without recalculating CWT.
        _stage_states = _stage_statuses(task)
        reuse_extrema_allowed = resume_points_allowed and _stage_states.get("extrema") == "ready"
        reuse_envelopes_allowed = resume_points_allowed and _stage_states.get("envelopes") == "ready"

        self.progress.update_progress(0.1, "Начало поиска экстремумов...")
        self.progress.log_info(
            "Запущен анализ всех осей признаков для каждого 1D CWT"
        )

        use_gpu_knn = bool(plan.knn and self.backend.use_gpu)
        if plan.knn:
            device = "GPU" if use_gpu_knn else "CPU"
            self.progress.log_info(f"Использование {device} для KNN вычислений")

        branches = []
        if row_var:
            branches.append({
                "type_data": 0,
                "cwt_axis": "row",
                "cwt_label": "Row CWT",
            })
        if col_var:
            branches.append({
                "type_data": 1,
                "cwt_axis": "col",
                "cwt_label": "Column CWT",
            })

        feature_axes = (
            {
                "axis": "row", "label": "по строкам",
                "max_key": "max_by_row", "min_key": "min_by_row",
            },
            {
                "axis": "col", "label": "по столбцам",
                "max_key": "max_by_column", "min_key": "min_by_column",
            },
        )

        available_count = (
            len(task.result[branches[0]["type_data"]]) if branches else 0
        )
        channel_codes = list(getattr(task, "analysis_channel_codes", []))
        colors = list(getattr(task, "analysis_channel_labels", []))
        if len(channel_codes) != available_count or len(colors) != available_count:
            fallback_codes = list(CHANNEL_CODES[:available_count])
            fallback_labels = {
                "r": "Красный (R)", "g": "Зелёный (G)", "b": "Синий (B)",
                "gray": "Оттенки серого (Gray)",
                "gs1": "Gram–Schmidt 1 (GS1)",
                "gs2": "Gram–Schmidt 2 (GS2)",
                "gs3": "Gram–Schmidt 3 (GS3)",
            }
            channel_codes = fallback_codes
            colors = [fallback_labels.get(code, code) for code in fallback_codes]
        channels_to_process = list(range(available_count))
        total_operations = len(branches) * len(channels_to_process) * task.num_scale
        current_operation = 0
        extremes = []
        total_extrema_points = 0

        scale_block_sizes = getattr(task, "scale_block_sizes", [5])
        if isinstance(scale_block_sizes, int):
            scale_block_sizes = [scale_block_sizes]

        row_sync_stride = int(getattr(task, "row_sync_stride", 1))
        row_sync_tolerance = int(getattr(task, "row_sync_tolerance", 1))
        row_sync_metrics = getattr(task, "row_sync_metrics", ["jaccard"])
        if isinstance(row_sync_metrics, str):
            row_sync_metrics = [row_sync_metrics]

        from compute.extremes.interpolator import Interpolator
        interpolator = Interpolator(self.backend) if plan.envelopes else None

        for branch in branches:
            type_data = branch["type_data"]
            cwt_axis = branch["cwt_axis"]
            cwt_label = branch["cwt_label"]

            for channel in channels_to_process:
                channel_name = colors[channel]
                channel_code = channel_codes[channel]
                statistics_by_axis = {
                    feature["axis"]: {
                        "scales": [], "counts": [], "slice_index": None,
                    }
                    for feature in feature_axes
                }
                row_upper_points_by_scale = {}

                for scale_index in range(task.num_scale):
                    current_operation += 1
                    scale_value = task.scales[scale_index]
                    progress = 0.1 + (
                        current_operation / max(total_operations, 1)
                    ) * 0.8
                    self.progress.update_progress(
                        progress,
                        (
                            f"{cwt_label}, все оси: {channel_name}, "
                            f"масштаб {scale_value}"
                        ),
                    )

                    # Every source branch reads only its own coefficient tensor.
                    coefs_2d = task.result[type_data][channel][scale_index]

                    detection_kwargs = {
                        "row_var": True,
                        "col_var": True,
                        "max_var": max_var,
                        "min_var": min_var,
                    }
                    extrema_stage = (
                        f"extrema:{cwt_axis}:{channel_code}:{float(scale_value):g}"
                    )

                    # Incremental execution: reuse raw extrema when every
                    # requested point kind is already available for this
                    # CWT/channel/scale.  New runs always write an internal
                    # cache, while legacy TXT/NPZ exports are also understood.
                    cached_extrema = {}
                    extrema_cache_complete = reuse_extrema_allowed
                    if extrema_cache_complete:
                        for feature_axis in ("row", "col"):
                            if max_var:
                                cached_extrema[(feature_axis, "max")] = _load_point_cache(
                                    resume_cache_folder, "ext", cwt_axis, feature_axis,
                                    channel_code, scale_value, "max"
                                )
                            if min_var:
                                cached_extrema[(feature_axis, "min")] = _load_point_cache(
                                    resume_cache_folder, "ext", cwt_axis, feature_axis,
                                    channel_code, scale_value, "min"
                                )
                        extrema_cache_complete = all(
                            points is not None for points in cached_extrema.values()
                        )

                    if extrema_cache_complete:
                        pmaxr = cached_extrema.get(("row", "max"), [])
                        pmaxc = cached_extrema.get(("col", "max"), [])
                        pminr = cached_extrema.get(("row", "min"), [])
                        pminc = cached_extrema.get(("col", "min"), [])
                        if protocol is not None:
                            protocol.record_stage(
                                extrema_stage, actual_backend="disk-cache", dtype="int32",
                                details={"reused": True},
                            )
                    else:
                        if self.backend.use_gpu:
                            try:
                                detected = ExtremesFinder.find_extremes_gpu(
                                    coefs_2d, **detection_kwargs
                                )
                                if protocol is not None:
                                    protocol.record_stage(
                                        extrema_stage, actual_backend="gpu", dtype=COMPUTE_DTYPE_NAME
                                    )
                            except Exception as exc:
                                fallback_error_names = {
                                    "GPUUnavailableError", "GPUOutOfMemoryError",
                                    "GPUDeviceLostError", "GPUCompatibilityError",
                                }
                                if exc.__class__.__name__ not in fallback_error_names:
                                    raise
                                if (protocol is not None and protocol.strict_backend) or getattr(self.backend, "strict_backend", False):
                                    raise
                                if protocol is not None:
                                    protocol.record_fallback(
                                        extrema_stage, reason=exc.__class__.__name__,
                                        message=str(exc),
                                    )
                                self.progress.log_error(
                                    f"GPU extrema недоступны ({exc.__class__.__name__}), переход на CPU"
                                )
                                detected = self.find_extremes(
                                    coefs=coefs_2d, **detection_kwargs
                                )
                                if protocol is not None:
                                    protocol.record_stage(
                                        extrema_stage, actual_backend="cpu", dtype=COMPUTE_DTYPE_NAME,
                                        fallback=True,
                                        details={"fallback_reason": exc.__class__.__name__},
                                    )
                        else:
                            detected = self.find_extremes(
                                coefs=coefs_2d, **detection_kwargs
                            )
                            if protocol is not None:
                                protocol.record_stage(
                                    extrema_stage, actual_backend="cpu", dtype=COMPUTE_DTYPE_NAME
                                )
                        coefs_2d, pmaxr, pmaxc, pminr, pminc = detected
                    scale_folder = self.find_scale_folder(task, scale_value)
                    knn_extremes = {
                        "type_data": type_data,
                        "cwt_axis": cwt_axis,
                        "channel": channel_code,
                        "scale": scale_value,
                        "max_by_row": [],
                        "max_by_column": [],
                        "min_by_row": [],
                        "min_by_column": [],
                    }

                    raw_by_axis = {
                        "row": (pmaxr, pminr),
                        "col": (pmaxc, pminc),
                    }
                    for feature in feature_axes:
                        feature_axis = feature["axis"]
                        raw_max, raw_min = raw_by_axis[feature_axis]
                        self.progress.log_info(
                            f"{cwt_label}, {feature['label']}, "
                            f"масштаб {scale_value}, {channel_name}: "
                            f"максимумов={len(raw_max)}, минимумов={len(raw_min)}"
                        )
                        total_extrema_points += len(raw_max) + len(raw_min)

                        # Persist raw extrema independently of user export
                        # settings so a later stage can be calculated without
                        # repeating detection.  Empty results are cached too.
                        if max_var and resume_cache_folder:
                            _save_point_cache(
                                resume_cache_folder, "ext", cwt_axis, feature_axis,
                                channel_code, scale_value, "max", raw_max
                            )
                        if min_var and resume_cache_folder:
                            _save_point_cache(
                                resume_cache_folder, "ext", cwt_axis, feature_axis,
                                channel_code, scale_value, "min", raw_min
                            )

                        upper_points, lower_points = [], []
                        envelope_stage = (
                            f"envelopes:{cwt_axis}:{feature_axis}:"
                            f"{channel_code}:{float(scale_value):g}"
                        )
                        if plan.envelopes:
                            # Envelope cache may be reused only when this run is
                            # genuinely resuming a matching point pipeline.  On a
                            # fresh calculation None is the sentinel for "not
                            # loaded"; using [] here used to incorrectly mark a
                            # missing cache as complete and skipped interpolation.
                            cached_upper = None
                            cached_lower = None
                            envelope_cache_complete = False
                            if reuse_envelopes_allowed and extrema_cache_complete:
                                if max_var:
                                    cached_upper = _load_point_cache(
                                        resume_cache_folder, "env", cwt_axis, feature_axis,
                                        channel_code, scale_value, "upper"
                                    )
                                if min_var:
                                    cached_lower = _load_point_cache(
                                        resume_cache_folder, "env", cwt_axis, feature_axis,
                                        channel_code, scale_value, "lower"
                                    )
                                envelope_cache_complete = (
                                    (not max_var or cached_upper is not None)
                                    and (not min_var or cached_lower is not None)
                                )

                            if envelope_cache_complete:
                                upper_points = cached_upper or []
                                lower_points = cached_lower or []
                                if protocol is not None:
                                    protocol.record_stage(
                                        envelope_stage, actual_backend="disk-cache", dtype="int32",
                                        details={"reused": True},
                                    )
                            else:
                                upper_points, lower_points = interpolator.get_envelopes(
                                    coefs_2d, raw_max, raw_min,
                                    direction=feature_axis,
                                    protocol=protocol,
                                    stage=envelope_stage,
                                )
                        upper_points = self._valid_point_collection(upper_points)
                        lower_points = self._valid_point_collection(lower_points)
                        if plan.envelopes and max_var and resume_cache_folder:
                            _save_point_cache(
                                resume_cache_folder, "env", cwt_axis, feature_axis,
                                channel_code, scale_value, "upper", upper_points
                            )
                        if plan.envelopes and min_var and resume_cache_folder:
                            _save_point_cache(
                                resume_cache_folder, "env", cwt_axis, feature_axis,
                                channel_code, scale_value, "lower", lower_points
                            )

                        if max_var:
                            knn_extremes[feature["max_key"]] = (
                                upper_points if plan.envelopes else raw_max
                            )
                        if min_var:
                            knn_extremes[feature["min_key"]] = (
                                lower_points if plan.envelopes else raw_min
                            )

                        if plan.synchronization and (
                            cwt_axis == "row" and feature_axis == "row"
                        ):
                            row_upper_points_by_scale[scale_value] = upper_points

                        stats = statistics_by_axis[feature_axis]
                        if plan.statistics:
                            if stats["slice_index"] is None:
                                dimension = 0 if feature_axis == "row" else 1
                                stats["slice_index"] = (
                                    coefs_2d.shape[dimension] // 2
                                )
                            selected_upper = (
                                upper_points if plan.envelopes else raw_max
                            )
                            count = self.count_points_on_axis(
                                selected_upper,
                                stats["slice_index"],
                                feature_axis,
                            )
                            stats["scales"].append(scale_value)
                            stats["counts"].append(count)

                        exports = (
                            ("ext", "max", raw_max, max_var,
                             task.output_extremes_text,
                             task.output_extremes_image),
                            ("ext", "min", raw_min, min_var,
                             task.output_extremes_text,
                             task.output_extremes_image),
                            ("env", "upper", upper_points,
                             plan.envelopes and max_var,
                             task.output_envelopes_text,
                             task.output_envelopes_image),
                            ("env", "lower", lower_points,
                             plan.envelopes and min_var,
                             task.output_envelopes_text,
                             task.output_envelopes_image),
                        )
                        for artifact, kind, points, enabled, save_text, save_image in exports:
                            if not enabled or not points:
                                continue
                            title = point_stem(
                                artifact, cwt_axis, feature_axis,
                                channel_code, scale_value, kind,
                            )
                            if save_text:
                                self.save_extremes_to_file(
                                    scale_folder, title, points
                                )
                            if save_image:
                                self.save_extremes_graphic(
                                    scale_folder, title, points,
                                    coefs_2d=coefs_2d,
                                )

                    extremes.append(knn_extremes)

                    if plan.knn:
                        self.progress.update_progress(
                            progress, f"KNN для {cwt_label}, обе оси..."
                        )
                        from compute.knn.knn_cpu import (
                            process_extremes_with_knn,
                        )
                        knn_result = process_extremes_with_knn(
                            knn_extremes,
                            scale_folder,
                            knn_var,
                            task.original_image,
                            knn_bool_text_var,
                            knn_bool_image_var,
                            use_gpu=use_gpu_knn,
                            source_direction=cwt_axis,
                            progress_callback=self.progress.update_progress,
                            log_callback=self.progress.log_info,
                            protocol=protocol,
                        )
                        for point_type, payload in knn_result.items():
                            if payload:
                                task.knn_results[
                                    (
                                        cwt_axis,
                                        channel_code,
                                        float(scale_value),
                                        point_type,
                                    )
                                ] = payload

                if plan.statistics:
                    for feature in feature_axes:
                        stats = statistics_by_axis[feature["axis"]]
                        if not stats["scales"]:
                            continue
                        self._save_direction_statistics(
                            task=task,
                            branch=branch,
                            feature=feature,
                            channel_name=channel_name,
                            channel_code=channel_code,
                            scales=stats["scales"],
                            counts=stats["counts"],
                            slice_index=stats["slice_index"],
                            scale_block_sizes=scale_block_sizes,
                        )

                # Synchronization retains its existing, explicitly row-based
                # scientific meaning. It is not silently applied to columns.
                if (
                    plan.synchronization
                    and cwt_axis == "row"
                    and row_upper_points_by_scale
                ):
                    self._save_row_synchronization(
                        task=task,
                        channel_name=channel_name,
                        channel_code=channel_code,
                        scales=statistics_by_axis["row"]["scales"]
                        or list(row_upper_points_by_scale),
                        upper_points_by_scale=row_upper_points_by_scale,
                        scale_block_sizes=scale_block_sizes,
                        row_sync_stride=row_sync_stride,
                        row_sync_tolerance=row_sync_tolerance,
                        row_sync_metrics=row_sync_metrics,
                    )

        self.progress.update_progress(
            0.95, "Завершение поиска экстремумов..."
        )
        self.progress.log_info(
            f"Всего найдено точек экстремумов: {total_extrema_points}"
        )
        self.progress.update_progress(1.0, "Поиск экстремумов завершен")
        return extremes

    @staticmethod
    def _valid_point_collection(points):
        if not isinstance(points, (list, np.ndarray)) or len(points) == 0:
            return []
        return points.tolist() if isinstance(points, np.ndarray) else points

    def _save_direction_statistics(
            self, *, task, branch, feature, channel_name, channel_code, scales,
            counts, slice_index, scale_block_sizes):
        cwt_axis = branch["cwt_axis"]
        feature_axis = feature["axis"]
        coordinate = "y" if feature_axis == "row" else "x"
        histogram_dir = os.path.join(
            task.task_folder_path,
            "statistics",
        )
        file_base = statistics_stem(
            cwt_axis, feature_axis, channel_code,
            coordinate, slice_index,
        )
        result_key = (cwt_axis, feature_axis, channel_code)
        task.statistics_results[result_key] = {
            "cwt_direction": cwt_axis,
            "feature_axis": feature_axis,
            "slice_index": slice_index,
            "scales": list(scales),
            "upper_envelope_maxima_counts": list(counts),
            "scale_blocks": {},
        }

        png_path, csv_path = (
            self.save_upper_envelope_maxima_row_histogram(
                histogram_dir,
                file_base,
                scales,
                counts,
                slice_index,
                channel_name=channel_name,
                save_png=task.statistics_output_image,
                save_csv=task.statistics_output_csv,
                axis=feature_axis,
                cwt_direction=cwt_axis,
            )
        )
        if png_path:
            self.progress.log_info(f"Гистограмма сохранена: {png_path}")
        if csv_path:
            self.progress.log_info(f"CSV статистики сохранён: {csv_path}")

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

            labels, block_counts, ranges = (
                self.compress_scale_counts_to_blocks(
                    scales, counts, block_size
                )
            )
            task.statistics_results[result_key]["scale_blocks"][
                block_size
            ] = {
                "labels": list(labels),
                "counts": list(block_counts),
                "ranges": list(ranges),
            }
            block_base = statistics_stem(
                cwt_axis, feature_axis, channel_code,
                coordinate, slice_index, block_size=block_size,
            )
            block_png, block_csv = (
                self.save_upper_envelope_maxima_block_histogram(
                    histogram_dir,
                    block_base,
                    labels,
                    block_counts,
                    ranges,
                    slice_index,
                    block_size,
                    channel_name=channel_name,
                    save_png=task.statistics_output_image,
                    save_csv=task.statistics_output_csv,
                    axis=feature_axis,
                    cwt_direction=cwt_axis,
                )
            )
            if block_png:
                self.progress.log_info(
                    f"Сжатая гистограмма сохранена: {block_png}"
                )
            if block_csv:
                self.progress.log_info(
                    f"CSV блочной статистики сохранён: {block_csv}"
                )

    def _save_row_synchronization(
            self, *, task, channel_name, channel_code, scales,
            upper_points_by_scale, scale_block_sizes, row_sync_stride,
            row_sync_tolerance, row_sync_metrics):
        image_height, image_width = task.original_image.shape[:2]
        rows_to_analyze = list(
            range(0, image_height, max(1, row_sync_stride))
        )
        middle_row = image_height // 2
        if middle_row not in rows_to_analyze:
            rows_to_analyze.append(middle_row)
            rows_to_analyze.sort()

        sync_dir = os.path.join(
            task.task_folder_path, "synchronization"
        )
        sync_title = synchronization_prefix("row", channel_code)
        for block_size in scale_block_sizes:
            try:
                block_size = int(block_size)
            except (TypeError, ValueError):
                continue
            if block_size <= 0:
                continue
            for metric_name in row_sync_metrics:
                results = (
                    self.calculate_row_synchronization_for_scale_block(
                        path=sync_dir,
                        title=sync_title,
                        points_by_scale=upper_points_by_scale,
                        scales=scales,
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
                        ),
                    )
                )
                task.synchronization_results[
                    (channel_code, block_size, metric_name)
                ] = results

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

    @staticmethod
    def save_extremes_graphic(path, title, points_local, coefs_2d=None, original_img_shape=None):
        """
        Сохранение изображения с точками экстремумов огибающих.

        Если передана матрица coefs_2d, точки накладываются на результат
        вейвлет-преобразования. Если coefs_2d не передана, рисуется только
        поле с точками.
        """
        import matplotlib.pyplot as plt
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
    def count_points_on_axis(points, slice_index, axis):
        """Count points on a row (y) or column (x) image slice."""
        if points is None or len(points) == 0:
            return 0
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 2:
            return 0
        coordinate = 1 if axis == "row" else 0
        return int(np.sum(points[:, coordinate] == slice_index))

    @staticmethod
    def save_upper_envelope_maxima_row_histogram(
            path,
            file_base_name,
            scales,
            max_counts,
            row_index,
            channel_name=None,
            save_png=True,
            save_csv=True,
            axis="row",
            cwt_direction=None,
    ):
        """
        Строит и сохраняет гистограмму распределения количества максимумов
        верхней огибающей на выбранной строке изображения в зависимости от масштаба.

        Возвращает:
            (png_path, csv_path)
        """
        import matplotlib.pyplot as plt
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

            direction = "rows" if axis == "row" else "columns"
            cwt_direction = cwt_direction or axis
            coordinate_name = "row_index" if axis == "row" else "column_index"
            coordinate_label = "строка изображения y" if axis == "row" else "столбец изображения x"

            # Дополнительно сохраняем численные данные, чтобы можно было проверить график.
            if save_csv:
                csv_path = os.path.join(path, f"{file_base_name}.csv")
                with open(csv_path, 'w', encoding='utf-8') as file:
                    file.write(
                        f"scale,upper_envelope_maxima_count,{coordinate_name},"
                        "cwt_direction,feature_axis\n"
                    )
                    for scale, count in zip(scales, max_counts):
                        file.write(
                            f"{scale},{int(count)},{row_index},"
                            f"{cwt_direction},{direction}\n"
                        )

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
                f"{title_channel}, {coordinate_label} = {row_index}"
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
            save_csv=True,
            axis="row",
            cwt_direction=None,
    ):
        """
        Сохраняет сжатую гистограмму по блокам масштабов.

        Каждый столбец соответствует диапазону масштабов.
        Высота столбца — сумма количества максимумов верхней огибающей
        по всем масштабам внутри блока.

        Возвращает:
            (png_path, csv_path)
        """
        import matplotlib.pyplot as plt
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

            direction = "rows" if axis == "row" else "columns"
            cwt_direction = cwt_direction or axis
            coordinate_name = "row_index" if axis == "row" else "column_index"
            coordinate_label = "строка изображения y" if axis == "row" else "столбец изображения x"

            if save_csv:
                csv_path = os.path.join(path, f"{file_base_name}.csv")
                with open(csv_path, 'w', encoding='utf-8') as file:
                    file.write(
                        "block_label,block_start_scale,block_end_scale,"
                        f"scales_in_block,upper_envelope_maxima_sum,{coordinate_name},"
                        "block_size,cwt_direction,feature_axis\n"
                    )
                    for label, count, block_range in zip(
                            block_labels, block_counts, block_ranges):
                        scale_start, scale_end, block_scales = block_range
                        scales_as_text = "|".join(str(s) for s in block_scales)
                        file.write(
                            f"{label},{scale_start},{scale_end},"
                            f"{scales_as_text},{int(count)},{row_index},{block_size},"
                            f"{cwt_direction},{direction}\n"
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
                f"{title_channel}, {coordinate_label} = {row_index}, "
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
        import matplotlib.pyplot as plt
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

        scale_folder_path = os.path.join(
            task.task_folder_path, "scales", scale_folder_name(scale)
        )
        if os.path.exists(scale_folder_path) and os.path.isdir(scale_folder_path):
            return scale_folder_path
        else:
            print(f"Directory {scale_folder_path} is not found")
            return None

    def compute_for_task(self, task):
        """Выполнение вычислений для конкретной задачи"""
        try:
            if not self.backend._initialized:
                self.progress.log_info("Подготовка вычислительного устройства...")
                if self.backend.use_gpu:
                    setup_cuda_environment()
                self.backend.ensure_initialized()
                self.progress.log_info(
                    f"Вычислительный бэкенд: {self.backend.get_backend_info()['device_name']}")
            requested_backend = getattr(self.backend, "requested_backend", "gpu" if self.backend.use_gpu else "cpu")
            task.requested_backend = requested_backend
            task.strict_backend = bool(getattr(task, "strict_backend", False))
            self.backend.set_strict_backend(task.strict_backend)
            task.execution_protocol.reset(
                requested_backend=requested_backend,
                strict_backend=task.strict_backend,
            )
            if requested_backend == "gpu" and not self.backend.use_gpu:
                if task.strict_backend:
                    raise GPUUnavailableError(
                        "Строгий режим: запрошен GPU, но CUDA backend недоступен"
                    )
                task.execution_protocol.record_fallback(
                    "backend_initialization",
                    reason="GPUUnavailableError",
                    message="Requested GPU backend is unavailable; CPU backend selected.",
                )
                task.execution_protocol.record_stage(
                    "backend_initialization", actual_backend="cpu",
                    dtype=COMPUTE_DTYPE_NAME, fallback=True,
                    details={"fallback_reason": "GPUUnavailableError"},
                )

            self.progress.begin_stage(f"{task.task_name} · Подготовка данных")
            task_folder = self.create_task_folder(task)
            self.progress.log_info(f"Создана папка для {task.task_name}: {task_folder}")
            prepared = prepare_task_channels(task)
            self.progress.log_info(
                "Каналы анализа: " + ", ".join(item.key for item in prepared)
                + f" ({task.channel_summary()})"
            )
            manifest_path = write_channel_manifest(task, task_folder)
            self.progress.log_info(f"Manifest запуска: {manifest_path}")

            # Сохранение исходных каналов - 5%
            self.progress.update_progress(0.05, "Сохранение исходных каналов...")
            self.save_orig_channels_txt(task, task.save_source_channels)

            current_cwt_signature = cwt_signature(task)
            can_reuse_pipeline = bool(
                getattr(task, "reuse_existing_results", True)
                and getattr(task, "last_completed_cwt_signature", None)
                == current_cwt_signature
                and getattr(task, "task_folder_path", "")
                and os.path.isdir(task.task_folder_path)
            )

            task.result = {}
            pipeline_plan = task.resolve_pipeline()
            # KNN/statistics are materialized again for the current request,
            # but their upstream CWT/extrema/envelopes may be restored.
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
            stage_states_before = stage_statuses(task)
            execution_plan = minimal_execution_plan(task)
            self.progress.log_info(
                "Состояние этапов: " + ", ".join(
                    f"{name}={state}" for name, state in stage_states_before.items()
                )
            )
            self.progress.log_info(format_execution_plan(task))
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
                point_execution = [s for s in execution_plan if s in {"extrema", "envelopes", "knn", "statistics"}]
                need_cwt_in_memory = bool("wavelet" in execution_plan or point_execution)
                if need_cwt_in_memory:
                    restored_cwt = None
                    if can_reuse_pipeline and "wavelet" not in execution_plan:
                        restored_cwt = load_complete_cwt(task)
                    if restored_cwt is not None:
                        task.result = restored_cwt
                        self.progress.log_info(
                            "CWT уже рассчитан для этой задачи — коэффициенты "
                            "загружены из lossless cache, повторный Morlet пропущен"
                        )
                        task.execution_protocol.record_stage(
                            "wavelet:resume", actual_backend="disk-cache",
                            dtype=COMPUTE_DTYPE_NAME,
                            details={"reused": True},
                        )
                        self.progress.update_progress(1.0, "Готовый CWT восстановлен")
                    else:
                        self.compute_wavelets(task, info_out)
                        mark_stage_ready(task, "wavelet")
                else:
                    self.progress.log_info(
                        "Вейвлеты уже актуальны; загрузка коэффициентов не требуется"
                    )
                    self.progress.update_progress(1.0, "Вейвлеты актуальны")

                if point_execution:
                    # Ready downstream stages are removed from the executable plan.
                    # Their prerequisites stay enabled so compute_points can restore
                    # the required cached inputs without recomputing them.
                    effective_plan = replace(
                        pipeline_plan,
                        knn=bool(pipeline_plan.knn and "knn" in execution_plan),
                        statistics=bool(pipeline_plan.statistics and "statistics" in execution_plan),
                    )
                    selected = []
                    if "extrema" in execution_plan:
                        selected.append("экстремумы")
                    if "envelopes" in execution_plan:
                        selected.append("огибающие")
                    if "knn" in execution_plan:
                        selected.append("KNN")
                    if "statistics" in execution_plan:
                        selected.append("статистики")
                    self.progress.update_progress(0.7, "Начало анализа точек...")
                    self.progress.begin_stage(
                        f"{task.task_name} · Досчёт: {', '.join(selected)}"
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
                        pipeline_plan=effective_plan
                    )
                    if "extrema" in execution_plan:
                        mark_stage_ready(task, "extrema")
                    if "envelopes" in execution_plan:
                        mark_stage_ready(task, "envelopes")
                    if "knn" in execution_plan:
                        mark_stage_ready(task, "knn")
                    if "statistics" in execution_plan:
                        mark_stage_ready(task, "statistics")
                elif pipeline_plan.point_pipeline_required:
                    self.progress.log_info(
                        "Все выбранные этапы анализа точек уже актуальны — пересчёт пропущен"
                    )

            task.last_completed_cwt_signature = current_cwt_signature
            manifest_path = write_channel_manifest(task, task_folder)
            self.progress.log_info(
                f"Протокол вычислительного backend обновлён: {manifest_path}"
            )

            completion = 1.0
            message = f"Вычисления для {task.task_name} завершены успешно"
            self.progress.update_progress(completion, message)
            self.progress.log_info(message)

        except Exception as e:
            import traceback
            error_msg = f"Ошибка при обработке задачи {task.task_name}: {str(e)}\n{traceback.format_exc()}"
            self.progress.log_error(error_msg)
            try:
                if getattr(task, "task_folder_path", ""):
                    write_channel_manifest(task, task.task_folder_path)
            except Exception:
                pass
            raise
        finally:
            # Очищаем тяжёлые временные коэффициенты; KNN/статистики остаются
            # доступными для ML и истории. Исходный RGB не меняется.
            task.result = {}
            clear_prepared_channels(task)
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

    def __init__(self, master, tab_names, navigation_master=None, hidden_tabs=()):
        super().__init__(master, fg_color="transparent", corner_radius=0)
        self._pages = {}
        self._buttons = {}
        self._underlines = {}
        self._current_name = None
        self._hidden_tabs = set(hidden_tabs)
        self._visible_tab_names = [
            name for name in tab_names if name not in self._hidden_tabs
        ]

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._navigation_height = 32 if navigation_master is not None else AppTheme.NAV_HEIGHT

        self.navigation = ctk.CTkFrame(
            navigation_master if navigation_master is not None else self,
            height=self._navigation_height,
            corner_radius=0,
            fg_color=AppTheme.NAV_BACKGROUND
        )
        if navigation_master is None:
            self.grid_rowconfigure(0, weight=0)
            self.grid_rowconfigure(1, weight=1)
            self.navigation.grid(row=0, column=0, sticky="ew")
        else:
            self.navigation.pack(side="left", fill="x", expand=True, padx=(12, 0))
        self.navigation.grid_propagate(False)
        self.navigation.pack_propagate(False)

        self.tabs_strip = ctk.CTkFrame(
            self.navigation,
            fg_color=AppTheme.NAV_BACKGROUND,
            corner_radius=0
        )
        self.tabs_strip.pack(side="left", fill="y", padx=4)

        # Keep all pages accessible when the window is too narrow for the tabs.
        self.compact_navigation = ctk.CTkOptionMenu(
            self.navigation, values=self._visible_tab_names, command=self.set,
            width=210, height=self._navigation_height - 4,
            fg_color=AppTheme.NAV_BACKGROUND,
            button_color=AppTheme.NAV_HOVER,
            button_hover_color=AppTheme.BORDER,
        )
        self.navigation.bind("<Configure>", self._fit_navigation, add="+")

        self.page_container = ctk.CTkFrame(
            self, fg_color="transparent", corner_radius=0
        )
        self.page_container.grid(
            row=1 if navigation_master is None else 0, column=0, sticky="nsew"
        )
        self.page_container.grid_columnconfigure(0, weight=1)
        self.page_container.grid_rowconfigure(0, weight=1)
        self.page_container.grid_propagate(False)

        for name in tab_names:
            self.add(name, show_in_navigation=name not in self._hidden_tabs)

    def _fit_navigation(self, event):
        if event.width < self.tabs_strip.winfo_reqwidth() + 8:
            self.tabs_strip.pack_forget()
            self.compact_navigation.pack(side="left", padx=4, pady=2)
        else:
            self.compact_navigation.pack_forget()
            self.tabs_strip.pack(side="left", fill="y", padx=4)

    def add(self, name, show_in_navigation=True):
        item = None
        button = None
        underline = None
        if show_in_navigation:
            item = ctk.CTkFrame(
                self.tabs_strip, fg_color="transparent", corner_radius=0
            )
            item.pack(side="left", fill="y")

            button = ctk.CTkButton(
                item,
                text=name,
                command=lambda tab_name=name: self.set(tab_name),
                width=max(60, 9 * len(name) + 20),
                height=self._navigation_height - 4,
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
        if button is not None:
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

    def current_tab(self):
        return self._current_name

    def set(self, name):
        if name not in self._pages:
            return
        # Переключаем саму страницу до её обновления. Если код конкретной
        # вкладки завершится ошибкой, навигация всё равно не должна оставаться
        # на предыдущей странице и создавать впечатление «заблокированной».
        self._pages[name].tkraise()
        self._current_name = name
        if name in self._visible_tab_names:
            self.compact_navigation.set(name)
        for tab_name, button in self._buttons.items():
            active = tab_name == name
            button.configure(
                fg_color="transparent",
                text_color=(
                    AppTheme.NAV_TEXT_ACTIVE if active
                    else AppTheme.NAV_TEXT_INACTIVE
                )
            )
            underline = self._underlines.get(tab_name)
            if underline is None:
                continue
            if active:
                underline.configure(fg_color=AppTheme.NAV_ACTIVE)
                underline.place(
                    relx=0.08, rely=1.0, relwidth=0.84, y=-4
                )
            else:
                underline.place_forget()
        if getattr(self, 'on_select', None) is not None:
            try:
                self.on_select(name)
            except Exception:
                # Передаём исключение штатному обработчику Tk уже после того,
                # как состояние навигации стало согласованным.
                self.winfo_toplevel().report_callback_exception(*sys.exc_info())

    def set_enabled(self, enabled):
        state = "normal" if enabled else "disabled"
        self.compact_navigation.configure(state=state)
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

    def __init__(self, *, show_splash=False):
        super().__init__()
        self._compute_thread = None
        self._ml_thread = None
        # Explicit UI lock states. Never infer widget availability from
        # Thread.is_alive(): worker shutdown and Tk callbacks are asynchronous.
        self._ml_ui_locked = False
        # Explicit UI lock state. Do not infer it from Thread.is_alive():
        # the worker may still be technically alive for a few milliseconds
        # after the Tk unlock callback has already run.
        self._compute_ui_locked = False
        self._startup_pending = True
        self._startup_steps = self._initialize_workspace()
        if show_splash:
            from utils.startup import StartupSplash
            self.withdraw()
            self._splash = StartupSplash(self, self.safe_destroy)
            self.after_safe(16, self._advance_startup)
        else:
            for _status in self._startup_steps:
                pass
            self._startup_pending = False
            self.after_safe(100, self._maximize_properly)

    def _advance_startup(self):
        try:
            self._splash.set_status(next(self._startup_steps))
        except StopIteration:
            self._startup_pending = False
            self.deiconify()
            self._maximize_properly()
            self.after_safe(0, self._splash.destroy)
        except Exception as error:
            traceback.print_exc()
            self._splash.show_error(str(error))
        else:
            # Return control to Tk between construction stages for painting and close.
            self.after_safe(1, self._advance_startup)

    def _initialize_workspace(self):
        yield "Подготовка рабочей области…"
        self._compute_thread = None
        self._loading_task_settings = False
        self.run_history = RunHistoryStore()
        self.title("Wavelets Analysis Studio")
        self.resizable(True, True)
        # Открываем главное окно крупным, но не растянутым на весь экран,
        # и всегда центрируем его относительно текущего монитора.
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        window_width = min(max(1200, int(screen_width * 0.90)), max(800, screen_width - 80))
        window_height = min(max(700, int(screen_height * 0.88)), max(600, screen_height - 80))
        window_x = max(0, (screen_width - window_width) // 2)
        window_y = max(0, (screen_height - window_height) // 2)
        self.geometry(
            f"{window_width}x{window_height}+{window_x}+{window_y}"
        )

        # настройка темы
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # инициализация всех атрибутов UI
        self._initialize_ui_variables()

        # Начальный экран создаётся в том же Tk-окне, что и рабочая область.
        # Это позволяет не плодить mainloop/Tk и безопасно сохранять состояние.
        self.welcome_frame = self._create_welcome_page()

        self._tasks_visible = True
        self._progress_visible = True
        self.layout_controls = ctk.CTkFrame(
            self, height=32, corner_radius=0, fg_color=AppTheme.NAV_BACKGROUND
        )
        self.layout_controls.pack(fill='x', padx=8, pady=2)
        self.tasks_toggle = IconButton(self.layout_controls, 'sidebar', 'Панель задач', self._toggle_tasks_panel)
        self.tasks_toggle.pack(side='left', padx=3)
        self.progress_toggle = IconButton(self.layout_controls, 'bottom', 'Панель выполнения', self._toggle_progress_panel)
        self.progress_toggle.pack(side='left', padx=3)
        IconButton(self.layout_controls, 'focus', 'Фокус на рабочей области', self._toggle_focus_layout).pack(side='left', padx=3)

        # Глобальная навигация находится отдельно от этапов текущего исследования.
        nav_button_style = dict(
            height=28,
            corner_radius=0,
            fg_color="transparent",
            hover_color=AppTheme.NAV_HOVER,
            text_color=AppTheme.NAV_TEXT_INACTIVE,
            border_width=0,
            font=ctk.CTkFont(size=14),
        )
        self.home_button = ctk.CTkButton(
            self.layout_controls,
            text="Главная",
            width=100,
            command=self._show_welcome,
            **nav_button_style,
        )
        self.home_button.pack(side="right", padx=(4, 3))
        self.history_button = ctk.CTkButton(
            self.layout_controls,
            text="История",
            width=95,
            command=self._show_history_page,
            **nav_button_style,
        )
        self.history_button.pack(side="right", padx=3)

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

        yield from self._create_workspace_tabs()

        yield "Подготовка панели выполнения…"
        self.progress_manager = ProgressManager(self)
        self.cancel_compute_button = ctk.CTkButton(self.progress_manager.frame, text='Отменить расчёт',
                                                   command=self._request_cancel, state='disabled', width=150)
        self.cancel_compute_button.pack(anchor='e', padx=12, pady=3)

        # менеджер изображений
        self.image_processor = ImageProcessor(self.progress_manager)

        # текущая задача
        self.current_task = None
        self.knn_text_var.trace('w', self.update_knn_for_current_task)
        for variable in (
                self.row_var, self.col_var, self.max_var, self.min_var,
                self.wp_var1, self.wp_var2, self.p_ex_var1, self.p_ex_var2,
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
                self.ml_eps_var, self.ml_min_samples_var, self.ml_dataset_var,
                self.ml_dbscan_tile_size_var, self.ml_dbscan_tile_overlap_var,
                self.ml_dbscan_max_points_var):
            variable.trace_add('write', self._store_ml_settings_for_current_task)

        # Переменная для GPU/CPU переключения
        self.use_gpu_var = tk.BooleanVar(value=True)
        self.compute_device_var = tk.StringVar(value="cpu")
        self.strict_backend_var = tk.BooleanVar(value=False)

        # Обновляем UI после инициализации image_processor
        yield "Завершение подготовки…"
        self._update_gpu_section()
        self._detect_compute_devices_async()
        self._update_action_availability()
        self._show_welcome(force=True)

    def _create_welcome_page(self):
        """Создать Research Hub: новый эксперимент и последние запуски."""
        page = ctk.CTkFrame(self, fg_color="transparent", corner_radius=0)

        outer = ctk.CTkFrame(page, fg_color="transparent")
        outer.pack(fill="both", expand=True, padx=48, pady=36)
        outer.grid_columnconfigure(0, weight=0, minsize=360)
        outer.grid_columnconfigure(1, weight=1, minsize=560)
        outer.grid_rowconfigure(1, weight=1)

        title_block = ctk.CTkFrame(outer, fg_color="transparent")
        title_block.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(10, 30))
        ctk.CTkLabel(
            title_block,
            text="Wavelets Analysis Studio",
            font=ctk.CTkFont(size=30, weight="bold"),
            anchor="w",
        ).pack(fill="x")
        ctk.CTkLabel(
            title_block,
            text="Студия исследования изображений и вейвлет-анализа",
            font=ctk.CTkFont(size=14),
            text_color=AppTheme.TEXT_SECONDARY,
            anchor="w",
        ).pack(fill="x", pady=(5, 0))

        new_card = ctk.CTkFrame(
            outer, border_width=1, border_color=AppTheme.BORDER, corner_radius=10
        )
        new_card.grid(row=1, column=0, sticky="nsew", padx=(0, 18))
        ctk.CTkLabel(
            new_card,
            text="Новое исследование",
            font=ctk.CTkFont(size=21, weight="bold"),
            anchor="w",
        ).pack(fill="x", padx=24, pady=(30, 8))
        ctk.CTkLabel(
            new_card,
            text=(
                "Создайте новую задачу, загрузите изображение и настройте "
                "вейвлет-преобразование и этапы обработки."
            ),
            font=AppTheme.body_font(),
            text_color=AppTheme.TEXT_SECONDARY,
            justify="left",
            anchor="nw",
            wraplength=300,
        ).pack(fill="x", padx=24, pady=(0, 22))
        ctk.CTkButton(
            new_card,
            text="+  Новое исследование",
            command=self._start_new_research,
            height=48,
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color=AppTheme.PRIMARY,
            hover_color=AppTheme.PRIMARY_HOVER,
        ).pack(fill="x", padx=24, pady=(0, 12))
        ctk.CTkLabel(
            new_card,
            text="Текущие задачи в этой сессии сохранятся в рабочей области.",
            font=AppTheme.caption_font(),
            text_color=AppTheme.MUTED,
            justify="left",
            anchor="w",
            wraplength=300,
        ).pack(fill="x", padx=24, pady=(0, 24))

        recent_card = ctk.CTkFrame(
            outer, border_width=1, border_color=AppTheme.BORDER, corner_radius=10
        )
        recent_card.grid(row=1, column=1, sticky="nsew")
        recent_header = ctk.CTkFrame(recent_card, fg_color="transparent")
        recent_header.pack(fill="x", padx=22, pady=(18, 7))
        ctk.CTkLabel(
            recent_header,
            text="Последние исследования",
            font=ctk.CTkFont(size=20, weight="bold"),
            anchor="w",
        ).pack(side="left", fill="x", expand=True)
        ctk.CTkButton(
            recent_header,
            text="Все исследования →",
            width=155,
            height=30,
            fg_color="transparent",
            hover_color=AppTheme.NAV_HOVER,
            command=self._show_history_page,
        ).pack(side="right")

        self.welcome_recent_container = ctk.CTkFrame(
            recent_card, fg_color="transparent"
        )
        self.welcome_recent_container.pack(
            fill="both", expand=True, padx=18, pady=(0, 12)
        )
        return page

    def _welcome_recent_limit(self):
        """Показывать 4 карточки на обычном Full HD и 3 на низких экранах."""
        try:
            return 6 if self.winfo_screenheight() >= 900 else 5
        except Exception:
            return 5

    def _refresh_welcome_recent_runs(self):
        """Обновить последние исследования на главном экране."""
        container = getattr(self, "welcome_recent_container", None)
        if container is None:
            return
        for child in container.winfo_children():
            child.destroy()

        try:
            runs = self.run_history.list_runs(limit=self._welcome_recent_limit())
        except Exception as error:
            ctk.CTkLabel(
                container,
                text=f"Не удалось загрузить историю: {error}",
                text_color=AppTheme.DANGER,
                anchor="w",
            ).pack(fill="x", padx=8, pady=12)
            return

        if not runs:
            empty = ctk.CTkFrame(container, fg_color="transparent")
            empty.pack(fill="both", expand=True, padx=8, pady=16)
            ctk.CTkLabel(
                empty,
                text="История исследований пока пуста",
                font=AppTheme.section_title_font(),
                anchor="w",
            ).pack(fill="x", pady=(8, 4))
            ctk.CTkLabel(
                empty,
                text="Первый завершённый запуск появится здесь автоматически.",
                font=AppTheme.body_font(),
                text_color=AppTheme.TEXT_SECONDARY,
                anchor="w",
            ).pack(fill="x")
            return

        for run in runs:
            self._create_welcome_run_card(container, run)

    def _create_welcome_run_card(self, parent, run):
        """Компактная карточка запуска для Research Hub."""
        try:
            details = self._format_history_snapshot(
                self.run_history.get_settings(run["id"])
            )
        except Exception:
            details = {
                "wavelet": run.get("analysis_mode", "—"),
                "scales": "—",
                "stages": run.get("pipeline_preset", "—"),
                "nuances": "",
            }

        try:
            created = time.strftime(
                "%d.%m.%Y  %H:%M",
                time.localtime(
                    datetime.fromisoformat(run["created_at"]).timestamp()
                ),
            )
        except Exception:
            created = str(run.get("created_at", ""))[:16].replace("T", "  ")

        status_raw = str(run.get("status", "") or "")
        status_text = {
            "completed": "Завершён",
            "cancelled": "Отменён",
        }.get(status_raw, "Ошибка" if status_raw else "")
        status_color = (
            AppTheme.SUCCESS if status_raw == "completed"
            else AppTheme.TEXT_SECONDARY if status_raw == "cancelled"
            else AppTheme.DANGER if status_raw
            else AppTheme.TEXT_SECONDARY
        )

        card = ctk.CTkFrame(
            parent, border_width=1, border_color=AppTheme.BORDER,
            corner_radius=8, fg_color=AppTheme.NAV_SELECTED_BACKGROUND,
        )
        card.pack(fill="x", padx=4, pady=3)

        top = ctk.CTkFrame(card, fg_color="transparent")
        top.pack(fill="x", padx=12, pady=(6, 2))
        ctk.CTkLabel(
            top, text=f"Эксперимент №{run['id']}",
            font=ctk.CTkFont(size=15, weight="bold"), anchor="w"
        ).pack(side="left")
        ctk.CTkLabel(
            top, text=created, font=AppTheme.caption_font(),
            text_color=AppTheme.TEXT_SECONDARY
        ).pack(side="right")

        middle = ctk.CTkFrame(card, fg_color="transparent")
        middle.pack(fill="x", padx=12, pady=(0, 2))
        ctk.CTkLabel(
            middle, text=details.get("wavelet", "—"),
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=AppTheme.NAV_HOVER, corner_radius=6,
            padx=8, pady=2
        ).pack(side="left")
        if status_text:
            ctk.CTkLabel(
                middle, text=status_text, font=AppTheme.caption_font(),
                text_color=status_color, fg_color=AppTheme.NAV_BACKGROUND,
                corner_radius=6, padx=7, pady=2
            ).pack(side="left", padx=(6, 0))

        # Действия держим на той же строке, что и бейджи: это экономит
        # почти треть высоты карточки, не ухудшая читаемость.
        if run.get("output_path"):
            ctk.CTkButton(
                middle, text="Результаты", width=96, height=25,
                fg_color="transparent", hover_color=AppTheme.NAV_HOVER,
                text_color=AppTheme.NAV_TEXT_ACTIVE, font=ctk.CTkFont(size=13),
                command=lambda folder=run["output_path"]: self._open_results_from_welcome(folder),
            ).pack(side="right", padx=(6, 0))
        ctk.CTkButton(
            middle, text="Продолжить", width=102, height=25,
            font=ctk.CTkFont(size=13),
            command=lambda run_id=run["id"]: self._continue_research(run_id)
        ).pack(side="right")

        meta = [
            f"Масштабы: {details.get('scales', '—')}",
            f"Сценарий: {run.get('pipeline_preset', '—')}"
        ]
        if run.get("duration_seconds"):
            meta.append(format_time(run["duration_seconds"]))
        ctk.CTkLabel(
            card, text="  ·  ".join(meta), font=AppTheme.caption_font(),
            text_color=AppTheme.TEXT_SECONDARY, anchor="w", wraplength=900
        ).pack(fill="x", padx=12, pady=(0, 5))

        # Заметка полезна для идентификации запуска, но на главном экране
        # показываем её одной короткой строкой, чтобы карточки не разрастались.
        note = str(run.get("researcher_note", "") or "").strip()
        if note:
            short_note = note if len(note) <= 110 else note[:107].rstrip() + "..."
            ctk.CTkLabel(
                card, text=f"Заметка: {short_note}", font=AppTheme.caption_font(),
                text_color=AppTheme.MUTED, anchor="w", wraplength=900
            ).pack(fill="x", padx=12, pady=(0, 5))

    def _workspace_is_busy(self):
        return any(
            thread is not None and thread.is_alive()
            for thread in (
                getattr(self, "_compute_thread", None),
                getattr(self, "_ml_thread", None),
            )
        )

    def _set_global_navigation_state(self, active=None):
        """Стилизовать глобальные кнопки навигации в духе верхних вкладок."""
        mapping = {
            "home": getattr(self, "home_button", None),
            "history": getattr(self, "history_button", None),
        }
        for name, button in mapping.items():
            if button is None:
                continue
            is_active = name == active
            button.configure(
                text_color=(
                    AppTheme.NAV_TEXT_ACTIVE if is_active
                    else AppTheme.NAV_TEXT_INACTIVE
                ),
                fg_color="transparent",
            )

    def _show_welcome(self, force=False):
        """Показать главный экран, не уничтожая рабочую сессию."""
        if not force and self._workspace_is_busy():
            mb.showwarning(
                "Выполнение",
                "Нельзя перейти на главный экран, пока выполняется расчёт.",
            )
            return

        for widget_name in ("layout_controls", "main_container"):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                widget.pack_forget()
        progress_manager = getattr(self, "progress_manager", None)
        if progress_manager is not None:
            progress_manager.frame.pack_forget()

        self._refresh_welcome_recent_runs()
        self._set_global_navigation_state("home")
        self.welcome_frame.pack(fill="both", expand=True)

    def _show_workspace(self):
        """Вернуться к рабочей области с сохранением текущих задач."""
        if getattr(self, "welcome_frame", None) is not None:
            self.welcome_frame.pack_forget()

        if not self.layout_controls.winfo_manager():
            self.layout_controls.pack(fill="x", padx=8, pady=2)
        if not self.main_container.winfo_manager():
            self.main_container.pack(
                fill="both",
                expand=True,
                padx=AppTheme.WINDOW_PADDING,
                pady=(2, AppTheme.SECTION_GAP),
            )
        if hasattr(self, "progress_manager") and not self.progress_manager.frame.winfo_manager():
            self.progress_manager.frame.pack(fill="x", padx=20, pady=(4, 10))
        current_tab = None
        if hasattr(self, "workspace_tabs"):
            current_tab = self.workspace_tabs.current_tab()
        self._set_global_navigation_state(
            "history" if current_tab == "Предыдущие запуски" else None
        )

    def _start_new_research(self):
        self._show_workspace()
        self.add_new_task()

    def _continue_research(self, run_id):
        self._show_workspace()
        self._restore_history_run(run_id)

    def _open_results_from_welcome(self, folder):
        self._show_workspace()
        self._show_saved_results(folder)

    def _show_history_page(self):
        self._show_workspace()
        self.workspace_tabs.set("Предыдущие запуски")

    def _update_gpu_section(self):
        """Обновление секции GPU после инициализации image_processor"""
        if hasattr(self, 'compute_settings_section') and self.compute_settings_section:
            # Очищаем секцию
            for widget in self.compute_settings_section.content.winfo_children():
                widget.destroy()

            # Пересоздаем содержимое
            self._setup_compute_settings_section()

    def _detect_compute_devices_async(self):
        """Проверить GPU в фоне и после проверки автоматически выбрать его."""
        if not hasattr(self, "image_processor") or self.image_processor is None:
            return
        if getattr(self, "_device_detection_started", False):
            return
        self._device_detection_started = True

        def worker():
            try:
                backend = self.image_processor.backend
                backend.use_gpu = True
                backend.ensure_initialized()
                info = backend.get_backend_info()
                if info.get("gpu_available"):
                    backend.set_use_gpu(True)
                else:
                    backend.set_use_gpu(False)
            except Exception as error:
                try:
                    self.progress_manager.log_error(f"Проверка GPU: {error}")
                except Exception:
                    pass
            finally:
                def finish():
                    info = self.image_processor.get_backend_info()
                    enabled = bool(info.get("gpu_available") and info.get("use_gpu"))
                    self.use_gpu_var.set(enabled)
                    self.compute_device_var.set("gpu" if enabled else "cpu")
                    self._update_gpu_section()
                self.after(0, finish)

        threading.Thread(target=worker, daemon=True, name="device-detection").start()

    def _maximize_properly(self):
        """Развернуть главное окно приложения на весь экран."""
        try:
            if self.tk.call("tk", "windowingsystem") == "win32":
                self.state("zoomed")
            else:
                self.attributes("-zoomed", True)
        except tk.TclError:
            # Fallback: занять доступную область экрана.
            screen_width = self.winfo_screenwidth()
            screen_height = self.winfo_screenheight()
            self.geometry(f"{screen_width}x{screen_height}+0+0")

        # Даём Tk/CustomTkinter завершить перерасчёт размеров интерфейса.
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
        self.channel_representation_var = tk.StringVar(value="RGB")
        self.rgb_channel_mode_var = tk.StringVar(value="Все RGB")
        self.rgb_single_channel_var = tk.StringVar(value="R")
        self.calculate_extrema_var = tk.BooleanVar(value=False)
        self.calculate_envelopes_var = tk.BooleanVar(value=False)
        self.calculate_knn_var = tk.BooleanVar(value=False)
        self.row_var = tk.BooleanVar(value=True)
        self.col_var = tk.BooleanVar(value=True)
        self.max_var = tk.BooleanVar(value=True)
        self.min_var = tk.BooleanVar(value=True)
        self.wp_var1 = tk.BooleanVar(value=False)
        self.wp_var2 = tk.BooleanVar(value=True)
        self.wavelet_numpy_var = tk.BooleanVar(value=False)
        self.p_ex_var1 = tk.BooleanVar(value=True)
        self.p_ex_var2 = tk.BooleanVar(value=False)
        self.envelope_text_var = tk.BooleanVar(value=True)
        self.envelope_image_var = tk.BooleanVar(value=False)
        self.knn_bool_text_var = tk.BooleanVar(value=True)
        self.knn_bool_image_var = tk.BooleanVar(value=False)
        self.print_channels_txt_var = tk.BooleanVar(value=False)
        self.centering_means_var = tk.BooleanVar(value=False)
        self.calculate_statistics_var = tk.BooleanVar(value=False)
        self.statistics_image_var = tk.BooleanVar(value=False)
        self.statistics_csv_var = tk.BooleanVar(value=True)
        self.calculate_sync_var = tk.BooleanVar(value=False)
        self.sync_heatmap_var = tk.BooleanVar(value=False)
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
        self.ml_dataset_var = tk.StringVar(value="")
        self._ml_dataset_options = {}
        self.ml_point_filter_var = tk.StringVar(value="Все точки KNN")
        self.ml_feature_set_var = tk.StringVar(value="Геометрия KNN")
        self.ml_standardize_var = tk.BooleanVar(value=True)
        self.ml_n_clusters_var = tk.StringVar(value="5")
        self.ml_random_state_var = tk.StringVar(value="42")
        self.ml_eps_var = tk.StringVar(value="0.8")
        self.ml_min_samples_var = tk.StringVar(value="5")
        self.ml_dbscan_tile_size_var = tk.StringVar(value="256")
        self.ml_dbscan_tile_overlap_var = tk.StringVar(value="32")
        self.ml_dbscan_max_points_var = tk.StringVar(value="30000")
        self._ml_thread = None
        self._ml_preview_image = None

        # Journal filters/pagination are interface state, not analysis parameters.
        self.history_search_var = tk.StringVar(value="")
        self.history_status_var = tk.StringVar(value="Все статусы")
        self.history_page_size_var = tk.StringVar(value="5")
        self.history_page = 1
        self.history_total_pages = 1
        self._history_search_after_id = None

        # Widget references
        self.load_button = None
        self.print_load_image = None
        self.pipette_button = None
        self.gram_shmidt_button = None
        self.channel_representation_menu = None
        self.rgb_mode_menu = None
        self.rgb_single_channel_menu = None
        self.channel_detection_label = None
        self.channel_effective_label = None
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
        self.cpu_radio = None
        self.gpu_radio = None
        self.pipeline_preset_tooltip = None
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
        yield "Подготовка навигации и загрузки данных…"
        self.workspace_tabs = WorkspaceTabs(
            self.workspace_frame,
            ("Данные", "Параметры расчёта", "Результаты", "ML", "Предыдущие запуски"),
            navigation_master=self.layout_controls,
            hidden_tabs=("Предыдущие запуски",),
        )
        self.workspace_tabs.grid(
            row=0, column=0, sticky="nsew"
        )

        self.data_panel = self._create_data_tab(
            self.workspace_tabs.tab("Данные")
        )
        yield "Подготовка параметров анализа…"
        self.analysis_panel = self._create_analysis_tab(
            self.workspace_tabs.tab("Параметры расчёта")
        )
        self.workspace_tabs.on_select = self._on_workspace_tab_selected
        self.workspace_tabs.set("Данные")

    def _on_workspace_tab_selected(self, name):
        self._set_global_navigation_state(
            "history" if name == "Предыдущие запуски" else None
        )
        self._ensure_workspace_tab(name)

        if name == "ML" and hasattr(self, "ml_panel"):
            # Состояние ML-контролов нельзя брать из старого состояния виджетов.
            # KNN мог завершиться уже после создания вкладки, а общий unlock UI
            # мог ещё находиться в очереди Tk. Пересчитываем доступность при
            # КАЖДОМ входе на вкладку и ещё раз в следующем цикле event loop.
            self._update_ml_tab_state()
            self.after_safe(0, self._refresh_ml_controls_state)

    def _ensure_workspace_tab(self, name):
        """Build optional pages once, retaining their state and fixed containers."""
        if name == "Результаты" and not hasattr(self, 'results_panel'):
            from utils.result_viewer import ResultsPanel
            self.results_panel = ResultsPanel(self.workspace_tabs.tab(name))
            self.results_panel.pack(fill="both", expand=True)
            task = getattr(self, 'current_task', None)
            self.results_panel.load_folder(task.task_folder_path if task else '')
        elif name == "ML" and not hasattr(self, 'ml_panel'):
            self.ml_panel = self._create_ml_tab(self.workspace_tabs.tab(name))
            self._update_ml_tab_state()
        elif name == "Предыдущие запуски":
            if not hasattr(self, 'history_panel'):
                self.history_panel = self._create_history_tab(self.workspace_tabs.tab(name))
            elif getattr(self, '_history_dirty', False):
                self._refresh_history_tab(force=True)
            if getattr(self, '_compute_ui_locked', False):
                self._lock_controls_in(self.history_panel)

    def _create_history_tab(self, parent):
        """Create a researcher-friendly paginated view of persistent previous runs."""
        panel = ctk.CTkFrame(parent, fg_color="transparent")
        panel.pack(fill="both", expand=True)

        header = ctk.CTkFrame(panel, fg_color="transparent")
        header.pack(fill="x", padx=12, pady=(12, 6))
        ctk.CTkLabel(
            header, text="История исследований",
            font=AppTheme.panel_title_font(), anchor="w"
        ).pack(side="left", fill="x", expand=True)
        ctk.CTkButton(
            header, text="Обновить", width=110,
            height=AppTheme.SMALL_BUTTON_HEIGHT,
            command=lambda: self._refresh_history_tab(force=True)
        ).pack(side="right")

        ctk.CTkLabel(
            panel,
            text=("Здесь сохраняются настройки и результаты исследований. "
                  "Любой запуск можно повторить как новую задачу."),
            font=AppTheme.caption_font(),
            text_color=AppTheme.TEXT_SECONDARY,
            anchor="w", justify="left"
        ).pack(fill="x", padx=12, pady=(0, 8))

        filters = ctk.CTkFrame(panel, fg_color="transparent")
        filters.pack(fill="x", padx=12, pady=(0, 8))

        self.history_search_entry = ctk.CTkEntry(
            filters,
            textvariable=self.history_search_var,
            placeholder_text="Поиск по № запуска, изображению, сценарию или задаче",
            width=360
        )
        self.history_search_entry.pack(side="left", fill="x", expand=True)

        # История доступна для чтения/поиска даже во время вычислений.
        self.history_search_entry._read_only_during_compute = True

        # Не полагаемся только на trace StringVar: в CustomTkinter на Windows
        # после программных обновлений/смены вкладок он может отрабатывать
        # неудобно. KeyRelease гарантированно запускает реактивный поиск.
        self.history_search_entry.bind(
            "<KeyRelease>",
            self._on_history_search_key_event,
            add="+",
        )
        self.history_search_entry.bind(
            "<<Paste>>",
            self._on_history_search_key_event,
            add="+",
        )
        self.history_search_entry.bind(
            "<<Cut>>",
            self._on_history_search_key_event,
            add="+",
        )

        self.history_status_menu = ctk.CTkOptionMenu(
            filters,
            variable=self.history_status_var,
            values=["Все статусы", "Завершённые", "С ошибкой", "Отменённые"],
            command=lambda _value: self._on_history_filter_changed(),
            width=145
        )
        self.history_status_menu.pack(side="left", padx=(8, 0))
        self.history_status_menu._read_only_during_compute = True

        ctk.CTkLabel(
            filters,
            text="На странице:",
            font=AppTheme.caption_font(),
            text_color=AppTheme.TEXT_SECONDARY
        ).pack(side="left", padx=(12, 5))

        self.history_page_size_menu = ctk.CTkOptionMenu(
            filters,
            variable=self.history_page_size_var,
            values=["5", "10", "20", "50"],
            command=lambda _value: self._on_history_filter_changed(),
            width=78
        )
        self.history_page_size_menu.pack(side="left")
        self.history_page_size_menu._read_only_during_compute = True

        self.history_scrollable = ScrollableFrame(panel)
        self.history_scrollable.pack(
            fill="both", expand=True, padx=6, pady=(0, 4)
        )
        self.history_cards = []

        pagination = ctk.CTkFrame(panel, fg_color="transparent")
        pagination.pack(fill="x", padx=12, pady=(2, 8))

        self.history_prev_button = ctk.CTkButton(
            pagination,
            text="← Предыдущая",
            width=120,
            height=AppTheme.COMPACT_CONTROL_HEIGHT,
            command=lambda: self._change_history_page(-1)
        )
        self.history_prev_button.pack(side="left")
        self.history_prev_button._read_only_during_compute = True

        self.history_page_label = ctk.CTkLabel(
            pagination,
            text="Страница 1 из 1",
            font=AppTheme.caption_font(),
            text_color=AppTheme.TEXT_SECONDARY
        )
        self.history_page_label.pack(side="left", fill="x", expand=True)

        self.history_next_button = ctk.CTkButton(
            pagination,
            text="Следующая →",
            width=120,
            height=AppTheme.COMPACT_CONTROL_HEIGHT,
            command=lambda: self._change_history_page(1)
        )
        self.history_next_button.pack(side="right")
        self.history_next_button._read_only_during_compute = True

        self._refresh_history_tab(force=True)
        return panel

    def _on_history_search_key_event(self, _event=None):
        """Запустить реактивный поиск после ввода/вставки текста."""
        self.history_page = 1

        callback_id = getattr(self, "_history_search_after_id", None)
        if callback_id:
            self.after_cancel_safe(callback_id)

        # Для Paste/Cut значение Entry обновляется после события,
        # поэтому даём Tk закончить стандартную обработку.
        self._history_search_after_id = self.after_safe(
            120, self._apply_reactive_history_search
        )

    def _apply_reactive_history_search(self):
        self._history_search_after_id = None

        # Берём текст непосредственно из виджета — это устраняет рассинхронизацию
        # между CTkEntry и StringVar на некоторых версиях CustomTkinter/Tk.
        if hasattr(self, "history_search_entry"):
            current_text = self.history_search_entry.get()
            if current_text != self.history_search_var.get():
                self.history_search_var.set(current_text)

        self._refresh_history_tab(force=True)

    def _on_history_filter_changed(self):
        self.history_page = 1
        self._refresh_history_tab(force=True)

    def _change_history_page(self, delta):
        target = self.history_page + int(delta)
        target = max(1, min(target, max(1, self.history_total_pages)))
        if target == self.history_page:
            return
        self.history_page = target
        self._refresh_history_tab(force=True)

    def _refresh_history_tab(self, force=False):
        if not hasattr(self, "history_scrollable"):
            return
        if (not force and hasattr(self, 'history_panel') and
                self.workspace_tabs._current_name != "Предыдущие запуски"):
            self._history_dirty = True
            return
        self._history_dirty = False
        for card in self.history_cards:
            card.destroy()
        self.history_cards.clear()
        # Берём достаточно записей для локальной фильтрации и пагинации.
        # Store возвращает их от новых к старым.
        runs = self.run_history.list_runs(limit=1000)
        query_source = (
            self.history_search_entry.get()
            if hasattr(self, "history_search_entry")
            else self.history_search_var.get()
        )
        query = query_source.strip().casefold()
        status_filter = self.history_status_var.get()

        filtered_runs = []
        for run in runs:
            if status_filter == "Завершённые" and run["status"] != "completed":
                continue
            if status_filter == "С ошибкой" and run["status"] != "failed":
                continue
            if status_filter == "Отменённые" and run["status"] != "cancelled":
                continue

            # В поиск обязательно включаем номер эксперимента: запрос "11"
            # должен находить "Эксперимент №11".
            searchable = " ".join((
                str(run.get("id", "")),
                str(run.get("created_at", "")),
                run.get("task_name", ""),
                run.get("image_path", ""),
                run.get("pipeline_preset", ""),
                run.get("ml_algorithm", ""),
                run.get("researcher_note", ""),
                run.get("status", ""),
            )).casefold()
            if query and query not in searchable:
                continue
            filtered_runs.append(run)

        total_runs = len(filtered_runs)
        try:
            page_size = max(1, int(self.history_page_size_var.get()))
        except (TypeError, ValueError):
            page_size = 5
            self.history_page_size_var.set("5")

        self.history_total_pages = max(
            1, (total_runs + page_size - 1) // page_size
        )
        self.history_page = max(
            1, min(self.history_page, self.history_total_pages)
        )

        start_index = (self.history_page - 1) * page_size
        end_index = start_index + page_size
        runs = filtered_runs[start_index:end_index]

        if hasattr(self, "history_page_label"):
            if total_runs:
                first_number = start_index + 1
                last_number = min(end_index, total_runs)
                self.history_page_label.configure(
                    text=(
                        f"Страница {self.history_page} из "
                        f"{self.history_total_pages} · "
                        f"{first_number}–{last_number} из {total_runs}"
                    )
                )
            else:
                self.history_page_label.configure(text="Нет записей")

        if hasattr(self, "history_prev_button"):
            self.history_prev_button.configure(
                state="normal" if self.history_page > 1 else "disabled"
            )
        if hasattr(self, "history_next_button"):
            self.history_next_button.configure(
                state=(
                    "normal"
                    if self.history_page < self.history_total_pages
                    else "disabled"
                )
            )

        parent = self.history_scrollable.scrollable_frame
        if not runs:
            empty = ctk.CTkLabel(
                parent, text="По заданным условиям запусков не найдено",
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
            status = {'completed': 'Завершён', 'cancelled': 'Отменён'}.get(run['status'], 'Ошибка')
            status_color = (AppTheme.SUCCESS if run['status'] == 'completed' else
                            AppTheme.TEXT_SECONDARY if run['status'] == 'cancelled' else AppTheme.DANGER)
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
            view_results = ctk.CTkButton(
                actions, text="Результаты", width=110,
                command=lambda folder=run['output_path']: self._show_saved_results(folder)
            )
            view_results._read_only_during_compute = True
            view_results.pack(side="left", padx=(0, 6))
            ctk.CTkButton(
                actions, text="Продолжить исследование", width=190,
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
        import cv2
        self._show_workspace()
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
                    # Представление будет подготовлено перед новым расчётом;
                    # исходный RGB при восстановлении истории не преобразуется.
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
            text="Далее: параметры расчёта",
            command=lambda: self.workspace_tabs.set("Параметры расчёта"),
            width=AppTheme.ACTION_BUTTON_WIDTH,
            height=AppTheme.BUTTON_HEIGHT,
            state="disabled"
        )
        self.data_next_button.pack(anchor="e", padx=12, pady=(14, 10))
        return panel

    def _create_analysis_tab(self, parent):
        """Единая страница параметров расчёта с двухколоночной компоновкой."""
        panel = ctk.CTkFrame(parent, fg_color="transparent")
        panel.pack(fill="both", expand=True)
        scrollable = ScrollableFrame(panel)
        scrollable.pack(fill="both", expand=True)
        content = scrollable.scrollable_frame
        self.analysis_content = content

        self._add_tab_heading(
            content,
            "Параметры расчёта",
            "Настройте режим анализа, этапы вычислений и форматы сохранения результатов."
        )

        # Строка 1: вычислительное устройство + режим вейвлет-анализа.
        top_row = ctk.CTkFrame(content, fg_color="transparent")
        top_row.pack(fill="x", padx=5, pady=(0, 4))
        top_row.grid_columnconfigure((0, 1), weight=1, uniform="analysis_top")

        self.compute_settings_section = CollapsibleFrame(
            top_row,
            title="Вычислительное устройство",
            fg_color=AppTheme.NAV_HOVER,
            corner_radius=10,
        )
        self.compute_settings_section.grid(
            row=0, column=0, sticky="nsew", padx=(0, 4)
        )
        ctk.CTkLabel(
            self.compute_settings_section.content,
            text="Определение устройств...",
            font=AppTheme.body_font(),
            text_color=AppTheme.MUTED
        ).pack(pady=10)

        self.analysis_mode_section = CollapsibleFrame(
            top_row,
            title="Режим вейвлет-анализа",
            fg_color=AppTheme.NAV_HOVER,
            corner_radius=10,
        )
        self.analysis_mode_section.grid(
            row=0, column=1, sticky="nsew", padx=(4, 0)
        )
        self._populate_analysis_mode_section(self.analysis_mode_section.content)

        # Строка 2: масштабы + настройки выбранного вейвлет-преобразования.
        transform_row = ctk.CTkFrame(content, fg_color="transparent")
        transform_row.pack(fill="x", padx=5, pady=4)
        transform_row.grid_columnconfigure((0, 1), weight=1, uniform="analysis_transform")

        self.scales_section = CollapsibleFrame(
            transform_row,
            title="Масштабы",
            fg_color=AppTheme.NAV_HOVER,
            corner_radius=10,
        )
        self.scales_section.grid(
            row=0, column=0, sticky="nsew", padx=(0, 4)
        )
        self._setup_scales_section()

        self.transform_slot = ctk.CTkFrame(transform_row, fg_color="transparent")
        self.transform_slot.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        self.transform_slot.grid_columnconfigure(0, weight=1)

        self.extremes_section = CollapsibleFrame(
            self.transform_slot,
            title="Настройки 1-D преобразования",
            fg_color=AppTheme.NAV_HOVER,
            corner_radius=10,
        )
        self.extremes_section.grid(row=0, column=0, sticky="nsew")
        self._setup_extremes_section()

        self.two_d_section = CollapsibleFrame(
            self.transform_slot,
            title="Настройки 2D Morlet",
            fg_color=AppTheme.NAV_HOVER,
            corner_radius=10,
        )
        self.two_d_section.grid(row=0, column=0, sticky="nsew")
        self._setup_2d_section()
        self.two_d_section.grid_remove()

        # Строка 3: сценарий анализа.
        self.pipeline_section = CollapsibleFrame(content, title="Сценарий анализа")
        self.pipeline_section.pack(fill="x", padx=5, pady=4)
        self._setup_pipeline_section()

        # Строка 4: параметры выбранных этапов.
        self.stage_settings_section = CollapsibleFrame(
            content, title="Настройки выбранных этапов"
        )
        self.stage_settings_section.pack(fill="x", padx=5, pady=4)
        self._setup_stage_settings_section()

        # Строка 5: экспорт результатов.
        self.export_section = CollapsibleFrame(content, title="Экспорт результатов")
        self.export_section.pack(fill="x", padx=5, pady=4)
        self._setup_export_results_section()

        # Строка 6: дополнительные файлы.
        self.intermediate_section = CollapsibleFrame(
            content, title="Дополнительные файлы"
        )
        self.intermediate_section.pack(fill="x", padx=5, pady=4)
        self._setup_intermediate_section()

        self.compute_section = ctk.CTkFrame(content, fg_color="transparent")
        self.compute_section.pack(fill="x", padx=5, pady=20)
        self._setup_compute_section()

        self.analysis_next_button = None
        self.after_idle(self._setup_compute_settings_section)
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
        source_select = ctk.CTkFrame(source_frame, fg_color="transparent")
        source_select.pack(fill="x", padx=12, pady=(0, 10))
        ctk.CTkLabel(source_select, text="KNN-набор", font=AppTheme.body_font(), anchor="w").pack(side="left")
        self.ml_dataset_selector = ctk.CTkOptionMenu(
            source_select, values=["Нет KNN-наборов"], variable=self.ml_dataset_var,
            command=lambda _value: self._store_ml_settings_for_current_task(), width=560,
            state="disabled")
        self.ml_dataset_selector.pack(side="right")

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
        self.ml_algorithm_selector = ctk.CTkOptionMenu(
            common_grid, values=["K-means", "DBSCAN"],
            variable=self.ml_algorithm_var,
            command=self._on_ml_algorithm_changed, width=220
        )
        self.ml_feature_set_selector = ctk.CTkOptionMenu(
            common_grid,
            values=["Геометрия KNN", "Координаты + KNN"],
            variable=self.ml_feature_set_var, width=260
        )
        labels_and_widgets = [
            ("Алгоритм", self.ml_algorithm_selector),
            ("Набор признаков", self.ml_feature_set_selector),
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
        ctk.CTkLabel(
            results_frame,
            text="Пространственный результат сохраняется как ML-слой. Откройте вкладку «Результаты» и добавьте его через «+ Слой».",
            font=AppTheme.caption_font(), text_color=AppTheme.TEXT_SECONDARY,
            anchor="w", justify="left", wraplength=760
        ).pack(fill="x", padx=12, pady=(0, 6))

        preview_controls = ctk.CTkFrame(results_frame, fg_color="transparent")
        preview_controls.pack(fill="x", padx=12, pady=(2, 6))
        self.ml_preview_selector = ctk.CTkOptionMenu(
            preview_controls,
            values=[
                "Кластеры на изображении", "Кластеры в признаках",
                "По масштабам", "По RGB-каналам"
            ],
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
        # CTkLabel не во всех версиях CustomTkinter корректно освобождает
        # прежний PhotoImage при configure(image=None). Оставляем постоянную
        # прозрачную заглушку, чтобы Tk никогда не ссылался на уже удалённый
        # pyimage после смены параметров или повторной кластеризации.
        empty_preview = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        self._ml_empty_preview_image = ctk.CTkImage(
            light_image=empty_preview,
            dark_image=empty_preview,
            size=(1, 1),
        )
        self.ml_preview_label.configure(image=self._ml_empty_preview_image)
        self._on_ml_algorithm_changed(self.ml_algorithm_var.get())
        return panel

    def _update_ml_tab_state(self):
        """Обновить источник, настройки и последний результат активной задачи."""
        page_created = hasattr(self, "ml_source_label")
        task = getattr(self, "current_task", None)
        if task is None:
            if not page_created:
                return
            self.ml_source_label.configure(
                text="Активная задача не выбрана",
                text_color=AppTheme.TEXT_SECONDARY
            )
            self._show_ml_result(None)
            self._refresh_ml_controls_state()
            return

        self._loading_task_settings = True
        has_knn = bool(getattr(task, "knn_results", None))
        if has_knn:
            from ml.clustering import knn_dataset_options
            dataset_options = knn_dataset_options(task.knn_results)
        else:
            dataset_options = []
        self._ml_dataset_options = {item["label"]: item["key"] for item in dataset_options}
        if page_created:
            values = list(self._ml_dataset_options) or ["Нет KNN-наборов"]
            self.ml_dataset_selector.configure(values=values, state="normal" if dataset_options else "disabled")
            wanted = getattr(task, "ml_dataset_key", None)
            selected_label = None
            if wanted is not None:
                selected_label = next((label for label, key in self._ml_dataset_options.items()
                                       if tuple(key) == tuple(wanted)), None)
            if selected_label is None and dataset_options:
                selected_label = dataset_options[0]["label"]
                task.ml_dataset_key = dataset_options[0]["key"]
            self.ml_dataset_var.set(selected_label or "Нет KNN-наборов")
        source_status = (
            "KNN-признаки готовы к кластеризации" if has_knn else
            "KNN-признаки ещё не рассчитаны. На вкладке «Параметры расчёта» "
            "выберите сценарий «Подготовка данных для ML» и запустите расчёт."
        )
        if page_created:
            self.ml_source_label.configure(
                text=f"{task.task_name}: {source_status}",
                text_color=AppTheme.TEXT_ON_DARK if has_knn else AppTheme.TEXT_SECONDARY
            )

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
        self.ml_dbscan_tile_size_var.set(str(getattr(task, "ml_dbscan_tile_size", 256)))
        self.ml_dbscan_tile_overlap_var.set(str(getattr(task, "ml_dbscan_tile_overlap", 32)))
        self.ml_dbscan_max_points_var.set(str(getattr(task, "ml_dbscan_max_points_per_tile", 30000)))
        self._loading_task_settings = False
        self._on_ml_algorithm_changed(self.ml_algorithm_var.get())
        self._show_ml_result(task.ml_result)
        self._refresh_ml_controls_state()

    def _on_ml_algorithm_changed(self, _value=None):
        """Показать параметры выбранного алгоритма без уничтожения CTkEntry.

        Важно: CTkEntry, привязанный к постоянному StringVar, нельзя постоянно
        destroy()/создавать заново. В некоторых версиях CustomTkinter trace
        переменной переживает уничтожение внутреннего tk.Entry, после чего
        следующий variable.set() вызывает TclError ``invalid command name``.
        Поэтому обе группы параметров создаются один раз и затем только
        переключаются через pack_forget()/pack().
        """
        frame = getattr(self, "ml_algorithm_parameters", None)
        if frame is None:
            return

        if not getattr(self, "_ml_parameter_widgets_built", False):
            self._ml_kmeans_parameters_frame = ctk.CTkFrame(
                frame, fg_color="transparent"
            )
            self._ml_dbscan_parameters_frame = ctk.CTkFrame(
                frame, fg_color="transparent"
            )

            self._ml_kmeans_parameter_entries = []
            self._ml_dbscan_parameter_entries = []

            kmeans_fields = [
                ("Количество кластеров", self.ml_n_clusters_var),
                ("Начальное состояние", self.ml_random_state_var),
            ]
            dbscan_fields = [
                ("Радиус соседства ε", self.ml_eps_var),
                ("Минимум точек", self.ml_min_samples_var),
                ("Размер пространственного тайла, px", self.ml_dbscan_tile_size_var),
                ("Перекрытие тайлов, px", self.ml_dbscan_tile_overlap_var),
                ("Максимум точек в тайле", self.ml_dbscan_max_points_var),
            ]

            def build_group(container, fields, target_entries):
                container.grid_columnconfigure(1, weight=1)
                for row, (label, variable) in enumerate(fields):
                    ctk.CTkLabel(
                        container,
                        text=label,
                        font=AppTheme.body_font(),
                        anchor="w",
                    ).grid(
                        row=row, column=0, sticky="w",
                        padx=(0, 14), pady=4,
                    )
                    entry = ctk.CTkEntry(
                        container, textvariable=variable, width=160
                    )
                    entry.grid(row=row, column=1, sticky="w", pady=4)
                    target_entries.append(entry)

            build_group(
                self._ml_kmeans_parameters_frame,
                kmeans_fields,
                self._ml_kmeans_parameter_entries,
            )
            build_group(
                self._ml_dbscan_parameters_frame,
                dbscan_fields,
                self._ml_dbscan_parameter_entries,
            )
            self._ml_parameter_entries = (
                self._ml_kmeans_parameter_entries
                + self._ml_dbscan_parameter_entries
            )
            self._ml_parameter_widgets_built = True

        # Никаких destroy(): только переключение уже существующих групп.
        self._ml_kmeans_parameters_frame.pack_forget()
        self._ml_dbscan_parameters_frame.pack_forget()

        if self.ml_algorithm_var.get() == "K-means":
            self._ml_kmeans_parameters_frame.pack(fill="x")
            help_text = (
                "K-means делит все точки на заранее заданное число групп. "
                "Шумовые точки отдельно не выделяются."
            )
        else:
            self._ml_dbscan_parameters_frame.pack(fill="x")
            help_text = (
                "DBSCAN выполняется локально по пространственным тайлам с перекрытием. "
                "Все точки участвуют в анализе; перекрытие уменьшает граничные артефакты. "
                "Номера кластеров разных тайлов не являются глобальными классами."
            )

        if getattr(self, "ml_algorithm_help_label", None) is not None:
            self.ml_algorithm_help_label.configure(text=help_text)
        self._refresh_ml_controls_state()

    def _refresh_ml_controls_state(self):
        """Пересчитать доступность ML-контролов из фаз UI-операций.

        Нельзя использовать ``Thread.is_alive()`` для состояния виджетов:
        callback завершения уже выполняется в Tk-потоке, когда worker ещё может
        оставаться alive до выхода из своей функции. Именно это окно гонки
        оставляло параметры ML заблокированными после готового KNN/ML.
        Проверки ``is_alive()`` остаются только в командах запуска, где они
        защищают от реального параллельного старта второй операции.
        """
        if not hasattr(self, "ml_run_button"):
            return

        task = getattr(self, "current_task", None)
        has_knn = bool(task is not None and getattr(task, "knn_results", None))
        compute_locked = bool(getattr(self, "_compute_ui_locked", False))
        ml_locked = bool(getattr(self, "_ml_ui_locked", False))
        editable = ml_controls_editable(
            knn_ready=has_knn,
            compute_locked=compute_locked,
            ml_locked=ml_locked,
        )
        state = "normal" if editable else "disabled"

        for widget_name in (
            "ml_dataset_selector",
            "ml_algorithm_selector",
            "ml_feature_set_selector",
            "ml_standardize_checkbox",
        ):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                widget.configure(state=state)

        for entry in getattr(self, "_ml_parameter_entries", []):
            try:
                if entry.winfo_exists():
                    entry.configure(state=state)
            except Exception:
                pass

        self.ml_run_button.configure(
            state=state,
            text=("Выполняется..." if ml_locked else "Выполнить кластеризацию")
        )

    def _store_ml_settings(self, task):
        point_filters = {
            "Все точки KNN": "all", "Максимумы по строкам": "max_by_row",
            "Минимумы по строкам": "min_by_row",
            "Максимумы по столбцам": "max_by_column",
            "Минимумы по столбцам": "min_by_column",
        }
        task.ml_algorithm = "kmeans" if self.ml_algorithm_var.get() == "K-means" else "dbscan"
        selected_dataset = self._ml_dataset_options.get(self.ml_dataset_var.get())
        if selected_dataset is None:
            raise ValueError("Не выбран KNN-набор")
        task.ml_dataset_key = tuple(selected_dataset)
        task.ml_point_filter = "all"
        task.ml_feature_set = (
            "knn" if self.ml_feature_set_var.get() == "Геометрия KNN"
            else "coordinates_knn"
        )
        task.ml_standardize = bool(self.ml_standardize_var.get())
        task.ml_n_clusters = int(self.ml_n_clusters_var.get())
        task.ml_random_state = int(self.ml_random_state_var.get())
        task.ml_eps = float(self.ml_eps_var.get().replace(",", "."))
        task.ml_min_samples = int(self.ml_min_samples_var.get())
        task.ml_dbscan_tile_size = int(self.ml_dbscan_tile_size_var.get())
        task.ml_dbscan_tile_overlap = int(self.ml_dbscan_tile_overlap_var.get())
        task.ml_dbscan_max_points_per_tile = int(self.ml_dbscan_max_points_var.get())

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
        self._ml_ui_locked = True
        self._refresh_ml_controls_state()
        self.progress_manager.run_control.reset()
        self.ml_result_label.configure(
            text="Подготовка признаков и кластеризация...",
            text_color=AppTheme.TEXT_SECONDARY,
        )
        ml_stages = [
            "Подготовка признаков",
            "Стандартизация",
            "Кластеризация",
            "Метрики качества",
            "Сохранение и визуализация",
        ]
        self.progress_manager.begin_run(
            f"{task.task_name} · повторная ML-кластеризация",
            ml_stages,
        )
        self.cancel_compute_button.configure(state="normal", text="Отменить расчёт")
        self._ml_thread = threading.Thread(
            target=self._ml_clustering_worker, args=(task,), daemon=True
        )
        self._ml_thread.start()

    def _ml_clustering_worker(self, task):
        from ml.clustering import run_clustering, knn_dataset_options
        started = time.time()
        snapshot = task.settings_snapshot()
        try:
            save_run_image(task)
            snapshot = task.settings_snapshot()
            stage_names = {
                "features": "Подготовка признаков",
                "scaling": "Стандартизация",
                "clustering": "Кластеризация",
                "metrics": "Метрики качества",
                "saving": "Сохранение и визуализация",
            }
            active_stage = {"key": None}

            def ml_progress(stage_key, value, message):
                stage_name = stage_names.get(stage_key, stage_key)
                if active_stage["key"] != stage_key:
                    self.progress_manager.begin_stage(stage_name)
                    active_stage["key"] = stage_key
                self.progress_manager.update_progress(value, message)

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
                dataset_key=task.ml_dataset_key,
                dbscan_tile_size=getattr(task, "ml_dbscan_tile_size", 256),
                dbscan_tile_overlap=getattr(task, "ml_dbscan_tile_overlap", 32),
                dbscan_max_points_per_tile=getattr(task, "ml_dbscan_max_points_per_tile", 30000),
                progress_callback=ml_progress,
                cancel_callback=self.progress_manager.run_control.check,
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
        except RunCancelled:
            self.run_history.add_run(
                task=task, status="cancelled",
                duration_seconds=time.time() - started,
                settings=snapshot, ml_algorithm=task.ml_algorithm,
            )
            self.after_safe(0, self._refresh_history_tab)
            self.after_safe(0, self._cancel_ml_clustering)
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

    def _cancel_ml_clustering(self):
        self._ml_ui_locked = False
        self._refresh_ml_controls_state()
        self.cancel_compute_button.configure(state="disabled", text="Отменить расчёт")
        self.progress_manager.cancel_run()
        if hasattr(self, "ml_result_label"):
            self.ml_result_label.configure(
                text="ML-расчёт отменён пользователем",
                text_color=AppTheme.TEXT_SECONDARY,
            )

    def _finish_ml_clustering(self, task, result):
        self._ml_ui_locked = False
        self.cancel_compute_button.configure(state="disabled", text="Отменить расчёт")
        self._show_saved_results(task.task_folder_path)
        if self.current_task is task:
            self._show_ml_result(result)
        self._refresh_ml_controls_state()
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
        self._ml_ui_locked = False
        self.cancel_compute_button.configure(state="disabled", text="Отменить расчёт")
        self._refresh_ml_controls_state()
        self.ml_result_label.configure(text=message, text_color=AppTheme.DANGER)
        self.progress_manager.fail_run(f"Ошибка ML: {message}")

    @staticmethod
    def _format_metric(value):
        return "не рассчитывается" if value is None else f"{value:.4f}"

    def _show_ml_result(self, result):
        if not hasattr(self, 'ml_result_label'):
            return
        if not result:
            self.ml_result_label.configure(
                text="Кластеризация ещё не выполнялась",
                text_color=AppTheme.TEXT_SECONDARY,
            )
            self.ml_preview_selector.configure(state="disabled")
            self.ml_open_folder_button.configure(state="disabled")
            self._set_ml_preview_image(None)
            return
        metrics = result["metrics"]
        if result.get("tiled"):
            stats = result.get("dbscan_statistics") or {}
            anomaly_mean = stats.get("anomaly_mean", result["metrics"].get("anomaly_mean"))
            counts = (
                f"A>0: {stats.get('anomaly_nonzero', 0):,}; "
                f"устойчивые A≥{stats.get('anomaly_threshold', 0.75):.2f}: {stats.get('anomaly_stable', 0):,}; "
                f"A=1: {stats.get('anomaly_full', 0):,}; "
                f"средняя A: {anomaly_mean:.3f}" if anomaly_mean is not None else
                f"A>0: {stats.get('anomaly_nonzero', 0):,}"
            )
        else:
            counts = ", ".join(
                (f"шум: {count}" if label == -1 else f"кластер {label + 1}: {count}")
                for label, count in result["counts"].items()
            )
        timings = result.get("timings", {})
        timing_text = "; ".join(
            f"{name}: {timings[name]:.2f} c"
            for name in ("feature_extraction", "clustering", "quality_metrics", "saving_and_visualization")
            if name in timings
        )
        silhouette_sample = metrics.get("silhouette_sample_size", 0)
        sample_text = (
            f" (выборка {silhouette_sample:,})" if silhouette_sample else ""
        )
        self.ml_result_label.configure(
            text=(
                f"Алгоритм: {result['algorithm_name']}; точек: {result.get('point_count', len(result.get('records', []))):,}\n"
                f"Набор: {result.get('dataset', {}).get('channel', '—')} · a={result.get('dataset', {}).get('scale', '—')} · "
                f"{result.get('dataset', {}).get('point_type_name', '—')} · {result.get('dataset', {}).get('direction', '—')}\n"
                + ((f"Локальный сеточный DBSCAN: тайлов {result.get('dbscan_statistics', {}).get('tiles', '—')}\n")
                   if result.get("tiled") else
                   (f"Кластеров: {metrics['cluster_count']}; шумовых точек: {metrics['noise_count']}\n"))
                + f"Результат: {counts}\n"
                + (("DBSCAN: локальные ID тайлов не являются глобальными кластерами; "
                    "основной показатель — устойчивость аномалии A∈[0,1].\n") if result.get("tiled") else
                   (f"Silhouette: {self._format_metric(metrics['silhouette'])}{sample_text}; "
                    f"Davies–Bouldin: {self._format_metric(metrics['davies_bouldin'])}; "
                    f"Calinski–Harabasz: {self._format_metric(metrics['calinski_harabasz'])}\n"))
                + (f"Время: {timing_text}\n" if timing_text else "")
                + (("DBSCAN: анализируйте отдельные ML-слои по масштабу, каналу, типу точек и направлению во вкладке «Результаты»."
                    ) if result.get("tiled") else
                   "Silhouette: ближе к 1 — лучше; Davies–Bouldin: меньше — лучше; Calinski–Harabasz: больше — лучше.")
            ),
            text_color=AppTheme.TEXT_ON_DARK,
        )
        preview_values = ["Аномальность в признаках" if result.get("tiled") else "Кластеры в признаках"]
        self.ml_preview_selector.configure(values=preview_values, state="normal")
        if self.ml_preview_selector.get() not in preview_values:
            self.ml_preview_selector.set(preview_values[0])
        self.ml_open_folder_button.configure(state="normal")
        self._refresh_ml_preview()

    def _set_ml_preview_image(self, image=None, text=""):
        """Атомарно заменить ML-превью, не оставляя Tk со старым pyimage."""
        if not hasattr(self, "ml_preview_label"):
            return
        replacement = image or getattr(self, "_ml_empty_preview_image", None)
        # Старое изображение должно оставаться достижимым до того момента,
        # когда Tk действительно переключит image-option на новое.
        previous = getattr(self, "_ml_preview_image", None)
        self.ml_preview_label.configure(image=replacement, text=text)
        self._ml_preview_image = image
        del previous

    def _refresh_ml_preview(self):
        task = self.current_task
        result = task.ml_result if task is not None else None
        if not result:
            return
        preview_paths = {
            "Кластеры на изображении": result.get("image_plot_path"),
            "Кластеры в признаках": result.get("feature_plot_path"),
            "Аномальность в признаках": result.get("feature_plot_path"),
            "По масштабам": result.get("scale_plot_path"),
            "По RGB-каналам": result.get("channel_plot_path"),
            "Шум DBSCAN": result.get("noise_plot_path"),
        }
        path = preview_paths.get(self.ml_preview_selector.get())
        if not path:
            self._set_ml_preview_image(None, "Визуализация недоступна")
            return
        try:
            with Image.open(path) as source_image:
                image = source_image.convert("RGBA")
                image.thumbnail((760, 430), Image.Resampling.LANCZOS)
            preview_image = ctk.CTkImage(
                light_image=image.copy(), dark_image=image.copy(), size=image.size
            )
            self._set_ml_preview_image(preview_image)
        except Exception as error:
            self._set_ml_preview_image(
                None, f"Не удалось показать изображение: {error}"
            )

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

    def _setup_analysis_mode_section(self, parent, manage_geometry=True):
        """Создать отдельный блок выбора режима анализа."""
        mode_frame = CollapsibleFrame(parent, title="Режим вейвлет-анализа")
        if manage_geometry:
            mode_frame.pack(fill="x", padx=5, pady=(0, 8))
        self._populate_analysis_mode_section(mode_frame.content)
        return mode_frame

    def _populate_analysis_mode_section(self, parent):
        """Заполнить содержимое блока выбора режима активной задачи."""
        self.analysis_source_label = ctk.CTkLabel(
            parent,
            text="Источник: сначала загрузите изображение",
            font=AppTheme.body_font(),
            text_color=AppTheme.TEXT_SECONDARY,
            anchor="w"
        )
        self.analysis_source_label.pack(fill="x", padx=2, pady=(0, 6))

        self.analysis_mode_selector = ctk.CTkSegmentedButton(
            parent,
            values=list(self.ANALYSIS_MODE_LABELS.keys()),
            variable=self.analysis_mode_var,
            command=self.on_analysis_mode_changed,
            height=AppTheme.BUTTON_HEIGHT,
            state="disabled"
        )
        self.analysis_mode_selector.pack(fill="x", padx=2, pady=(0, 8))
        self.analysis_mode_selector.set("1D-анализ")

        self.analysis_mode_description = ctk.CTkLabel(
            parent,
            text="Создайте или активируйте задачу, чтобы выбрать режим.",
            font=AppTheme.caption_font(),
            text_color=AppTheme.MUTED,
            anchor="w",
            justify="left",
            wraplength=520
        )
        self.analysis_mode_description.pack(fill="x", padx=2, pady=(0, 2))

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
        """Показать настройки, применимые к текущему режиму анализа."""
        task = getattr(self, "current_task", None)
        if task is None:
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
            if hasattr(self, "two_d_section"):
                self.two_d_section.grid_remove()
            if hasattr(self, "extremes_section"):
                self.extremes_section.grid()
            return

        mode = task.analysis_mode
        mode_label = self.ANALYSIS_MODE_NAMES[mode]
        self.analysis_mode_var.set(mode_label)
        self.analysis_mode_selector.set(mode_label)
        source_loaded = bool(task.image_path)
        self.analysis_mode_selector.configure(
            state="normal" if source_loaded else "disabled"
        )
        self.analysis_source_label.configure(
            text=(f"Источник: {os.path.basename(task.image_path)}"
                  if source_loaded else "Источник: сначала загрузите изображение"),
            text_color=(AppTheme.TEXT_ON_DARK if source_loaded
                        else AppTheme.TEXT_SECONDARY)
        )

        if mode == "1d":
            self.analysis_mode_description.configure(
                text="Одномерное преобразование Морле по строкам и/или столбцам изображения.",
                text_color=AppTheme.TEXT_SECONDARY
            )
            self.two_d_section.grid_remove()
            self.extremes_section.grid()
            for section in (
                self.pipeline_section,
                self.stage_settings_section,
                self.export_section,
            ):
                if not section.winfo_manager():
                    section.pack(fill="x", padx=5, pady=4, before=self.intermediate_section)
        else:
            self.analysis_mode_description.configure(
                text=("Двумерное комплексное преобразование Морле по масштабам "
                      "и ориентациям. Результат содержит модуль и фазу."),
                text_color=AppTheme.TEXT_SECONDARY
            )
            self.extremes_section.grid_remove()
            self.two_d_section.grid()
            for section in (
                self.pipeline_section,
                self.stage_settings_section,
            ):
                section.pack_forget()
            # Экспорт оставляем: для 2D доступны вейвлетные результаты.
            if not self.export_section.winfo_manager():
                self.export_section.pack(fill="x", padx=5, pady=4, before=self.intermediate_section)

        self._update_pipeline_controls_state()

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
        """Показать CPU/GPU и выбрать GPU по умолчанию, когда он доступен."""
        for widget in self.compute_settings_section.content.winfo_children():
            widget.destroy()

        if not hasattr(self, "image_processor") or self.image_processor is None:
            ctk.CTkLabel(
                self.compute_settings_section.content,
                text="Определение устройств...",
                font=AppTheme.body_font(), text_color=AppTheme.MUTED
            ).pack(anchor="w", pady=8)
            return

        info = self.image_processor.get_backend_info()
        pending = info.get("status") == "pending" or not info.get("gpu_checked", False)
        gpu_available = bool(info.get("gpu_available", False))

        cpu_name, cpu_details = self._get_cpu_display_info()
        self.compute_device_var.set("gpu" if info.get("use_gpu") and gpu_available else "cpu")

        cpu_row = ctk.CTkFrame(self.compute_settings_section.content, fg_color="transparent")
        cpu_row.pack(fill="x", pady=(0, 8))
        self.cpu_radio = ctk.CTkRadioButton(
            cpu_row, text="CPU", variable=self.compute_device_var, value="cpu",
            command=self._on_compute_device_changed
        )
        self.cpu_radio.pack(anchor="w")
        ctk.CTkLabel(
            cpu_row, text=f"{cpu_name} | {cpu_details}",
            font=AppTheme.caption_font(), text_color=AppTheme.TEXT_SECONDARY,
            anchor="w", justify="left"
        ).pack(fill="x", padx=(28, 0), pady=(2, 0))

        gpu_row = ctk.CTkFrame(self.compute_settings_section.content, fg_color="transparent")
        gpu_row.pack(fill="x")
        self.gpu_radio = ctk.CTkRadioButton(
            gpu_row, text="GPU (CUDA)", variable=self.compute_device_var, value="gpu",
            command=self._on_compute_device_changed,
            state="disabled" if pending or not gpu_available else "normal"
        )
        self.gpu_radio.pack(anchor="w")

        if pending:
            gpu_text = "Проверка доступности GPU..."
            gpu_color = AppTheme.MUTED
        elif gpu_available:
            gpu_text = info.get("gpu_device_name", "GPU")
            memory = info.get("gpu_memory", "N/A")
            if memory != "N/A":
                gpu_text += f" | Память: {memory}"
            gpu_color = AppTheme.GPU_AVAILABLE
        else:
            gpu_text = "Совместимый GPU не обнаружен"
            gpu_color = AppTheme.WARNING

        ctk.CTkLabel(
            gpu_row, text=gpu_text,
            font=AppTheme.caption_font(), text_color=gpu_color,
            anchor="w", justify="left"
        ).pack(fill="x", padx=(28, 0), pady=(2, 0))

        strict_row = ctk.CTkFrame(
            self.compute_settings_section.content, fg_color="transparent"
        )
        strict_row.pack(fill="x", pady=(10, 0))
        self.strict_backend_checkbox = ctk.CTkCheckBox(
            strict_row,
            text="Строгая воспроизводимость: не переходить GPU → CPU",
            variable=self.strict_backend_var,
            command=self._on_strict_backend_changed,
        )
        self.strict_backend_checkbox.pack(anchor="w")
        ctk.CTkLabel(
            strict_row,
            text=(
                "При ошибке CUDA расчёт остановится вместо автоматического "
                "fallback на CPU."
            ),
            font=AppTheme.caption_font(), text_color=AppTheme.TEXT_SECONDARY,
            anchor="w", justify="left", wraplength=520,
        ).pack(fill="x", padx=(28, 0), pady=(2, 0))

    @staticmethod
    def _get_cpu_display_info():
        """Короткое человекочитаемое описание CPU без обязательных зависимостей."""
        name = platform.processor().strip()
        if not name and sys.platform.startswith("win"):
            try:
                import winreg
                with winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
                ) as key:
                    name = str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
            except Exception:
                name = ""
        if not name:
            name = platform.machine() or "CPU"
        logical = os.cpu_count() or 1
        physical = None
        try:
            import psutil
            physical = psutil.cpu_count(logical=False)
        except Exception:
            pass
        if physical:
            details = f"{physical} ядер, {logical} потоков"
        else:
            details = f"{logical} логических потоков"
        return name, details

    def _on_compute_device_changed(self):
        requested_gpu = self.compute_device_var.get() == "gpu"
        enabled, _ = self.image_processor.set_gpu_enabled(requested_gpu)
        self.use_gpu_var.set(enabled)
        self.compute_device_var.set("gpu" if enabled else "cpu")
        self.progress_manager.log_info(
            f"Для вычислений выбран {'GPU' if enabled else 'CPU'}"
        )
        self._update_compute_settings_display()
        self._update_gpu_section()

    def _on_strict_backend_changed(self):
        enabled = bool(self.strict_backend_var.get())
        self.image_processor.backend.set_strict_backend(enabled)
        if self.current_task is not None:
            self.current_task.strict_backend = enabled
        self.progress_manager.log_info(
            "Строгая воспроизводимость включена: GPU fallback запрещён"
            if enabled else
            "Строгая воспроизводимость выключена: GPU fallback разрешён"
        )

    def toggle_gpu_backend(self):
        """Обратная совместимость со старым переключателем GPU."""
        self.compute_device_var.set("gpu" if self.use_gpu_var.get() else "cpu")
        self._on_compute_device_changed()

    def _update_compute_settings_display(self):
        """Обновление отображения настроек вычислений"""
        if not hasattr(self, 'image_processor') or self.image_processor is None:
            return

        knn_gpu_available = (
            self.image_processor.backend.use_gpu and
            self.image_processor._knn_processor is not None and
            self.image_processor._knn_processor.is_gpu_available()
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
        """Явный выбор представления и каналов исследования."""
        representation_frame = ctk.CTkFrame(
            self.channel_section.content, fg_color="transparent"
        )
        self.channel_section.add_widget(representation_frame, pady=(2, 6))
        ctk.CTkLabel(
            representation_frame,
            text="Представление изображения:",
            font=AppTheme.body_font(),
            anchor="w",
        ).pack(fill="x")
        self.channel_representation_menu = ctk.CTkOptionMenu(
            representation_frame,
            values=["RGB", "Grayscale", "Gram–Schmidt"],
            variable=self.channel_representation_var,
            command=self._on_channel_settings_changed,
            width=AppTheme.ACTION_BUTTON_WIDTH,
        )
        self.channel_representation_menu.pack(anchor="w", pady=(5, 0))
        self.channel_detection_label = ctk.CTkLabel(
            representation_frame,
            text="Тип изображения ещё не определён",
            font=AppTheme.caption_font(),
            text_color=AppTheme.TEXT_SECONDARY,
            anchor="w",
        )
        self.channel_detection_label.pack(fill="x", pady=(5, 0))

        rgb_frame = ctk.CTkFrame(
            self.channel_section.content, fg_color="transparent"
        )
        self.channel_section.add_widget(rgb_frame, pady=(2, 6))
        ctk.CTkLabel(
            rgb_frame, text="RGB-каналы:", font=AppTheme.body_font(), anchor="w"
        ).pack(fill="x")
        controls = ctk.CTkFrame(rgb_frame, fg_color="transparent")
        controls.pack(fill="x", pady=(5, 0))
        self.rgb_mode_menu = ctk.CTkOptionMenu(
            controls,
            values=["Все RGB", "Один канал"],
            variable=self.rgb_channel_mode_var,
            command=self._on_channel_settings_changed,
            width=155,
        )
        self.rgb_mode_menu.pack(side="left")
        self.rgb_single_channel_menu = ctk.CTkOptionMenu(
            controls,
            values=["R", "G", "B"],
            variable=self.rgb_single_channel_var,
            command=self._on_channel_settings_changed,
            width=80,
        )
        self.rgb_single_channel_menu.pack(side="left", padx=(8, 0))

        self.channel_effective_label = ctk.CTkLabel(
            self.channel_section.content,
            text="Будут рассчитаны: R, G, B",
            font=AppTheme.caption_font(),
            text_color=AppTheme.TEXT_SECONDARY,
            anchor="w",
        )
        self.channel_section.add_widget(
            self.channel_effective_label, pady=(0, 8)
        )

        pipette_frame = ctk.CTkFrame(
            self.channel_section.content, fg_color="transparent"
        )
        self.channel_section.add_widget(pipette_frame, pady=2)
        ctk.CTkLabel(
            pipette_frame,
            text="Цветовой базис Gram–Schmidt:",
            font=AppTheme.body_font(),
            anchor="w",
        ).pack(fill="x")
        self.pipette_button = ctk.CTkButton(
            pipette_frame,
            text="Выбрать цвета пипеткой",
            command=self.pipette_channel,
            width=AppTheme.ACTION_BUTTON_WIDTH,
            height=AppTheme.BUTTON_HEIGHT,
            font=AppTheme.body_font(),
        )
        self.pipette_button.pack(anchor="w", pady=(5, 0))

        gram_frame = ctk.CTkFrame(
            self.channel_section.content, fg_color="transparent"
        )
        self.channel_section.add_widget(gram_frame, pady=(8, 2))
        self.gram_shmidt_button = ctk.CTkButton(
            gram_frame,
            text="Использовать Gram–Schmidt",
            command=self.gramm_shmidt_transform,
            width=AppTheme.ACTION_BUTTON_WIDTH,
            height=AppTheme.BUTTON_HEIGHT,
            font=AppTheme.body_font(),
        )
        self.gram_shmidt_button.pack(anchor="w")
        self._update_channel_controls_state()

    def _on_channel_settings_changed(self, _value=None):
        task = getattr(self, "current_task", None)
        if task is None or self._loading_task_settings:
            self._update_channel_controls_state()
            return

        representation = {
            "RGB": "rgb",
            "Grayscale": "grayscale",
            "Gram–Schmidt": "gram_schmidt",
        }.get(self.channel_representation_var.get(), "rgb")
        rgb_mode = (
            "single" if self.rgb_channel_mode_var.get() == "Один канал"
            else "all"
        )
        try:
            task.set_channel_analysis(
                representation,
                rgb_mode=rgb_mode,
                single_channel=self.rgb_single_channel_var.get(),
            )
        except ValueError as error:
            mb.showwarning("Каналы анализа", str(error), parent=self)
            return

        self._update_channel_controls_state()
        self._update_tasks_display()
        self._update_ml_tab_state()
        self._update_action_availability()

    def _update_channel_controls_state(self):
        task = getattr(self, "current_task", None)
        has_source = bool(task is not None and task.image_path)
        representation = self.channel_representation_var.get()
        is_rgb = representation == "RGB"
        is_single = is_rgb and self.rgb_channel_mode_var.get() == "Один канал"
        has_colors = bool(task is not None and task.has_colors_selected())

        if self.channel_representation_menu is not None:
            self.channel_representation_menu.configure(
                state="normal" if has_source else "disabled"
            )
        if self.rgb_mode_menu is not None:
            self.rgb_mode_menu.configure(
                state="normal" if has_source and is_rgb else "disabled"
            )
        if self.rgb_single_channel_menu is not None:
            self.rgb_single_channel_menu.configure(
                state="normal" if has_source and is_single else "disabled"
            )

        if task is None:
            detection = "Тип изображения ещё не определён"
            effective = "Будут рассчитаны: —"
        else:
            if task.detected_color_type == "grayscale":
                similarity = task.grayscale_similarity
                suffix = (
                    f" · совпадение RGB {similarity:.2%}"
                    if similarity is not None else ""
                )
                detection = "Обнаружено: grayscale" + suffix
            elif task.detected_color_type == "color":
                detection = "Обнаружено: цветное изображение"
            else:
                detection = "Тип изображения ещё не определён"

            codes = task.analysis_channel_codes_for_settings()
            readable = {
                "r": "R", "g": "G", "b": "B", "gray": "Gray",
                "gs1": "GS1", "gs2": "GS2", "gs3": "GS3",
            }
            effective = "Будут рассчитаны: " + ", ".join(
                readable.get(code, code) for code in codes
            )
            if task.channel_representation == "gram_schmidt" and not has_colors:
                effective += " · требуется цветовой базис"

        if self.channel_detection_label is not None:
            self.channel_detection_label.configure(text=detection)
        if self.channel_effective_label is not None:
            self.channel_effective_label.configure(text=effective)
        if self.pipette_button is not None:
            self.pipette_button.configure(
                text="Изменить цвета пипеткой" if has_colors else "Выбрать цвета пипеткой",
                state="normal" if has_source else "disabled",
                fg_color=AppTheme.PRIMARY,
                hover_color=AppTheme.PRIMARY_HOVER,
            )
        if self.gram_shmidt_button is not None:
            selected = bool(task is not None and task.channel_representation == "gram_schmidt")
            self.gram_shmidt_button.configure(
                text="Gram–Schmidt выбран" if selected else "Использовать Gram–Schmidt",
                state="normal" if has_source and has_colors else "disabled",
                fg_color=AppTheme.SUCCESS if selected else AppTheme.PRIMARY,
                hover_color=AppTheme.SUCCESS_HOVER if selected else AppTheme.PRIMARY_HOVER,
            )

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
            text="Применить масштабы",
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
        """Параметры 1D-направлений и типов экстремумов в две колонки."""
        grid = ctk.CTkFrame(self.extremes_section.content, fg_color="transparent")
        self.extremes_section.add_widget(grid, pady=2)
        grid.grid_columnconfigure((0, 1), weight=1, uniform="transform_options")

        direction_frame = ctk.CTkFrame(grid, fg_color="transparent")
        direction_frame.grid(row=0, column=0, sticky="nw", padx=(0, 10))
        ctk.CTkLabel(
            direction_frame, text="Источники 1D CWT:",
            font=AppTheme.body_font(), anchor="w"
        ).pack(fill="x", pady=(0, 5))
        self.row_checkbox = ctk.CTkCheckBox(
            direction_frame, text="CWT по строкам", variable=self.row_var,
            command=self._on_direction_changed
        )
        self.row_checkbox.pack(anchor="w", pady=3)
        self.col_checkbox = ctk.CTkCheckBox(
            direction_frame, text="CWT по столбцам", variable=self.col_var,
            command=self._on_direction_changed
        )
        self.col_checkbox.pack(anchor="w", pady=3)

        type_frame = ctk.CTkFrame(grid, fg_color="transparent")
        type_frame.grid(row=0, column=1, sticky="nw", padx=(10, 0))
        ctk.CTkLabel(
            type_frame, text="Тип экстремумов:",
            font=AppTheme.body_font(), anchor="w"
        ).pack(fill="x", pady=(0, 5))
        self.max_checkbox = ctk.CTkCheckBox(
            type_frame, text="Максимумы", variable=self.max_var,
            command=self._store_settings_for_current_task
        )
        self.max_checkbox.pack(anchor="w", pady=3)
        self.min_checkbox = ctk.CTkCheckBox(
            type_frame, text="Минимумы", variable=self.min_var,
            command=self._store_settings_for_current_task
        )
        self.min_checkbox.pack(anchor="w", pady=3)

        self.extremes_hint_label = ctk.CTkLabel(
            self.extremes_section.content,
            text=("Для каждого выбранного CWT экстремумы ищутся "
                  "и по строкам, и по столбцам."),
            font=AppTheme.caption_font(),
            text_color=AppTheme.TEXT_SECONDARY,
            anchor="w", justify="left", wraplength=500,
        )
        self.extremes_section.add_widget(
            self.extremes_hint_label, pady=(6, 0)
        )

    def _on_direction_changed(self):
        """Сразу синхронизировать направления и доступность зависимых этапов."""
        self._store_settings_for_current_task()
        self._update_pipeline_controls_state()
        self._update_action_availability()

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
        lines = [f"Каналы анализа: {task.channel_summary()}"]

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
            self.wavelet_numpy_var.set(False)
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
            self.channel_representation_var.set({
                "rgb": "RGB",
                "grayscale": "Grayscale",
                "gram_schmidt": "Gram–Schmidt",
            }.get(self.current_task.channel_representation, "RGB"))
            self.rgb_channel_mode_var.set(
                "Один канал"
                if self.current_task.rgb_channel_mode == "single"
                else "Все RGB"
            )
            self.rgb_single_channel_var.set(self.current_task.rgb_single_channel)
            self.strict_backend_var.set(
                bool(getattr(self.current_task, "strict_backend", False))
            )
            self.image_processor.backend.set_strict_backend(
                self.strict_backend_var.get()
            )
            self._loading_task_settings = False
            self._update_channel_controls_state()
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
            self._update_channel_controls_state()
            self.button_save_scales.configure(
                text="Применить масштабы",
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
        if hasattr(self, 'memory_label'):
            self._refresh_memory_estimate()
        task = self.current_task
        self._update_navigation_buttons()
        self._update_channel_controls_state()
        if task is None:
            self._set_point_analysis_controls_enabled(False)
            self.analysis_mode_selector.configure(state="disabled")
            self.load_button.configure(state="disabled")
            self.button_save_scales.configure(state="disabled")
            self.button_load_scales_file.configure(state="disabled")
            self.app_start_button.configure(state="disabled")
            self._set_workflow_status(
                "Шаг 1 из 3: создайте задачу и загрузите изображение",
                AppTheme.TEXT_SECONDARY
            )
            return

        self._set_point_analysis_controls_enabled(
            task.analysis_mode == "1d"
            and (task.process_rows or task.process_columns)
        )
        self.load_button.configure(state="normal")
        if not task.image_path:
            self.analysis_mode_selector.configure(state="disabled")
            self.button_save_scales.configure(state="disabled")
            self.button_load_scales_file.configure(state="disabled")
            self.app_start_button.configure(state="disabled")
            self._set_workflow_status(
                "Шаг 1 из 3: загрузите изображение",
                AppTheme.WARNING
            )
            return

        self.analysis_mode_selector.configure(state="normal")
        self.button_save_scales.configure(state="normal")
        self.button_load_scales_file.configure(state="normal")

        if (task.channel_representation == "gram_schmidt"
                and not task.has_colors_selected()):
            self.app_start_button.configure(state="disabled")
            self._set_workflow_status(
                "Шаг 2 из 3: задайте цветовой базис Gram–Schmidt пипеткой",
                AppTheme.WARNING
            )
            return

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
            text="Запустить задачу",
            fg_color=AppTheme.PRIMARY,
            hover_color=AppTheme.PRIMARY_HOVER
        )
        self._set_workflow_status(
            f"Шаг 3 из 3: {task.channel_summary()} · можно запускать вычисления",
            AppTheme.TEXT_ON_DARK
        )

    def _set_workflow_status(self, text, color):
        """Показать текущий этап в общей строке состояния журнала."""
        progress_manager = getattr(self, "progress_manager", None)
        if progress_manager is not None:
            progress_manager.set_status(text, color)

    def _set_point_analysis_controls_enabled(self, enabled):
        """Enable point stages when either independent 1D branch exists."""
        state = "normal" if enabled else "disabled"
        for widget in (
                self.max_checkbox, self.min_checkbox, self.entry_near_point,
        ):
            if widget is not None:
                widget.configure(state=state)

        if self.extremes_hint_label is not None:
            self.extremes_hint_label.configure(
                text=(("Для каждого выбранного CWT экстремумы ищутся "
                       "по строкам и по столбцам.")
                      if enabled else
                      "Недоступно: включите CWT по строкам и/или столбцам."),
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
            loaded = self.image_processor.load_image_for_task(self.current_task, self)
            # wait_window runs events, including closing the main application.
            if self._is_destroyed:
                return
            if loaded:
                self._update_ui_for_current_task()
                self._update_tasks_display()
        except Exception as e:
            if self._is_destroyed:
                return
            self.progress_manager.log_error(f"Ошибка при загрузке изображения: {e}")
            mb.showerror("Ошибка", f"Не удалось загрузить изображение: {e}", parent=self)

    def pipette_channel(self):
        """Обработчик выбора пипетки для текущей задачи"""
        if not self.current_task:
            mb.showwarning("Внимание", "Сначала создайте задачу")
            return

        if not self.current_task.image_path:
            mb.showwarning("Внимание", "Сначала загрузите изображение")
            return

        try:
            selected = self.image_processor.pipette_channel_for_task(self.current_task, self)
            if self._is_destroyed:
                return
            if selected:
                self._update_ui_for_current_task()
                self._update_tasks_display()
        except Exception as error:
            if self._is_destroyed:
                return
            self.progress_manager.log_error(f'Ошибка пипетки: {error}')
            mb.showerror('Ошибка', f'Не удалось открыть пипетку: {error}', parent=self)

    def gramm_shmidt_transform(self):
        """Выбрать Gram–Schmidt как представление текущего исследования."""
        if not self.current_task:
            mb.showwarning("Внимание", "Сначала создайте задачу")
            return
        if (getattr(self.current_task, "color1", None) is None or
                getattr(self.current_task, "color2", None) is None):
            mb.showwarning("Внимание", "Сначала выберите цвета пипеткой")
            return
        try:
            self.image_processor.gram_shmidt_transform_for_task(self.current_task)
        except ValueError as error:
            self.progress_manager.log_error(str(error))
            mb.showwarning("Gram–Schmidt", str(error))
            return
        self.channel_representation_var.set("Gram–Schmidt")
        self._update_channel_controls_state()
        self._update_tasks_display()
        self._update_ml_tab_state()
        self._update_action_availability()

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
        """Explicit input enables KNN; restoring task settings never does."""
        task = self.current_task
        if (task is None or self._loading_task_settings or task.analysis_mode != '1d'
                or self.entry_near_point.cget('state') == 'disabled'):
            return
        if not self.calculate_knn_var.get():
            self.calculate_knn_var.set(True)
            self._on_pipeline_controls_changed()
            self._set_workflow_status('KNN включён: будут рассчитаны необходимые предыдущие этапы', AppTheme.TEXT_ON_DARK)

    def update_knn_for_current_task(self, *args):
        """Обновление KNN для текущей задачи при изменении поля"""
        if self._loading_task_settings:
            return
        if self.current_task and self.knn_text_var.get().isdigit():
            self.current_task.k_neighbors = int(self.knn_text_var.get())
            self._update_tasks_display()
            pending = getattr(self, '_knn_memory_job', None)
            if pending:
                self.after_cancel_safe(pending)
            self._knn_memory_job = self.after_safe(250, self._refresh_memory_estimate)

    def _store_settings_for_current_task(self, *args):
        """Сохранить значения элементов управления в активной задаче."""
        task = getattr(self, 'current_task', None)
        if task is None or self._loading_task_settings:
            return

        task.channel_representation = {
            "RGB": "rgb",
            "Grayscale": "grayscale",
            "Gram–Schmidt": "gram_schmidt",
        }.get(self.channel_representation_var.get(), "rgb")
        task.rgb_channel_mode = (
            "single" if self.rgb_channel_mode_var.get() == "Один канал" else "all"
        )
        task.rgb_single_channel = self.rgb_single_channel_var.get()
        task.gram_schmidt_applied = task.channel_representation == "gram_schmidt"

        task.process_rows = bool(self.row_var.get())
        task.process_columns = bool(self.col_var.get())
        task.find_maxima = bool(self.max_var.get())
        task.find_minima = bool(self.min_var.get())
        task.calculate_extrema = bool(self.calculate_extrema_var.get())
        task.calculate_envelopes = bool(self.calculate_envelopes_var.get())
        task.calculate_knn = bool(self.calculate_knn_var.get())
        task.output_wavelet_image = bool(self.wp_var1.get())
        task.output_wavelet_text = bool(self.wp_var2.get())
        task.output_wavelet_numpy = False
        task.output_extremes_text = bool(self.p_ex_var1.get())
        task.output_extremes_image = bool(self.p_ex_var2.get())
        task.output_envelopes_text = bool(self.envelope_text_var.get())
        task.output_envelopes_image = bool(self.envelope_image_var.get())
        task.output_knn_text = bool(self.knn_bool_text_var.get())
        task.output_knn_image = bool(self.knn_bool_image_var.get())
        task.save_source_channels = bool(self.print_channels_txt_var.get())
        task.save_centering_means = bool(self.centering_means_var.get())
        task.strict_backend = bool(self.strict_backend_var.get())
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
        """Управление готовым сценарием и ручными флажками этапов."""
        top = ctk.CTkFrame(self.pipeline_section.content, fg_color="transparent")
        self.pipeline_section.add_widget(top, pady=(0, 8))
        ctk.CTkLabel(
            top, text="Готовый сценарий:",
            font=AppTheme.body_font(), anchor="w"
        ).pack(side="left", padx=(0, 12))

        preset_values = ["Пользовательский", *self.current_pipeline_preset_names()]
        self.pipeline_preset_selector = ctk.CTkOptionMenu(
            top,
            values=preset_values,
            variable=self.pipeline_preset_var,
            command=self.on_pipeline_preset_changed,
            width=260
        )
        self.pipeline_preset_selector.pack(side="left")
        self.pipeline_preset_tooltip = HoverTooltip(
            self.pipeline_preset_selector,
            text_provider=self._current_pipeline_tooltip_text,
            delay_ms=450,
            wraplength=390,
        )

        stages = ctk.CTkFrame(self.pipeline_section.content, fg_color="transparent")
        self.pipeline_section.add_widget(stages, pady=(4, 2))
        stages.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)

        self.wavelet_required_var = tk.BooleanVar(value=True)
        self.wavelet_required_checkbox = ctk.CTkCheckBox(
            stages, text="Вейвлеты", variable=self.wavelet_required_var,
            state="disabled"
        )
        self.wavelet_required_checkbox.grid(row=0, column=0, sticky="w", padx=(0, 10))

        self.calculate_extrema_switch = ctk.CTkCheckBox(
            stages,
            text="Экстремумы",
            variable=self.calculate_extrema_var,
            command=lambda: self._on_pipeline_controls_changed("extrema")
        )
        self.calculate_extrema_switch.grid(row=0, column=1, sticky="w", padx=10)

        self.calculate_envelopes_switch = ctk.CTkCheckBox(
            stages,
            text="Огибающие",
            variable=self.calculate_envelopes_var,
            command=lambda: self._on_pipeline_controls_changed("envelopes")
        )
        self.calculate_envelopes_switch.grid(row=0, column=2, sticky="w", padx=10)

        self.calculate_knn_switch = ctk.CTkCheckBox(
            stages,
            text="KNN и углы",
            variable=self.calculate_knn_var,
            command=lambda: self._on_pipeline_controls_changed("knn")
        )
        self.calculate_knn_switch.grid(row=0, column=3, sticky="w", padx=10)

        self.calculate_statistics_switch = ctk.CTkCheckBox(
            stages,
            text="Статистики",
            variable=self.calculate_statistics_var,
            command=lambda: self._on_pipeline_controls_changed("statistics")
        )
        self.calculate_statistics_switch.grid(row=0, column=4, sticky="w", padx=(10, 0))

        self.pipeline_stage_status_label = ctk.CTkLabel(
            self.pipeline_section.content, text="", anchor="w", justify="left",
            font=AppTheme.body_font(),
        )
        self.pipeline_section.add_widget(self.pipeline_stage_status_label, pady=(6, 1))
        self.pipeline_execution_plan_label = ctk.CTkLabel(
            self.pipeline_section.content, text="", anchor="w", justify="left",
            font=AppTheme.body_font(),
        )
        self.pipeline_section.add_widget(self.pipeline_execution_plan_label, pady=(0, 4))

        # Синхронизация пока остаётся частью вычислительной модели, но не входит
        # в набор пользовательских сценариев этой страницы.
        self.calculate_sync_switch = None
        self.pipeline_chain_label = None
        self.pipeline_dependency_label = None
        self.after_idle(self._update_pipeline_controls_state)

    def _current_pipeline_tooltip_text(self):
        preset = self.pipeline_preset_var.get() or "Пользовательский"
        return ProcessingTask.PIPELINE_DESCRIPTIONS.get(
            preset,
            ProcessingTask.PIPELINE_DESCRIPTIONS["Пользовательский"]
        )

    def current_pipeline_preset_names(self):
        return list(ProcessingTask.PIPELINE_PRESETS.keys())

    def on_pipeline_preset_changed(self, preset_name):
        """Применить пресет к активной задаче без изменения форматов файлов."""
        if self.current_task is None:
            self.pipeline_preset_var.set("Пользовательский")
            return
        self.current_task.apply_pipeline_preset(preset_name)
        self._update_ui_for_current_task()

    def _on_pipeline_controls_changed(self, changed_stage=None):
        """Обработать ручное изменение этапов с учётом зависимостей.

        Граф зависимостей 1-D pipeline:

            Вейвлеты (всегда)
                ↓
            Экстремумы
                ↓
            Огибающие
             ↙       ↘
           KNN     Статистики

        Правила:
        - включение нижележащего этапа автоматически включает prerequisites;
        - выключение prerequisite каскадно выключает зависимые этапы;
        - после нормализации комбинации автоматически подбирается готовый
          сценарий, либо «Пользовательский».
        """
        if self._loading_task_settings:
            return

        task = getattr(self, "current_task", None)
        if task is None:
            return

        extrema = bool(self.calculate_extrema_var.get())
        envelopes = bool(self.calculate_envelopes_var.get())
        knn = bool(self.calculate_knn_var.get())
        statistics = bool(self.calculate_statistics_var.get())

        status_message = None

        # -------------------------------------------------------------
        # Включение этапа: автоматически включаем все необходимые этапы
        # перед ним. Невалидная цепочка в интерфейсе не допускается.
        # -------------------------------------------------------------
        if changed_stage == "envelopes" and envelopes:
            if not extrema:
                extrema = True
                status_message = "Для огибающих автоматически включены экстремумы."

        elif changed_stage == "knn" and knn:
            added = []
            if not extrema:
                extrema = True
                added.append("экстремумы")
            if not envelopes:
                envelopes = True
                added.append("огибающие")
            if added:
                status_message = (
                    "Для KNN автоматически включены: " + ", ".join(added) + "."
                )

        elif changed_stage == "statistics" and statistics:
            added = []
            if not extrema:
                extrema = True
                added.append("экстремумы")
            if not envelopes:
                envelopes = True
                added.append("огибающие")
            if added:
                status_message = (
                    "Для статистик автоматически включены: "
                    + ", ".join(added)
                    + "."
                )

        # -------------------------------------------------------------
        # Выключение prerequisite: зависимые этапы выключаются каскадно.
        # Это безопаснее, чем оставлять KNN/статистики на данных другого
        # смысла или незаметно возвращать снятый пользователем флажок.
        # -------------------------------------------------------------
        if changed_stage == "extrema" and not extrema:
            disabled = []
            if envelopes:
                envelopes = False
                disabled.append("огибающие")
            if knn:
                knn = False
                disabled.append("KNN")
            if statistics:
                statistics = False
                disabled.append("статистики")
            if disabled:
                status_message = (
                    "Экстремумы отключены → также отключены: "
                    + ", ".join(disabled)
                    + "."
                )

        elif changed_stage == "envelopes" and not envelopes:
            disabled = []
            if knn:
                knn = False
                disabled.append("KNN")
            if statistics:
                statistics = False
                disabled.append("статистики")
            if disabled:
                status_message = (
                    "Огибающие отключены → также отключены: "
                    + ", ".join(disabled)
                    + "."
                )

        # Финальная страховка: комбинация всегда должна удовлетворять DAG.
        if knn or statistics:
            envelopes = True
            extrema = True
        elif envelopes:
            extrema = True

        self.calculate_extrema_var.set(extrema)
        self.calculate_envelopes_var.set(envelopes)
        self.calculate_knn_var.set(knn)
        self.calculate_statistics_var.set(statistics)

        # На этой странице синхронизация пока не является этапом сценария.
        self.calculate_sync_var.set(False)

        self._store_settings_for_current_task()
        task.calculate_synchronization = False

        matched = task.match_pipeline_preset()
        task.pipeline_preset = matched
        self.pipeline_preset_var.set(matched)

        if status_message:
            self._set_workflow_status(status_message, AppTheme.TEXT_ON_DARK)

        self._update_pipeline_controls_state()

    def _update_pipeline_controls_state(self):
        """Синхронизировать доступность этапов, настроек и экспортов."""
        task = getattr(self, "current_task", None)
        stage_widgets = (
            self.calculate_extrema_switch,
            self.calculate_envelopes_switch,
            self.calculate_knn_switch,
            self.calculate_statistics_switch,
        )
        if task is None:
            if self.pipeline_preset_selector is not None:
                self.pipeline_preset_selector.configure(state="disabled")
            for widget in stage_widgets:
                if widget is not None:
                    widget.configure(state="disabled")
            return

        is_1d = task.analysis_mode == "1d"
        if self.pipeline_preset_selector is not None:
            self.pipeline_preset_selector.configure(state="normal" if is_1d else "disabled")

        direction_available = is_1d and bool(
            task.process_rows or task.process_columns
        )
        stage_state = "normal" if direction_available else "disabled"
        for widget in stage_widgets:
            if widget is not None:
                widget.configure(state=stage_state)

        plan = task.resolve_pipeline()
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
            self.entry_near_point.configure(state="normal" if plan.knn else "disabled")

        if getattr(self, "pipeline_stage_status_label", None) is not None:
            try:
                self.pipeline_stage_status_label.configure(text=format_stage_status(task))
                self.pipeline_execution_plan_label.configure(text=format_execution_plan(task))
            except Exception:
                self.pipeline_stage_status_label.configure(text="Состояние этапов: ещё не рассчитано")
                self.pipeline_execution_plan_label.configure(text="")

        self._update_statistics_controls_state()
        self._update_export_rows_visibility()

    def _setup_stage_settings_section(self):
        """Карточки параметров этапов, которые требуют дополнительных настроек."""
        holder = ctk.CTkFrame(self.stage_settings_section.content, fg_color="transparent")
        self.stage_settings_section.add_widget(holder, pady=0)
        holder.grid_columnconfigure((0, 1), weight=1, uniform="stage_settings")

        self.knn_settings_card = ctk.CTkFrame(holder)
        self.knn_settings_card.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        ctk.CTkLabel(
            self.knn_settings_card, text="KNN и углы",
            font=AppTheme.section_title_font(), anchor="w"
        ).pack(fill="x", padx=12, pady=(10, 6))
        row = ctk.CTkFrame(self.knn_settings_card, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(0, 12))
        ctk.CTkLabel(row, text="Количество ближайших точек:", anchor="w").pack(side="left")
        self.entry_near_point = ctk.CTkEntry(
            row, textvariable=self.knn_text_var, placeholder_text="5", width=130
        )
        self.entry_near_point.pack(side="right")
        self.entry_near_point.bind("<Button-1>", self.on_entry_click)
        self.entry_near_point.bind("<KeyPress>", self.on_entry_click)
        self.entry_near_point.bind("<<Paste>>", self.on_entry_click)

        self.statistics_settings_card = ctk.CTkFrame(holder)
        self.statistics_settings_card.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        ctk.CTkLabel(
            self.statistics_settings_card, text="Статистики",
            font=AppTheme.section_title_font(), anchor="w"
        ).pack(fill="x", padx=12, pady=(10, 4))
        ctk.CTkLabel(
            self.statistics_settings_card,
            text="Интеграция и детальная настройка статистик будет расширена позже.",
            font=AppTheme.caption_font(), text_color=AppTheme.TEXT_SECONDARY,
            anchor="w", justify="left", wraplength=430
        ).pack(fill="x", padx=12, pady=(0, 6))
        stats_row = ctk.CTkFrame(self.statistics_settings_card, fg_color="transparent")
        stats_row.pack(fill="x", padx=12, pady=(0, 12))
        ctk.CTkLabel(stats_row, text="Размер блока масштабов:", anchor="w").pack(side="left")
        self.scale_block_sizes_entry = ctk.CTkEntry(
            stats_row, textvariable=self.scale_block_sizes_var,
            placeholder_text="5", width=130
        )
        self.scale_block_sizes_entry.pack(side="right")
        self.statistics_parameter_widgets = [self.scale_block_sizes_entry]
        self.synchronization_parameter_widgets = []
        self.synchronization_output_widgets = []

    def _setup_export_results_section(self):
        """Единая таблица экспорта: текстовый файл и PNG."""
        table = ctk.CTkFrame(self.export_section.content, fg_color="transparent")
        self.export_section.add_widget(table, pady=0)
        table.grid_columnconfigure(0, weight=2)
        table.grid_columnconfigure((1, 2), weight=1, uniform="export_formats")

        headers = ("Результат", "Текстовый файл", "PNG")
        for column, title in enumerate(headers):
            ctk.CTkLabel(
                table, text=title, font=AppTheme.section_title_font(),
                anchor="w" if column == 0 else "center"
            ).grid(row=0, column=column, sticky="ew", padx=8, pady=(4, 7))

        rows = [
            ("Вейвлеты", self.wp_var2, self.wp_var1, "wavelet"),
            ("Экстремумы", self.p_ex_var1, self.p_ex_var2, "extrema"),
            ("Огибающие", self.envelope_text_var, self.envelope_image_var, "envelopes"),
            ("KNN и углы", self.knn_bool_text_var, self.knn_bool_image_var, "knn"),
            ("Статистики", self.statistics_csv_var, self.statistics_image_var, "statistics"),
        ]
        self.extremes_output_widgets = []
        self.envelopes_output_widgets = []
        self.knn_output_widgets = []
        self.statistics_output_widgets = []
        self.export_result_rows = {}

        for row_index, (label, text_var, png_var, kind) in enumerate(rows, start=1):
            label_widget = ctk.CTkLabel(table, text=label, anchor="w")
            label_widget.grid(
                row=row_index, column=0, sticky="ew", padx=8, pady=3
            )

            text_cb = ctk.CTkCheckBox(
                table, text="", variable=text_var, width=24
            )
            text_cb.grid(row=row_index, column=1, pady=3)

            png_cb = ctk.CTkCheckBox(
                table, text="", variable=png_var, width=24
            )
            png_cb.grid(row=row_index, column=2, pady=3)

            self.export_result_rows[kind] = (
                label_widget,
                text_cb,
                png_cb,
            )

            if kind == "extrema":
                self.extremes_output_widgets.extend([text_cb, png_cb])
            elif kind == "envelopes":
                self.envelopes_output_widgets.extend([text_cb, png_cb])
            elif kind == "knn":
                self.knn_output_widgets.extend([text_cb, png_cb])
            elif kind == "statistics":
                self.statistics_output_widgets.extend([text_cb, png_cb])

        # Вейвлеты рассчитываются всегда, поэтому их строка всегда видима.
        # Остальные строки покажет _update_export_rows_visibility().
        self._update_export_rows_visibility()

        # NPY больше не является пользовательским форматом экспорта.
        self.wavelet_numpy_var.set(False)

    def _update_export_rows_visibility(self):
        """Показывать в таблице экспорта только реально вычисляемые этапы."""
        rows = getattr(self, "export_result_rows", None)
        if not rows:
            return

        task = getattr(self, "current_task", None)

        visibility = {
            "wavelet": True,
            "extrema": False,
            "envelopes": False,
            "knn": False,
            "statistics": False,
        }

        if task is not None:
            plan = task.resolve_pipeline()
            visibility.update({
                "extrema": bool(plan.extrema),
                "envelopes": bool(plan.envelopes),
                "knn": bool(plan.knn),
                "statistics": bool(plan.statistics),
            })

        for kind, widgets in rows.items():
            visible = visibility.get(kind, False)
            for widget in widgets:
                if visible:
                    widget.grid()
                else:
                    widget.grid_remove()

    def _setup_wavelet_section(self):
        """Настройка секции вейвлет-преобразования"""
        info_label = ctk.CTkLabel(
            self.wavelet_section.content,
            text="Вейвлет-преобразование выполняется всегда. Форматы экспорта:",
            font=AppTheme.body_font(),
            anchor="w",
            wraplength=0
        )
        self.wavelet_section.add_widget(info_label, pady=(0, 10))

        self.wp1_checkbox = ctk.CTkCheckBox(
            self.wavelet_section.content,
            text="Изображение PNG",
            variable=self.wp_var1
        )
        self.wavelet_section.add_widget(self.wp1_checkbox, fill="x")

        self.wp2_checkbox = ctk.CTkCheckBox(
            self.wavelet_section.content,
            text="Данные TXT (рекомендуется для визуализатора)",
            variable=self.wp_var2
        )
        self.wavelet_section.add_widget(self.wp2_checkbox, fill="x")

        self.wavelet_numpy_checkbox = ctk.CTkCheckBox(
            self.wavelet_section.content,
            text="Экспорт исходных коэффициентов NPY (с точностью расчёта)",
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
            text="Изображение PNG",
            variable=self.p_ex_var2
        )
        self.output_extremes_section.add_widget(self.p_ex2_checkbox, fill="x")
        self.p_ex1_checkbox = ctk.CTkCheckBox(
            self.output_extremes_section.content,
            text="Данные TXT",
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
            text="Изображение PNG",
            variable=self.envelope_image_var
        )
        self.output_extremes_section.add_widget(
            self.envelope_image_checkbox, fill="x"
        )
        self.envelope_text_checkbox = ctk.CTkCheckBox(
            self.output_extremes_section.content,
            text="Данные TXT",
            variable=self.envelope_text_var
        )
        self.output_extremes_section.add_widget(
            self.envelope_text_checkbox, fill="x"
        )
        self.envelopes_output_widgets = [
            self.envelope_image_checkbox, self.envelope_text_checkbox
        ]

    def _setup_statistics_section(self):
        """Параметры и экспорт статистик/синхронизаций без дублирования этапов."""
        ctk.CTkLabel(
            self.statistics_section.content, text="Статистики экстремумов",
            font=AppTheme.section_title_font(), anchor="w"
        ).pack(fill="x", padx=5, pady=(0, 6))

        block_size_label = ctk.CTkLabel(
            self.statistics_section.content,
            text="Размеры блоков масштабов (через запятую)",
            font=AppTheme.caption_font(), anchor="w"
        )
        self.statistics_section.add_widget(block_size_label, pady=(2, 2))
        self.scale_block_sizes_entry = ctk.CTkEntry(
            self.statistics_section.content, textvariable=self.scale_block_sizes_var,
            placeholder_text="5"
        )
        self.statistics_section.add_widget(self.scale_block_sizes_entry, pady=(0, 6))
        self.statistics_parameter_widgets = [self.scale_block_sizes_entry]

        statistics_csv = ctk.CTkCheckBox(
            self.statistics_section.content, text="Данные CSV",
            variable=self.statistics_csv_var
        )
        self.statistics_section.add_widget(statistics_csv)
        statistics_image = ctk.CTkCheckBox(
            self.statistics_section.content, text="Гистограммы PNG",
            variable=self.statistics_image_var
        )
        self.statistics_section.add_widget(statistics_image, pady=(2, 10))
        self.statistics_output_widgets = [statistics_csv, statistics_image]

        ctk.CTkLabel(
            self.statistics_section.content, text="Межстрочная синхронизация",
            font=AppTheme.section_title_font(), anchor="w"
        ).pack(fill="x", padx=5, pady=(8, 6))

        sync_stride_label = ctk.CTkLabel(
            self.statistics_section.content, text="Шаг выбора строк",
            font=AppTheme.caption_font(), anchor="w"
        )
        self.statistics_section.add_widget(sync_stride_label, pady=(2, 2))
        self.sync_stride_entry = ctk.CTkEntry(
            self.statistics_section.content, textvariable=self.sync_stride_var,
            placeholder_text="1"
        )
        self.statistics_section.add_widget(self.sync_stride_entry, pady=(0, 4))

        sync_tolerance_label = ctk.CTkLabel(
            self.statistics_section.content, text="Допустимое смещение по X, пиксели",
            font=AppTheme.caption_font(), anchor="w"
        )
        self.statistics_section.add_widget(sync_tolerance_label, pady=(2, 2))
        self.sync_tolerance_entry = ctk.CTkEntry(
            self.statistics_section.content, textvariable=self.sync_tolerance_var,
            placeholder_text="1"
        )
        self.statistics_section.add_widget(self.sync_tolerance_entry, pady=(0, 4))

        sync_metric_label = ctk.CTkLabel(
            self.statistics_section.content, text="Метрика синхронизации",
            font=AppTheme.caption_font(), anchor="w"
        )
        self.statistics_section.add_widget(sync_metric_label, pady=(2, 2))
        self.sync_metric_selector = ctk.CTkOptionMenu(
            self.statistics_section.content, values=["jaccard", "dice", "phi"],
            variable=self.sync_metric_var, width=180
        )
        self.statistics_section.add_widget(
            self.sync_metric_selector, fill="none", anchor="w", pady=(0, 6)
        )
        self.synchronization_parameter_widgets = [
            self.sync_stride_entry, self.sync_tolerance_entry, self.sync_metric_selector
        ]

        sync_matrix = ctk.CTkCheckBox(
            self.statistics_section.content, text="Матрица синхронизаций CSV",
            variable=self.sync_matrix_csv_var
        )
        self.statistics_section.add_widget(sync_matrix)
        sync_pairs = ctk.CTkCheckBox(
            self.statistics_section.content, text="Метрики пар строк CSV",
            variable=self.sync_pairs_csv_var
        )
        self.statistics_section.add_widget(sync_pairs)
        sync_heatmap = ctk.CTkCheckBox(
            self.statistics_section.content, text="Heatmap PNG",
            variable=self.sync_heatmap_var
        )
        self.statistics_section.add_widget(sync_heatmap)
        self.synchronization_output_widgets = [sync_matrix, sync_pairs, sync_heatmap]

        ctk.CTkLabel(
            self.statistics_section.content,
            text=("Синхронизации используют максимумы верхней огибающей "
                  "построчного 1D-преобразования."),
            font=AppTheme.caption_font(), text_color=AppTheme.TEXT_SECONDARY,
            anchor="w", justify="left", wraplength=500
        ).pack(fill="x", padx=5, pady=(8, 0))
        self._update_statistics_controls_state()

    def _on_statistics_stage_changed(self):
        self._on_pipeline_controls_changed()

    def _update_statistics_controls_state(self):
        """Блокировать настройки/экспорт статистик, пока этап не выбран."""
        task = getattr(self, "current_task", None)
        point_analysis_available = bool(
            task is not None
            and task.analysis_mode == "1d"
            and (task.process_rows or task.process_columns)
        )
        plan = task.resolve_pipeline() if task is not None else None
        statistics_enabled = bool(
            point_analysis_available and plan and plan.statistics
        )
        state = "normal" if statistics_enabled else "disabled"
        for widget in self.statistics_output_widgets:
            widget.configure(state=state)
        for widget in self.statistics_parameter_widgets:
            widget.configure(state=state)

    def _setup_knn_section(self):
        """Параметры KNN и форматы экспорта."""
        ctk.CTkLabel(
            self.knn_section.content, text="Количество ближайших точек:",
            font=AppTheme.body_font(), anchor="w"
        ).pack(fill="x", padx=5, pady=(0, 4))
        self.entry_near_point = ctk.CTkEntry(
            self.knn_section.content, textvariable=self.knn_text_var,
            placeholder_text="5"
        )
        self.knn_section.add_widget(self.entry_near_point, pady=(0, 10))
        self.entry_near_point.bind("<Button-1>", self.on_entry_click)
        self.entry_near_point.bind("<KeyPress>", self.on_entry_click)
        self.entry_near_point.bind("<<Paste>>", self.on_entry_click)

        ctk.CTkLabel(
            self.knn_section.content, text="Экспорт результатов KNN и углов:",
            font=AppTheme.body_font(), anchor="w"
        ).pack(fill="x", padx=5, pady=(0, 6))
        self.knn_text_checkbox = ctk.CTkCheckBox(
            self.knn_section.content, text="Данные TXT",
            variable=self.knn_bool_text_var
        )
        self.knn_section.add_widget(self.knn_text_checkbox, fill="x")
        self.knn_image_checkbox = ctk.CTkCheckBox(
            self.knn_section.content, text="Изображение PNG",
            variable=self.knn_bool_image_var
        )
        self.knn_section.add_widget(self.knn_image_checkbox, fill="x")
        self.knn_output_widgets = [self.knn_text_checkbox, self.knn_image_checkbox]

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
        self.memory_label = ctk.CTkLabel(self.compute_section, text='Оценка памяти появится после загрузки данных',
                                         anchor='w', justify='left', wraplength=650)
        self.memory_label.grid(row=1, column=0, sticky='ew', padx=12)

        self.app_start_button = ctk.CTkButton(
            self.compute_section,
            text="Запустить задачу",
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

    def _refresh_memory_estimate(self):
        try:
            self.memory_label.configure(text=estimate_label([self.current_task] if self.current_task else []))
        except (ValueError, TypeError, OverflowError) as error:
            self.memory_label.configure(text=f'Для оценки памяти проверьте параметры: {error}')

    def _request_cancel(self):
        compute_running = self._compute_thread is not None and self._compute_thread.is_alive()
        ml_running = self._ml_thread is not None and self._ml_thread.is_alive()
        if not compute_running and not ml_running:
            return
        self.progress_manager.run_control.cancel()
        self.cancel_compute_button.configure(state='disabled', text='Ожидание остановки…')
        self.progress_manager.progress_label.configure(
            text='Отмена запрошена: ожидаем завершения текущей операции…'
        )

    def safe_destroy(self):
        running = lambda: any(t is not None and t.is_alive() for t in
                              (getattr(self, '_compute_thread', None), getattr(self, '_ml_thread', None)))
        if running():
            if not getattr(self, '_waiting_to_close', False):
                self._waiting_to_close = True
                if ((self._compute_thread is not None and self._compute_thread.is_alive())
                        or (self._ml_thread is not None and self._ml_thread.is_alive())):
                    self._request_cancel()
                def finish_close():
                    if running():
                        self.after_safe(100, finish_close)
                    else:
                        super(App, self).safe_destroy()
                self.after_safe(100, finish_close)
            return
        super().safe_destroy()

    def safe_compute(self):
        """Безопасный запуск вычислений в отдельном потоке"""
        # Инициализируем _compute_thread если он None
        if self._compute_thread is None:
            self._compute_thread = threading.Thread()

        if self._compute_thread.is_alive():
            mb.showwarning("Внимание", "Вычисления уже выполняются")
            return

        if self._ml_thread is not None and self._ml_thread.is_alive():
            mb.showwarning('Выполнение', 'Дождитесь завершения отдельного ML-расчёта')
            return

        task = self.current_task
        if task is None or task not in self.image_processor.tasks:
            mb.showwarning('Выполнение', 'Выберите задачу для расчёта')
            return
        self._store_settings_for_current_task()
        self._refresh_memory_estimate()
        self.progress_manager.run_control.reset()
        self.cancel_compute_button.configure(state='normal', text='Отменить расчёт')

        # Блокируем UI на время вычислений
        self._disable_ui_during_compute(True)

        self._compute_thread = threading.Thread(target=self._compute_wrapper, args=((task,),))
        self._compute_thread.daemon = True
        self._compute_thread.start()

    def _compute_wrapper(self, tasks=None):
        """Обертка для безопасного выполнения в потоке"""
        try:
            self.compute(tasks)
        except RunCancelled:
            self.progress_manager.log_info('Расчёт отменён пользователем; следующие задачи не запущены')
            self.progress_manager.cancel_run()
            self.after_safe(0, self._refresh_history_tab)
        except Exception as e:
            error_msg = f"Ошибка вычислений: {str(e)}\n{traceback.format_exc()}"
            self.progress_manager.log_error(error_msg)
            self.progress_manager.fail_run(f"Вычисления остановлены: {str(e)}")
        finally:
            try:
                self.image_processor.clear_gpu_memory()
            except Exception as error:
                self.progress_manager.log_error(f'Очистка GPU: {error}')
            self.after_safe(0, lambda: self.cancel_compute_button.configure(state='disabled', text='Отменить расчёт'))
            try:
                self.after_safe(0, lambda: self._disable_ui_during_compute(False))
            except Exception as e:
                self.progress_manager.log_error(f"Ошибка при отключении UI: {str(e)}")

    def _disable_ui_during_compute(self, disable: bool):
        """Блокировка/разблокировка UI во время вычислений"""
        self._compute_ui_locked = bool(disable)
        if disable:
            self._locked_controls = []
            self._lock_controls_in(self.workspace_frame)
            self._lock_controls_in(self.tasks_panel)
        state = "disabled" if disable else "normal"

        widgets_to_disable = [
            self.load_button,
            self.pipette_button,
            self.gram_shmidt_button,
            self.button_save_scales,
            self.button_load_scales_file,
            self.app_start_button,
            self.add_task_btn,
            self.analysis_mode_selector,
            getattr(self, "cpu_radio", None),
            getattr(self, "gpu_radio", None),
        ]

        for widget in widgets_to_disable:
            if widget is None:
                continue
            try:
                # GPU может быть принципиально недоступен. После расчёта
                # не делаем его активным только ради общего unlock.
                if (
                    not disable
                    and widget is getattr(self, "gpu_radio", None)
                    and hasattr(self, "image_processor")
                ):
                    info = self.image_processor.get_backend_info()
                    gpu_enabled = bool(
                        info.get("gpu_available", False)
                        and info.get("gpu_checked", False)
                    )
                    widget.configure(
                        state="normal" if gpu_enabled else "disabled"
                    )
                else:
                    widget.configure(state=state)
            except Exception as e:
                self.progress_manager.log_error(
                    f"Не удалось изменить состояние элемента интерфейса: {e}"
                )

        try:
            self.workspace_tabs.set_enabled(True)
        except (tk.TclError, RuntimeError) as error:
            self.progress_manager.log_error(
                f"Не удалось обновить навигацию после расчёта: {error}"
            )

        if not disable:
            # Состояния, сохранённые ДО расчёта, уже могут быть неактуальны.
            # В частности, ML-поля могли быть disabled лишь потому, что KNN на
            # тот момент ещё не существовал. Ошибка одного старого Tk-виджета
            # также не должна обрывать всю процедуру разблокировки.
            for widget, old_state in getattr(self, '_locked_controls', []):
                try:
                    if widget.winfo_exists():
                        widget.configure(state=old_state)
                except (tk.TclError, RuntimeError) as error:
                    self.progress_manager.log_error(
                        f"Пропущен устаревший элемент при разблокировке UI: {error}"
                    )
            self._locked_controls = []

            try:
                if hasattr(self, 'results_panel'):
                    for pane in (self.results_panel.left, self.results_panel.right):
                        pane.refresh_control_states()
                    self.results_panel.update_colors()
            except (tk.TclError, RuntimeError) as error:
                self.progress_manager.log_error(
                    f"Не удалось обновить панель результатов после расчёта: {error}"
                )

            try:
                self._update_gpu_section()
            except (tk.TclError, RuntimeError) as error:
                self.progress_manager.log_error(
                    f"Не удалось обновить секцию устройства после расчёта: {error}"
                )

            try:
                self._apply_analysis_mode_ui()
                self._update_action_availability()
            except (tk.TclError, RuntimeError) as error:
                self.progress_manager.log_error(
                    f"Не удалось полностью обновить основной UI после расчёта: {error}"
                )
            finally:
                # Главное правило: готовый KNN + нет реально работающего worker =
                # параметры ML активны. Не восстанавливаем их из old_state.
                if hasattr(self, "ml_panel"):
                    try:
                        self._update_ml_tab_state()
                        self.after_safe(0, self._refresh_ml_controls_state)
                    except (tk.TclError, RuntimeError) as error:
                        self.progress_manager.log_error(
                            f"Не удалось обновить состояние ML-вкладки: {error}"
                        )

    def _lock_controls_in(self, parent):
        if not hasattr(self, '_locked_controls'):
            self._locked_controls = []
        locked = {widget for widget, _state in self._locked_controls}
        stack = [parent]
        control_types = (ctk.CTkButton, ctk.CTkCheckBox, ctk.CTkEntry, ctk.CTkComboBox,
                         ctk.CTkOptionMenu, ctk.CTkSwitch, ctk.CTkSlider)
        while stack:
            widget = stack.pop()
            if widget is getattr(self, 'results_panel', None):
                continue
            stack.extend(widget.winfo_children())
            if (widget not in locked and isinstance(widget, control_types)
                    and not getattr(widget, '_read_only_during_compute', False)):
                self._locked_controls.append((widget, widget.cget('state')))
                widget.configure(state='disabled')

    def _run_integrated_ml(self, task):
        """Устаревшая точка входа: ML больше не является этапом анализа."""
        return None

    def _history_ml_summary(self, task):
        result = task.ml_result
        if not result:
            return ""
        metrics = result.get("metrics", {})
        if result.get("tiled"):
            stats = result.get("dbscan_statistics") or {}
            return (
                f"DBSCAN: A>0 — {stats.get('anomaly_nonzero', 0)}; "
                f"устойчивые — {stats.get('anomaly_stable', 0)}; "
                f"тайлов — {stats.get('tiles', 0)}"
            )
        return (
            f"Кластеров: {metrics.get('cluster_count', 0)}; "
            f"шумовых точек: {metrics.get('noise_count', 0)}"
        )

    def compute(self, tasks=None):
        """Run the captured selection; other tasks are never queued implicitly."""
        tasks = tuple(tasks) if tasks is not None else ((self.current_task,) if self.current_task else ())
        if not tasks:
            mb.showwarning("Внимание", "Нет задач для обработки")
            return

        # Validate only the tasks explicitly included in this run.
        for i, task in enumerate(tasks):
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
            if (task.channel_representation == "gram_schmidt"
                    and not task.has_colors_selected()):
                mb.showerror(
                    "Каналы анализа",
                    f"Задача {i + 1}: для Gram–Schmidt не задан цветовой базис"
                )
                return
            if task.analysis_mode == "1d" and not (
                    task.process_rows or task.process_columns):
                mb.showerror(
                    "Ошибка",
                    f"Задача {i + 1}: не выбран источник 1D CWT"
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
            total_tasks = len(tasks)
            current_task_num = 0
            execution_stages = []
            for planned_task in tasks:
                execution_stages.extend(
                    f"{planned_task.task_name} · {stage}"
                    for stage in planned_task.execution_stage_names()
                )
            self.progress_manager.begin_run(
                f"Исследование · {', '.join(task.task_name for task in tasks)}", execution_stages
            )

            for task in tasks:
                self.progress_manager.run_control.check()
                current_task_num += 1
                self.progress_manager.log_info(f"Обработка задачи {current_task_num}/{total_tasks}: {task.task_name}")

                task_started = time.time()
                snapshot = task.settings_snapshot()
                try:
                    # Продолжение той же задачи переиспользует её каталог,
                    # пока научные входы CWT не изменились. Изменение источника,
                    # масштабов, каналов или параметров Morlet создаёт новый каталог.
                    current_signature = cwt_signature(task)
                    resume_same_task = bool(
                        getattr(task, "reuse_existing_results", True)
                        and getattr(task, "last_completed_cwt_signature", None)
                        == current_signature
                        and getattr(task, "task_folder_path", "")
                        and os.path.isdir(task.task_folder_path)
                    )
                    if resume_same_task:
                        self.progress_manager.log_info(
                            f"Продолжение {task.task_name}: используются готовые "
                            "результаты предыдущих этапов"
                        )
                    else:
                        task.task_folder_path = ""
                        self.image_processor.create_task_folder(task)
                        save_run_image(task)
                    snapshot = task.settings_snapshot()
                    # На странице параметров расчёта выполняется только подготовка
                    # признаков. ML-алгоритмы запускаются исключительно на вкладке «ML».
                    self.image_processor.compute_for_task(task)
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
                    self.progress_manager.run_control.check()
                    self.run_history.add_run(
                        task=task,
                        status="completed",
                        duration_seconds=time.time() - task_started,
                        settings=snapshot,
                        ml_summary=self._history_ml_summary(task),
                    )
                    self.progress_manager.log_info(f'Результаты {task.task_name} сохранены')
                except RunCancelled as error:
                    self.run_history.add_run(task=task, status='cancelled',
                                             duration_seconds=time.time() - task_started,
                                             settings=snapshot, error_message=str(error))
                    self.progress_manager.save_run_log(task.task_folder_path, header='Запуск отменён; результаты частичные')
                    raise
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
            self.after_safe(0, self._update_pipeline_controls_state)
            self.after_safe(0, lambda: self.show_success_message(elapsed_time, total_tasks))

        except RunCancelled:
            raise
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
        self._show_workspace()
        self._ensure_workspace_tab("Результаты")
        self.results_panel.category.set('Все результаты')
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
            history_callback=self._show_history_page,
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
            app = App(show_splash=True)
            app.mainloop()
        except Exception as e:
            print(f"Critical error: {e}")
            import traceback
            traceback.print_exc()


    main()
