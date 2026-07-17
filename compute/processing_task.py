import numpy as np
import os


class ProcessingTask:
    VALID_ANALYSIS_MODES = {"1d", "2d"}

    def __init__(self):
        self.task_id = 0
        self.task_name = ""
        # Режим является частью задачи, а не глобальной настройкой интерфейса.
        # Текущий вычислительный конвейер реализует режим 1d; 2d зарезервирован
        # для полноценного двумерного преобразования Морле.
        self.analysis_mode = "1d"
        self.image_path = ""
        self.original_image = None
        self.data = []  # данные изображения
        self.data_copy = []  # копия данных для сравнения
        self.color1 = None
        self.color2 = None
        self.scales = np.array([])
        self.num_scale = 0

        # Параметры 1D-анализа
        self.process_rows = True
        self.process_columns = True
        self.find_maxima = True
        self.find_minima = True

        # Параметры 2D Morlet
        self.orientations = [0.0, 45.0, 90.0, 135.0]
        self.morlet_omega0 = 6.0
        self.morlet_anisotropy = 1.0

        # Форматы результатов
        self.output_wavelet_image = True
        self.output_wavelet_text = False
        self.output_extremes_text = False
        self.output_extremes_image = True
        self.output_knn_text = False
        self.output_knn_image = False
        self.save_source_channels = False

        self.k_neighbors = 5
        self.task_folder_path = ""
        self.result = {}

    def to_dict(self):
        """Преобразование задачи в словарь для отображения"""
        has_colors = (self.color1 is not None and
                      self.color2 is not None and
                      isinstance(self.color1, np.ndarray) and
                      isinstance(self.color2, np.ndarray) and
                      self.color1.size > 0 and
                      self.color2.size > 0)

        return {
            'task_id': self.task_id,
            'task_name': self.task_name,
            'analysis_mode': self.analysis_mode,
            'image_name': os.path.basename(self.image_path) if self.image_path else "Не загружено",
            'scales_count': self.num_scale,
            'k_neighbors': self.k_neighbors,
            'process_rows': self.process_rows,
            'process_columns': self.process_columns,
            'orientations': self.orientations.copy(),
            'colors_selected': has_colors,
        }

    def set_analysis_mode(self, mode):
        """Установить поддерживаемый режим анализа для задачи."""
        if mode not in self.VALID_ANALYSIS_MODES:
            raise ValueError(f"Неизвестный режим анализа: {mode}")
        self.analysis_mode = mode

    def get_image_dimensions(self):
        """Получить размеры изображения"""
        if self.original_image is not None:
            return self.original_image.shape[:2]
        return None

    def has_colors_selected(self):
        """Проверка, выбраны ли цвета пипеткой"""
        return (self.color1 is not None and self.color2 is not None and
                isinstance(self.color1, np.ndarray) and
                isinstance(self.color2, np.ndarray) and
                self.color1.size > 0 and self.color2.size > 0)

    def is_gram_schmidt_applied(self):
        """Проверка, применено ли преобразование Грамма-Шмидта"""
        if not self.data or not self.data_copy:
            return False
        if len(self.data) != len(self.data_copy):
            return False
        return any(not np.array_equal(self.data[i], self.data_copy[i])
                   for i in range(len(self.data)))
