import numpy as np
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class PipelinePlan:
    """Разрешённый план вычислений для одной задачи.

    Поля ``*_requested`` отражают выбор пользователя, а остальные поля —
    фактически необходимые этапы с учётом зависимостей.
    """

    mode: str
    extrema_requested: bool
    envelopes_requested: bool
    knn_requested: bool
    statistics_requested: bool
    synchronization_requested: bool
    ml_requested: bool
    extrema: bool
    envelopes: bool
    knn: bool
    statistics: bool
    synchronization: bool
    ml: bool
    maxima_required: bool

    @property
    def point_pipeline_required(self):
        return self.mode == "1d" and self.extrema

    def automatic_reasons(self):
        """Вернуть пояснения для этапов, включённых зависимостями."""
        reasons = {}
        downstream_envelopes = []
        if self.knn_requested:
            downstream_envelopes.append("KNN")
        if self.statistics_requested:
            downstream_envelopes.append("статистик")
        if self.synchronization_requested:
            downstream_envelopes.append("синхронизаций")
        if self.envelopes and not self.envelopes_requested:
            reasons["envelopes"] = "требуется для " + ", ".join(downstream_envelopes)

        downstream_extrema = []
        if self.envelopes:
            downstream_extrema.append("огибающих")
        if self.extrema and not self.extrema_requested:
            reasons["extrema"] = "требуется для " + ", ".join(downstream_extrema)
        return reasons


class ProcessingTask:
    VALID_ANALYSIS_MODES = {"1d", "2d"}
    PIPELINE_PRESETS = {
        "Только вейвлет": {
            "calculate_extrema": False,
            "calculate_envelopes": False,
            "calculate_knn": False,
            "calculate_statistics": False,
            "calculate_synchronization": False,
        },
        "Вейвлет и экстремумы": {
            "calculate_extrema": True,
            "calculate_envelopes": False,
            "calculate_knn": False,
            "calculate_statistics": False,
            "calculate_synchronization": False,
        },
        "Вейвлет, экстремумы и огибающие": {
            "calculate_extrema": True,
            "calculate_envelopes": True,
            "calculate_knn": False,
            "calculate_statistics": False,
            "calculate_synchronization": False,
        },
        "Огибающие и статистики": {
            "calculate_extrema": True,
            "calculate_envelopes": True,
            "calculate_knn": False,
            "calculate_statistics": True,
            "calculate_synchronization": False,
        },
        "Подготовка данных для ML": {
            "calculate_extrema": True,
            "calculate_envelopes": True,
            "calculate_knn": True,
            "calculate_statistics": False,
            "calculate_synchronization": False,
        },
        "Полный 1D-анализ": {
            "calculate_extrema": True,
            "calculate_envelopes": True,
            "calculate_knn": True,
            "calculate_statistics": True,
            "calculate_synchronization": False,
        },
    }

    PIPELINE_DESCRIPTIONS = {
        "Только вейвлет": (
            "Выполняется только вейвлет-преобразование. "
            "Вейвлеты рассчитываются всегда."
        ),
        "Вейвлет и экстремумы": (
            "Вейвлет-преобразование и поиск локальных экстремумов."
        ),
        "Вейвлет, экстремумы и огибающие": (
            "Вейвлеты → экстремумы → построение огибающих."
        ),
        "Огибающие и статистики": (
            "Вейвлеты → экстремумы → огибающие плюс статистики."
        ),
        "Подготовка данных для ML": (
            "Цикл вычислений от вейвлетов до KNN включительно. "
            "Полученные KNN-признаки затем используются на отдельной странице ML."
        ),
        "Полный 1D-анализ": (
            "Цикл от вейвлетов до KNN включительно плюс статистики. "
            "Настройка и интеграция статистик будет расширена позже."
        ),
        "Пользовательский": (
            "Набор этапов не совпадает ни с одним готовым сценарием."
        ),
    }

    def __init__(self):
        self.task_id = 0
        self.task_name = ""
        # Режим является частью задачи, а не глобальной настройкой интерфейса.
        # Для 1D доступен управляемый конвейер последующих этапов; 2D пока
        # выполняет отдельный базовый Morlet без 1D-зависимостей.
        self.analysis_mode = "1d"
        self.image_path = ""
        self.original_image = None
        self.data = []  # данные изображения
        self.data_copy = []  # копия данных для сравнения
        self.color1 = None
        self.color2 = None
        self.gram_schmidt_applied = False
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
        self.pipeline_preset = "Только вейвлет"
        self.calculate_extrema = False
        self.calculate_envelopes = False
        self.calculate_knn = False
        self.output_wavelet_image = False
        self.output_wavelet_text = True
        self.output_wavelet_numpy = False  # устаревший формат; в новом UI не используется
        self.output_extremes_text = True
        self.output_extremes_image = False
        self.output_envelopes_text = True
        self.output_envelopes_image = False
        self.output_knn_text = True
        self.output_knn_image = False
        self.save_source_channels = False
        self.save_centering_means = False

        # Статистики экстремумов и межстрочная синхронизация
        self.calculate_statistics = False
        self.statistics_output_image = False
        self.statistics_output_csv = True
        self.calculate_synchronization = False
        self.synchronization_output_heatmap = False
        self.synchronization_output_matrix_csv = True
        self.synchronization_output_pairs_csv = True
        self.scale_block_sizes = [5]
        self.row_sync_stride = 1
        self.row_sync_tolerance = 1
        self.row_sync_metrics = ["jaccard"]

        self.k_neighbors = 5
        self.task_folder_path = ""
        self.result = {}
        self.knn_results = {}
        self.statistics_results = {}
        self.synchronization_results = {}

        # Настройки и результаты ML принадлежат конкретной задаче.
        self.ml_algorithm = "kmeans"
        self.ml_dataset_key = None
        self.ml_point_filter = "all"
        self.ml_feature_set = "knn"
        self.ml_standardize = True
        self.ml_n_clusters = 5
        self.ml_random_state = 42
        self.ml_eps = 0.8
        self.ml_min_samples = 5
        self.ml_dbscan_tile_size = 256
        self.ml_dbscan_tile_overlap = 32
        self.ml_dbscan_max_points_per_tile = 30000
        self.ml_result = None

    def invalidate_source_results(self):
        """Detach derived data and old output paths after replacing the source."""
        self.result = {}
        self.knn_results = {}
        self.statistics_results = {}
        self.synchronization_results = {}
        self.ml_result = None
        self.task_folder_path = ""
        self.color1 = None
        self.color2 = None
        self.gram_schmidt_applied = False

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
            'calculate_statistics': self.calculate_statistics,
            'calculate_synchronization': self.calculate_synchronization,
            'calculate_extrema': self.calculate_extrema,
            'calculate_envelopes': self.calculate_envelopes,
            'calculate_knn': self.calculate_knn,
            'pipeline_preset': self.pipeline_preset,
            'colors_selected': has_colors,
        }

    def set_analysis_mode(self, mode):
        """Установить поддерживаемый режим анализа для задачи."""
        if mode not in self.VALID_ANALYSIS_MODES:
            raise ValueError(f"Неизвестный режим анализа: {mode}")
        self.analysis_mode = mode

    def apply_pipeline_preset(self, preset_name):
        """Применить один из безопасных предустановленных сценариев."""
        if preset_name == "Пользовательский":
            self.pipeline_preset = preset_name
            return
        settings = self.PIPELINE_PRESETS.get(preset_name)
        if settings is None:
            raise ValueError(f"Неизвестный сценарий расчёта: {preset_name}")
        for field_name, value in settings.items():
            setattr(self, field_name, bool(value))
        self.pipeline_preset = preset_name

    def match_pipeline_preset(self):
        """Вернуть готовый сценарий, точно соответствующий ручному набору этапов."""
        current = {
            "calculate_extrema": bool(self.calculate_extrema),
            "calculate_envelopes": bool(self.calculate_envelopes),
            "calculate_knn": bool(self.calculate_knn),
            "calculate_statistics": bool(self.calculate_statistics),
            "calculate_synchronization": bool(self.calculate_synchronization),
        }
        for preset_name, settings in self.PIPELINE_PRESETS.items():
            if all(current.get(key) == bool(value) for key, value in settings.items()):
                return preset_name
        return "Пользовательский"

    def resolve_pipeline(self):
        """Построить непротиворечивый план без изменения выбора пользователя."""
        if self.analysis_mode == "2d":
            return PipelinePlan(
                mode="2d",
                extrema_requested=False,
                envelopes_requested=False,
                knn_requested=False,
                statistics_requested=False,
                synchronization_requested=False,
                ml_requested=False,
                extrema=False,
                envelopes=False,
                knn=False,
                statistics=False,
                synchronization=False,
                ml=False,
                maxima_required=False,
            )

        row_pipeline_available = bool(self.process_rows)
        knn = bool(self.calculate_knn and row_pipeline_available)
        statistics = bool(self.calculate_statistics and self.process_rows)
        synchronization = bool(
            self.calculate_synchronization and self.process_rows
        )
        ml_requested = False
        envelopes = bool(row_pipeline_available and (
            self.calculate_envelopes or knn or statistics or synchronization
        ))
        extrema = bool(
            row_pipeline_available and (self.calculate_extrema or envelopes)
        )
        return PipelinePlan(
            mode="1d",
            extrema_requested=bool(self.calculate_extrema),
            envelopes_requested=bool(self.calculate_envelopes),
            knn_requested=bool(self.calculate_knn),
            statistics_requested=bool(self.calculate_statistics),
            synchronization_requested=bool(self.calculate_synchronization),
            ml_requested=ml_requested,
            extrema=extrema,
            envelopes=envelopes,
            knn=knn,
            statistics=statistics,
            synchronization=synchronization,
            ml=False,
            maxima_required=bool(
                statistics
                or synchronization
                or (extrema and not (self.find_maxima or self.find_minima))
            ),
        )

    def settings_snapshot(self):
        """Return only reproducible settings; image arrays and results are excluded."""
        fields = (
            "analysis_mode", "image_path", "process_rows", "process_columns",
            "find_maxima", "find_minima", "gram_schmidt_applied",
            "orientations", "morlet_omega0",
            "morlet_anisotropy", "pipeline_preset", "calculate_extrema",
            "calculate_envelopes", "calculate_knn", "calculate_statistics",
            "calculate_synchronization", "output_wavelet_image",
            "output_wavelet_text", "output_wavelet_numpy", "output_extremes_text",
            "output_extremes_image", "output_envelopes_text",
            "output_envelopes_image", "output_knn_text", "output_knn_image",
            "statistics_output_image", "statistics_output_csv",
            "synchronization_output_heatmap", "synchronization_output_matrix_csv",
            "synchronization_output_pairs_csv", "scale_block_sizes",
            "row_sync_stride", "row_sync_tolerance", "row_sync_metrics",
            "k_neighbors", "ml_algorithm", "ml_point_filter", "ml_feature_set",
            "ml_standardize", "ml_n_clusters", "ml_random_state", "ml_eps",
            "ml_min_samples", "ml_dataset_key", "ml_dbscan_tile_size", "ml_dbscan_tile_overlap", "ml_dbscan_max_points_per_tile", "save_source_channels", "save_centering_means",
        )
        snapshot = {name: getattr(self, name) for name in fields}
        snapshot["scales"] = np.asarray(self.scales).tolist()
        snapshot["color1"] = None if self.color1 is None else np.asarray(self.color1).tolist()
        snapshot["color2"] = None if self.color2 is None else np.asarray(self.color2).tolist()
        return snapshot

    def execution_stage_names(self):
        """Return the user-facing execution plan used by the status panel."""
        plan = self.resolve_pipeline()
        stages = ["Подготовка данных", "2D Morlet" if plan.mode == "2d" else "Вейвлет-преобразование"]
        if plan.point_pipeline_required:
            selected = ["экстремумы"]
            if plan.envelopes:
                selected.append("огибающие")
            if plan.knn:
                selected.append("KNN")
            if plan.statistics:
                selected.append("статистики")
            if plan.synchronization:
                selected.append("синхронизации")
            # These operations are currently calculated in one scale/channel
            # traversal, therefore they form one honest progress stage.
            stages.append("Анализ точек: " + ", ".join(selected))
        stages.append("Сохранение результатов")
        return stages

    def scales_summary(self):
        values = np.asarray(self.scales, dtype=float)
        if values.size == 0:
            return "не заданы"
        if values.size == 1:
            return f"{values[0]:g} · 1 масштаб"
        differences = np.diff(values)
        regular = np.allclose(differences, differences[0])
        step = f", шаг {differences[0]:g}" if regular else ""
        return f"{values[0]:g}–{values[-1]:g}{step} · {values.size} масштабов"

    def executed_stage_summary(self):
        plan = self.resolve_pipeline()
        stages = ["Morlet 2D" if plan.mode == "2d" else "вейвлеты"]
        if plan.extrema:
            stages.append("экстремумы")
        if plan.envelopes:
            stages.append("огибающие")
        if plan.knn:
            stages.append("KNN")
        if plan.statistics:
            stages.append("статистики")
        if plan.synchronization:
            stages.append("синхронизации")
        return " → ".join(stages)

    def nuances_summary(self):
        details = []
        if self.analysis_mode == "1d":
            directions = []
            if self.process_rows:
                directions.append("строки")
            if self.process_columns:
                directions.append("столбцы")
            details.append(" + ".join(directions) if directions else "направление не выбрано")
        else:
            details.append(f"ориентации: {', '.join(f'{value:g}°' for value in self.orientations)}")
        point_types = []
        if self.find_maxima:
            point_types.append("максимумы")
        if self.find_minima:
            point_types.append("минимумы")
        if self.resolve_pipeline().extrema and point_types:
            details.append(" + ".join(point_types))
        if self.resolve_pipeline().statistics or self.resolve_pipeline().synchronization:
            details.append("блоки масштабов: " + ", ".join(map(str, self.scale_block_sizes)))
        if self.resolve_pipeline().synchronization:
            details.append("метрика: " + ", ".join(self.row_sync_metrics))
        if self.gram_schmidt_applied:
            details.append("Грамм–Шмидт")
        return " · ".join(details)

    def apply_settings_snapshot(self, snapshot):
        """Apply a saved configuration without silently reusing old result data."""
        for name, value in snapshot.items():
            if name in {"scales", "color1", "color2", "image_path"}:
                continue
            if hasattr(self, name):
                setattr(self, name, value)
        self.scales = np.asarray(snapshot.get("scales", []), dtype=float)
        self.num_scale = len(self.scales)
        for name in ("color1", "color2"):
            value = snapshot.get(name)
            setattr(self, name, None if value is None else np.asarray(value))
        # Source is restored by the application only if the file is still readable.
        self.image_path = snapshot.get("image_path", "")
        self.task_folder_path = ""
        self.result = {}
        self.knn_results = {}
        self.statistics_results = {}
        self.synchronization_results = {}
        self.ml_result = None

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
        return bool(self.gram_schmidt_applied)
