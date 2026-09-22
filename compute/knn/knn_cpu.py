import gc
import logging
import os
from typing import Dict, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from sklearn.neighbors import NearestNeighbors
from result_naming import knn_stem, point_type_parts
from compute.backend_policy import GPU_FALLBACK_ERRORS, GPUUnavailableError, ExecutionProtocol
from compute.numerics import COMPUTE_DTYPE, COMPUTE_DTYPE_NAME

logger = logging.getLogger("WaveletApp")

class KNNProcessor:
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self.gpu_processor = None
        self._initialize_processors()

    def _initialize_processors(self):
        """Инициализация CPU и GPU процессоров"""
        # Всегда доступен CPU
        self.cpu_available = True

        # Пытаемся инициализировать GPU
        if self.use_gpu:
            try:
                from .knn_gpu import KNN_GPU
                self.gpu_processor = KNN_GPU(
                    use_sqrt=True,
                    cleanup_each_ref_batch=False,
                    cleanup_each_query_batch=False,
                    log_timing=True
                )
                if self.gpu_processor.is_available():
                    logger.info("GPU KNN процессор инициализирован")
                else:
                    logger.info("GPU KNN недоступен, используется CPU")
                    self.gpu_processor = None
            except ImportError as e:
                logger.warning(f"Не удалось импортировать GPU KNN: {e}")
                self.gpu_processor = None

    def find_k_nearest_neighbors(self, points: np.ndarray, k: int,
                                 progress_callback=None, log_callback=None,
                                 protocol: ExecutionProtocol | None = None,
                                 stage: str = "knn") -> Optional[Dict]:
        """Унифицированный поиск соседей (GPU или CPU)"""
        if log_callback:
            log_callback(f"Поиск {k}-ближайших соседей для {len(points)} точек")

        points = np.asarray(points, dtype=COMPUTE_DTYPE)

        if len(points) < 2:
            if log_callback:
                log_callback("Недостаточно точек для поиска соседей")
            return {}

        # Пытаемся использовать GPU если он был запрошен и доступен.
        if self.use_gpu and (self.gpu_processor is None or not self.gpu_processor.is_available()):
            exc = GPUUnavailableError("GPU KNN processor is unavailable")
            if protocol is not None and protocol.strict_backend:
                raise exc
            if protocol is not None:
                protocol.record_fallback(
                    stage, reason=exc.__class__.__name__, message=str(exc)
                )
            if log_callback:
                log_callback("GPU KNN недоступен, переход на CPU")

        if self.use_gpu and self.gpu_processor and self.gpu_processor.is_available():
            if log_callback:
                log_callback("Использование GPU для KNN...")

            try:
                result = self.gpu_processor.find_k_nearest_neighbors(points, k)
                if protocol is not None:
                    protocol.record_stage(stage, actual_backend="gpu", dtype=COMPUTE_DTYPE_NAME)
                if log_callback:
                    log_callback("KNN вычислен на GPU")
                return result
            except GPU_FALLBACK_ERRORS as exc:
                if protocol is not None and protocol.strict_backend:
                    raise
                if protocol is not None:
                    protocol.record_fallback(
                        stage, reason=exc.__class__.__name__, message=str(exc)
                    )
                if log_callback:
                    log_callback(
                        f"GPU KNN недоступен ({exc.__class__.__name__}), переход на CPU"
                    )

        # Fallback to CPU
        if log_callback:
            log_callback("Использование CPU для KNN...")
        result = self._find_k_nearest_neighbors_cpu(points, k, progress_callback, log_callback)
        if protocol is not None:
            protocol.record_stage(
                stage, actual_backend="cpu", dtype=COMPUTE_DTYPE_NAME,
                fallback=any(w.stage == stage for w in protocol.warnings),
            )
        return result

    @staticmethod
    def _find_k_nearest_neighbors_cpu(points: np.ndarray, k: int,
                                      progress_callback=None, log_callback=None) -> Dict:
        """CPU реализация KNN"""
        points = np.asarray(points, dtype=COMPUTE_DTYPE)
        if len(points) < 2:
            return {}

        if len(points) <= k:
            k = len(points) - 1

        try:
            if progress_callback:
                progress_callback(0.1, "Инициализация алгоритма NearestNeighbors...")

            nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='kd_tree').fit(points)

            if progress_callback:
                progress_callback(0.4, "Вычисление расстояний между точками...")

            distances, indices = nbrs.kneighbors(points)
            distances = np.asarray(distances, dtype=COMPUTE_DTYPE)

            if progress_callback:
                progress_callback(0.8, "Формирование словаря соседей...")

            neighbors_dict = {}
            for i in range(len(points)):
                neighbors_dict[i] = {
                    'indices': indices[i][1:].tolist(),  # исключаем саму точку
                    'distances': distances[i][1:].tolist()
                }

            if progress_callback:
                progress_callback(1.0, "Поиск соседей завершен")

            return neighbors_dict

        except Exception as e:
            if log_callback:
                log_callback(f"Ошибка при поиске соседей на CPU: {e}")
            return {}

    def is_gpu_available(self) -> bool:
        """Проверка доступности GPU"""
        return self.gpu_processor is not None and self.gpu_processor.is_available()

    def toggle_gpu(self, use_gpu: bool) -> bool:
        """Переключение между GPU и CPU"""
        self.use_gpu = use_gpu
        if use_gpu and self.gpu_processor is None:
            self._initialize_processors()
        return self.is_gpu_available()

_knn_processor = None

def get_knn_processor(use_gpu: bool = True):
    global _knn_processor

    if _knn_processor is None:
        _knn_processor = KNNProcessor(use_gpu)
    else:
        _knn_processor.toggle_gpu(use_gpu)

    return _knn_processor

def find_k_nearest_neighbors(points, k, progress_callback=None, log_callback=None):
    processor = get_knn_processor()
    return processor.find_k_nearest_neighbors(points, k, progress_callback, log_callback)

def process_extremes_with_knn(extreme_dict, scale_folder_path, k, original_image,
                              print_text_var, print_image_var, use_gpu=True,
                              source_direction=None,
                              progress_callback=None, log_callback=None,
                              protocol: ExecutionProtocol | None = None):
    processor = get_knn_processor(use_gpu)

    if log_callback:
        device = "GPU" if processor.is_gpu_available() and use_gpu else "CPU"
        log_callback(f"Начало обработки KNN на {device} для масштаба {extreme_dict['scale']}")

    type_data = extreme_dict['type_data']
    channel = extreme_dict['channel']
    scale = extreme_dict['scale']
    source_direction = (
        source_direction
        or extreme_dict.get("cwt_axis")
        or ("row" if type_data == 0 else "col")
    )

    extreme_types = ['max_by_row', 'max_by_column', 'min_by_row', 'min_by_column']
    total_extreme_types = len(extreme_types)
    processed_types = 0

    results = {}
    for extreme_type in extreme_types:
        processed_types += 1
        points = np.array(extreme_dict[extreme_type])

        if len(points) < 2:
            continue

        current_type_progress = processed_types / total_extreme_types

        def update_progress(stage_progress, message):
            if progress_callback:
                overall_progress = (current_type_progress - 1 / total_extreme_types) + (stage_progress / total_extreme_types)
                progress_callback(overall_progress, f"KNN {extreme_type}: {message}")

        neighbors_dict = processor.find_k_nearest_neighbors(
            points, k,
            progress_callback=lambda p, m: update_progress(p * 0.4, m),
            log_callback=log_callback,
            protocol=protocol,
            stage=f"knn:{source_direction}:{channel}:{scale}:{extreme_type}")

        if not neighbors_dict:
            continue

        neighbors_with_angles = compute_angles_for_neighbors(
            points,
            neighbors_dict,
            image_coords=True)
        results[extreme_type] = {
            "points": points,
            "neighbors": neighbors_with_angles,
        }

        feature_axis, point_kind = point_type_parts(extreme_type)
        file_stem = knn_stem(
            source_direction, feature_axis, channel, scale, point_kind, k
        )
        graph_filename = file_stem + ".png"
        info_filename = file_stem + ".txt"

        if print_image_var:
            draw_knn_graph(
                points, neighbors_dict, scale, channel, extreme_type,
                os.path.join(scale_folder_path, graph_filename), k, original_image,
                progress_callback=lambda p, m: update_progress(0.4 + p * 0.3, m),
                log_callback=log_callback)

        if print_text_var:
            save_knn_info(
                os.path.join(scale_folder_path, info_filename),
                points, neighbors_with_angles, scale, channel, extreme_type, k,
                progress_callback=lambda p, m: update_progress(0.7 + p * 0.3, m),
                log_callback=log_callback)

    if log_callback:
        log_callback(f"Завершена обработка KNN для масштаба {scale}")
    return results


def compute_angles_for_neighbors(points: np.ndarray, neighbors_dict: dict, image_coords: bool = True):
    points = np.asarray(points, dtype=COMPUTE_DTYPE)
    result = {}

    for i, data in neighbors_dict.items():
        i = int(i)
        xi, yi = points[i]
        angles = []

        for neighbor_id in data["indices"]:
            neighbor_id = int(neighbor_id)
            xj, yj = points[neighbor_id]

            dx = xj - xi
            dy = yj - yi

            if dx == 0 and dy == 0:
                angle_deg = np.nan
            else:
                if image_coords:
                    angle = np.arctan2(dx, -dy)
                else:
                    angle = np.arctan2(dx, dy)

                angle = (angle + 2 * np.pi) % (2 * np.pi)
                angle_deg = float(np.degrees(angle))

            angles.append(angle_deg)

        result[i] = {
            "indices": [int(idx) for idx in data["indices"]],
            "distances": [float(dist) for dist in data["distances"]],
            "angles": angles
        }

    return result

def draw_knn_graph(points, neighbors_dict, scale, channel, extreme_type, filename, k, original_image,
                   progress_callback=None, log_callback=None):
    if len(points) == 0:
        if log_callback:
            log_callback("Нет точек для отрисовки графа")
        return

    if log_callback:
        log_callback(f"Начало отрисовки графа KNN: {len(points)} точек")

    try:
        if progress_callback:
            progress_callback(0.1, "Подготовка данных...")

        fig, ax = plt.subplots(figsize=(7, 8))
        if original_image is not None:
            if progress_callback:
                progress_callback(0.25, "Добавление фонового изображения...")
            ax.imshow(original_image)

        h, w = original_image.shape[:2] if original_image is not None else (None, None)

        ax.invert_yaxis()
        ax.set_autoscale_on(False)
        if progress_callback:
            progress_callback(0.4, "Подготовка соединений...")

        edges = []
        seen = set()

        for i, neighbors in neighbors_dict.items():
            for j in neighbors['indices']:
                edge = tuple(sorted((i, j)))
                if edge in seen:
                    continue
                seen.add(edge)
                edges.append(edge)

        if edges:
            idx = np.array(edges, dtype=np.int32)
            lines = np.stack([
                points[idx[:, 0]],
                points[idx[:, 1]]
            ], axis=1).astype(np.float32)

            lc = LineCollection(
                lines,
                colors='blue',
                linewidths=0.5,
                alpha=0.6
            )

            ax.add_collection(lc)
        if progress_callback:
            progress_callback(0.75, "Отрисовка точек...")

        ax.scatter(
            points[:, 0],
            points[:, 1],
            c='red',
            s=2,
            alpha=1
        )

        if w is not None and h is not None:
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)

        channel_label = {
            "r": "Красный (R)", "g": "Зелёный (G)", "b": "Синий (B)",
            "gray": "Gray", "gs1": "GS1", "gs2": "GS2", "gs3": "GS3",
            0: "Красный (R)", 1: "Зелёный (G)", 2: "Синий (B)",
        }.get(channel, str(channel))
        ax.set_title(
            f'Граф {k}-ближайших соседей\n'
            f'Масштаб: {scale}, Канал: {channel_label}, Тип: {extreme_type}'
        )
        ax.set_xlabel('X (пиксели)')
        ax.set_ylabel('Y (пиксели)')

        if progress_callback:
            progress_callback(0.9, "Сохранение...")

        fig.savefig(filename, dpi=180, bbox_inches='tight')
        save_knn_info(os.path.splitext(filename)[0] + '.txt', points, neighbors_dict,
                      scale, channel, extreme_type, k, log_callback=log_callback)
        plt.close(fig)

        if log_callback:
            log_callback(f"График KNN сохранён: {filename}")

        if progress_callback:
            progress_callback(1.0, "График сохранён")

        gc.collect()

    except Exception as e:
        if log_callback:
            log_callback(f"Ошибка при отрисовке KNN: {e}")

def save_knn_info(
        filename,
        points,
        neighbors_dict,
        scale,
        channel,
        extreme_type,
        k,
        progress_callback=None,
        log_callback=None):

    if log_callback:
        log_callback(f"Сохранение KNN: {filename}")

    try:
        if progress_callback:
            progress_callback(0.2, "Запись заголовка...")

        with open(filename, 'w', encoding='utf-8') as f:
            f.write("# META\n")
            f.write(f"# scale={scale}\n")
            f.write(f"# channel={channel}\n")
            f.write(f"# extreme_type={extreme_type}\n")
            f.write(f"# k={k}\n")
            f.write(f"# total_points={len(points)}\n")
            f.write("# angle_definition=from_vertical_up_clockwise\n")
            f.write("# coordinate_system=image_screen\n")
            f.write("\n")
            f.write("point_id,x,y,neighbor_id,distance,angle_deg\n")

            if progress_callback:
                progress_callback(0.5, "Запись связей...")

            for i, data in neighbors_dict.items():
                x, y = points[i]

                angles = data.get("angles", [None] * len(data["indices"]))

                for neighbor_id, dist, angle in zip(
                        data['indices'],
                        data['distances'],
                        angles):

                    if angle is None:
                        angle_str = ""
                    else:
                        angle_str = f"{angle:.6f}"

                    f.write(f"{i},{x},{y},{neighbor_id},{dist:.6f},{angle_str}\n")

        if log_callback:
            log_callback("Файл KNN успешно сохранён")

        if progress_callback:
            progress_callback(1.0, "Файл сохранён")

    except Exception as e:
        if log_callback:
            log_callback(f"Ошибка сохранения KNN: {e}")
