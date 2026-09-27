import gc
import logging
import os
from typing import Dict, Optional
from contextlib import nullcontext
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from sklearn.neighbors import NearestNeighbors
from result_naming import knn_stem, point_type_parts
from compute.backend_policy import GPU_FALLBACK_ERRORS, GPUUnavailableError, ExecutionProtocol
from compute.numerics import COMPUTE_DTYPE, COMPUTE_DTYPE_NAME
from compute.knn.result_format import make_compact_neighbors, neighbor_arrays

logger = logging.getLogger("WaveletApp")


def _profile_context(profiling, key):
    """Return a StageProfile measurement context when KNN profiling is enabled."""
    if not profiling:
        return nullcontext()
    profile = profiling.get(key)
    return profile.measure() if profile is not None else nullcontext()


def _profile_stat(stats, key, increment=1):
    if stats is not None:
        stats[key] = stats.get(key, 0) + increment

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
                                 stage: str = "knn", profiling=None) -> Optional[Dict]:
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
        result = self._find_k_nearest_neighbors_cpu(
            points, k, progress_callback, log_callback, profiling=profiling
        )
        if protocol is not None:
            protocol.record_stage(
                stage, actual_backend="cpu", dtype=COMPUTE_DTYPE_NAME,
                fallback=any(w.stage == stage for w in protocol.warnings),
            )
        return result

    @staticmethod
    def _find_k_nearest_neighbors_cpu(points: np.ndarray, k: int,
                                      progress_callback=None, log_callback=None,
                                      profiling=None) -> Dict:
        """CPU реализация KNN"""
        points = np.asarray(points, dtype=COMPUTE_DTYPE)
        if len(points) < 2:
            return {}

        if len(points) <= k:
            k = len(points) - 1

        try:
            if progress_callback:
                progress_callback(0.1, "Инициализация алгоритма NearestNeighbors...")

            with _profile_context(profiling, "cpu_fit"):
                nbrs = NearestNeighbors(
                    n_neighbors=k + 1, algorithm='kd_tree'
                ).fit(points)

            if progress_callback:
                progress_callback(0.4, "Вычисление расстояний между точками...")

            with _profile_context(profiling, "cpu_kneighbors"):
                distances, indices = nbrs.kneighbors(points)
                distances = np.asarray(distances, dtype=COMPUTE_DTYPE)

            if progress_callback:
                progress_callback(0.8, "Формирование компактного результата...")

            # NearestNeighbors returns k+1 items because the first one is the
            # query point itself.  Keep only real neighbours and immediately
            # compress the result to int32/float32 NumPy matrices.  This avoids
            # N Python dictionaries and 2*N*k Python scalar objects.
            with _profile_context(profiling, "cpu_compact_result"):
                compact = make_compact_neighbors(
                    indices[:, 1:].astype(np.int32, copy=False),
                    distances[:, 1:].astype(np.float32, copy=False),
                )

            if progress_callback:
                progress_callback(1.0, "Поиск соседей завершен")

            return compact

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
                              protocol: ExecutionProtocol | None = None,
                              profiling=None, profiling_stats=None,
                              print_npz_var=True, png_max_points=50000):
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
        with _profile_context(profiling, "points_prepare"):
            # Preserve the historical production conversion exactly; the
            # backend then normalizes to COMPUTE_DTYPE inside the search call.
            points = np.asarray(extreme_dict[extreme_type], dtype=COMPUTE_DTYPE)

        if len(points) < 2:
            continue

        _profile_stat(profiling_stats, "groups", 1)
        _profile_stat(profiling_stats, "points", int(len(points)))

        current_type_progress = processed_types / total_extreme_types

        def update_progress(stage_progress, message):
            if progress_callback:
                overall_progress = (current_type_progress - 1 / total_extreme_types) + (stage_progress / total_extreme_types)
                progress_callback(overall_progress, f"KNN {extreme_type}: {message}")

        with _profile_context(profiling, "search_total"):
            neighbors = processor.find_k_nearest_neighbors(
                points, k,
                progress_callback=lambda p, m: update_progress(p * 0.4, m),
                log_callback=log_callback,
                protocol=protocol,
                stage=f"knn:{source_direction}:{channel}:{float(scale):g}:{extreme_type}",
                profiling=profiling,
            )

        if not neighbors:
            continue

        neighbor_indices, _, _ = neighbor_arrays(neighbors, point_count=len(points))
        _profile_stat(profiling_stats, "links", int(np.count_nonzero(neighbor_indices >= 0)))

        with _profile_context(profiling, "angles"):
            neighbors_with_angles = compute_angles_for_neighbors(
                points,
                neighbors,
                image_coords=True)
        compact_indices, compact_distances, compact_angles = neighbor_arrays(
            neighbors_with_angles, point_count=len(points)
        )
        compact_bytes = (
            points.nbytes
            + compact_indices.nbytes
            + compact_distances.nbytes
            + (0 if compact_angles is None else compact_angles.nbytes)
        )
        _profile_stat(profiling_stats, "stored_bytes", int(compact_bytes))
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
        npz_filename = file_stem + ".npz"

        # NPZ is the primary complete machine-readable representation.  It
        # stores the exact KNN arrays without converting millions of links to
        # decimal text and is independent from the optional TXT/PNG exports.
        if print_npz_var:
            save_knn_npz(
                os.path.join(scale_folder_path, npz_filename),
                points, neighbors_with_angles, scale, channel, extreme_type, k,
                original_image=original_image, source_direction=source_direction,
                log_callback=log_callback, profiling=profiling,
                profiling_stats=profiling_stats)

        if print_image_var:
            draw_knn_graph(
                points, neighbors_with_angles, scale, channel, extreme_type,
                os.path.join(scale_folder_path, graph_filename), k, original_image,
                progress_callback=lambda p, m: update_progress(0.4 + p * 0.3, m),
                log_callback=log_callback,
                profiling=profiling,
                profiling_stats=profiling_stats,
                max_visualization_points=png_max_points)

        if print_text_var:
            save_knn_info(
                os.path.join(scale_folder_path, info_filename),
                points, neighbors_with_angles, scale, channel, extreme_type, k,
                progress_callback=lambda p, m: update_progress(0.7 + p * 0.3, m),
                log_callback=log_callback,
                profiling=profiling,
                profiling_stats=profiling_stats)

    # Matplotlib and temporary Python containers are collected once per
    # (CWT direction, channel, scale) group instead of after every PNG.
    # NumPy arrays retained in ``results`` are reference-counted and are not
    # affected by this collection.
    with _profile_context(profiling, "gc"):
        gc.collect()

    if log_callback:
        log_callback(f"Завершена обработка KNN для масштаба {scale}")
    return results


def compute_angles_for_neighbors(points: np.ndarray, neighbors, image_coords: bool = True):
    """Vectorized KNN direction angles for all N*k links.

    Image coordinates use +x to the right and +y downward.  The project angle
    is measured clockwise from the upward vertical, therefore
    ``theta = atan2(dx, -dy) mod 2*pi``.
    """
    points = np.asarray(points, dtype=COMPUTE_DTYPE)
    indices, distances, _ = neighbor_arrays(neighbors, point_count=len(points))
    if indices.size == 0:
        return make_compact_neighbors(indices, distances, np.empty_like(distances))

    valid = (indices >= 0) & (indices < len(points))
    safe_indices = indices if np.all(valid) else np.where(valid, indices, 0)

    # Work with two (N, k) coordinate-difference matrices instead of one
    # (N, k, 2) target tensor.  ``out=`` keeps angle normalization in-place,
    # which bounds temporary memory on multi-million-link groups.
    dx = points[safe_indices, 0] - points[:, None, 0]
    dy = points[safe_indices, 1] - points[:, None, 1]
    zero_length = (dx == 0) & (dy == 0)

    if image_coords:
        np.negative(dy, out=dy)
    angles = np.empty_like(dx, dtype=np.float32)
    np.arctan2(dx, dy, out=angles)
    np.remainder(angles, np.float32(2.0 * np.pi), out=angles)
    np.multiply(angles, np.float32(180.0 / np.pi), out=angles)

    angles[~valid | zero_length] = np.nan

    return make_compact_neighbors(indices, distances, angles)


def _visualization_source_indices(point_count: int, max_points: int) -> np.ndarray:
    """Return deterministic approximately uniform source indices for PNG rendering.

    The full KNN remains untouched.  Sampling is applied only to the static
    visualization.  ``max_points <= 0`` explicitly requests full rendering.
    """
    n = int(point_count)
    limit = int(max_points or 0)
    if n <= 0:
        return np.empty(0, dtype=np.int32)
    if limit <= 0 or n <= limit:
        return np.arange(n, dtype=np.int32)

    # One representative source from each equal-width interval of the ordered
    # point array.  For n >= limit the integer formula is strictly increasing,
    # deterministic, and requires no RNG state.
    sample = (np.arange(limit, dtype=np.int64) * n // limit).astype(np.int32)
    return sample


def _unique_undirected_edges(neighbors, point_count: int,
                             source_indices=None) -> np.ndarray:
    """Build unique undirected edge pairs without Python ``set`` objects.

    ``source_indices`` limits only visualization sources.  Neighbor targets are
    still taken from the full exact KNN, so rendered segments remain genuine
    KNN links rather than a recomputed graph on the sampled subset.
    """
    indices, _, _ = neighbor_arrays(neighbors, point_count=point_count)
    if indices.size == 0:
        return np.empty((0, 2), dtype=np.int32)

    n, k = indices.shape
    if source_indices is None:
        source_rows = np.arange(n, dtype=np.int32)
    else:
        source_rows = np.asarray(source_indices, dtype=np.int32).reshape(-1)
        source_rows = source_rows[(source_rows >= 0) & (source_rows < n)]
    if source_rows.size == 0:
        return np.empty((0, 2), dtype=np.int32)

    sources = np.repeat(source_rows, k)
    targets = indices[source_rows].reshape(-1)
    valid = (targets >= 0) & (targets < point_count) & (sources != targets)
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.int32)

    sources = sources[valid]
    targets = targets[valid].astype(np.int32, copy=False)
    lo = np.minimum(sources, targets)
    hi = np.maximum(sources, targets)

    # Encode pair (lo, hi) into one uint64 key. ``point_count`` is the radix,
    # therefore each valid undirected pair has one unique scalar key.
    keys = lo.astype(np.uint64) * np.uint64(point_count) + hi.astype(np.uint64)
    _, first = np.unique(keys, return_index=True)
    return np.column_stack((lo[first], hi[first])).astype(np.int32, copy=False)


def save_knn_npz(filename, points, neighbors_dict, scale, channel, extreme_type, k,
                 original_image=None, source_direction=None, log_callback=None,
                 profiling=None, profiling_stats=None):
    """Save the complete lossless KNN result as an uncompressed NPZ archive.

    ``np.savez`` is intentionally used instead of ``np.savez_compressed``:
    for multi-million-link KNN datasets the priority is fast exact export and
    reload.  The arrays are already compact int32/float32 matrices.
    """
    if log_callback:
        log_callback(f"Сохранение KNN NPZ: {filename}")

    try:
        with _profile_context(profiling, "npz_export"):
            points_arr = np.ascontiguousarray(points, dtype=np.float32)
            indices, distances, angles = neighbor_arrays(
                neighbors_dict, point_count=len(points_arr)
            )
            if angles is None:
                angles = np.full(distances.shape, np.nan, dtype=np.float32)

            if original_image is None:
                image_shape = np.array([0, 0], dtype=np.int32)
            else:
                image_shape = np.asarray(original_image.shape[:2], dtype=np.int32)

            np.savez(
                filename,
                format_version=np.array([1], dtype=np.int16),
                points=points_arr,
                indices=np.ascontiguousarray(indices, dtype=np.int32),
                distances=np.ascontiguousarray(distances, dtype=np.float32),
                angles=np.ascontiguousarray(angles, dtype=np.float32),
                shape=image_shape,
                scale=np.array([float(scale)], dtype=np.float32),
                k=np.array([int(k)], dtype=np.int32),
                channel=np.array(str(channel)),
                extreme_type=np.array(str(extreme_type)),
                source_direction=np.array(str(source_direction or "")),
                angle_definition=np.array("from_vertical_up_clockwise"),
                coordinate_system=np.array("image_screen"),
            )

            _profile_stat(profiling_stats, "npz_files", 1)
            try:
                _profile_stat(profiling_stats, "npz_bytes", os.path.getsize(filename))
            except OSError:
                pass

        if log_callback:
            log_callback("KNN NPZ успешно сохранён")
    except Exception as e:
        if log_callback:
            log_callback(f"Ошибка сохранения KNN NPZ: {e}")


def draw_knn_graph(points, neighbors_dict, scale, channel, extreme_type, filename, k, original_image,
                   progress_callback=None, log_callback=None,
                   profiling=None, profiling_stats=None,
                   max_visualization_points=50000):
    if len(points) == 0:
        if log_callback:
            log_callback("Нет точек для отрисовки графа")
        return

    if log_callback:
        log_callback(f"Начало отрисовки графа KNN: {len(points)} точек")

    fig = None
    try:
        # The PNG path is measured separately from the TXT sidecar written
        # afterwards. This makes it possible to distinguish Matplotlib/render
        # cost from text serialization in production logs.
        with _profile_context(profiling, "image_export"):
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

            source_indices = _visualization_source_indices(
                len(points), max_visualization_points
            )
            sampled = len(source_indices) < len(points)
            if sampled and log_callback:
                log_callback(
                    "PNG KNN: визуализация "
                    f"{len(source_indices)} из {len(points)} исходных точек; "
                    "полный KNN сохранён без прореживания"
                )

            with _profile_context(profiling, "image_edges"):
                idx = _unique_undirected_edges(
                    neighbors_dict, len(points), source_indices=source_indices
                )

            _profile_stat(profiling_stats, "image_source_points", int(len(source_indices)))
            _profile_stat(profiling_stats, "image_full_points", int(len(points)))
            _profile_stat(profiling_stats, "image_edges_rendered", int(len(idx)))
            if sampled:
                _profile_stat(profiling_stats, "image_sampled_files", 1)

            if len(idx):
                lines = np.stack([
                    points[idx[:, 0]],
                    points[idx[:, 1]]
                ], axis=1).astype(np.float32, copy=False)

                lc = LineCollection(
                    lines,
                    colors='blue',
                    linewidths=0.5,
                    alpha=0.6
                )
                ax.add_collection(lc)

            if progress_callback:
                progress_callback(0.75, "Отрисовка точек...")

            visual_points = points[source_indices]
            ax.scatter(
                visual_points[:, 0],
                visual_points[:, 1],
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
            sampling_note = (
                f"; PNG: {len(source_indices)}/{len(points)} точек"
                if sampled else ""
            )
            ax.set_title(
                f'Граф {k}-ближайших соседей\n'
                f'Масштаб: {scale}, Канал: {channel_label}, Тип: {extreme_type}'
                f'{sampling_note}'
            )
            ax.set_xlabel('X (пиксели)')
            ax.set_ylabel('Y (пиксели)')

            if progress_callback:
                progress_callback(0.9, "Сохранение PNG...")

            fig.savefig(filename, dpi=180, bbox_inches='tight')
            _profile_stat(profiling_stats, "image_files", 1)

        # PNG and TXT are independent export formats.  PNG generation must
        # not create/overwrite a hidden TXT sidecar.
        if log_callback:
            log_callback(f"График KNN сохранён: {filename}")

        if progress_callback:
            progress_callback(1.0, "График сохранён")

    except Exception as e:
        if log_callback:
            log_callback(f"Ошибка при отрисовке KNN: {e}")
    finally:
        if fig is not None:
            plt.close(fig)


def save_knn_info(
        filename,
        points,
        neighbors_dict,
        scale,
        channel,
        extreme_type,
        k,
        progress_callback=None,
        log_callback=None,
        profiling=None,
        profiling_stats=None):

    if log_callback:
        log_callback(f"Сохранение KNN: {filename}")

    try:
        with _profile_context(profiling, "text_export"):
            if progress_callback:
                progress_callback(0.2, "Запись заголовка...")

            rows_written = 0
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

                indices, distances, angles = neighbor_arrays(
                    neighbors_dict, point_count=len(points)
                )
                buffer = []
                flush_rows = 8192
                for i in range(len(points)):
                    x, y = points[i]
                    for j in range(indices.shape[1]):
                        neighbor_id = int(indices[i, j])
                        if neighbor_id < 0:
                            continue
                        dist = float(distances[i, j])
                        angle = None if angles is None else float(angles[i, j])
                        angle_str = "" if angle is None else f"{angle:.6f}"
                        buffer.append(
                            f"{i},{x},{y},{neighbor_id},{dist:.6f},{angle_str}\n"
                        )
                        rows_written += 1
                        if len(buffer) >= flush_rows:
                            f.writelines(buffer)
                            buffer.clear()
                if buffer:
                    f.writelines(buffer)

            _profile_stat(profiling_stats, "text_files", 1)
            _profile_stat(profiling_stats, "text_rows", rows_written)

        if log_callback:
            log_callback("Файл KNN успешно сохранён")

        if progress_callback:
            progress_callback(1.0, "Файл сохранён")

    except Exception as e:
        if log_callback:
            log_callback(f"Ошибка сохранения KNN: {e}")
