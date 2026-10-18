"""Internal reusable checkpoints for researcher runs.

Only application-created files from a run result directory are loaded. Raw
wavelet tensors are deliberately excluded here because they can be many
gigabytes for large images; this checkpoint targets inexpensive repeated ML.
"""

import gzip
import os
import pickle
import zipfile
import zlib

import numpy as np
from history.atomic_io import atomic_binary_writer
from compute.knn.result_format import materialize_knn_payload, neighbor_arrays


CHECKPOINT_FILENAME = "feature_checkpoint.pkl.gz"
LEGACY_CHECKPOINT_FILENAMES = ("Контрольная_точка_признаков.pkl.gz",)
CHECKPOINT_VERSION = 1


def checkpoint_path(output_dir):
    return os.path.join(output_dir, CHECKPOINT_FILENAME)


def existing_checkpoint_path(output_dir):
    candidates = (CHECKPOINT_FILENAME, *LEGACY_CHECKPOINT_FILENAMES)
    return next(
        (os.path.join(output_dir, name) for name in candidates
         if os.path.isfile(os.path.join(output_dir, name))),
        checkpoint_path(output_dir),
    )


def feature_checkpoint_available(output_dir):
    return bool(
        output_dir and os.path.isfile(existing_checkpoint_path(output_dir))
    )


def save_feature_checkpoint(task, *, cancel_check=None):
    if not task.task_folder_path:
        return ""
    payload = {
        "version": CHECKPOINT_VERSION,
        "knn_results": task.knn_results,
        "statistics_results": task.statistics_results,
        "synchronization_results": task.synchronization_results,
        "completed": {
            "knn": bool(task.knn_results),
            "statistics": bool(task.statistics_results),
            "synchronization": bool(task.synchronization_results),
        },
    }
    path = checkpoint_path(task.task_folder_path)
    with atomic_binary_writer(path, cancel_check=cancel_check) as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=5) as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def _valid_knn_group(group):
    loaded = materialize_knn_payload(group)
    points = np.asarray(loaded['points'])
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("некорректные координаты KNN")
    indices, distances, angles = neighbor_arrays(loaded['neighbors'], point_count=len(points))
    if indices.shape != distances.shape or indices.ndim != 2 or indices.shape[0] != len(points):
        raise ValueError("несогласованные размеры KNN")
    valid = indices >= 0
    if ((indices < -1).any() or (indices[valid] >= len(points)).any()
            or not np.isfinite(distances[valid]).all() or (distances[valid] < 0).any()):
        raise ValueError("некорректные соседи KNN")
    if angles is not None and (angles.shape != indices.shape or not np.isfinite(angles[valid]).all()):
        raise ValueError("некорректные углы KNN")


_RECOVERY_ERRORS = (OSError, EOFError, ValueError, TypeError, KeyError,
                    AttributeError, pickle.UnpicklingError, zipfile.BadZipFile, zlib.error)


def restore_feature_checkpoint(task, output_dir):
    """Restore independent sources; report unavailable ones without selecting substitutes."""
    task.recovery_warnings = []
    task.knn_results = {}
    task.statistics_results = {}
    task.synchronization_results = {}
    task.task_folder_path = output_dir
    path = existing_checkpoint_path(output_dir)
    try:
        with gzip.open(path, "rb") as stream:
            payload = pickle.load(stream)
        if not isinstance(payload, dict) or payload.get("version") != CHECKPOINT_VERSION:
            raise ValueError("версия или структура контрольной точки не поддерживается")
        if not isinstance(payload.get('knn_results', {}), dict):
            raise ValueError("некорректный список KNN-источников")
    except _RECOVERY_ERRORS as error:
        task.recovery_warnings.append(
            f"Контрольная точка недоступна ({path}): {error}. "
            "Для ML требуется повторный расчёт KNN; остальные файлы можно открыть в результатах.")
        return []
    recovered = {}
    for key, group in payload.get('knn_results', {}).items():
        # Keep the original source keys, including legacy nested checkpoints.
        groups = group.items() if isinstance(key, tuple) and len(key) == 2 and isinstance(group, dict) else [(None, group)]
        for kind, item in groups:
            try:
                _valid_knn_group(item)
            except _RECOVERY_ERRORS as error:
                task.recovery_warnings.append(
                    f"KNN {key} {kind or ''}: {error}. Источник исключён из ML; требуется пересчитать KNN.")
                continue
            if kind is None:
                recovered[key] = item
            else:
                recovered.setdefault(key, {})[kind] = item
    task.knn_results = recovered
    labels = ['KNN-признаки'] if recovered else []
    for name, label in (('statistics', 'статистики'), ('synchronization', 'синхронизации')):
        value = payload.get(name + '_results', {})
        if isinstance(value, dict):
            setattr(task, name + '_results', value)
            if value:
                labels.append(label)
        else:
            task.recovery_warnings.append(f"Не удалось восстановить {label}: неверная структура данных.")
    return labels
