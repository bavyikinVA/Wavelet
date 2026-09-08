"""Internal reusable checkpoints for researcher runs.

Only application-created files from a run result directory are loaded. Raw
wavelet tensors are deliberately excluded here because they can be many
gigabytes for large images; this checkpoint targets inexpensive repeated ML.
"""

import gzip
import os
import pickle


CHECKPOINT_FILENAME = "Контрольная_точка_признаков.pkl.gz"
CHECKPOINT_VERSION = 1


def checkpoint_path(output_dir):
    return os.path.join(output_dir, CHECKPOINT_FILENAME)


def feature_checkpoint_available(output_dir):
    return bool(output_dir and os.path.isfile(checkpoint_path(output_dir)))


def save_feature_checkpoint(task):
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
    if not any(payload["completed"].values()):
        return ""
    path = checkpoint_path(task.task_folder_path)
    with gzip.open(path, "wb", compresslevel=5) as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def restore_feature_checkpoint(task, output_dir):
    path = checkpoint_path(output_dir)
    if not os.path.isfile(path):
        return []
    with gzip.open(path, "rb") as stream:
        payload = pickle.load(stream)
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError("Версия контрольной точки не поддерживается")
    task.knn_results = payload.get("knn_results", {})
    task.statistics_results = payload.get("statistics_results", {})
    task.synchronization_results = payload.get("synchronization_results", {})
    task.task_folder_path = output_dir
    labels = {
        "knn": "KNN-признаки",
        "statistics": "статистики",
        "synchronization": "синхронизации",
    }
    return [
        labels[name] for name, ready in payload.get("completed", {}).items()
        if ready and name in labels
    ]
