"""Clustering of point features produced by the in-memory KNN stage.

This module deliberately has no GUI dependencies.  It accepts the structures
stored in ``ProcessingTask.knn_results`` and returns plain Python/NumPy data so
the calculations can be tested independently of Tkinter.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


POINT_TYPE_LABELS = {
    "max_by_row": "Максимумы по строкам",
    "min_by_row": "Минимумы по строкам",
    "max_by_column": "Максимумы по столбцам",
    "min_by_column": "Минимумы по столбцам",
}


class ClusteringError(ValueError):
    """A user-correctable clustering configuration or data error."""


@dataclass
class FeatureTable:
    values: np.ndarray
    feature_names: list[str]
    records: list[dict]


def _safe_number(value, default=0.0):
    try:
        number = float(value)
        return number if np.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _selected_point_types(point_filter: str) -> Iterable[str]:
    if point_filter == "all":
        return POINT_TYPE_LABELS.keys()
    if point_filter not in POINT_TYPE_LABELS:
        raise ClusteringError("Неизвестный тип точек для ML")
    return (point_filter,)


def extract_knn_features(
    knn_results: dict,
    point_filter: str = "all",
    feature_set: str = "knn",
) -> FeatureTable:
    """Build one feature vector per extreme point from KNN results.

    Angles are represented through mean sine/cosine, which avoids the false
    discontinuity between 359 and 0 degrees.
    """
    records = []
    point_types = tuple(_selected_point_types(point_filter))
    channel_order = {"red": 0.0, "green": 1.0, "blue": 2.0}

    for (channel, scale), scale_result in sorted(
        knn_results.items(), key=lambda item: (str(item[0][0]), float(item[0][1]))
    ):
        if not isinstance(scale_result, dict):
            continue
        for point_type in point_types:
            payload = scale_result.get(point_type)
            if not payload:
                continue
            points = np.asarray(payload.get("points", []), dtype=float)
            neighbors = payload.get("neighbors", {})
            if points.ndim != 2 or points.shape[1] < 2:
                continue
            for point_id, point in enumerate(points):
                neighbor = neighbors.get(point_id, neighbors.get(str(point_id), {}))
                distances = np.asarray(neighbor.get("distances", []), dtype=float)
                angles = np.asarray(neighbor.get("angles", []), dtype=float)
                distances = distances[np.isfinite(distances)]
                angles = angles[np.isfinite(angles)]
                radians = np.deg2rad(angles)
                mean_sin = float(np.mean(np.sin(radians))) if angles.size else 0.0
                mean_cos = float(np.mean(np.cos(radians))) if angles.size else 0.0
                concentration = float(np.hypot(mean_sin, mean_cos))
                records.append({
                    "channel": str(channel),
                    "channel_index": channel_order.get(str(channel), 0.0),
                    "scale": _safe_number(scale),
                    "point_type": point_type,
                    "point_type_name": POINT_TYPE_LABELS[point_type],
                    "point_id": int(point_id),
                    "x": _safe_number(point[0]),
                    "y": _safe_number(point[1]),
                    "distance_mean": float(np.mean(distances)) if distances.size else 0.0,
                    "distance_std": float(np.std(distances)) if distances.size else 0.0,
                    "angle_sin_mean": mean_sin,
                    "angle_cos_mean": mean_cos,
                    "angle_concentration": concentration,
                })

    if len(records) < 3:
        raise ClusteringError(
            "Недостаточно KNN-точек. Выполните 1D-анализ с включённым этапом KNN."
        )

    if feature_set == "knn":
        feature_names = [
            "distance_mean", "distance_std", "angle_sin_mean",
            "angle_cos_mean", "angle_concentration",
        ]
    elif feature_set == "coordinates_knn":
        feature_names = [
            "x", "y", "scale", "channel_index", "distance_mean",
            "distance_std", "angle_sin_mean", "angle_cos_mean",
            "angle_concentration",
        ]
    else:
        raise ClusteringError("Неизвестный набор признаков")

    values = np.asarray(
        [[record[name] for name in feature_names] for record in records],
        dtype=float,
    )
    if not np.all(np.isfinite(values)):
        raise ClusteringError("В признаках обнаружены некорректные значения")
    return FeatureTable(values=values, feature_names=feature_names, records=records)


def _quality_metrics(values: np.ndarray, labels: np.ndarray) -> dict:
    mask = labels != -1
    evaluated_values = values[mask]
    evaluated_labels = labels[mask]
    cluster_count = len(set(evaluated_labels.tolist()))
    metrics = {
        "cluster_count": cluster_count,
        "noise_count": int(np.sum(labels == -1)),
        "silhouette": None,
        "davies_bouldin": None,
        "calinski_harabasz": None,
    }
    if cluster_count < 2 or len(evaluated_values) <= cluster_count:
        return metrics
    metrics["silhouette"] = float(
        silhouette_score(evaluated_values, evaluated_labels)
    )
    metrics["davies_bouldin"] = float(
        davies_bouldin_score(evaluated_values, evaluated_labels)
    )
    metrics["calinski_harabasz"] = float(
        calinski_harabasz_score(evaluated_values, evaluated_labels)
    )
    return metrics


def _cluster_colors(labels: np.ndarray):
    unique = sorted(set(labels.tolist()))
    cmap = plt.get_cmap("tab20", max(len([x for x in unique if x >= 0]), 1))
    colors = []
    for label in labels:
        colors.append("#777777" if label == -1 else cmap(int(label) % cmap.N))
    return colors


def _plot_feature_space(values, labels, algorithm_name, output_path):
    projected = PCA(n_components=2, random_state=42).fit_transform(values)
    fig, ax = plt.subplots(figsize=(9, 7), dpi=130)
    ax.scatter(
        projected[:, 0], projected[:, 1], c=_cluster_colors(labels),
        s=22, alpha=0.82, linewidths=0.15, edgecolors="#202020",
    )
    ax.set_title(f"{algorithm_name}: кластеры в пространстве признаков")
    ax.set_xlabel("Главная компонента 1")
    ax.set_ylabel("Главная компонента 2")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_on_image(records, labels, image, algorithm_name, output_path):
    fig, ax = plt.subplots(figsize=(10, 8), dpi=130)
    if image is not None:
        ax.imshow(image)
    else:
        ax.set_facecolor("#303030")
        ax.invert_yaxis()
    x = np.asarray([record["x"] for record in records])
    y = np.asarray([record["y"] for record in records])
    ax.scatter(
        x, y, c=_cluster_colors(labels), s=18, alpha=0.78,
        linewidths=0.12, edgecolors="#111111",
    )
    ax.set_title(f"{algorithm_name}: расположение кластеров на изображении")
    ax.set_xlabel("X, пиксели")
    ax.set_ylabel("Y, пиксели")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _write_csv(records, labels, output_path):
    columns = [
        ("cluster", "Кластер"), ("channel", "Канал"),
        ("scale", "Масштаб"), ("point_type_name", "Тип точки"),
        ("point_id", "Номер точки"), ("x", "X"), ("y", "Y"),
        ("distance_mean", "Среднее расстояние KNN"),
        ("distance_std", "Разброс расстояний KNN"),
        ("angle_sin_mean", "Средний синус направления"),
        ("angle_cos_mean", "Средний косинус направления"),
        ("angle_concentration", "Согласованность направлений"),
    ]
    fieldnames = [title for _key, title in columns]
    with open(output_path, "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record, label in zip(records, labels):
            source = dict(record)
            source["cluster"] = int(label)
            row = {title: source.get(key) for key, title in columns}
            writer.writerow(row)


def run_clustering(
    knn_results: dict,
    algorithm: str,
    output_root: str,
    image=None,
    point_filter: str = "all",
    feature_set: str = "knn",
    standardize: bool = True,
    n_clusters: int = 5,
    random_state: int = 42,
    eps: float = 0.8,
    min_samples: int = 5,
) -> dict:
    """Run clustering, calculate quality metrics and save readable outputs."""
    table = extract_knn_features(knn_results, point_filter, feature_set)
    values = table.values
    model_values = StandardScaler().fit_transform(values) if standardize else values

    if algorithm == "kmeans":
        if n_clusters < 2 or n_clusters >= len(model_values):
            raise ClusteringError(
                "Для K-means число кластеров должно быть от 2 до числа точек минус 1"
            )
        model = KMeans(
            n_clusters=int(n_clusters), random_state=int(random_state), n_init=10
        )
        labels = model.fit_predict(model_values)
        algorithm_name = "K-means"
    elif algorithm == "dbscan":
        if eps <= 0:
            raise ClusteringError("Радиус DBSCAN должен быть больше нуля")
        if min_samples < 2:
            raise ClusteringError("Минимум точек DBSCAN должен быть не меньше 2")
        model = DBSCAN(eps=float(eps), min_samples=int(min_samples))
        labels = model.fit_predict(model_values)
        algorithm_name = "DBSCAN"
    else:
        raise ClusteringError("Неизвестный алгоритм кластеризации")

    metrics = _quality_metrics(model_values, labels)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = os.path.join(output_root, "ML_кластеризация", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    table_path = os.path.join(output_dir, "кластеры.csv")
    feature_plot_path = os.path.join(output_dir, "кластеры_признаки.png")
    image_plot_path = os.path.join(output_dir, "кластеры_изображение.png")
    _write_csv(table.records, labels, table_path)
    _plot_feature_space(model_values, labels, algorithm_name, feature_plot_path)
    _plot_on_image(table.records, labels, image, algorithm_name, image_plot_path)

    counts = {
        int(label): int(np.sum(labels == label))
        for label in sorted(set(labels.tolist()))
    }
    return {
        "algorithm": algorithm,
        "algorithm_name": algorithm_name,
        "labels": labels,
        "records": table.records,
        "feature_names": table.feature_names,
        "metrics": metrics,
        "counts": counts,
        "output_dir": output_dir,
        "table_path": table_path,
        "feature_plot_path": feature_plot_path,
        "image_plot_path": image_plot_path,
    }
