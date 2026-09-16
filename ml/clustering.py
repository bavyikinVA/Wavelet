"""Clustering of point features produced by the in-memory KNN stage.

The module has no GUI dependencies.  Long-running operations expose optional
progress and cancellation callbacks so a GUI can stay responsive and show the
real stage currently being executed.
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from .errors import ClusteringError


POINT_TYPE_LABELS = {
    "max_by_row": "Максимумы по строкам",
    "min_by_row": "Минимумы по строкам",
    "max_by_column": "Максимумы по столбцам",
    "min_by_column": "Минимумы по столбцам",
}

ProgressCallback = Callable[[str, float, str], None]
CancelCallback = Callable[[], None]


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


def _notify(callback: ProgressCallback | None, stage: str, value: float, message: str):
    if callback is not None:
        callback(stage, max(0.0, min(1.0, float(value))), message)


def _check_cancel(callback: CancelCallback | None):
    if callback is not None:
        callback()



def _iter_knn_groups(knn_results: dict):
    """Yield normalized KNN groups.

    Current format uses keys ``(direction, channel, scale, point_type)`` and a
    payload with ``points``/``neighbors``.  Legacy checkpoints with
    ``(channel, scale) -> {point_type: payload}`` remain readable.
    """
    for key, value in knn_results.items():
        if isinstance(key, tuple) and len(key) == 4:
            direction, channel, scale, point_type = key
            if isinstance(value, dict):
                yield str(direction), str(channel), float(scale), str(point_type), value
            continue
        if isinstance(key, tuple) and len(key) == 2 and isinstance(value, dict):
            channel, scale = key
            for point_type, payload in value.items():
                if point_type in POINT_TYPE_LABELS and isinstance(payload, dict):
                    # Legacy data did not retain Str/Col as a separate key.
                    direction = "rows" if point_type.endswith("_row") else "columns"
                    yield direction, str(channel), float(scale), str(point_type), payload


def knn_dataset_options(knn_results: dict):
    """Return stable descriptors used by the GUI to select one ML source."""
    result = []
    for direction, channel, scale, point_type, payload in _iter_knn_groups(knn_results):
        points = np.asarray(payload.get("points", []))
        count = int(len(points)) if points.ndim >= 1 else 0
        key = (direction, channel, float(scale), point_type)
        direction_name = "построчно" if direction in {"rows", "row", "str", "0"} else "по столбцам"
        channel_name = {"red": "Красный", "green": "Зелёный", "blue": "Синий"}.get(channel, channel)
        label = f"{channel_name} · a={float(scale):g} · {POINT_TYPE_LABELS.get(point_type, point_type)} · {direction_name} · {count:,} точек"
        result.append({"key": key, "label": label, "count": count})
    return sorted(result, key=lambda x: (x["key"][1], x["key"][2], x["key"][0], x["key"][3]))


def select_knn_dataset(knn_results: dict, dataset_key):
    """Return a one-group KNN mapping for clustering."""
    if dataset_key is None:
        return knn_results
    wanted = tuple(dataset_key)
    selected = {}
    for direction, channel, scale, point_type, payload in _iter_knn_groups(knn_results):
        key = (direction, channel, float(scale), point_type)
        if key == wanted:
            selected[key] = payload
            break
    if not selected:
        raise ClusteringError("Выбранный KNN-набор больше недоступен")
    return selected

def extract_knn_features(
    knn_results: dict,
    point_filter: str = "all",
    feature_set: str = "knn",
    *,
    progress_callback: ProgressCallback | None = None,
    cancel_callback: CancelCallback | None = None,
) -> FeatureTable:
    """Build one feature vector per point from one or more normalized KNN groups."""
    records = []
    selected_types = set(_selected_point_types(point_filter))
    channel_order = {"red": 0.0, "green": 1.0, "blue": 2.0}
    groups = [g for g in _iter_knn_groups(knn_results) if g[3] in selected_types]
    total_groups = max(1, len(groups))

    for group_index, (direction, channel, scale, point_type, payload) in enumerate(groups, 1):
        _check_cancel(cancel_callback)
        points = np.asarray(payload.get("points", []), dtype=float)
        neighbors = payload.get("neighbors", {})
        if points.ndim != 2 or points.shape[1] < 2:
            continue
        for point_id, point in enumerate(points):
            if point_id % 1000 == 0:
                _check_cancel(cancel_callback)
            neighbor = neighbors.get(point_id, neighbors.get(str(point_id), {}))
            distances = np.asarray(neighbor.get("distances", []), dtype=float)
            angles = np.asarray(neighbor.get("angles", []), dtype=float)
            distances = distances[np.isfinite(distances)]
            angles = angles[np.isfinite(angles)]
            radians = np.deg2rad(angles)
            mean_sin = float(np.mean(np.sin(radians))) if angles.size else 0.0
            mean_cos = float(np.mean(np.cos(radians))) if angles.size else 0.0
            records.append({
                "direction": direction,
                "channel": str(channel),
                "channel_index": channel_order.get(str(channel), 0.0),
                "scale": _safe_number(scale),
                "point_type": point_type,
                "point_type_name": POINT_TYPE_LABELS.get(point_type, point_type),
                "point_id": int(point_id),
                "x": _safe_number(point[0]),
                "y": _safe_number(point[1]),
                "distance_mean": float(np.mean(distances)) if distances.size else 0.0,
                "distance_std": float(np.std(distances)) if distances.size else 0.0,
                "angle_sin_mean": mean_sin,
                "angle_cos_mean": mean_cos,
                "angle_concentration": float(np.hypot(mean_sin, mean_cos)),
            })
        _notify(progress_callback, "features", group_index / total_groups,
                f"Подготовка признаков: {len(records):,} точек")

    if len(records) < 3:
        raise ClusteringError("Недостаточно KNN-точек в выбранном наборе")

    if feature_set == "knn":
        feature_names = ["distance_mean", "distance_std", "angle_sin_mean",
                         "angle_cos_mean", "angle_concentration"]
    elif feature_set == "coordinates_knn":
        feature_names = ["x", "y", "distance_mean", "distance_std",
                         "angle_sin_mean", "angle_cos_mean", "angle_concentration"]
    else:
        raise ClusteringError("Неизвестный набор признаков")

    values = np.asarray([[r[name] for name in feature_names] for r in records], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ClusteringError("В признаках обнаружены некорректные значения")
    return FeatureTable(values=values, feature_names=feature_names, records=records)

def _quality_metrics(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    silhouette_max_samples: int = 5000,
) -> dict:
    """Calculate clustering metrics without quadratic work on the full large set."""
    mask = labels != -1
    evaluated_values = values[mask]
    evaluated_labels = labels[mask]
    cluster_count = len(set(evaluated_labels.tolist()))
    metrics = {
        "cluster_count": cluster_count,
        "noise_count": int(np.sum(labels == -1)),
        "silhouette": None,
        "silhouette_sample_size": 0,
        "davies_bouldin": None,
        "calinski_harabasz": None,
    }
    if cluster_count < 2 or len(evaluated_values) <= cluster_count:
        return metrics

    sample_size = min(int(silhouette_max_samples), len(evaluated_values))
    metrics["silhouette"] = float(
        silhouette_score(
            evaluated_values,
            evaluated_labels,
            sample_size=sample_size if sample_size < len(evaluated_values) else None,
            random_state=42,
        )
    )
    metrics["silhouette_sample_size"] = sample_size
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
    return ["#777777" if label == -1 else cmap(int(label) % cmap.N) for label in labels]


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


def _plot_dbscan_noise(records, labels, image, output_path):
    fig, ax = plt.subplots(figsize=(10, 8), dpi=130)
    if image is not None:
        ax.imshow(image)
    else:
        ax.set_facecolor("#303030")
        ax.invert_yaxis()
    x = np.asarray([record["x"] for record in records])
    y = np.asarray([record["y"] for record in records])
    noise = labels == -1
    if np.any(~noise):
        ax.scatter(x[~noise], y[~noise], c="#8a8a8a", s=8, alpha=0.25, linewidths=0)
    if np.any(noise):
        ax.scatter(x[noise], y[noise], c="#d62728", s=24, alpha=0.9, linewidths=0.15,
                   edgecolors="#111111", label="Шум / потенциально необычные точки")
        ax.legend(loc="upper right")
    ax.set_title("DBSCAN: шумовые точки на изображении")
    ax.set_xlabel("X, пиксели")
    ax.set_ylabel("Y, пиксели")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _write_csv(records, labels, output_path, anomaly_score=None):
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
    if anomaly_score is not None:
        columns.append(("anomaly_score", "Аномальность DBSCAN"))
    fieldnames = [title for _key, title in columns]
    with open(output_path, "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, (record, label) in enumerate(zip(records, labels)):
            source = dict(record)
            source["cluster"] = int(label)
            if anomaly_score is not None:
                source["anomaly_score"] = float(anomaly_score[row_index])
            writer.writerow({title: source.get(key) for key, title in columns})



def _write_dbscan_csv(records, anomaly_score, output_path, anomaly_threshold=0.75):
    """Write a spatial DBSCAN layer without pretending tile-local ids are global clusters."""
    columns = [
        ("channel", "Канал"), ("scale", "Масштаб"),
        ("point_type_name", "Тип точки"), ("direction", "Направление"),
        ("point_id", "Номер точки"), ("x", "X"), ("y", "Y"),
        ("distance_mean", "Среднее расстояние KNN"),
        ("distance_std", "Разброс расстояний KNN"),
        ("angle_sin_mean", "Средний синус направления"),
        ("angle_cos_mean", "Средний косинус направления"),
        ("angle_concentration", "Согласованность направлений"),
        ("anomaly_score", "Аномальность DBSCAN"),
        ("dbscan_state", "Состояние DBSCAN"),
    ]
    with open(output_path, "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[title for _, title in columns])
        writer.writeheader()
        for record, score in zip(records, anomaly_score):
            source = dict(record)
            source["anomaly_score"] = float(score)
            if score >= anomaly_threshold:
                source["dbscan_state"] = "устойчивая аномалия"
            elif score > 0:
                source["dbscan_state"] = "неустойчивая область"
            else:
                source["dbscan_state"] = "плотная структура"
            writer.writerow({title: source.get(key) for key, title in columns})


def _dbscan_statistics(anomaly_score, *, point_count, tile_count, eps, min_samples,
                       anomaly_threshold=0.75):
    scores = np.asarray(anomaly_score, dtype=float)
    return {
        "points": int(point_count),
        "tiles": int(tile_count),
        "eps": float(eps),
        "min_samples": int(min_samples),
        "anomaly_mean": float(scores.mean()) if len(scores) else 0.0,
        "anomaly_nonzero": int(np.count_nonzero(scores > 0)),
        "anomaly_stable": int(np.count_nonzero(scores >= anomaly_threshold)),
        "anomaly_full": int(np.count_nonzero(scores >= 1.0 - 1e-12)),
        "anomaly_threshold": float(anomaly_threshold),
    }


def _write_dbscan_statistics(stats, output_path):
    columns = [
        ("points", "Всего точек"), ("tiles", "Количество тайлов"),
        ("eps", "eps"), ("min_samples", "min_samples"),
        ("anomaly_mean", "Средняя аномальность"),
        ("anomaly_nonzero", "Точек с аномальностью > 0"),
        ("anomaly_stable", "Устойчивых аномалий"),
        ("anomaly_full", "Точек с аномальностью = 1"),
        ("anomaly_threshold", "Порог устойчивой аномалии"),
    ]
    with open(output_path, "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[title for _, title in columns])
        writer.writeheader()
        writer.writerow({title: stats[key] for key, title in columns})


def _plot_anomaly_feature_space(values, anomaly_score, output_path):
    reduced = PCA(n_components=2, random_state=42).fit_transform(values)
    fig, ax = plt.subplots(figsize=(10, 7), dpi=130)
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=anomaly_score,
                         cmap="inferno", vmin=0.0, vmax=1.0, s=9, alpha=.72, linewidths=0)
    fig.colorbar(scatter, ax=ax, label="Аномальность DBSCAN")
    ax.set_title("DBSCAN: локальная аномальность в пространстве признаков")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.grid(alpha=.18)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

def _cluster_summary(records, labels):
    summary = []
    for label in sorted(set(labels.tolist())):
        indices = np.flatnonzero(labels == label)
        selected = [records[int(index)] for index in indices]
        summary.append({
            "cluster": int(label),
            "count": int(len(selected)),
            "distance_mean": float(np.mean([r["distance_mean"] for r in selected])),
            "distance_std_mean": float(np.mean([r["distance_std"] for r in selected])),
            "angle_concentration_mean": float(np.mean([r["angle_concentration"] for r in selected])),
            "x_mean": float(np.mean([r["x"] for r in selected])),
            "y_mean": float(np.mean([r["y"] for r in selected])),
        })
    return summary


def _write_cluster_summary(summary, output_path):
    columns = [
        ("cluster", "Кластер"), ("count", "Количество точек"),
        ("distance_mean", "Среднее KNN-расстояние"),
        ("distance_std_mean", "Средний разброс KNN-расстояний"),
        ("angle_concentration_mean", "Средняя согласованность направлений"),
        ("x_mean", "Средний X"), ("y_mean", "Средний Y"),
    ]
    with open(output_path, "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[title for _, title in columns])
        writer.writeheader()
        for row in summary:
            writer.writerow({title: row[key] for key, title in columns})


def _distribution_matrix(records, labels, key):
    categories = sorted({record[key] for record in records})
    cluster_labels = sorted({int(label) for label in labels if int(label) >= 0})
    matrix = np.zeros((len(categories), len(cluster_labels)), dtype=int)
    cat_index = {value: idx for idx, value in enumerate(categories)}
    cluster_index = {value: idx for idx, value in enumerate(cluster_labels)}
    for record, label in zip(records, labels):
        label = int(label)
        if label < 0:
            continue
        matrix[cat_index[record[key]], cluster_index[label]] += 1
    return categories, cluster_labels, matrix


def _plot_distribution(records, labels, key, title, xlabel, output_path):
    categories, cluster_labels, matrix = _distribution_matrix(records, labels, key)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=130)
    if cluster_labels and categories:
        x = np.arange(len(categories))
        bottom = np.zeros(len(categories), dtype=float)
        for index, label in enumerate(cluster_labels):
            values = matrix[:, index]
            ax.bar(x, values, bottom=bottom, label=f"Кластер {label + 1}")
            bottom += values
        ax.set_xticks(x)
        ax.set_xticklabels([f"{value:g}" if isinstance(value, (int, float, np.number)) else str(value)
                            for value in categories], rotation=35 if len(categories) > 8 else 0)
        ax.legend(ncol=min(4, max(1, len(cluster_labels))))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Количество точек")
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)



def _tiled_dbscan(values, records, eps, min_samples, *, tile_size=256, overlap=32,
                  max_points_per_tile=30000, progress_callback=None, cancel_callback=None):
    """Memory-bounded local DBSCAN over adaptive spatial tiles.

    The spatial domain is first split into regular cells and a dense cell is
    recursively divided until the expanded (overlapped) working set fits the
    configured point budget.  Thus every source point is analysed; there is no
    random sampling.  Overlap votes produce a stable anomaly score in [0, 1].
    """
    x = np.asarray([r["x"] for r in records], dtype=float)
    y = np.asarray([r["y"] for r in records], dtype=float)
    if len(values) == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=float), 0
    tile_size = max(32, int(tile_size))
    overlap = max(0, min(int(overlap), tile_size // 2))
    max_points_per_tile = max(1000, int(max_points_per_tile))
    xmin, xmax = int(np.floor(x.min())), int(np.ceil(x.max())) + 1
    ymin, ymax = int(np.floor(y.min())), int(np.ceil(y.max())) + 1

    pending = []
    for y0 in range(ymin, ymax, tile_size):
        for x0 in range(xmin, xmax, tile_size):
            pending.append((x0, min(x0 + tile_size, xmax), y0, min(y0 + tile_size, ymax)))
    cells = []
    while pending:
        _check_cancel(cancel_callback)
        x0, x1, y0, y1 = pending.pop()
        expanded = ((x >= x0-overlap) & (x < x1+overlap) &
                    (y >= y0-overlap) & (y < y1+overlap))
        count = int(np.count_nonzero(expanded))
        width, height = x1-x0, y1-y0
        if count > max_points_per_tile and max(width, height) > 32:
            if width >= height and width > 32:
                mid = (x0+x1)//2
                pending.extend([(x0, mid, y0, y1), (mid, x1, y0, y1)])
            elif height > 32:
                mid = (y0+y1)//2
                pending.extend([(x0, x1, y0, mid), (x0, x1, mid, y1)])
            else:
                cells.append((x0, x1, y0, y1))
        else:
            cells.append((x0, x1, y0, y1))

    labels = np.full(len(values), -1, dtype=int)
    noise_votes = np.zeros(len(values), dtype=np.float32)
    votes = np.zeros(len(values), dtype=np.int16)
    cluster_offset = 0
    total = max(1, len(cells))
    for tile_no, (x0, x1, y0, y1) in enumerate(cells, 1):
        _check_cancel(cancel_callback)
        expanded = ((x >= x0-overlap) & (x < x1+overlap) &
                    (y >= y0-overlap) & (y < y1+overlap))
        idx = np.flatnonzero(expanded)
        if len(idx) == 0:
            _notify(progress_callback, "clustering", tile_no/total,
                    f"DBSCAN: тайл {tile_no}/{total} — нет точек")
            continue
        if len(idx) > max_points_per_tile * 1.5:
            raise ClusteringError(
                f"Слишком плотный DBSCAN-тайл: {len(idx):,} точек. "
                "Уменьшите размер тайла или перекрытие."
            )
        try:
            local = DBSCAN(eps=float(eps), min_samples=int(min_samples)).fit_predict(values[idx])
        except MemoryError as error:
            raise ClusteringError(
                f"DBSCAN исчерпал память даже на локальном тайле ({len(idx):,} точек). "
                "Уменьшите размер тайла/перекрытие."
            ) from error
        votes[idx] += 1
        noise_votes[idx] += (local == -1)
        core = ((x[idx] >= x0) & (x[idx] < x1) & (y[idx] >= y0) & (y[idx] < y1))
        core_idx = idx[core]
        core_labels = local[core]
        positive = core_labels >= 0
        if np.any(positive):
            mapped = core_labels.copy()
            mapped[positive] += cluster_offset
            labels[core_idx] = mapped
            cluster_offset = int(mapped[positive].max()) + 1
        labels[core_idx[~positive]] = -1
        _notify(progress_callback, "clustering", tile_no/total,
                f"DBSCAN: тайл {tile_no}/{total} · {len(idx):,} точек")
    anomaly = np.divide(noise_votes, np.maximum(votes, 1), dtype=np.float32)
    return labels, anomaly, total

def _dataset_slug(records):
    if not records:
        return "dataset"
    r = records[0]
    channel = str(r.get("channel", "unknown")).replace(" ", "_")
    scale = f"{float(r.get('scale', 0)):g}"
    point_type = str(r.get("point_type", "points"))
    direction = str(r.get("direction", "direction"))
    return f"Scale_{scale}_Channel_{channel}_{point_type}_{direction}"

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
    *,
    dataset_key=None,
    progress_callback: ProgressCallback | None = None,
    cancel_callback: CancelCallback | None = None,
    silhouette_max_samples: int = 5000,
    minibatch_threshold: int = 50000,
    minibatch_size: int = 4096,
    dbscan_tile_size: int = 256,
    dbscan_tile_overlap: int = 32,
    dbscan_max_points_per_tile: int = 30000,
) -> dict:
    """Cluster one selected KNN dataset with bounded-memory large-data paths."""
    timings = {}
    def timed(name, fn):
        started = time.perf_counter()
        try:
            return fn()
        finally:
            timings[name] = time.perf_counter() - started

    selected = select_knn_dataset(knn_results, dataset_key)
    _check_cancel(cancel_callback)
    _notify(progress_callback, "features", 0.0, "Подготовка ML-признаков…")
    table = timed("feature_extraction", lambda: extract_knn_features(
        selected, point_filter, feature_set,
        progress_callback=progress_callback, cancel_callback=cancel_callback))
    values = table.values
    point_count = len(values)

    _notify(progress_callback, "scaling", 0.0, f"Стандартизация {point_count:,} точек…")
    model_values = timed("standardization",
                         lambda: StandardScaler().fit_transform(values) if standardize else values)
    _notify(progress_callback, "scaling", 1.0, "Признаки подготовлены")
    _check_cancel(cancel_callback)

    anomaly_score = None
    tiled = False
    model_kind = algorithm
    if algorithm == "kmeans":
        if n_clusters < 2 or n_clusters >= point_count:
            raise ClusteringError("Для K-means число кластеров должно быть от 2 до числа точек минус 1")
        if point_count >= int(minibatch_threshold):
            model = MiniBatchKMeans(n_clusters=int(n_clusters), random_state=int(random_state),
                                    batch_size=int(minibatch_size), n_init="auto")
            algorithm_name = "MiniBatch K-means"
        else:
            model = KMeans(n_clusters=int(n_clusters), random_state=int(random_state), n_init=10)
            algorithm_name = "K-means"
        _notify(progress_callback, "clustering", 0.0,
                f"{algorithm_name}: кластеризация {point_count:,} точек…")
        labels = timed("clustering", lambda: model.fit_predict(model_values))
        _notify(progress_callback, "clustering", 1.0, f"{algorithm_name}: кластеризация завершена")
    elif algorithm == "dbscan":
        if eps <= 0:
            raise ClusteringError("Радиус DBSCAN должен быть больше нуля")
        if min_samples < 2:
            raise ClusteringError("Минимум точек DBSCAN должен быть не меньше 2")
        algorithm_name = "DBSCAN (локальный сеточный)"
        tiled = True
        _notify(progress_callback, "clustering", 0.0,
                f"DBSCAN: сеточная обработка {point_count:,} точек…")
        def execute():
            return _tiled_dbscan(model_values, table.records, eps, min_samples,
                                 tile_size=dbscan_tile_size, overlap=dbscan_tile_overlap,
                                 max_points_per_tile=dbscan_max_points_per_tile,
                                 progress_callback=progress_callback, cancel_callback=cancel_callback)
        labels, anomaly_score, tile_count = timed("clustering", execute)
        _notify(progress_callback, "clustering", 1.0,
                f"DBSCAN: обработано тайлов: {tile_count}")
    else:
        raise ClusteringError("Неизвестный алгоритм кластеризации")

    _check_cancel(cancel_callback)
    _notify(progress_callback, "metrics", 0.0, "Расчёт метрик качества…")
    if tiled:
        # Local tile ids are deliberately not global clusters; cluster-shape
        # metrics across tiles would be misleading.
        metrics = {
            "cluster_count": int(len(set(labels[labels >= 0].tolist()))),
            "noise_count": int(np.sum(labels == -1)),
            "silhouette": None, "silhouette_sample_size": 0,
            "davies_bouldin": None, "calinski_harabasz": None,
            "anomaly_mean": float(np.mean(anomaly_score)) if anomaly_score is not None else None,
        }
        timings["quality_metrics"] = 0.0
    else:
        metrics = timed("quality_metrics", lambda: _quality_metrics(
            model_values, labels, silhouette_max_samples=silhouette_max_samples))
    _notify(progress_callback, "metrics", 1.0, "Метрики рассчитаны")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    slug = _dataset_slug(table.records)
    output_dir = os.path.join(output_root, "ML_кластеризация", slug, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    if tiled:
        base = f"ML_DBSCAN_аномальность_{slug}"
        summary_path = os.path.join(output_dir, "статистика_DBSCAN.csv")
        feature_plot_path = os.path.join(output_dir, "DBSCAN_аномальность_признаки.png")
    else:
        base = f"ML_KMeans_кластеры_{slug}"
        summary_path = os.path.join(output_dir, "характеристики_кластеров.csv")
        feature_plot_path = os.path.join(output_dir, "кластеры_признаки.png")
    table_path = os.path.join(output_dir, base + ".csv")

    _notify(progress_callback, "saving", 0.0, "Сохранение ML-слоя…")
    save_started = time.perf_counter()
    if tiled:
        _write_dbscan_csv(table.records, anomaly_score, table_path)
        dbscan_stats = _dbscan_statistics(
            anomaly_score, point_count=point_count, tile_count=tile_count,
            eps=eps, min_samples=min_samples)
        _write_dbscan_statistics(dbscan_stats, summary_path)
        summary = []
    else:
        _write_csv(table.records, labels, table_path)
        summary = _cluster_summary(table.records, labels)
        _write_cluster_summary(summary, summary_path)
        dbscan_stats = None
    # Keep a compact feature-space diagnostic. Spatial overlays are rendered
    # natively by the Results viewer from the CSV layer.
    _notify(progress_callback, "saving", 0.55, "Построение пространства признаков…")
    if point_count <= 100000:
        if tiled:
            _plot_anomaly_feature_space(model_values, anomaly_score, feature_plot_path)
        else:
            _plot_feature_space(model_values, labels, algorithm_name, feature_plot_path)
    else:
        rng = np.random.default_rng(random_state)
        idx = np.sort(rng.choice(point_count, size=100000, replace=False))
        if tiled:
            _plot_anomaly_feature_space(model_values[idx], anomaly_score[idx], feature_plot_path)
        else:
            _plot_feature_space(model_values[idx], labels[idx], algorithm_name, feature_plot_path)
    timings["saving_and_visualization"] = time.perf_counter() - save_started
    _notify(progress_callback, "saving", 1.0, "ML-слой сохранён; откройте его в «Результатах»")

    counts = {int(label): int(np.sum(labels == label)) for label in sorted(set(labels.tolist()))}
    timings["total"] = float(sum(v for k, v in timings.items() if k != "total"))
    dataset = table.records[0] if table.records else {}
    return {
        "algorithm": model_kind, "algorithm_name": algorithm_name,
        "labels": labels, "records": table.records, "feature_names": table.feature_names,
        "point_count": point_count, "metrics": metrics, "counts": counts,
        "cluster_summary": summary, "timings": timings, "output_dir": output_dir,
        "table_path": table_path, "summary_path": summary_path,
        "feature_plot_path": feature_plot_path,
        "image_plot_path": None, "scale_plot_path": None, "channel_plot_path": None,
        "noise_plot_path": None, "anomaly_score": anomaly_score,
        "dbscan_statistics": dbscan_stats,
        "dataset": {k: dataset.get(k) for k in ("direction", "channel", "scale", "point_type", "point_type_name")},
        "tiled": tiled,
        "dbscan_tile_size": int(dbscan_tile_size) if tiled else None,
        "dbscan_tile_overlap": int(dbscan_tile_overlap) if tiled else None,
        "dbscan_max_points_per_tile": int(dbscan_max_points_per_tile) if tiled else None,
    }

