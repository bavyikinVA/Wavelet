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
from result_naming import dated_folder_name, ml_dataset_slug
from compute.knn.result_format import (
    neighbor_arrays, knn_payload_point_count, materialize_knn_payload,
)

from .errors import ClusteringError
from .validation import validate_ml_parameters
from .dbscan_global import global_dbscan


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
        count = knn_payload_point_count(payload)
        key = (direction, channel, float(scale), point_type)
        direction_name = (
            "CWT по строкам" if direction in {"rows", "row", "str", "0"}
            else "CWT по столбцам"
        )
        channel_name = {
            "red": "Красный", "green": "Зелёный", "blue": "Синий",
            "r": "Красный", "g": "Зелёный", "b": "Синий",
            "gray": "Gray", "gs1": "GS1", "gs2": "GS2", "gs3": "GS3",
        }.get(channel, channel)
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

# Only the approved KNN Info schema is used for clustering.  Metadata IDs
# identify relationships; they are not geometric distances and must not be
# supplied as continuous numeric features.
KNN_FEATURE_SCHEMA_VERSION = "knn_info_raw_v1"


def extract_knn_features(
    knn_results: dict,
    point_filter: str = "all",
    feature_set: str = "knn_raw",
    *,
    progress_callback: ProgressCallback | None = None,
    cancel_callback: CancelCallback | None = None,
) -> FeatureTable:
    """Create one observation per point from raw KNN Info columns.

    KNN Info stores ``point_id, x, y, neighbor_id, distance, angle_deg`` in
    LONG form (k edges per point).  Clustering needs ONE fixed-width row for
    each point, therefore its feature vector is
    ``[x, y, distance_1, angle_deg_1, ..., distance_k, angle_deg_k]``.
    ``point_id`` and each ``neighbor_id`` remain in the exported records for
    exact traceability, never as arbitrary numeric ML features.

    The KNN link order and individual angles/distances are left unchanged:
    no means, standard deviations, trigonometric substitutions or other
    derived descriptors.  Legacy values of ``feature_set`` are deliberately
    normalized to the one approved raw representation.
    """
    if feature_set not in {"knn_raw", "knn", "coordinates_knn", "coordinates"}:
        raise ClusteringError("Неизвестный набор признаков KNN")

    records = []
    vectors = []
    selected_types = set(_selected_point_types(point_filter))
    groups = [g for g in _iter_knn_groups(knn_results) if g[3] in selected_types]
    total_groups = max(1, len(groups))
    expected_k = None
    feature_names = []

    for group_index, (direction, channel, scale, point_type, payload) in enumerate(groups, 1):
        _check_cancel(cancel_callback)
        try:
            materialized = materialize_knn_payload(payload)
        except (OSError, KeyError, ValueError) as error:
            raise ClusteringError(f"Не удалось загрузить KNN-набор для ML: {error}") from error
        points = np.asarray(materialized.get("points", []), dtype=np.float64)
        if points.ndim != 2 or points.shape[1] < 2:
            continue
        indices, distances, angles = neighbor_arrays(
            materialized.get("neighbors", {}), point_count=len(points)
        )
        if angles is None:
            raise ClusteringError("В KNN отсутствуют исходные углы; необходим пересчёт KNN")
        if indices.shape != distances.shape or angles.shape != indices.shape or indices.shape[0] != len(points):
            raise ClusteringError("Размеры массивов KNN не совпадают")
        k = indices.shape[1]
        if expected_k is None:
            expected_k = k
            feature_names = ["x", "y"]
            for rank in range(1, k + 1):
                feature_names.extend((f"neighbor_{rank}_distance", f"neighbor_{rank}_angle_deg"))
        elif k != expected_k:
            raise ClusteringError("Невозможно объединить наборы KNN с разными значениями k")

        # Invalid neighbor slots may occur when fewer than k points exist.
        # They are not fabricated into a distance/angle pair.
        invalid = (indices < 0) | (indices >= len(points))
        if np.any(invalid):
            raise ClusteringError("KNN-набор содержит отсутствующих соседей; проверьте k")
        if not (np.all(np.isfinite(points[:, :2])) and np.all(np.isfinite(distances))
                and np.all(np.isfinite(angles))):
            raise ClusteringError("KNN-набор содержит NaN/Inf; анализ остановлен")

        batch = np.empty((len(points), 2 + 2 * k), dtype=np.float64)
        batch[:, :2] = points[:, :2]
        batch[:, 2::2] = distances
        batch[:, 3::2] = angles
        vectors.append(batch)
        for point_id, point in enumerate(points):
            if point_id % 1000 == 0:
                _check_cancel(cancel_callback)
            record = {
                "direction": direction,
                "channel": str(channel),
                "scale": _safe_number(scale),
                "point_type": point_type,
                "point_type_name": POINT_TYPE_LABELS.get(point_type, point_type),
                "point_id": int(point_id),
                "x": float(point[0]),
                "y": float(point[1]),
            }
            for j in range(k):
                rank = j + 1
                record[f"neighbor_{rank}_id"] = int(indices[point_id, j])
                record[f"neighbor_{rank}_distance"] = float(distances[point_id, j])
                record[f"neighbor_{rank}_angle_deg"] = float(angles[point_id, j])
            records.append(record)
        _notify(progress_callback, "features", group_index / total_groups,
                f"Подготовка исходных KNN-данных: {len(records):,} точек")

    if len(records) < 3:
        raise ClusteringError("Недостаточно KNN-точек в выбранном наборе")
    values = np.concatenate(vectors, axis=0)
    return FeatureTable(values=values, feature_names=feature_names, records=records)


def _raw_knn_columns(records):
    """Columns of the KNN Info source, in original neighbor-rank order."""
    columns = [("point_id", "Номер точки"), ("x", "X"), ("y", "Y")]
    if records:
        rank = 1
        while f"neighbor_{rank}_id" in records[0]:
            columns.extend((
                (f"neighbor_{rank}_id", f"Сосед {rank}: номер"),
                (f"neighbor_{rank}_distance", f"Сосед {rank}: расстояние"),
                (f"neighbor_{rank}_angle_deg", f"Сосед {rank}: угол (градусы)"),
            ))
            rank += 1
    return columns

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
    # Avoid quadratic cluster-vs-cluster memory for pathologically many labels.
    # Silhouette itself compares samples pairwise and is bounded separately.
    sample_size = min(int(silhouette_max_samples), len(evaluated_values))
    if 2 <= cluster_count < sample_size and sample_size >= 2:
        try:
            metrics["silhouette"] = float(silhouette_score(
                evaluated_values, evaluated_labels,
                sample_size=sample_size if sample_size < len(evaluated_values) else None,
                random_state=42))
            metrics["silhouette_sample_size"] = sample_size
        except ValueError:
            # Random subsamples may omit members of rare clusters; report N/A.
            pass
    if cluster_count <= 1000:
        metrics["davies_bouldin"] = float(
            davies_bouldin_score(evaluated_values, evaluated_labels))
    metrics["calinski_harabasz"] = float(
        calinski_harabasz_score(evaluated_values, evaluated_labels))
    return metrics


def _cluster_colors(labels: np.ndarray):
    from ml.color_scheme import categorical_ml_rgba
    return categorical_ml_rgba(labels)


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
        ("direction", "Направление"),
        *_raw_knn_columns(records),
    ]
    with open(output_path, "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[title for _, title in columns])
        writer.writeheader()
        for record, label in zip(records, labels):
            source = dict(record)
            source["cluster"] = int(label)
            writer.writerow({title: source.get(key) for key, title in columns})


def _cluster_summary(records, labels):
    """Spatial diagnostics, not replacement ML features."""
    summary = []
    for label in sorted(set(labels.tolist())):
        indices = np.flatnonzero(labels == label)
        selected = [records[int(index)] for index in indices]
        summary.append({
            "cluster": int(label),
            "count": int(len(selected)),
            "x_mean": float(np.mean([r["x"] for r in selected])),
            "y_mean": float(np.mean([r["y"] for r in selected])),
        })
    return summary


def _write_cluster_summary(summary, output_path):
    columns = [
        ("cluster", "Кластер"), ("count", "Количество точек"),
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
            ax.bar(x, values, bottom=bottom, label=f"Кластер {label}")
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



def _dataset_slug(records):
    if not records:
        return "dataset"
    r = records[0]
    return ml_dataset_slug(
        r.get("direction", "row"),
        r.get("channel", "r"),
        r.get("scale", 0),
        r.get("point_type", "max_by_row"),
    )

def run_clustering(
    knn_results: dict,
    algorithm: str,
    output_root: str,
    image=None,
    point_filter: str = "all",
    feature_set: str = "knn_raw",
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
    dbscan_device: str = "cpu",
    dbscan_batch_size: int = 512,
    dbscan_small_threshold: int = 2500,
) -> dict:
    """Cluster one selected KNN dataset with bounded-memory large-data paths."""
    parameters = validate_ml_parameters(dict(
        algorithm=algorithm, n_clusters=n_clusters, random_state=random_state,
        eps=eps, min_samples=min_samples, dbscan_batch_size=dbscan_batch_size))
    algorithm = parameters['algorithm']
    if algorithm == 'kmeans':
        n_clusters = parameters['n_clusters']
        random_state = parameters['random_state']
    else:
        eps = parameters['eps']
        min_samples = parameters['min_samples']
        dbscan_batch_size = parameters['dbscan_batch_size']
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
    validate_ml_parameters(parameters, point_count)

    _notify(progress_callback, "scaling", 0.0, f"Стандартизация {point_count:,} точек…")
    model_values = timed("standardization",
                         lambda: StandardScaler().fit_transform(values) if standardize else values)
    _notify(progress_callback, "scaling", 1.0, "Признаки подготовлены")
    _check_cancel(cancel_callback)

    dbscan_backend = None
    dbscan_fallback_reason = None
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
        if min_samples < 1:
            raise ClusteringError("Минимум точек DBSCAN должен быть не меньше 1")
        algorithm_name = "DBSCAN (глобальный)"
        _notify(progress_callback, "clustering", 0.0,
                f"DBSCAN: единый набор из {point_count:,} точек ({dbscan_device.upper()})…")
        labels, dbscan_backend, dbscan_fallback_reason = timed("clustering", lambda: global_dbscan(
            model_values, eps, min_samples, device=dbscan_device,
            batch_size=dbscan_batch_size, small_threshold=dbscan_small_threshold,
            progress_callback=progress_callback, cancel_callback=cancel_callback))
    else:
        raise ClusteringError("Неизвестный алгоритм кластеризации")

    _check_cancel(cancel_callback)
    _notify(progress_callback, "metrics", 0.0, "Расчёт метрик качества…")
    metrics = timed("quality_metrics", lambda: _quality_metrics(
        model_values, labels, silhouette_max_samples=silhouette_max_samples))
    _notify(progress_callback, "metrics", 1.0, "Метрики рассчитаны")

    timestamp = dated_folder_name(datetime.now())
    slug = _dataset_slug(table.records)
    # Only attribute an ML run to a scale when ALL its observations
    # belong to that scale. Mixed-scale datasets stay at task level.
    from result_naming import scale_folder_name
    dataset_scales = {str(r.get("scale")) for r in table.records if r.get("scale") is not None}
    if len(dataset_scales) == 1:
        scale_value = next(r.get("scale") for r in table.records if r.get("scale") is not None)
        output_dir = os.path.join(output_root, "scales", scale_folder_name(scale_value), "ml", slug, timestamp)
    else:
        output_dir = os.path.join(output_root, "ml", slug, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    prefix = "ml_dbscan" if algorithm == "dbscan" else "ml_kmeans"
    base = f"{prefix}_{slug}"
    summary_path = os.path.join(output_dir, f"{prefix}_summary_{slug}.csv")
    feature_plot_path = os.path.join(output_dir, f"{prefix}_features_{slug}.png")
    table_path = os.path.join(output_dir, base + ".csv")

    _notify(progress_callback, "saving", 0.0, "Сохранение ML-слоя…")
    save_started = time.perf_counter()
    _write_csv(table.records, labels, table_path)
    summary = _cluster_summary(table.records, labels)
    _write_cluster_summary(summary, summary_path)
    # Keep a compact feature-space diagnostic. Spatial overlays are rendered
    # natively by the Results viewer from the CSV layer.
    _notify(progress_callback, "saving", 0.55, "Построение пространства признаков…")
    if point_count <= 100000:
        _plot_feature_space(model_values, labels, algorithm_name, feature_plot_path)
    else:
        rng = np.random.default_rng(random_state)
        idx = np.sort(rng.choice(point_count, size=100000, replace=False))
        _plot_feature_space(model_values[idx], labels[idx], algorithm_name, feature_plot_path)
    timings["saving_and_visualization"] = time.perf_counter() - save_started
    _notify(progress_callback, "saving", 1.0, "ML-слой сохранён; откройте его в «Результатах»")

    counts = {int(label): int(np.sum(labels == label)) for label in sorted(set(labels.tolist()))}
    timings["total"] = float(sum(v for k, v in timings.items() if k != "total"))
    dataset = table.records[0] if table.records else {}
    return {
        "algorithm": model_kind, "algorithm_name": algorithm_name,
        "labels": labels, "records": table.records, "feature_names": table.feature_names,
        "feature_schema_version": KNN_FEATURE_SCHEMA_VERSION,
        "point_count": point_count, "metrics": metrics, "counts": counts,
        "cluster_summary": summary, "timings": timings, "output_dir": output_dir,
        "table_path": table_path, "summary_path": summary_path,
        "feature_plot_path": feature_plot_path,
        "image_plot_path": None, "scale_plot_path": None, "channel_plot_path": None,
        "noise_plot_path": None, "dbscan_backend": dbscan_backend,
        "dbscan_fallback_reason": dbscan_fallback_reason,
        "dataset": {k: dataset.get(k) for k in ("direction", "channel", "scale", "point_type", "point_type_name")},
        "dbscan_batch_size": int(dbscan_batch_size) if algorithm == "dbscan" else None,
    }
