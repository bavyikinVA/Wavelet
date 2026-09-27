"""Explicit image-channel selection for one research task.

The source RGB image is immutable.  Every run prepares a separate list of
analysis channels from the task's declared representation.  The same prepared
channels are then used by all downstream stages.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from Gram_Shmidt import change_channels


REPRESENTATION_RGB = "rgb"
REPRESENTATION_GRAYSCALE = "grayscale"
REPRESENTATION_GRAM_SCHMIDT = "gram_schmidt"

RGB_MODE_ALL = "all"
RGB_MODE_SINGLE = "single"

VALID_REPRESENTATIONS = {
    REPRESENTATION_RGB,
    REPRESENTATION_GRAYSCALE,
    REPRESENTATION_GRAM_SCHMIDT,
}
VALID_RGB_MODES = {RGB_MODE_ALL, RGB_MODE_SINGLE}
VALID_RGB_CHANNELS = {"R", "G", "B"}


@dataclass(frozen=True)
class PreparedChannel:
    key: str
    code: str
    label: str
    data: np.ndarray


def detect_color_type(
    image_rgb: np.ndarray,
    *,
    tolerance: float = 2.0,
    grayscale_ratio: float = 0.999,
) -> tuple[str, float]:
    """Return (``grayscale`` | ``color``, RGB-similarity ratio).

    This is advisory detection only.  The user remains free to choose any
    supported representation in the UI.
    """
    image = np.asarray(image_rgb)
    if image.ndim == 2:
        return "grayscale", 1.0
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(
            "Ожидалось изображение H×W×3 либо двумерное grayscale-изображение"
        )

    rgb = image[..., :3].astype(np.float64, copy=False)
    spread = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    similarity = float(np.mean(spread <= float(tolerance)))
    detected = "grayscale" if similarity >= float(grayscale_ratio) else "color"
    return detected, similarity


def rgb_to_grayscale(source_channels: Iterable[np.ndarray]) -> np.ndarray:
    """Convert RGB to Rec.709 luminance without intermediate rounding."""
    channels = [np.asarray(channel) for channel in source_channels]
    if len(channels) != 3:
        raise ValueError("Для RGB → Gray требуется ровно три исходных канала")
    r, g, b = (channel.astype(np.float64, copy=False) for channel in channels)
    if r.shape != g.shape or r.shape != b.shape:
        raise ValueError("Размеры исходных RGB-каналов не совпадают")
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def clear_prepared_channels(task) -> None:
    task.analysis_data = []
    task.analysis_channel_keys = []
    task.analysis_channel_codes = []
    task.analysis_channel_labels = []


def apply_detection_defaults(task, image_rgb: np.ndarray) -> tuple[str, float]:
    """Detect the source type and set a transparent default for a *new* image."""
    detected, similarity = detect_color_type(image_rgb)
    task.detected_color_type = detected
    task.grayscale_similarity = similarity

    if detected == "grayscale":
        task.channel_representation = REPRESENTATION_GRAYSCALE
    else:
        task.channel_representation = REPRESENTATION_RGB
    task.rgb_channel_mode = RGB_MODE_ALL
    task.rgb_single_channel = "R"
    task.gram_schmidt_applied = False
    clear_prepared_channels(task)
    return detected, similarity


def _source_rgb(task) -> list[np.ndarray]:
    # data_copy is the immutable source contract in the current application.
    source = getattr(task, "data_copy", None)
    if not source:
        source = getattr(task, "data", None)
    if source is None or len(source) != 3:
        raise ValueError("У задачи нет трёх исходных RGB-каналов")
    channels = [np.asarray(channel) for channel in source]
    if any(channel.ndim != 2 for channel in channels):
        raise ValueError("Каждый исходный канал должен быть двумерной матрицей")
    if not (channels[0].shape == channels[1].shape == channels[2].shape):
        raise ValueError("Размеры исходных RGB-каналов не совпадают")
    return channels


def prepare_task_channels(task) -> list[PreparedChannel]:
    """Build the exact channels used by the whole computational pipeline.

    ``task.data`` and ``task.data_copy`` are never modified here.
    """
    source = _source_rgb(task)
    representation = getattr(task, "channel_representation", REPRESENTATION_RGB)
    rgb_mode = getattr(task, "rgb_channel_mode", RGB_MODE_ALL)
    single = str(getattr(task, "rgb_single_channel", "R")).upper()

    if representation not in VALID_REPRESENTATIONS:
        raise ValueError(f"Неизвестное представление изображения: {representation}")

    if representation == REPRESENTATION_RGB:
        if rgb_mode not in VALID_RGB_MODES:
            raise ValueError(f"Неизвестный режим RGB-каналов: {rgb_mode}")
        metadata = [
            ("R", "r", "Красный (R)", 0),
            ("G", "g", "Зелёный (G)", 1),
            ("B", "b", "Синий (B)", 2),
        ]
        if rgb_mode == RGB_MODE_SINGLE:
            if single not in VALID_RGB_CHANNELS:
                raise ValueError(f"Неизвестный RGB-канал: {single}")
            metadata = [item for item in metadata if item[0] == single]
        prepared = [
            PreparedChannel(key, code, label, source[index])
            for key, code, label, index in metadata
        ]
    elif representation == REPRESENTATION_GRAYSCALE:
        prepared = [
            PreparedChannel(
                "GRAY", "gray", "Оттенки серого (Gray)",
                rgb_to_grayscale(source),
            )
        ]
    else:
        if not task.has_colors_selected():
            raise ValueError(
                "Для Gram–Schmidt сначала выберите два цветовых вектора пипеткой"
            )
        transformed = change_channels(
            np.asarray(task.color1),
            np.asarray(task.color2),
            [channel.copy() for channel in source],
        )
        transformed = np.asarray(transformed)
        if transformed.shape[0] != 3:
            raise ValueError("Преобразование Gram–Schmidt должно вернуть три канала")
        prepared = [
            PreparedChannel("GS1", "gs1", "Gram–Schmidt 1 (GS1)", transformed[0]),
            PreparedChannel("GS2", "gs2", "Gram–Schmidt 2 (GS2)", transformed[1]),
            PreparedChannel("GS3", "gs3", "Gram–Schmidt 3 (GS3)", transformed[2]),
        ]

    task.analysis_data = [np.asarray(item.data) for item in prepared]
    task.analysis_channel_keys = [item.key for item in prepared]
    task.analysis_channel_codes = [item.code for item in prepared]
    task.analysis_channel_labels = [item.label for item in prepared]
    task.gram_schmidt_applied = representation == REPRESENTATION_GRAM_SCHMIDT
    return prepared


def prepared_channel_meta(task) -> list[tuple[str, str, str]]:
    keys = list(getattr(task, "analysis_channel_keys", []))
    codes = list(getattr(task, "analysis_channel_codes", []))
    labels = list(getattr(task, "analysis_channel_labels", []))
    if not (keys and len(keys) == len(codes) == len(labels)):
        prepared = prepare_task_channels(task)
        return [(item.key, item.code, item.label) for item in prepared]
    return list(zip(keys, codes, labels))


def _json_value(value):
    """Convert common NumPy/container values into JSON-safe Python values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _source_manifest(task) -> dict:
    image = getattr(task, "original_image", None)
    if image is None:
        return {
            "stored_as": "image.png",
            "sha256": None,
            "shape": None,
            "dtype": None,
        }
    array = np.ascontiguousarray(image)
    return {
        "stored_as": "image.png",
        "sha256": hashlib.sha256(array.view(np.uint8)).hexdigest(),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def _pipeline_manifest(task) -> dict:
    resolve = getattr(task, "resolve_pipeline", None)
    if not callable(resolve):
        return {}
    plan = resolve()
    requested = {
        "extrema": bool(plan.extrema_requested),
        "envelopes": bool(plan.envelopes_requested),
        "knn": bool(plan.knn_requested),
        "statistics": bool(plan.statistics_requested),
        "synchronization": bool(plan.synchronization_requested),
    }
    resolved = {
        "wavelet": True,
        "extrema": bool(plan.extrema),
        "envelopes": bool(plan.envelopes),
        "knn": bool(plan.knn),
        "statistics": bool(plan.statistics),
        "synchronization": bool(plan.synchronization),
    }
    return {
        "preset": str(getattr(task, "pipeline_preset", "Пользовательский")),
        "requested": requested,
        "resolved": resolved,
        "automatic_reasons": plan.automatic_reasons(),
    }


def _scientific_parameters_manifest(task) -> dict:
    mode = str(getattr(task, "analysis_mode", "1d"))
    wavelet = {
        "mode": mode,
        "family": "morlet",
        "scales": [float(x) for x in np.asarray(getattr(task, "scales", []), dtype=float)],
    }
    if mode == "1d":
        wavelet.update({
            "directions": {
                "rows": bool(getattr(task, "process_rows", True)),
                "columns": bool(getattr(task, "process_columns", True)),
            },
            "output_dtype": "float32",
        })
    else:
        wavelet.update({
            "omega0": float(getattr(task, "morlet_omega0", 6.0)),
            "anisotropy": float(getattr(task, "morlet_anisotropy", 1.0)),
            "orientations_deg": [float(x) for x in getattr(task, "orientations", [])],
            "output_dtype": "complex64",
        })

    return {
        "wavelet": wavelet,
        "extrema": {
            "enabled_types": [
                name for name, enabled in (
                    ("max", bool(getattr(task, "find_maxima", True))),
                    ("min", bool(getattr(task, "find_minima", True))),
                ) if enabled
            ],
            "distance": int(getattr(task, "extrema_distance", 1)),
            "prominence": float(getattr(task, "extrema_prominence", 0.0)),
            "algorithm": "strict-local-extrema-distance-prominence-v2",
        },
        "envelopes": {
            "interpolation": "PCHIP",
            "algorithm": "pchip-grouped-v2",
            "boundary_policy": "constant-nearest-support",
            "meaning": "interpolational-envelope-through-extrema",
        },
        "knn": {
            "k_neighbors": int(getattr(task, "k_neighbors", 5)),
            "algorithm": "knn-angles-v1",
            "self_neighbor": "excluded",
            "distance": "euclidean",
            "angle_convention": "clockwise-from-vertical-up-image-coordinates",
        },
        "statistics": {
            "scale_block_sizes": [int(x) for x in getattr(task, "scale_block_sizes", [])],
        },
        "synchronization": {
            "row_stride": int(getattr(task, "row_sync_stride", 1)),
            "row_tolerance": int(getattr(task, "row_sync_tolerance", 1)),
            "metrics": list(getattr(task, "row_sync_metrics", [])),
        },
    }


def _exports_manifest(task) -> dict:
    return {
        "wavelet": {
            "txt": bool(getattr(task, "output_wavelet_text", False)),
            "png": bool(getattr(task, "output_wavelet_image", False)),
            "npy_legacy": bool(getattr(task, "output_wavelet_numpy", False)),
        },
        "extrema": {
            "txt": bool(getattr(task, "output_extremes_text", False)),
            "png": bool(getattr(task, "output_extremes_image", False)),
        },
        "envelopes": {
            "txt": bool(getattr(task, "output_envelopes_text", False)),
            "png": bool(getattr(task, "output_envelopes_image", False)),
        },
        "knn": {
            "npz": bool(getattr(task, "output_knn_npz", True)),
            "txt": bool(getattr(task, "output_knn_text", False)),
            "png": bool(getattr(task, "output_knn_image", False)),
            "png_max_points": int(getattr(task, "knn_png_max_points", 50000)),
        },
        "statistics": {
            "csv": bool(getattr(task, "statistics_output_csv", False)),
            "png": bool(getattr(task, "statistics_output_image", False)),
        },
        "synchronization": {
            "heatmap": bool(getattr(task, "synchronization_output_heatmap", False)),
            "matrix_csv": bool(getattr(task, "synchronization_output_matrix_csv", False)),
            "pairs_csv": bool(getattr(task, "synchronization_output_pairs_csv", False)),
        },
    }


def _stage_manifest(task) -> dict:
    try:
        from history.pipeline_state import stage_signatures, stage_statuses
        signatures = stage_signatures(task)
        statuses = stage_statuses(task)
        return {
            name: {"status": statuses[name], "signature": signatures[name]}
            for name in signatures
        }
    except Exception:
        return {}


def _execution_summary(protocol) -> dict:
    """Compact backend summary for the human-facing research manifest."""
    raw = protocol.to_dict()
    grouped: dict[str, dict] = {}
    for execution in raw.get("stages", {}).values():
        stage_name = str(execution.get("stage", ""))
        family = stage_name.split(":", 1)[0] or "unknown"
        item = grouped.setdefault(family, {
            "operations": 0,
            "actual_backends": set(),
            "dtypes": set(),
            "fallback_operations": 0,
            "algorithms": set(),
        })
        item["operations"] += 1
        if execution.get("actual_backend"):
            item["actual_backends"].add(str(execution["actual_backend"]))
        if execution.get("dtype"):
            item["dtypes"].add(str(execution["dtype"]))
        if execution.get("fallback"):
            item["fallback_operations"] += 1
        algorithm = (execution.get("details") or {}).get("algorithm")
        if algorithm:
            item["algorithms"].add(str(algorithm))

    stages = {}
    for family, item in grouped.items():
        stages[family] = {
            "operations": item["operations"],
            "actual_backends": sorted(item["actual_backends"]),
            "dtypes": sorted(item["dtypes"]),
            "fallback_operations": item["fallback_operations"],
            "algorithms": sorted(item["algorithms"]),
        }

    return {
        "requested_backend": raw.get("requested_backend"),
        "strict_backend": bool(raw.get("strict_backend", False)),
        "stages": stages,
        "warnings": raw.get("warnings", []),
    }


def _write_json_atomic(path: Path, document: dict) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(_json_value(document), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(path)


def write_channel_manifest(task, task_folder: str | Path) -> Path:
    """Write the research manifest and the detailed execution protocol atomically.

    Schema v4 keeps ``manifest.json`` compact and human-readable. Per-operation
    backend records are stored in ``execution_manifest.json`` so the audit trail
    remains complete without dominating the scientific passport.
    """
    folder = Path(task_folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "manifest.json"
    execution_path = folder / "execution_manifest.json"
    document: dict = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                document = loaded
        except (OSError, json.JSONDecodeError):
            backup = path.with_name("manifest.invalid.json")
            try:
                backup.write_bytes(path.read_bytes())
            except OSError:
                pass

    document["schema_version"] = 4
    document["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    document["task"] = {
        "id": _json_value(getattr(task, "task_id", None)),
        "name": str(getattr(task, "task_name", "")),
    }
    document["source"] = _source_manifest(task)
    document["channel_analysis"] = task.channel_manifest()
    document["pipeline"] = _pipeline_manifest(task)
    document["parameters"] = _scientific_parameters_manifest(task)
    document["exports"] = _exports_manifest(task)

    protocol = getattr(task, "execution_protocol", None)
    if protocol is not None:
        detailed_execution = protocol.to_dict()
        execution_document = {
            "schema_version": 1,
            "generated_at_utc": document["generated_at_utc"],
            "task": dict(document["task"]),
            **detailed_execution,
        }
        _write_json_atomic(execution_path, execution_document)
        document["computation"] = _execution_summary(protocol)
        document["execution"] = {
            "file": execution_path.name,
            "schema_version": 1,
            "operation_records": len(detailed_execution.get("stages", {})),
        }
    else:
        document.pop("computation", None)
        document.pop("execution", None)

    try:
        from history.pipeline_resume import cwt_signature, cwt_signature_payload
        document["resume"] = {
            "cwt_signature": cwt_signature(task),
            "cwt_inputs": cwt_signature_payload(task),
            "reuse_existing_results": bool(getattr(task, "reuse_existing_results", True)),
        }
    except Exception:
        pass

    document["stages"] = _stage_manifest(task)
    _write_json_atomic(path, document)
    return path

