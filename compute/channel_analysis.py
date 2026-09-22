"""Explicit image-channel selection for one research task.

The source RGB image is immutable.  Every run prepares a separate list of
analysis channels from the task's declared representation.  The same prepared
channels are then used by all downstream stages.
"""
from __future__ import annotations

from dataclasses import dataclass
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


def write_channel_manifest(task, task_folder: str | Path) -> Path:
    """Merge the channel methodology into ``manifest.json`` atomically."""
    folder = Path(task_folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "manifest.json"
    document: dict = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                document = loaded
        except (OSError, json.JSONDecodeError):
            # Preserve a malformed pre-existing manifest for diagnosis.
            backup = path.with_name("manifest.invalid.json")
            try:
                backup.write_bytes(path.read_bytes())
            except OSError:
                pass

    document["schema_version"] = max(2, int(document.get("schema_version", 1)))
    document["channel_analysis"] = task.channel_manifest()
    protocol = getattr(task, "execution_protocol", None)
    if protocol is not None:
        document["computation"] = protocol.to_dict()

    # Incremental execution metadata.  The signature intentionally excludes
    # downstream stage toggles, so enabling KNN after envelopes can reuse the
    # already computed CWT of the same task.
    try:
        from history.pipeline_resume import cwt_signature, cwt_signature_payload
        document["resume"] = {
            "cwt_signature": cwt_signature(task),
            "cwt_inputs": cwt_signature_payload(task),
            "reuse_existing_results": bool(
                getattr(task, "reuse_existing_results", True)
            ),
        }
    except Exception:
        # Manifest persistence must not turn an otherwise valid computation
        # into a failed run.
        pass

    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(path)
    return path
