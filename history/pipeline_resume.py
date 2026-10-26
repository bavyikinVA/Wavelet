"""Reusable artifacts for incremental execution inside one research task.

The module deliberately separates *scientific inputs* from requested downstream
stages.  If the CWT signature did not change, previously saved coefficients,
extrema and envelopes may be reused when the researcher enables a later stage.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np

from result_naming import cwt1d_stem, point_stem, scale_folder_name

CACHE_DIRNAME = ".pipeline_cache_v2"
CACHE_SCHEMA_VERSION = 2


def _normalise(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def cwt_signature_payload(task) -> dict:
    """Return only settings that can change 1D/2D wavelet coefficients."""
    image = getattr(task, "original_image", None)
    source_digest = None
    source_shape = None
    if image is not None:
        array = np.ascontiguousarray(image)
        source_shape = list(array.shape)
        source_digest = hashlib.sha256(array.view(np.uint8)).hexdigest()

    return {
        "analysis_mode": str(getattr(task, "analysis_mode", "1d")),
        "source_sha256": source_digest,
        "source_shape": source_shape,
        "scales": [float(x) for x in np.asarray(getattr(task, "scales", []), dtype=float)],
        "channel_representation": str(getattr(task, "channel_representation", "rgb")),
        "rgb_channel_mode": str(getattr(task, "rgb_channel_mode", "all")),
        "rgb_single_channel": str(getattr(task, "rgb_single_channel", "R")),
        "color1": None if getattr(task, "color1", None) is None else _normalise(np.asarray(task.color1)),
        "color2": None if getattr(task, "color2", None) is None else _normalise(np.asarray(task.color2)),
        "process_rows": bool(getattr(task, "process_rows", True)),
        "process_columns": bool(getattr(task, "process_columns", True)),
        "orientations": [float(x) for x in getattr(task, "orientations", [])],
        "morlet_omega0": float(getattr(task, "morlet_omega0", 6.0)),
        "morlet_anisotropy": float(getattr(task, "morlet_anisotropy", 1.0)),
    }


def cwt_signature(task) -> str:
    payload = cwt_signature_payload(task)
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def point_cache_path(task_folder, artifact, cwt_axis, feature_axis, channel, scale, kind) -> Path:
    stem = point_stem(artifact, cwt_axis, feature_axis, channel, scale, kind)
    return Path(task_folder) / CACHE_DIRNAME / "points" / f"{stem}.npy"


def save_point_cache(task_folder, artifact, cwt_axis, feature_axis, channel, scale, kind, points) -> Path:
    path = point_cache_path(task_folder, artifact, cwt_axis, feature_axis, channel, scale, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(points if points is not None else [], dtype=np.int32)
    if array.size == 0:
        array = np.empty((0, 2), dtype=np.int32)
    else:
        array = array.reshape(-1, 2)
    from history.atomic_io import atomic_binary_writer
    with atomic_binary_writer(path) as stream:
        np.save(stream, array, allow_pickle=False)
    return path


def _validated_points(data):
    array = np.asarray(data)
    if array.ndim != 2 or array.shape[1] != 2 or array.dtype.kind not in "iuf":
        return None
    if not np.isfinite(array).all() or (array < 0).any():
        return None
    if (array > np.iinfo(np.int32).max).any() or (array != np.floor(array)).any():
        return None
    return np.ascontiguousarray(array, dtype=np.int32)


def load_point_cache(task_folder, artifact, cwt_axis, feature_axis, channel, scale, kind, *, as_array=False):
    """Malformed canonical caches are missing, never silently empty results."""
    cache = point_cache_path(task_folder, artifact, cwt_axis, feature_axis, channel, scale, kind)
    stem = point_stem(artifact, cwt_axis, feature_axis, channel, scale, kind)
    scale_dir = Path(task_folder) / "scales" / scale_folder_name(scale)
    stage = "envelopes" if artifact == "env" else "extremes"
    candidates = [cache]
    for directory in (scale_dir / stage, scale_dir):
        candidates.extend([directory / f"{stem}.txt", directory / f"{stem}.npz"])
    for path in candidates:
        if not path.is_file():
            continue
        try:
            if path.suffix == ".txt":
                data = (np.empty((0, 2), dtype=np.int32) if path.stat().st_size == 0
                        else np.loadtxt(path, delimiter=",", ndmin=2))
            elif path.suffix == ".npz":
                with np.load(path, allow_pickle=False) as payload:
                    data = payload["points"]
            else:
                data = np.load(path, allow_pickle=False)
            points = _validated_points(data)
            return None if points is None else (points if as_array else points.tolist())
        except (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile):
            return None
    return None


def _expected_point_kinds(find_maxima: bool, find_minima: bool, *, envelope: bool):
    kinds = []
    if find_maxima:
        kinds.append("upper" if envelope else "max")
    if find_minima:
        kinds.append("lower" if envelope else "min")
    return kinds


def point_bundle_available(task, *, envelope: bool) -> bool:
    folder = getattr(task, "task_folder_path", "")
    if not folder:
        return False
    artifact = "env" if envelope else "ext"
    kinds = _expected_point_kinds(
        bool(getattr(task, "find_maxima", True)),
        bool(getattr(task, "find_minima", True)),
        envelope=envelope,
    )
    if not kinds:
        return False
    axes = []
    if getattr(task, "process_rows", True):
        axes.append("row")
    if getattr(task, "process_columns", True):
        axes.append("col")
    channels = list(getattr(task, "analysis_channel_codes", [])) or list(task.analysis_channel_codes_for_settings())
    if not axes or not channels or not len(task.scales):
        return False
    for cwt_axis in axes:
        for channel in channels:
            for scale in np.asarray(task.scales):
                for feature_axis in ("row", "col"):
                    for kind in kinds:
                        points = load_point_cache(folder, artifact, cwt_axis, feature_axis, channel, scale, kind, as_array=True)
                        if points is None:
                            return False
                        image = getattr(task, "original_image", None)
                        if image is not None and len(points):
                            height, width = image.shape[:2]
                            if (points[:, 0] >= width).any() or (points[:, 1] >= height).any():
                                return False
    return True


def load_complete_cwt(task, *, validate_only=False):
    """Restore the full selected 1D CWT from the lossless preview .npy files."""
    folder = Path(getattr(task, "task_folder_path", ""))
    if not folder.is_dir() or getattr(task, "analysis_mode", "1d") != "1d":
        return None

    axes = []
    if getattr(task, "process_rows", True):
        axes.append((0, "row"))
    if getattr(task, "process_columns", True):
        axes.append((1, "col"))
    channels = list(getattr(task, "analysis_channel_codes", [])) or list(task.analysis_channel_codes_for_settings())
    if not axes or not channels or not len(task.scales):
        return None
    image = getattr(task, "original_image", None)
    expected_shape = tuple(image.shape[:2]) if image is not None else None
    result = {}
    for type_data, cwt_axis in axes:
        per_channel = []
        for channel in channels:
            per_scale = []
            for scale in np.asarray(task.scales):
                stem = cwt1d_stem(cwt_axis, channel, scale)
                path = folder / "scales" / scale_folder_name(scale) / "wavelets" / f"{stem}.npy"
                if not path.is_file():
                    legacy = folder / "previews" / f"{stem}.npy"
                    path = legacy if legacy.is_file() else path
                if not path.is_file():
                    return None
                try:
                    array = np.load(path, allow_pickle=False, mmap_mode="r")
                    if (array.ndim != 2 or array.dtype.kind not in "fiu"
                            or (expected_shape is not None and array.shape != expected_shape)):
                        return None
                    if expected_shape is None:
                        expected_shape = array.shape
                    # Bounded temporary memory even for large coefficient matrices.
                    for start in range(0, len(array), 256):
                        if not np.isfinite(array[start:start + 256]).all():
                            return None
                    if not validate_only:
                        per_scale.append(np.array(array, dtype=np.float32))
                except (OSError, ValueError, EOFError):
                    return None
                finally:
                    if "array" in locals() and isinstance(array, np.memmap):
                        array._mmap.close()
            if not validate_only:
                per_channel.append(np.stack(per_scale, axis=0))
        if not validate_only:
            result[type_data] = np.stack(per_channel, axis=0)
    return True if validate_only else result
