"""Stable ASCII naming contract for result files and directories."""

from __future__ import annotations

import math
import re
from datetime import datetime


CHANNEL_CODES = ("r", "g", "b", "gray", "gs1", "gs2", "gs3")
VALID_AXES = {"row", "col"}


def number_token(value) -> str:
    """Encode a finite number without filesystem-sensitive punctuation."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite result parameter: {value}")
    text = f"{number:.12g}".lower()
    return text.replace("-", "m").replace("+", "").replace(".", "p")


def number_from_token(token: str) -> str:
    """Decode a filename number token to a compact display value."""
    text = str(token).lower()
    negative = text.startswith("m")
    if negative:
        text = text[1:]
    text = text.replace("p", ".").replace("em", "e-")
    try:
        number = float(("-" if negative else "") + text)
    except ValueError:
        return str(token)
    return f"{number:g}"


def safe_slug(value, fallback="item") -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")
    return text or fallback


def _axis(value: str) -> str:
    value = str(value).casefold()
    value = {
        "rows": "row", "columns": "col", "column": "col",
        "str": "row", "tr": "col", "0": "row", "1": "col",
    }.get(value, value)
    if value not in VALID_AXES:
        raise ValueError(f"Unknown axis: {value}")
    return value


def _channel(value) -> str:
    if isinstance(value, int):
        return CHANNEL_CODES[value]
    value = str(value).casefold()
    aliases = {
        "red": "r", "green": "g", "blue": "b",
        "красный": "r", "зелёный": "g", "зеленый": "g", "синий": "b",
        "grayscale": "gray", "grey": "gray", "серый": "gray",
        "оттенки серого": "gray",
        "gram-schmidt 1": "gs1", "gram-schmidt 2": "gs2",
        "gram-schmidt 3": "gs3",
    }
    value = aliases.get(value, value)
    if value not in CHANNEL_CODES:
        raise ValueError(f"Unknown channel: {value}")
    return value


def scale_folder_name(scale) -> str:
    return f"scale_{number_token(scale)}"


def run_root_folder_name(moment: datetime) -> str:
    """Return a readable, filesystem-safe folder name for one studio run."""
    return f"wavelet_studio_{moment.strftime('%d_%m_%Y_%H_%M')}"


def task_run_folder_name(task_id, moment: datetime, unique_suffix: str) -> str:
    """Return a collision-safe folder name for a task within one run."""
    timestamp = moment.strftime("%d_%m_%Y_%H_%M_%S")
    milliseconds = moment.microsecond // 1000
    return (
        f"task_{int(task_id):03d}_{timestamp}_{milliseconds:03d}_"
        f"{safe_slug(unique_suffix)}"
    )


def dated_folder_name(moment: datetime) -> str:
    """Return a readable timestamp for nested result folders."""
    return moment.strftime("%d_%m_%Y_%H_%M_%S_%f")


def source_channel_name(channel) -> str:
    return f"source_channel_{_channel(channel)}"


def centering_mean_name(cwt_axis, channel) -> str:
    return f"centering_mean_{_axis(cwt_axis)}_{_channel(channel)}"


def cwt1d_stem(cwt_axis, channel, scale) -> str:
    return f"cwt1d_{_axis(cwt_axis)}_{_channel(channel)}_s{number_token(scale)}"


def cwt2d_stem(channel, scale, angle) -> str:
    return (
        f"cwt2d_xy_{_channel(channel)}_s{number_token(scale)}_"
        f"a{number_token(angle)}"
    )


def point_stem(artifact, cwt_axis, feature_axis, channel, scale, point_kind) -> str:
    artifact = str(artifact).casefold()
    if artifact not in {"ext", "env"}:
        raise ValueError(f"Unknown point artifact: {artifact}")
    point_kind = safe_slug(point_kind, "points")
    return (
        f"{artifact}_cwt{_axis(cwt_axis)}_axis{_axis(feature_axis)}_"
        f"{_channel(channel)}_s{number_token(scale)}_{point_kind}"
    )


def knn_stem(cwt_axis, feature_axis, channel, scale, point_kind, k) -> str:
    return (
        f"knn_cwt{_axis(cwt_axis)}_axis{_axis(feature_axis)}_"
        f"{_channel(channel)}_s{number_token(scale)}_"
        f"{safe_slug(point_kind, 'points')}_k{int(k)}"
    )


def statistics_stem(
        cwt_axis, feature_axis, channel, coordinate_axis, coordinate,
        block_size=None) -> str:
    stem = (
        f"stat_cwt{_axis(cwt_axis)}_axis{_axis(feature_axis)}_"
        f"{_channel(channel)}_{coordinate_axis}{int(coordinate)}"
    )
    if block_size is not None:
        stem += f"_block{int(block_size)}"
    return stem


def synchronization_prefix(cwt_axis, channel) -> str:
    return f"sync_cwt{_axis(cwt_axis)}_axisrow_{_channel(channel)}"


def point_type_parts(point_type: str) -> tuple[str, str]:
    mapping = {
        "max_by_row": ("row", "max"),
        "min_by_row": ("row", "min"),
        "max_by_column": ("col", "max"),
        "min_by_column": ("col", "min"),
    }
    try:
        return mapping[str(point_type)]
    except KeyError as error:
        raise ValueError(f"Unknown point type: {point_type}") from error


def ml_dataset_slug(cwt_axis, channel, scale, point_type) -> str:
    feature_axis, point_kind = point_type_parts(point_type)
    return (
        f"cwt{_axis(cwt_axis)}_axis{feature_axis}_{_channel(channel)}_"
        f"s{number_token(scale)}_{point_kind}"
    )
