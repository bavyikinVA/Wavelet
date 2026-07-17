"""
Запуск:

    python visualize_wavelet_sync_results.py "C:\\Users\\...\\Задача 1 04_06_2026_12_58"

или если нужно явно указать папку результатов:

    python visualize_wavelet_sync_results.py "C:\\Users\\...\\Задача 1" --results-folder WaveletSyncResults

Результаты сохраняются в:

    WaveletSyncResults/05_visualizations
"""

from __future__ import annotations
import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHANNEL_ORDER = {
    "Red": 0,
    "Green": 1,
    "Blue": 2,
    "R": 0,
    "G": 1,
    "B": 2,
    "Красный": 0,
    "Зелёный": 1,
    "Зеленый": 1,
    "Синий": 2,
}

def safe_name(text: str) -> str:
    return (
        str(text)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(";", "_")
        .replace(",", "_")
        .replace("=", "_")
        .replace("|", "_")
        .replace("(", "_")
        .replace(")", "_")
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_matrix_txt(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data[None, :]
    return data


def save_heatmap(
    matrix: np.ndarray,
    output_path: Path,
    title: str,
    xlabel: str = "X",
    ylabel: str = "Y",
    robust: bool = True,
) -> None:
    """
    Сохраняет 2D-матрицу как тепловую карту.

    robust=True обрезает отображение по 1 и 99 процентилю,
    чтобы единичные огромные значения не портили весь рисунок.
    """

    ensure_dir(output_path.parent)

    arr = np.asarray(matrix, dtype=float)

    if arr.size == 0:
        return

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return

    if robust:
        vmin = float(np.percentile(finite, 1))
        vmax = float(np.percentile(finite, 99))
        if vmin == vmax:
            vmin = float(np.min(finite))
            vmax = float(np.max(finite))
    else:
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))

    fig, ax = plt.subplots(figsize=(9, 7))

    if vmin != vmax:
        im = ax.imshow(arr, interpolation="nearest", aspect="auto", vmin=vmin, vmax=vmax)
    else:
        im = ax.imshow(arr, interpolation="nearest", aspect="auto")

    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_float(value: object, default: float = math.nan) -> float:
    try:
        if value is None:
            return default
        text = str(value).strip().replace(",", ".")
        if text == "":
            return default
        return float(text)
    except Exception:
        return default


def parse_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return default
        text = str(value).strip()
        if text == "":
            return default
        return int(float(text.replace(",", ".")))
    except Exception:
        return default

def visualize_matrix_folder(
    source_dir: Path,
    output_dir: Path,
    title_prefix: str,
    max_files: int) -> int:
    if not source_dir.exists():
        print(f"Папка не найдена: {source_dir}")
        return 0

    files = sorted(source_dir.rglob("*.txt"))
    if max_files > 0:
        files = files[:max_files]

    count = 0

    for file_path in files:
        try:
            matrix = read_matrix_txt(file_path)

            rel = file_path.relative_to(source_dir)
            out_path = output_dir / rel.with_suffix(".png")
            title = f"{title_prefix}: {file_path.stem}"

            save_heatmap(
                matrix=matrix,
                output_path=out_path,
                title=title,
                xlabel="координата X",
                ylabel="координата Y",
                robust=True,
            )
            count += 1

        except Exception as exc:
            print(f"Не удалось визуализировать {file_path}: {exc}")

    return count

EVENT_RE = re.compile(
    r"^Events_(?P<direction>rows|cols)_Channel_(?P<channel>[^_]+)_Scale_(?P<scale>[-+]?\d+(?:\.\d+)?)\.txt$",
    re.IGNORECASE,
)


def normalize_channel_name(channel: str) -> str:
    ch = str(channel).strip()
    aliases = {
        "red": "Red",
        "r": "Red",
        "красный": "Red",
        "green": "Green",
        "g": "Green",
        "зелёный": "Green",
        "зеленый": "Green",
        "blue": "Blue",
        "b": "Blue",
        "синий": "Blue",
    }
    return aliases.get(ch.lower(), ch)


def channel_short(channel: str) -> str:
    ch = normalize_channel_name(channel)
    return {"Red": "R", "Green": "G", "Blue": "B"}.get(ch, ch)


def channel_ru(channel: str) -> str:
    ch = normalize_channel_name(channel)
    return {
        "Red": "Красный канал R",
        "Green": "Зелёный канал G",
        "Blue": "Синий канал B",
    }.get(ch, ch)


def parse_event_files(event_dir: Path) -> List[Tuple[str, float, str, Path]]:
    parsed_files: List[Tuple[str, float, str, Path]] = []

    for file_path in sorted(event_dir.glob("Events_*.txt")):
        match = EVENT_RE.match(file_path.name)
        if not match:
            continue

        channel = normalize_channel_name(match.group("channel"))
        scale = match.group("scale")
        parsed_files.append((channel, float(scale), scale, file_path))

    channel_order = {"Red": 0, "Green": 1, "Blue": 2}
    parsed_files.sort(key=lambda x: (channel_order.get(x[0], 99), x[1]))
    return parsed_files


def load_events_for_direction(event_dir: Path) -> Tuple[np.ndarray, List[str]]:
    rows: List[np.ndarray] = []
    labels: List[str] = []
    parsed_files = parse_event_files(event_dir)
    expected_len: Optional[int] = None

    for channel, _, scale_label, file_path in parsed_files:
        data = read_matrix_txt(file_path).astype(int).ravel()

        if expected_len is None:
            expected_len = len(data)
        elif len(data) != expected_len:
            print(f"Пропуск {file_path.name}: длина {len(data)} != {expected_len}")
            continue

        rows.append(data)
        labels.append(f"{channel_short(channel)}, scale={scale_label}")

    if not rows:
        return np.empty((0, 0), dtype=int), []

    return np.vstack(rows), labels


def load_events_by_channel(event_dir: Path) -> Dict[str, Tuple[np.ndarray, List[str]]]:
    parsed_files = parse_event_files(event_dir)
    grouped: Dict[str, List[Tuple[float, str, Path]]] = defaultdict(list)

    for channel, scale_float, scale_label, file_path in parsed_files:
        grouped[channel].append((scale_float, scale_label, file_path))

    result: Dict[str, Tuple[np.ndarray, List[str]]] = {}

    for channel in ["Red", "Green", "Blue"]:
        items = sorted(grouped.get(channel, []), key=lambda x: x[0])
        rows: List[np.ndarray] = []
        labels: List[str] = []
        expected_len: Optional[int] = None

        for _, scale_label, file_path in items:
            data = read_matrix_txt(file_path).astype(int).ravel()

            if expected_len is None:
                expected_len = len(data)
            elif len(data) != expected_len:
                print(f"Пропуск {file_path.name}: длина {len(data)} != {expected_len}")
                continue

            rows.append(data)
            labels.append(f"scale={scale_label}")

        if rows:
            result[channel] = (np.vstack(rows), labels)

    return result


def save_event_raster(
    raster: np.ndarray,
    labels: Sequence[str],
    output_path: Path,
    title: str,
) -> None:
    ensure_dir(output_path.parent)

    if raster.size == 0:
        return

    fig_height = max(5, min(18, 0.45 * len(labels) + 3))
    fig, ax = plt.subplots(figsize=(13, fig_height))

    im = ax.imshow(raster, interpolation="nearest", aspect="auto")

    ax.set_title(title)
    ax.set_xlabel("пространственная координата t")
    ax.set_ylabel("канал / масштаб")

    if len(labels) <= 80:
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
    else:
        ax.set_yticks([])

    fig.colorbar(im, ax=ax, label="событие: 0 / 1")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_event_activity_plot(
    raster: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    ensure_dir(output_path.parent)

    if raster.size == 0:
        return

    activity = np.sum(raster, axis=0)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(np.arange(len(activity)), activity)
    ax.set_title(title)
    ax.set_xlabel("пространственная координата t")
    ax.set_ylabel("количество активных рядов")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def visualize_binary_events(results_root: Path, output_root: Path) -> int:
    event_root = results_root / "03_binary_events"
    if not event_root.exists():
        print(f"Папка бинарных событий не найдена: {event_root}")
        return 0

    count = 0

    for direction_dir in sorted(p for p in event_root.iterdir() if p.is_dir()):
        direction = direction_dir.name

        raster, labels = load_events_for_direction(direction_dir)

        if raster.size != 0:
            save_event_activity_plot(
                raster=raster,
                output_path=output_root / "binary_events" / "all_channels" / f"events_activity_{direction}_all_channels.png",
                title=f"Суммарная активность бинарных событий: {direction}, все каналы",
            )
            count += 1

        by_channel = load_events_by_channel(direction_dir)

        for channel, (channel_raster, channel_labels) in by_channel.items():
            short = channel_short(channel)

            save_event_raster(
                raster=channel_raster,
                labels=channel_labels,
                output_path=output_root / "binary_events" / f"channel_{short}" / f"events_raster_{direction}_{short}.png",
                title=f"Бинарные события пиков: {direction}, {channel_ru(channel)}",
            )
            count += 1

            save_event_activity_plot(
                raster=channel_raster,
                output_path=output_root / "binary_events" / f"channel_{short}" / f"events_activity_{direction}_{short}.png",
                title=f"Активность событий: {direction}, {channel_ru(channel)}",
            )
            count += 1

    return count


BLOCK_EVENT_RE = re.compile(
    r"^EventsBlock_(?P<direction>rows|cols)_Channel_(?P<channel>[^_]+)_(?P<block>Block_\d+_[^.]*)\.txt$",
    re.IGNORECASE,
)


def parse_block_event_files(event_dir: Path) -> Dict[str, Tuple[np.ndarray, List[str]]]:
    grouped: Dict[str, List[Tuple[str, Path]]] = defaultdict(list)

    for file_path in sorted(event_dir.glob("EventsBlock_*.txt")):
        match = BLOCK_EVENT_RE.match(file_path.name)
        if not match:
            continue
        channel = normalize_channel_name(match.group("channel"))
        block = match.group("block")
        grouped[channel].append((block, file_path))

    result: Dict[str, Tuple[np.ndarray, List[str]]] = {}

    for channel in ["Red", "Green", "Blue"]:
        items = sorted(grouped.get(channel, []), key=lambda x: parse_float(x[0].split("_")[-1].split("-")[0], 0))
        rows: List[np.ndarray] = []
        labels: List[str] = []
        expected_len: Optional[int] = None

        for block_label, file_path in items:
            data = read_matrix_txt(file_path).astype(int).ravel()
            if expected_len is None:
                expected_len = len(data)
            elif len(data) != expected_len:
                print(f"Пропуск {file_path.name}: длина {len(data)} != {expected_len}")
                continue
            rows.append(data)
            labels.append(block_label)

        if rows:
            result[channel] = (np.vstack(rows), labels)

    return result


def visualize_binary_block_events(results_root: Path, output_root: Path) -> int:
    event_root = results_root / "03_binary_events_blocks"
    if not event_root.exists():
        return 0

    count = 0

    for direction_dir in sorted(p for p in event_root.iterdir() if p.is_dir()):
        direction = direction_dir.name
        by_channel = parse_block_event_files(direction_dir)

        for channel, (channel_raster, channel_labels) in by_channel.items():
            short = channel_short(channel)

            save_event_raster(
                raster=channel_raster,
                labels=channel_labels,
                output_path=output_root / "binary_event_blocks" / f"channel_{short}" / f"events_block_raster_{direction}_{short}.png",
                title=f"Бинарные события блоков масштабов: {direction}, {channel_ru(channel)}",
            )
            count += 1

            save_event_activity_plot(
                raster=channel_raster,
                output_path=output_root / "binary_event_blocks" / f"channel_{short}" / f"events_block_activity_{direction}_{short}.png",
                title=f"Активность блоков масштабов: {direction}, {channel_ru(channel)}",
            )
            count += 1

    return count


def read_statistics_csv(csv_path: Path) -> List[Dict[str, object]]:
    if not csv_path.exists():
        print(f"CSV не найден: {csv_path}")
        return []

    rows: List[Dict[str, object]] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            rows.append(dict(row))

    return rows


def pair_label(row: Dict[str, object]) -> str:
    return (
        f"{row.get('channel_a')}:{row.get('band_a')} "
        f"→ {row.get('channel_b')}:{row.get('band_b')} "
        f"lag={row.get('lag')}"
    )


def window_label(row: Dict[str, object]) -> str:
    return f"{row.get('start')}-{row.get('end')}"


def group_rows(
    rows: Sequence[Dict[str, object]],
    direction: str,
    model: str,
    band_type: Optional[str] = None,
) -> List[Dict[str, object]]:
    result = []
    for r in rows:
        if str(r.get("direction")) != direction or str(r.get("model")) != model:
            continue
        current_band_type = str(r.get("band_type") or "scale")
        if band_type is not None and current_band_type != band_type:
            continue
        result.append(r)
    return result


def unique_sorted(values: Iterable[str]) -> List[str]:
    return sorted(set(str(v) for v in values))


def unique_windows(rows: Sequence[Dict[str, object]]) -> List[Tuple[int, int, str]]:
    windows = set()

    for row in rows:
        start = parse_int(row.get("start"))
        end = parse_int(row.get("end"))
        windows.add((start, end, f"{start}-{end}"))

    return sorted(windows, key=lambda x: (x[0], x[1]))



def select_top_pairs(
    rows: Sequence[Dict[str, object]],
    metric: str,
    top_n: int,
    abs_metric: bool = False,
) -> List[str]:
    grouped: Dict[str, List[float]] = defaultdict(list)

    for row in rows:
        label = pair_label(row)
        value = parse_float(row.get(metric))

        if math.isnan(value) or math.isinf(value):
            continue

        grouped[label].append(abs(value) if abs_metric else value)

    scored = []

    for label, values in grouped.items():
        if values:
            scored.append((float(np.mean(values)), label))

    scored.sort(reverse=True, key=lambda x: x[0])

    return [label for _, label in scored[:top_n]]


def band_sort_key(band: object) -> Tuple[int, float, float, str]:
    text = str(band or "")

    block_match = re.search(r"Block_(\d+)_([-+]?\d+(?:\.\d+)?)-([-+]?\d+(?:\.\d+)?)", text)
    if block_match:
        return (
            int(block_match.group(1)),
            parse_float(block_match.group(2), 0.0),
            parse_float(block_match.group(3), 0.0),
            text,
        )

    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if nums:
        value = parse_float(nums[0], 0.0)
        return (0, value, value, text)

    return (999, 0.0, 0.0, text)


def pair_structural_sort_key(label: str) -> Tuple[int, Tuple[int, float, float, str], int, Tuple[int, float, float, str], int, str]:
    text = str(label)
    left, sep, right = text.partition("→")

    if not sep:
        return (999, (999, 0.0, 0.0, text), 999, (999, 0.0, 0.0, text), 0, text)

    left = left.strip()
    right = right.strip()

    channel_a, _, band_a = left.partition(":")
    right_main, _, lag_text = right.partition("lag=")
    channel_b, _, band_b = right_main.strip().partition(":")

    channel_a_norm = normalize_channel_name(channel_a.strip())
    channel_b_norm = normalize_channel_name(channel_b.strip())
    lag = parse_int(lag_text, 0)

    return (
        CHANNEL_ORDER.get(channel_a_norm, 99),
        band_sort_key(band_a.strip()),
        CHANNEL_ORDER.get(channel_b_norm, 99),
        band_sort_key(band_b.strip()),
        lag,
        text,
    )


def unique_pair_labels(rows: Sequence[Dict[str, object]]) -> List[str]:
    return sorted({pair_label(row) for row in rows}, key=pair_structural_sort_key)


def pair_count(rows: Sequence[Dict[str, object]]) -> int:
    return len({pair_label(row) for row in rows})


def compute_aligned_pair_limits(
    rows: Sequence[Dict[str, object]],
    directions: Sequence[str],
    models: Sequence[str],
    top_pairs: int,
) -> Dict[Tuple[str, str], int]:
    limits: Dict[Tuple[str, str], int] = {}

    for direction in directions:
        for model in models:
            counts = []
            for band_type in ["scale", "block"]:
                subset = group_rows(rows, direction, model, band_type=band_type)
                if subset:
                    counts.append(pair_count(subset))

            if counts:
                limits[(direction, model)] = max(1, min([top_pairs] + counts))

    return limits


def select_aligned_pairs(
    rows: Sequence[Dict[str, object]],
    top_n: int,
    score_metric: str = "phi",
    abs_metric: bool = True,
) -> List[str]:
    if top_n <= 0:
        return []

    grouped: Dict[str, List[float]] = defaultdict(list)

    for row in rows:
        label = pair_label(row)
        value = parse_float(row.get(score_metric))

        if math.isnan(value) or math.isinf(value):
            continue

        grouped[label].append(abs(value) if abs_metric else value)

    scored: List[Tuple[float, str]] = []
    for label, values in grouped.items():
        if values:
            scored.append((float(np.mean(values)), label))

    if scored:
        scored.sort(reverse=True, key=lambda x: x[0])
        selected = [label for _, label in scored[:top_n]]
    else:
        selected = unique_pair_labels(rows)[:top_n]

    return sorted(selected, key=pair_structural_sort_key)


def build_metric_matrix(
    rows: Sequence[Dict[str, object]],
    metric: str,
    top_pairs: Sequence[str],
    transform_pvalue: bool = False,
) -> Tuple[np.ndarray, List[str], List[str]]:
    window_pairs = sorted({
        (parse_int(row.get("start")), parse_int(row.get("end")))
        for row in rows
    }, key=lambda x: (x[0], x[1]))

    win_to_idx = {
        (start, end): idx
        for idx, (start, end) in enumerate(window_pairs)
    }

    pair_to_idx = {
        label: idx
        for idx, label in enumerate(top_pairs)
    }

    matrix = np.full((len(top_pairs), len(window_pairs)), np.nan, dtype=float)
    for row in rows:
        p_label = pair_label(row)
        if p_label not in pair_to_idx:
            continue

        start = parse_int(row.get("start"))
        end = parse_int(row.get("end"))
        w_key = (start, end)

        if w_key not in win_to_idx:
            continue

        value = parse_float(row.get(metric))
        if math.isnan(value):
            continue

        if transform_pvalue:
            value = max(value, 1e-300)
            value = -math.log10(value)

        pair_idx = pair_to_idx[p_label]
        win_idx = win_to_idx[w_key]
        if 0 <= pair_idx < matrix.shape[0] and 0 <= win_idx < matrix.shape[1]:
            matrix[pair_idx, win_idx] = value

    x_labels = [f"{start}-{end}" for start, end in window_pairs]
    y_labels = list(top_pairs)

    return matrix, x_labels, y_labels


def save_stat_heatmap(
    matrix: np.ndarray,
    x_labels: Sequence[str],
    y_labels: Sequence[str],
    output_path: Path,
    title: str,
    value_label: str,
) -> None:
    ensure_dir(output_path.parent)

    if matrix.size == 0:
        return

    fig_height = max(5, min(18, 0.35 * len(y_labels) + 3))
    fig_width = max(9, min(22, 0.28 * len(x_labels) + 6))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    masked = np.ma.masked_invalid(matrix)

    im = ax.imshow(masked, interpolation="nearest", aspect="auto")

    ax.set_title(title)
    ax.set_xlabel("окна")
    ax.set_ylabel("пара рядов")

    if len(x_labels) <= 40:
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=90, fontsize=7)
    else:
        step = max(1, len(x_labels) // 25)
        ticks = np.arange(0, len(x_labels), step)
        ax.set_xticks(ticks)
        ax.set_xticklabels([x_labels[i] for i in ticks], rotation=90, fontsize=7)

    if len(y_labels) <= 40:
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=7)
    else:
        ax.set_yticks([])

    fig.colorbar(im, ax=ax, label=value_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_top_pairs_bar(
    rows: Sequence[Dict[str, object]],
    metric: str,
    output_path: Path,
    title: str,
    top_n: int,
    abs_metric: bool = False,
) -> None:
    ensure_dir(output_path.parent)

    grouped: Dict[str, List[float]] = defaultdict(list)

    for row in rows:
        label = pair_label(row)
        value = parse_float(row.get(metric))

        if math.isnan(value) or math.isinf(value):
            continue

        grouped[label].append(abs(value) if abs_metric else value)

    scored = []

    for label, values in grouped.items():
        if values:
            scored.append((float(np.mean(values)), label))

    scored.sort(reverse=True, key=lambda x: x[0])
    scored = scored[:top_n]

    if not scored:
        return

    values = [v for v, _ in scored][::-1]
    labels = [l for _, l in scored][::-1]

    fig_height = max(5, min(16, 0.35 * len(labels) + 2))
    fig, ax = plt.subplots(figsize=(11, fig_height))

    ax.barh(np.arange(len(labels)), values)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel(f"среднее значение {metric}")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_pair_dynamics_lines(
    rows: Sequence[Dict[str, object]],
    metric: str,
    output_path: Path,
    title: str,
    top_pairs: Sequence[str],
) -> None:
    ensure_dir(output_path.parent)

    if not top_pairs:
        return

    windows = unique_windows(rows)
    win_labels = [label for _, _, label in windows]
    win_to_x = {label: i for i, label in enumerate(win_labels)}

    pair_values: Dict[str, Dict[int, float]] = {p: {} for p in top_pairs}

    for row in rows:
        p_label = pair_label(row)
        if p_label not in pair_values:
            continue

        w_label = window_label(row)
        if w_label not in win_to_x:
            continue

        value = parse_float(row.get(metric))
        if math.isnan(value) or math.isinf(value):
            continue

        pair_values[p_label][win_to_x[w_label]] = value

    fig, ax = plt.subplots(figsize=(13, 6))

    for p_label, values_by_x in pair_values.items():
        if not values_by_x:
            continue

        xs = sorted(values_by_x.keys())
        ys = [values_by_x[x] for x in xs]

        ax.plot(xs, ys, marker="o", linewidth=1, markersize=3, label=p_label)

    ax.set_title(title)
    ax.set_xlabel("номер окна")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)

    if len(top_pairs) <= 8:
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_significance_summary_bar(
    rows: Sequence[Dict[str, object]],
    p_metric: str,
    output_path: Path,
    title: str,
    alpha: float = 0.05,
    top_n: int = 25,
) -> None:
    """
    Для каждой пары считает долю окон, где p < alpha.
    """

    ensure_dir(output_path.parent)

    grouped_total: Dict[str, int] = defaultdict(int)
    grouped_sig: Dict[str, int] = defaultdict(int)

    for row in rows:
        label = pair_label(row)
        p_value = parse_float(row.get(p_metric))

        if math.isnan(p_value):
            continue

        grouped_total[label] += 1
        if p_value < alpha:
            grouped_sig[label] += 1

    scored = []

    for label, total in grouped_total.items():
        if total == 0:
            continue
        ratio = grouped_sig[label] / total
        scored.append((ratio, label))

    scored.sort(reverse=True, key=lambda x: x[0])
    scored = scored[:top_n]

    if not scored:
        return

    values = [v for v, _ in scored][::-1]
    labels = [l for _, l in scored][::-1]

    fig_height = max(5, min(16, 0.35 * len(labels) + 2))
    fig, ax = plt.subplots(figsize=(11, fig_height))

    ax.barh(np.arange(len(labels)), values)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel(f"доля окон с {p_metric} < {alpha}")
    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_pair_label(row: Dict[str, object]) -> str:
    return (
        f"{row.get('channel_a')}:{row.get('band_a')} "
        f"→ {row.get('channel_b')}:{row.get('band_b')} "
        f"lag={row.get('lag')}"
    )


def save_runs_bar(
    rows: Sequence[Dict[str, object]],
    metric: str,
    output_path: Path,
    title: str,
    top_n: int,
) -> None:
    ensure_dir(output_path.parent)

    scored = []
    for row in rows:
        value = parse_float(row.get(metric))
        if math.isnan(value) or math.isinf(value):
            continue
        scored.append((value, run_pair_label(row)))

    scored.sort(reverse=True, key=lambda x: x[0])
    scored = scored[:top_n]
    if not scored:
        return

    values = [v for v, _ in scored][::-1]
    labels = [l for _, l in scored][::-1]

    fig_height = max(5, min(16, 0.35 * len(labels) + 2))
    fig, ax = plt.subplots(figsize=(11, fig_height))
    ax.barh(np.arange(len(labels)), values)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel(metric)
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def visualize_run_statistics(
    results_root: Path,
    output_root: Path,
    top_pairs: int,
) -> int:
    csv_path = results_root / "04_synchronization_statistics" / "sync_significant_runs_all.csv"
    rows = read_statistics_csv(csv_path)
    if not rows:
        return 0

    count = 0
    directions = unique_sorted(row.get("direction", "") for row in rows)
    band_types = unique_sorted(row.get("band_type", "scale") or "scale" for row in rows)
    models = ["A", "B", "V", "G"]
    p_metrics = ["chi2_p", "fisher_p", "mcnemar_p", "binomial_p"]
    run_metrics = ["max_significant_run", "mean_significant_run", "number_of_runs"]

    def run_pair_count(subset: Sequence[Dict[str, object]]) -> int:
        return len({run_pair_label(row) for row in subset})

    run_limits: Dict[Tuple[str, str, str], int] = {}
    for direction in directions:
        for model in models:
            for p_metric in p_metrics:
                counts = []
                for band_type in ["scale", "block"]:
                    subset = [
                        row for row in rows
                        if str(row.get("band_type") or "scale") == band_type
                        and str(row.get("direction")) == direction
                        and str(row.get("model")) == model
                        and str(row.get("p_metric")) == p_metric
                    ]
                    if subset:
                        counts.append(run_pair_count(subset))
                if counts:
                    run_limits[(direction, model, p_metric)] = max(1, min([top_pairs] + counts))

    for band_type in band_types:
        for direction in directions:
            for model in models:
                for p_metric in p_metrics:
                    subset = [
                        row for row in rows
                        if str(row.get("band_type") or "scale") == band_type
                        and str(row.get("direction")) == direction
                        and str(row.get("model")) == model
                        and str(row.get("p_metric")) == p_metric
                    ]
                    if not subset:
                        continue

                    display_pair_count = run_limits.get(
                        (direction, model, p_metric),
                        min(top_pairs, run_pair_count(subset)),
                    )

                    for metric in run_metrics:
                        save_runs_bar(
                            rows=subset,
                            metric=metric,
                            output_path=output_root / "statistics_runs" / band_type / direction / f"model_{model}" / p_metric / f"top_{metric}.png",
                            title=f"{direction}, {band_type}, модель {model}, {p_metric}: {metric}",
                            top_n=display_pair_count,
                        )
                        count += 1

    return count


def visualize_statistics(
    results_root: Path,
    output_root: Path,
    top_pairs: int,
) -> int:
    csv_path = results_root / "04_synchronization_statistics" / "sync_statistics_all.csv"
    rows = read_statistics_csv(csv_path)

    if not rows:
        return 0

    count = 0

    directions = unique_sorted(row.get("direction", "") for row in rows)
    models = ["A", "B", "V", "G"]
    band_types = unique_sorted(row.get("band_type", "scale") or "scale" for row in rows)

    ordinary_metrics = [
        "phi",
        "jaccard",
        "dice",
        "odds_ratio",
        "n11_ratio",
    ]

    p_metrics = [
        "chi2_p",
        "fisher_p",
        "mcnemar_p",
        "binomial_p",
    ]

    aligned_pair_limits = compute_aligned_pair_limits(
        rows=rows,
        directions=directions,
        models=models,
        top_pairs=top_pairs,
    )

    for band_type in band_types:
        for direction in directions:
            for model in models:
                subset = group_rows(rows, direction, model, band_type=band_type)
                if not subset:
                    continue

                base_dir = output_root / "statistics" / band_type / direction / f"model_{model}"

                display_pair_count = aligned_pair_limits.get(
                    (direction, model),
                    min(top_pairs, pair_count(subset)),
                )

                aligned_pairs = select_aligned_pairs(
                    subset,
                    top_n=display_pair_count,
                    score_metric="phi",
                    abs_metric=True,
                )

                for metric in ordinary_metrics:
                    matrix, x_labels, y_labels = build_metric_matrix(
                        subset,
                        metric=metric,
                        top_pairs=aligned_pairs,
                        transform_pvalue=False,
                    )

                    save_stat_heatmap(
                        matrix=matrix,
                        x_labels=x_labels,
                        y_labels=y_labels,
                        output_path=base_dir / "heatmaps" / f"heatmap_{metric}.png",
                        title=f"{direction}, {band_type}, модель {model}: {metric}",
                        value_label=metric,
                    )
                    count += 1

                    save_top_pairs_bar(
                        rows=subset,
                        metric=metric,
                        output_path=base_dir / "bars" / f"top_pairs_mean_{metric}.png",
                        title=f"{direction}, {band_type}, модель {model}: топ пар по среднему {metric}",
                        top_n=display_pair_count,
                        abs_metric=(metric == "phi"),
                    )
                    count += 1

                save_pair_dynamics_lines(
                    rows=subset,
                    metric="phi",
                    output_path=base_dir / "lines" / "top_pairs_phi_by_windows.png",
                    title=f"{direction}, {band_type}, модель {model}: динамика phi по окнам",
                    top_pairs=aligned_pairs[:8],
                )
                count += 1

                save_pair_dynamics_lines(
                    rows=subset,
                    metric="jaccard",
                    output_path=base_dir / "lines" / "top_pairs_jaccard_by_windows.png",
                    title=f"{direction}, {band_type}, модель {model}: динамика Jaccard по окнам",
                    top_pairs=aligned_pairs[:8],
                )
                count += 1

                for p_metric in p_metrics:
                    matrix, x_labels, y_labels = build_metric_matrix(
                        subset,
                        metric=p_metric,
                        top_pairs=aligned_pairs,
                        transform_pvalue=True,
                    )

                    save_stat_heatmap(
                        matrix=matrix,
                        x_labels=x_labels,
                        y_labels=y_labels,
                        output_path=base_dir / "pvalues" / f"heatmap_minus_log10_{p_metric}.png",
                        title=f"{direction}, {band_type}, модель {model}: -log10({p_metric})",
                        value_label=f"-log10({p_metric})",
                    )
                    count += 1

                    save_significance_summary_bar(
                        rows=subset,
                        p_metric=p_metric,
                        output_path=base_dir / "pvalues" / f"significant_ratio_{p_metric}.png",
                        title=f"{direction}, {band_type}, модель {model}: доля значимых окон по {p_metric}",
                        alpha=0.05,
                        top_n=display_pair_count,
                    )
                    count += 1

    return count

def collect_pngs(root: Path) -> List[Path]:
    return sorted(root.rglob("*.png"))


MODEL_DESCRIPTIONS = {
    "model_A": "Модель A — внутрикальная межмасштабная синхронизация",
    "model_B": "Модель Б — межканальная синхронизация на одном масштабе",
    "model_V": "Модель В — сдвиговая внутрикальная межмасштабная синхронизация",
    "model_G": "Модель Г — сдвиговая межканальная межмасштабная синхронизация",
}

MODEL_EXPLANATIONS = {
    "model_A": (
        "Сравниваются разные масштабы внутри одного цветового канала. "
        "Модель показывает, возникают ли структурные признаки одного канала одновременно на разных масштабах."
    ),
    "model_B": (
        "Сравниваются разные цветовые каналы на одном и том же масштабе. "
        "Модель показывает, совпадают ли признаки между R, G и B при одинаковом масштабе анализа."
    ),
    "model_V": (
        "Сравниваются разные масштабы внутри одного канала, но со сдвигом по координате. "
        "Модель нужна для проверки запаздывающей или смещённой связи между масштабами."
    ),
    "model_G": (
        "Сравниваются разные каналы и разные масштабы со сдвигом. "
        "Модель показывает наиболее сложную связь: межканальную, межмасштабную и пространственно смещённую."
    ),
}

CROSS_MODEL_EXPLANATIONS = {
    "A": (
        "Кросс-спектр для модели A: один канал, две разные масштабные компоненты. "
        "На карте видны области, где в одном канале одновременно выражены два масштаба."
    ),
    "B": (
        "Кросс-спектр для модели Б: один масштаб, два разных цветовых канала. "
        "На карте видны области, где разные каналы дают сильный отклик на одном масштабе."
    ),
    "G": (
        "Кросс-спектр для модели Г: разные каналы и разные масштабы. "
        "На карте видны области совместного проявления межканальных и межмасштабных признаков."
    ),
}

DIRECTION_DESCRIPTIONS = {
    "rows": "Построчное направление",
    "cols": "Направление по столбцам",
}

METRIC_DESCRIPTIONS = {
    "phi": "Phi — коэффициент связи бинарных событий. Чем ближе |phi| к 1, тем сильнее связь.",
    "jaccard": "Jaccard — доля совместных событий среди всех событий хотя бы в одном ряду.",
    "dice": "Dice — мера совпадения событий, похожа на Jaccard, но сильнее учитывает совместные пики.",
    "n11_ratio": "n11_ratio — доля одновременных событий n11 относительно размера окна.",
    "odds_ratio": "Odds ratio — отношение шансов совместного появления событий.",
    "chi2_p": "χ² p-value — статистическая проверка зависимости по таблице 2×2.",
    "fisher_p": "Fisher p-value — точный критерий Фишера для таблицы 2×2.",
    "mcnemar_p": "McNemar p-value — проверка асимметрии несовпадающих событий.",
    "binomial_p": "Binomial p-value — биномиальная проверка совпадений событий.",
}

SECTION_DESCRIPTIONS = {
    "power_spectra": "Карты мощности P = W² для каждого канала, масштаба и направления.",
    "cross_spectra": "Карты произведений спектров C = P1 · P2, рассчитанные для сопоставляемых пар каналов и масштабов.",
    "binary_events": "Бинарные события пиков: 1 означает значимый пик мощности, 0 — отсутствие события.",
    "statistics": "",
}


def rel_link(path: Path, output_root: Path) -> str:
    return path.relative_to(output_root).as_posix()


def html_escape(text: object) -> str:
    s = str(text)
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def card_html(path: Path, output_root: Path, title: Optional[str] = None, small: bool = False) -> str:
    rel = rel_link(path, output_root)
    caption = title if title else rel
    cls = "card small-card" if small else "card"

    return f"""
    <div class="{cls}">
        <a href="{html_escape(rel)}" target="_blank">
            <img src="{html_escape(rel)}" loading="lazy">
        </a>
        <div class="caption">{html_escape(caption)}</div>
    </div>
    """


def section_open(title: str, anchor: str, description: str = "") -> str:
    desc_html = f"<p class='section-desc'>{html_escape(description)}</p>" if description else ""
    return f"""
    <section id="{html_escape(anchor)}" class="section-block">
        <h2>{html_escape(title)}</h2>
        {desc_html}
    """


def section_close() -> str:
    return "</section>"


def details_open(title: str, description: str = "", open_by_default: bool = False) -> str:
    opened = " open" if open_by_default else ""
    desc_html = f"<p class='details-desc'>{html_escape(description)}</p>" if description else ""
    return f"""
    <details{opened}>
        <summary>{html_escape(title)}</summary>
        {desc_html}
    """


def details_close() -> str:
    return "</details>"


def grid_html(cards: Sequence[str]) -> str:
    if not cards:
        return "<p class='empty'>Нет графиков для этого раздела.</p>"
    return "<div class='grid'>\n" + "\n".join(cards) + "\n</div>"


def get_pngs_under(output_root: Path, relative_folder: str) -> List[Path]:
    folder = output_root / relative_folder
    if not folder.exists():
        return []
    return sorted(folder.rglob("*.png"))


def build_power_section(output_root: Path) -> str:
    pngs = get_pngs_under(output_root, "power_spectra")

    by_direction: Dict[str, List[Path]] = defaultdict(list)
    for p in pngs:
        rel = p.relative_to(output_root)
        direction = rel.parts[1] if len(rel.parts) > 1 else "unknown"
        by_direction[direction].append(p)

    html = [
        section_open(
            "1. Мощности вейвлет-коэффициентов",
            "power",
            SECTION_DESCRIPTIONS["power_spectra"],
        )
    ]

    for direction in ["rows", "cols"]:
        paths = by_direction.get(direction, [])
        if not paths:
            continue

        title = DIRECTION_DESCRIPTIONS.get(direction, direction)
        html.append(details_open(title, f"Всего карт: {len(paths)}", open_by_default=(direction == "rows")))

        by_scale: Dict[str, List[Path]] = defaultdict(list)
        for p in paths:
            rel = p.relative_to(output_root)
            scale = rel.parts[2] if len(rel.parts) > 2 else "Scale_unknown"
            by_scale[scale].append(p)

        for scale in sorted(by_scale.keys(), key=lambda s: parse_float(s.replace("Scale_", ""), 0)):
            scale_paths = by_scale[scale]
            html.append(details_open(scale, f"Каналовые карты мощности: {len(scale_paths)}"))
            html.append(grid_html([card_html(p, output_root, title=p.name, small=True) for p in scale_paths]))
            html.append(details_close())

        html.append(details_close())

    html.append(section_close())
    return "\n".join(html)


def build_binary_section(output_root: Path) -> str:
    pngs = get_pngs_under(output_root, "binary_events")

    def paths_containing(*parts: str) -> List[Path]:
        result = []
        for p in pngs:
            rel = p.relative_to(output_root).as_posix()
            if all(part in rel for part in parts):
                result.append(p)
        return sorted(result)

    html = [
        section_open(
            "2. Бинарные события пиков",
            "binary",
            SECTION_DESCRIPTIONS["binary_events"],
        ),
        "<div class='note'><b>Как читать:</b> raster-график показывает, где сработали события-пики. "
        "Внутри каждой карты строки — это масштабы одного канала, "
        "а столбцы — пространственная координата t. Activity-график показывает количество активных масштабов в каждой координате.</div>",
    ]

    activity_all = [p for p in paths_containing("binary_events/all_channels") if "activity" in p.name]
    if activity_all:
        html.append(details_open("Суммарная активность всех каналов", "Сколько рядов канал+масштаб активны в каждой координате.", open_by_default=True))
        html.append(grid_html([card_html(p, output_root, title=p.name) for p in activity_all]))
        html.append(details_close())

    for short, ru in [
        ("R", "Красный канал R"),
        ("G", "Зелёный канал G"),
        ("B", "Синий канал B"),
    ]:
        channel_paths = paths_containing(f"binary_events/channel_{short}")
        if not channel_paths:
            continue

        activity = [p for p in channel_paths if "activity" in p.name]
        raster = [p for p in channel_paths if "raster" in p.name]

        html.append(details_open(ru, open_by_default=True))

        html.append(details_open("Активность по масштабам", "Сколько масштабов данного канала активны в каждой координате.", open_by_default=True))
        html.append(grid_html([card_html(p, output_root, title=p.name) for p in activity]))
        html.append(details_close())

        html.append(details_open("Raster-карты по масштабам", "Строки — масштабы, столбцы — координата t.", open_by_default=True))
        html.append(grid_html([card_html(p, output_root, title=p.name) for p in raster]))
        html.append(details_close())

        html.append(details_close())

    block_pngs = get_pngs_under(output_root, "binary_event_blocks")
    if block_pngs:
        html.append(details_open("Бинарные события после сжатия масштабов в блоки", "Блок равен 1, если хотя бы на одном масштабе внутри блока найден пик.", open_by_default=True))
        for short, ru in [
            ("R", "Красный канал R"),
            ("G", "Зелёный канал G"),
            ("B", "Синий канал B"),
        ]:
            channel_paths = [p for p in block_pngs if f"binary_event_blocks/channel_{short}" in p.relative_to(output_root).as_posix()]
            if not channel_paths:
                continue
            activity = [p for p in channel_paths if "activity" in p.name]
            raster = [p for p in channel_paths if "raster" in p.name]
            html.append(details_open(ru, open_by_default=True))
            html.append(details_open("Активность по блокам масштабов", open_by_default=True))
            html.append(grid_html([card_html(p, output_root, title=p.name) for p in activity]))
            html.append(details_close())
            html.append(details_open("Raster-карты по блокам масштабов", open_by_default=True))
            html.append(grid_html([card_html(p, output_root, title=p.name) for p in raster]))
            html.append(details_close())
            html.append(details_close())
        html.append(details_close())

    html.append(section_close())
    return "\n".join(html)


def parse_cross_parts(path: Path, output_root: Path) -> Tuple[str, str, str]:
    rel = path.relative_to(output_root)
    parts = rel.parts

    # cross_spectra / model_folder / direction / group / file.png
    model = parts[1] if len(parts) > 1 else "unknown_model"
    direction = parts[2] if len(parts) > 2 else "unknown_direction"
    group = parts[3] if len(parts) > 3 else "unknown_group"

    return model, direction, group


def readable_cross_model(model_folder: str) -> str:
    if model_folder.startswith("A_"):
        return "Модель A — внутрикальная межмасштабная синхронизация"
    if model_folder.startswith("B_"):
        return "Модель Б — межканальная синхронизация на одном масштабе"
    if model_folder.startswith("G_"):
        return "Модель Г — межканальная межмасштабная синхронизация"
    return model_folder


def cross_model_key(model_folder: str) -> str:
    if model_folder.startswith("A_"):
        return "A"
    if model_folder.startswith("B_"):
        return "B"
    if model_folder.startswith("G_"):
        return "G"
    return model_folder


def build_cross_section(output_root: Path) -> str:
    pngs = get_pngs_under(output_root, "cross_spectra")

    grouped: Dict[str, Dict[str, Dict[str, List[Path]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for p in pngs:
        model, direction, group = parse_cross_parts(p, output_root)
        grouped[model][direction][group].append(p)

    html = [
        section_open(
            "3. Кросс-спектры",
            "cross",
            SECTION_DESCRIPTIONS["cross_spectra"],
        )
]

    for model in sorted(grouped.keys()):
        model_title = readable_cross_model(model)
        model_key = cross_model_key(model)
        total = sum(len(paths) for d in grouped[model].values() for paths in d.values())
        html.append(details_open(
            model_title,
            f"{CROSS_MODEL_EXPLANATIONS.get(model_key, '')} Всего карт: {total}",
            open_by_default=False
        ))

        for direction in ["rows", "cols"]:
            if direction not in grouped[model]:
                continue

            direction_total = sum(len(paths) for paths in grouped[model][direction].values())
            html.append(details_open(DIRECTION_DESCRIPTIONS.get(direction, direction), f"Карт: {direction_total}"))

            for group in sorted(grouped[model][direction].keys()):
                paths = grouped[model][direction][group]
                html.append(details_open(group, f"Карт: {len(paths)}"))
                html.append(grid_html([card_html(p, output_root, title=p.name, small=True) for p in paths]))
                html.append(details_close())

            html.append(details_close())

        html.append(details_close())

    html.append(section_close())
    return "\n".join(html)


def classify_stat_file(path: Path) -> str:
    name = path.name

    if "heatmap_minus_log10" in name or "significant_ratio" in name:
        return "pvalues"
    if "heatmap_" in name:
        return "heatmaps"
    if "top_pairs_mean" in name:
        return "bars"
    if "by_windows" in name:
        return "lines"

    return "other"


def metric_from_stat_name(path: Path) -> str:
    name = path.stem

    replacements = [
        "heatmap_minus_log10_",
        "significant_ratio_",
        "top_pairs_mean_",
        "heatmap_",
        "top_pairs_",
        "_by_windows",
    ]

    metric = name
    for r in replacements:
        metric = metric.replace(r, "")

    return metric


def statistics_parameters_html(output_root: Path) -> str:
    rows = read_statistics_csv(output_root.parent / "04_synchronization_statistics" / "sync_statistics_all.csv")
    if not rows:
        return ""

    window_sizes = unique_sorted(row.get("window_size", "") for row in rows if row.get("window_size", "") != "")
    steps = unique_sorted(row.get("step", "") for row in rows if row.get("step", "") != "")
    directions = unique_sorted(row.get("direction", "") for row in rows)

    axis_parts = []
    if "rows" in directions:
        axis_parts.append("для построчного анализа окно перемещается вдоль оси X")
    if "cols" in directions:
        axis_parts.append("для анализа по столбцам окно перемещается вдоль оси Y")

    windows_by_direction = []
    for direction in ["rows", "cols"]:
        direction_rows = [row for row in rows if str(row.get("direction")) == direction]
        windows = {(row.get("start"), row.get("end")) for row in direction_rows}
        if windows:
            windows_by_direction.append(f"{DIRECTION_DESCRIPTIONS.get(direction, direction)}: {len(windows)} окон")

    return f"""
            <p>
                Параметры скользящего анализа: длина окна — {html_escape(', '.join(window_sizes) or 'не указана')} пикселей/отсчётов;
                шаг окна — {html_escape(', '.join(steps) or 'не указан')} пикселей/отсчётов.
                {html_escape('; '.join(axis_parts))}.
                Количество окон: {html_escape('; '.join(windows_by_direction) or 'не определено')}.
            </p>
    """


def build_statistics_section(output_root: Path) -> str:
    pngs = get_pngs_under(output_root, "statistics")
    run_pngs = get_pngs_under(output_root, "statistics_runs")

    grouped: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, List[Path]]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    )

    for p in pngs:
        rel = p.relative_to(output_root)
        parts = rel.parts
        if len(parts) < 6:
            continue
        band_type = parts[1]
        direction = parts[2]
        model = parts[3]
        subtype = classify_stat_file(p)
        metric = metric_from_stat_name(p)
        grouped[band_type][direction][model][subtype][metric].append(p)

    run_grouped: Dict[str, Dict[str, Dict[str, Dict[str, List[Path]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    for p in run_pngs:
        rel = p.relative_to(output_root)
        parts = rel.parts
        if len(parts) < 6:
            continue
        band_type = parts[1]
        direction = parts[2]
        model = parts[3]
        p_metric = parts[4]
        run_grouped[band_type][direction][model][p_metric].append(p)

    html = [
        section_open(
            "4. Статистика синхронизации",
            "statistics",
            SECTION_DESCRIPTIONS["statistics"],
        ),
        f"""
        <div class="note">
            <b>Методика статистического анализа синхронизации</b>
            <p>
                Статистический анализ выполняется после преобразования карт мощности вейвлет-коэффициентов
                в бинарные ряды событий. Значение 1 означает наличие локального максимума мощности,
                прошедшего пороговую фильтрацию, а значение 0 означает отсутствие события в данной пространственной позиции.
            </p>
            {statistics_parameters_html(output_root)}
            <p>
                Дополнительно выполняется сжатие масштабов в частотные блоки. Соседние масштабы объединяются
                в блоки вида Block_1, Block_2 и т.д.; бинарный ряд блока принимает значение 1, если хотя бы
                один масштаб внутри блока содержит пик в данной координате. Это позволяет анализировать не только
                отдельные масштабы, но и укрупнённые диапазоны масштабов.
            </p>
            <p>
                Для каждой пары рядов внутри каждого окна строится таблица сопряжённости 2×2: n11 — совместное
                наличие событий, n10 — событие есть только в первом ряду, n01 — событие есть только во втором ряду,
                n00 — события отсутствуют в обоих рядах. На основе таблицы рассчитываются Phi, Jaccard, Dice,
                n11_ratio, odds_ratio и p-value статистических критериев.
            </p>
            <p>
                Критерий серий оценивает устойчивость значимых окон. Для каждой пары рядов определяется,
                идут ли окна с p-value &lt; α подряд. Рассчитываются max_significant_run, mean_significant_run
                и number_of_runs.
            </p>
        </div>
        """
    ]

    band_titles = {
        "scale": "Отдельные масштабы",
        "block": "Сжатые блоки масштабов",
    }

    for band_type in ["scale", "block"]:
        if band_type not in grouped and band_type not in run_grouped:
            continue

        html.append(details_open(band_titles.get(band_type, band_type), open_by_default=(band_type == "scale")))

        for direction in ["rows", "cols"]:
            if direction not in grouped.get(band_type, {}) and direction not in run_grouped.get(band_type, {}):
                continue

            html.append(details_open(
                DIRECTION_DESCRIPTIONS.get(direction, direction),
                open_by_default=(direction == "rows"),
            ))

            for model in ["model_A", "model_B", "model_V", "model_G"]:
                model_data = grouped.get(band_type, {}).get(direction, {}).get(model, {})
                run_model_data = run_grouped.get(band_type, {}).get(direction, {}).get(model, {})
                if not model_data and not run_model_data:
                    continue

                html.append(details_open(
                    MODEL_DESCRIPTIONS.get(model, model),
                    MODEL_EXPLANATIONS.get(model, ""),
                    open_by_default=False,
                ))

                if "heatmaps" in model_data:
                    html.append(details_open("Основные метрики — heatmap по окнам", open_by_default=True))
                    for metric in ["phi", "jaccard", "dice", "n11_ratio", "odds_ratio"]:
                        paths = model_data["heatmaps"].get(metric, [])
                        if paths:
                            html.append(details_open(metric, METRIC_DESCRIPTIONS.get(metric, "")))
                            html.append(grid_html([card_html(p, output_root, title=p.name) for p in paths]))
                            html.append(details_close())
                    html.append(details_close())

                if "bars" in model_data:
                    html.append(details_open("Средние значения по парам — bar charts"))
                    for metric in ["phi", "jaccard", "dice", "n11_ratio", "odds_ratio"]:
                        paths = model_data["bars"].get(metric, [])
                        if paths:
                            html.append(details_open(metric, METRIC_DESCRIPTIONS.get(metric, "")))
                            html.append(grid_html([card_html(p, output_root, title=p.name) for p in paths]))
                            html.append(details_close())
                    html.append(details_close())

                if "lines" in model_data:
                    html.append(details_open("Динамика лучших пар по окнам"))
                    for metric, paths in sorted(model_data["lines"].items()):
                        html.append(details_open(metric, METRIC_DESCRIPTIONS.get(metric, "")))
                        html.append(grid_html([card_html(p, output_root, title=p.name) for p in paths]))
                        html.append(details_close())
                    html.append(details_close())

                if "pvalues" in model_data:
                    html.append(details_open("Статистическая значимость — p-value"))
                    for metric in ["chi2_p", "fisher_p", "mcnemar_p", "binomial_p"]:
                        paths = model_data["pvalues"].get(metric, [])
                        if paths:
                            html.append(details_open(metric, METRIC_DESCRIPTIONS.get(metric, "")))
                            html.append(grid_html([card_html(p, output_root, title=p.name) for p in paths]))
                            html.append(details_close())
                    html.append(details_close())

                if run_model_data:
                    html.append(details_open("Критерий серий значимых окон", "Оценивает длину и количество непрерывных серий окон, где p-value < α."))
                    for p_metric in ["chi2_p", "fisher_p", "mcnemar_p", "binomial_p"]:
                        paths = run_model_data.get(p_metric, [])
                        if paths:
                            html.append(details_open(p_metric))
                            html.append(grid_html([card_html(p, output_root, title=p.name) for p in paths]))
                            html.append(details_close())
                    html.append(details_close())

                html.append(details_close())

            html.append(details_close())

        html.append(details_close())

    html.append(section_close())
    return "\n".join(html)

def build_html_index(output_root: Path) -> None:
    html = [
        "<!doctype html>",
        "<html lang='ru'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<title>Wavelet Sync Visualizations</title>",
        "<style>",
        """
        :root {
            --bg: #f6f7fb;
            --card: #ffffff;
            --text: #1f2937;
            --muted: #6b7280;
            --border: #d8dee9;
            --accent: #2563eb;
            --accent-soft: #e8f0ff;
        }

        * {
            box-sizing: border-box;
        }

        body {
            font-family: Arial, sans-serif;
            margin: 0;
            background: var(--bg);
            color: var(--text);
        }

        header {
            position: sticky;
            top: 0;
            z-index: 20;
            background: rgba(255, 255, 255, 0.96);
            border-bottom: 1px solid var(--border);
            padding: 16px 28px;
            backdrop-filter: blur(8px);
        }

        h1 {
            margin: 0 0 8px 0;
            font-size: 24px;
        }

        .subtitle {
            margin: 0;
            color: var(--muted);
            font-size: 14px;
        }

        nav {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 14px;
        }

        nav a {
            text-decoration: none;
            color: var(--accent);
            background: var(--accent-soft);
            border: 1px solid #c8d9ff;
            padding: 8px 10px;
            border-radius: 999px;
            font-size: 13px;
        }

        main {
            padding: 24px 28px 60px;
            max-width: 1600px;
            margin: 0 auto;
        }

        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 14px;
            margin-bottom: 24px;
        }

        .summary-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
        }

        .summary-number {
            font-size: 28px;
            font-weight: bold;
            color: var(--accent);
        }

        .summary-title {
            color: var(--muted);
            margin-top: 4px;
        }

        .section-block {
            margin: 28px 0;
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 18px;
        }

        h2 {
            margin: 0 0 8px 0;
            padding-bottom: 8px;
            border-bottom: 1px solid var(--border);
        }

        .section-desc,
        .details-desc {
            color: var(--muted);
            font-size: 14px;
            line-height: 1.45;
        }

        .note {
            background: #f8fafc;
            border: 1px solid #cbd5e1;
            border-radius: 10px;
            padding: 16px 18px;
            margin: 16px 0;
            color: #1f2937;
            font-size: 14px;
            line-height: 1.65;
        }

        .note p {
            margin: 8px 0 10px;
        }

        .note ul {
            margin-top: 8px;
        }

        .note li {
            margin-bottom: 6px;
        }

        details {
            border: 1px solid var(--border);
            border-radius: 10px;
            margin: 12px 0;
            background: #fbfcff;
        }

        summary {
            cursor: pointer;
            padding: 12px 14px;
            font-weight: bold;
            color: #111827;
            user-select: none;
        }

        details[open] > summary {
            border-bottom: 1px solid var(--border);
            background: #f1f5ff;
            border-radius: 10px 10px 0 0;
        }

        details > p,
        details > .grid,
        details > details {
            margin-left: 12px;
            margin-right: 12px;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(430px, 1fr));
            gap: 14px;
            padding: 12px 0 14px;
        }

        .card {
            background: white;
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 10px;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
        }

        .small-card {
            padding: 8px;
        }

        .card img {
            width: 100%;
            max-height: 520px;
            object-fit: contain;
            display: block;
            background: white;
        }

        .small-card img {
            max-height: 380px;
        }

        .caption {
            margin-top: 8px;
            color: var(--muted);
            font-size: 12px;
            word-break: break-word;
            line-height: 1.35;
        }

        .empty {
            color: var(--muted);
            font-style: italic;
            padding: 8px 12px;
        }

        .backtop {
            position: fixed;
            right: 20px;
            bottom: 20px;
            background: var(--accent);
            color: white;
            text-decoration: none;
            padding: 10px 12px;
            border-radius: 999px;
            font-size: 13px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.18);
        }
        """,
        "</style>",
        "</head>",
        "<body>",
        "<header>",
        "<h1>Визуализация результатов синхронизации</h1>",
        "<nav>",
        "<a href='#power'>1. Мощности W²</a>",
        "<a href='#binary'>2. Бинарные события</a>",
        "<a href='#cross'>3. Кросс-спектры</a>",
        "<a href='#statistics'>4. Статистика</a>",
        "</nav>",
        "</header>",
        "<main>",
        build_power_section(output_root),
        build_binary_section(output_root),
        build_cross_section(output_root),
        build_statistics_section(output_root),
        "</main>",
        "<a class='backtop' href='#top'>Наверх</a>",
        "</body>",
        "</html>",
    ]

    (output_root / "index.html").write_text("\n".join(html), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Визуализация результатов WaveletSyncResults."
    )
    parser.add_argument(
        "task_folder",
        help="Папка задачи, внутри которой лежит WaveletSyncResults",
    )
    parser.add_argument(
        "--results-folder",
        default="WaveletSyncResults",
        help="Имя папки с результатами пайплайна",
    )
    parser.add_argument(
        "--max-matrix-files",
        type=int,
        default=0,
        help="Сколько матричных .txt максимум визуализировать для мощностей и кросс-спектров. 0 = все. По умолчанию строятся все, чтобы не обрезать модели Б/Г.",
    )
    parser.add_argument(
        "--top-pairs",
        type=int,
        default=25,
        help="Сколько лучших пар показывать на heatmap/bar графиках статистики.",
    )

    args = parser.parse_args()

    task_folder = Path(args.task_folder)
    results_root = task_folder / args.results_folder

    if not task_folder.exists():
        raise FileNotFoundError(f"Папка задачи не найдена: {task_folder}")

    if not results_root.exists():
        raise FileNotFoundError(
            f"Папка результатов не найдена: {results_root}\n"
            f"Сначала запусти wavelet_sync_pipeline.py."
        )

    output_root = results_root / "05_visualizations"
    ensure_dir(output_root)

    print("Визуализация мощностей W^2...")
    power_count = visualize_matrix_folder(
        source_dir=results_root / "01_power_spectra",
        output_dir=output_root / "power_spectra",
        title_prefix="Мощность W²",
        max_files=args.max_matrix_files,
    )
    print(f"  сохранено графиков мощностей: {power_count}")

    print("Визуализация кросс-спектров P1*P2...")
    cross_count = visualize_matrix_folder(
        source_dir=results_root / "02_cross_spectra",
        output_dir=output_root / "cross_spectra",
        title_prefix="Кросс-спектр P1·P2",
        max_files=args.max_matrix_files,
    )
    print(f"  сохранено графиков кросс-спектров: {cross_count}")

    print("Визуализация бинарных событий...")
    event_count = visualize_binary_events(
        results_root=results_root,
        output_root=output_root,
    )
    block_event_count = visualize_binary_block_events(
        results_root=results_root,
        output_root=output_root,
    )
    print(f"  сохранено графиков событий: {event_count}")
    print(f"  сохранено графиков блочных событий: {block_event_count}")

    print("Визуализация статистики синхронизации...")
    stat_count = visualize_statistics(
        results_root=results_root,
        output_root=output_root,
        top_pairs=args.top_pairs,
    )
    run_count = visualize_run_statistics(
        results_root=results_root,
        output_root=output_root,
        top_pairs=args.top_pairs,
    )
    print(f"  сохранено графиков статистики: {stat_count}")
    print(f"  сохранено графиков критерия серий: {run_count}")

    print("Создание index.html...")
    build_html_index(output_root)

    print("Готово.")
    print(f"Папка визуализаций: {output_root}")

if __name__ == "__main__":
    main()
