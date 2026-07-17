"""
1) Коэффициенты вейвлет-преобразования:
       W_c,s(x, y)

2) Мощности вейвлет-коэффициентов:
       P_c,s(x, y) = W_c,s(x, y)^2

3) Кросс-спектры как произведение спектров:
       C_ab(x, y) = P_a(x, y) * P_b(x, y)

4) Бинарные события появления пиков:
       B_c,s(t) in {0, 1}

   Для изображения временная ось заменяется пространственной осью:
   - файлы "построчно"    -> t = x, то есть движение по строкам слева направо;
   - файлы "по_столбцам" -> t = y, то есть движение по столбцам сверху вниз.

5) Модели синхронизации:
   A: один канал, одно окно, разные частотные диапазоны/масштабы;
   Б: один частотный диапазон/масштаб, одно окно, разные каналы;
   В: один канал, разные окна со сдвигом, разные диапазоны/масштабы;
   Г: разные каналы, разные диапазоны/масштабы, разные окна со сдвигом.

6) Для каждой пары бинарных рядов в каждом окне строится таблица 2x2:

            B=1   B=0
      A=1   n11   n10
      A=0   n01   n00

   и считаются статистики:
   - Jaccard;
   - Dice;
   - phi coefficient;
   - odds ratio;
   - chi-square p-value;
   - Fisher exact p-value;
   - McNemar p-value with Yates correction;
   - binomial p-value;
   - binomial confidence interval for n11 / n.

Запуск:
    python wavelet_sync_pipeline.py "C:\\Users\\...\\Задача 1 04_06_2026_12_58"

Пример с параметрами:
    python wavelet_sync_pipeline.py "C:\\Users\\...\\Задача 1" --window-size 100 --step 25 --lag 5 --peak-percentile 95

Результаты появятся в папке:
    WaveletSyncResults
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import chi2, chi2_contingency, fisher_exact, binomtest, beta
SCIPY_AVAILABLE = True


WAVELET_RE = re.compile(
    r"^Расчет_вейвлетов_(?P<direction>по_столбцам|построчно)_"
    r"Масштаб_(?P<scale>[-+]?\d+(?:\.\d+)?)_"
    r"(?P<channel>Красный|Зелёный|Зеленый|Синий|Red|Green|Blue)\.txt$",
    re.IGNORECASE,
)

CHANNEL_NORMALIZE = {
    "Красный": "Red",
    "Зелёный": "Green",
    "Зеленый": "Green",
    "Синий": "Blue",
    "Red": "Red",
    "Green": "Green",
    "Blue": "Blue",
}

CHANNEL_ORDER = {"Red": 0, "Green": 1, "Blue": 2}

DIRECTION_NORMALIZE = {
    "построчно": "rows",
    "по_столбцам": "cols",
}

DIRECTION_RU = {
    "rows": "построчно",
    "cols": "по_столбцам",
}


@dataclass(frozen=True)
class WaveletFile:
    path: Path
    direction: str
    scale_label: str
    scale_value: float
    channel: str


@dataclass(frozen=True)
class PairSpec:
    model: str
    channel_a: str
    band_a: str
    channel_b: str
    band_b: str
    lag: int


@dataclass
class PipelineData:
    files: List[WaveletFile]
    powers: Dict[Tuple[str, str, str], np.ndarray]
    channels: List[str]
    scales: List[str]
    directions: List[str]

def parse_wavelet_file(path: Path) -> Optional[WaveletFile]:
    match = WAVELET_RE.match(path.name)
    if not match:
        return None

    direction_raw = match.group("direction")
    scale_label = match.group("scale")
    channel_raw = match.group("channel")

    return WaveletFile(
        path=path,
        direction=DIRECTION_NORMALIZE[direction_raw],
        scale_label=scale_label,
        scale_value=float(scale_label),
        channel=CHANNEL_NORMALIZE[channel_raw],
    )


def find_wavelet_files(task_folder: Path) -> List[WaveletFile]:
    files: List[WaveletFile] = []
    for txt_path in task_folder.rglob("*.txt"):
        parsed = parse_wavelet_file(txt_path)
        if parsed is not None:
            files.append(parsed)

    return sorted(
        files,
        key=lambda f: (
            f.direction,
            f.scale_value,
            CHANNEL_ORDER.get(f.channel, 99),
            str(f.path),
        ),
    )


def load_matrix(path: Path) -> np.ndarray:
    matrix = np.loadtxt(path, delimiter=",")
    if matrix.ndim != 2:
        raise ValueError(f"Ожидалась 2D-матрица, получено {matrix.shape}: {path}")
    return matrix.astype(np.float64, copy=False)


def save_matrix(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, matrix, fmt="%.6f", delimiter=",")


def safe_name(text: str) -> str:
    return (
        text.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(";", "_")
    )


def compute_power_spectra(files: Sequence[WaveletFile], output_root: Path) -> Dict[Tuple[str, str, str], np.ndarray]:
    """
    Для каждого файла коэффициентов W считает P = W^2.

    Ключ словаря:
        (direction, scale_label, channel)
    """

    powers: Dict[Tuple[str, str, str], np.ndarray] = {}
    power_dir = output_root / "01_power_spectra"

    for wf in files:
        W = load_matrix(wf.path)
        P = W ** 2

        key = (wf.direction, wf.scale_label, wf.channel)
        powers[key] = P

        out_name = safe_name(
            f"Мощность_вейвлетов_{DIRECTION_RU[wf.direction]}_Масштаб_{wf.scale_label}_{wf.channel}.txt"
        )
        save_matrix(power_dir / wf.direction / f"Scale_{wf.scale_label}" / out_name, P)

    return powers


def compute_and_save_cross_spectra(
    powers: Dict[Tuple[str, str, str], np.ndarray],
    directions: Sequence[str],
    scales: Sequence[str],
    channels: Sequence[str],
    output_root: Path,
) -> int:
    """
    Считает C = P1 * P2.

    Сохраняются кросс-спектры для моделей:
    A: один канал, разные масштабы;
    Б: один масштаб, разные каналы;
    Г: разные каналы и разные масштабы.

    Для модели В отдельный кросс-спектр не создаётся, потому что отличие В от A
    появляется на этапе бинарных рядов через лаг/сдвиг.
    """

    count = 0
    cross_root = output_root / "02_cross_spectra"

    # A: same channel, different scales
    for direction in directions:
        for channel in channels:
            for s1, s2 in itertools.combinations(scales, 2):
                k1 = (direction, s1, channel)
                k2 = (direction, s2, channel)
                if k1 not in powers or k2 not in powers:
                    continue
                C = powers[k1] * powers[k2]
                out_name = safe_name(
                    f"Cross_A_{direction}_Channel_{channel}_Scale_{s1}_x_{s2}.txt"
                )
                save_matrix(cross_root / "A_same_channel_diff_scales" / direction / f"Channel_{channel}" / out_name, C)
                count += 1

    # Б: same scale, different channels
    for direction in directions:
        for scale in scales:
            for c1, c2 in itertools.combinations(channels, 2):
                k1 = (direction, scale, c1)
                k2 = (direction, scale, c2)
                if k1 not in powers or k2 not in powers:
                    continue
                C = powers[k1] * powers[k2]
                out_name = safe_name(
                    f"Cross_B_{direction}_Scale_{scale}_Channels_{c1}_x_{c2}.txt"
                )
                save_matrix(cross_root / "B_same_scale_diff_channels" / direction / f"Scale_{scale}" / out_name, C)
                count += 1

    # Г: different channels, different scales
    for direction in directions:
        for c1, c2 in itertools.combinations(channels, 2):
            for s1 in scales:
                for s2 in scales:
                    if s1 == s2:
                        continue
                    k1 = (direction, s1, c1)
                    k2 = (direction, s2, c2)
                    if k1 not in powers or k2 not in powers:
                        continue
                    C = powers[k1] * powers[k2]
                    out_name = safe_name(
                        f"Cross_G_{direction}_Channel_{c1}_Scale_{s1}_x_Channel_{c2}_Scale_{s2}.txt"
                    )
                    save_matrix(cross_root / "G_diff_channels_diff_scales" / direction / f"{c1}_x_{c2}" / out_name, C)
                    count += 1

    return count


def local_peak_events_from_power(P: np.ndarray, direction: str, percentile: float) -> np.ndarray:
    """
    Делает из 2D-карты мощности 1D бинарный ряд событий.

    Для direction='rows':
        анализ идёт вдоль строк, то есть локальные пики ищутся по оси X.
        Итоговый ряд имеет длину width.

    Для direction='cols':
        анализ идёт вдоль столбцов, то есть локальные пики ищутся по оси Y.
        Итоговый ряд имеет длину height.

    Порог нужен, чтобы не считать шумовые локальные максимумы событиями.
    По умолчанию берём 95-й процентиль мощности.
    """

    threshold = float(np.percentile(P, percentile))

    if direction == "rows":
        # P shape: (height, width), локальные максимумы вдоль width
        if P.shape[1] < 3:
            return np.zeros(P.shape[1], dtype=np.uint8)
        center = P[:, 1:-1]
        left = P[:, :-2]
        right = P[:, 2:]
        mask_inner = (center > left) & (center > right) & (center >= threshold)
        events = np.zeros(P.shape[1], dtype=np.uint8)
        events[1:-1] = np.any(mask_inner, axis=0).astype(np.uint8)
        return events

    if direction == "cols":
        # P shape: (height, width), локальные максимумы вдоль height
        if P.shape[0] < 3:
            return np.zeros(P.shape[0], dtype=np.uint8)
        center = P[1:-1, :]
        up = P[:-2, :]
        down = P[2:, :]
        mask_inner = (center > up) & (center > down) & (center >= threshold)
        events = np.zeros(P.shape[0], dtype=np.uint8)
        events[1:-1] = np.any(mask_inner, axis=1).astype(np.uint8)
        return events

    raise ValueError(f"Неизвестное направление: {direction}")


def build_event_cube(
    powers: Dict[Tuple[str, str, str], np.ndarray],
    direction: str,
    scales: Sequence[str],
    channels: Sequence[str],
    peak_percentile: float,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Формирует массив событий вида:
        events[channel_index, scale_index, t]
    """

    rows: List[np.ndarray] = []
    used_pairs: List[Tuple[str, str]] = []
    expected_length: Optional[int] = None

    for channel in channels:
        for scale in scales:
            key = (direction, scale, channel)
            if key not in powers:
                continue
            event_row = local_peak_events_from_power(powers[key], direction, peak_percentile)
            if expected_length is None:
                expected_length = len(event_row)
            elif len(event_row) != expected_length:
                raise ValueError(
                    f"Разные длины бинарных рядов для direction={direction}: "
                    f"{len(event_row)} != {expected_length}"
                )
            rows.append(event_row)
            used_pairs.append((channel, scale))

    if not rows:
        raise ValueError(f"Нет данных для построения событий direction={direction}")

    channel_list = sorted({ch for ch, _ in used_pairs}, key=lambda c: CHANNEL_ORDER.get(c, 99))
    scale_list = sorted({sc for _, sc in used_pairs}, key=float)

    events = np.zeros((len(channel_list), len(scale_list), expected_length), dtype=np.uint8)
    pair_to_row = {(ch, sc): row for (ch, sc), row in zip(used_pairs, rows)}

    for ci, ch in enumerate(channel_list):
        for si, sc in enumerate(scale_list):
            row = pair_to_row.get((ch, sc))
            if row is not None:
                events[ci, si] = row

    return events, channel_list, scale_list


def save_binary_events(
    events: np.ndarray,
    channels: Sequence[str],
    scales: Sequence[str],
    direction: str,
    output_root: Path,
) -> None:
    event_dir = output_root / "03_binary_events" / direction
    event_dir.mkdir(parents=True, exist_ok=True)

    for ci, channel in enumerate(channels):
        for si, scale in enumerate(scales):
            out_name = safe_name(f"Events_{direction}_Channel_{channel}_Scale_{scale}.txt")
            np.savetxt(event_dir / out_name, events[ci, si][None, :], fmt="%d", delimiter=",")


def make_scale_blocks(scales: Sequence[str], block_size: int) -> List[Tuple[str, List[str]]]:
    ordered = sorted(scales, key=float)
    blocks: List[Tuple[str, List[str]]] = []

    if block_size <= 1:
        return []

    for idx in range(0, len(ordered), block_size):
        part = list(ordered[idx:idx + block_size])
        if not part:
            continue
        label = f"Block_{len(blocks) + 1}_{part[0]}-{part[-1]}"
        blocks.append((label, part))

    return blocks


def build_block_event_cube(
    events: np.ndarray,
    scales: Sequence[str],
    block_size: int,
) -> Tuple[np.ndarray, List[str], Dict[str, str]]:
    blocks = make_scale_blocks(scales, block_size)
    if not blocks:
        return np.empty((events.shape[0], 0, events.shape[2]), dtype=np.uint8), [], {}

    scale_index = {scale: idx for idx, scale in enumerate(scales)}
    block_rows: List[np.ndarray] = []
    block_labels: List[str] = []
    block_map: Dict[str, str] = {}

    for label, block_scales in blocks:
        indices = [scale_index[s] for s in block_scales if s in scale_index]
        if not indices:
            continue
        block_events = np.any(events[:, indices, :].astype(bool), axis=1).astype(np.uint8)
        block_rows.append(block_events)
        block_labels.append(label)
        block_map[label] = ",".join(block_scales)

    if not block_rows:
        return np.empty((events.shape[0], 0, events.shape[2]), dtype=np.uint8), [], {}

    return np.stack(block_rows, axis=1), block_labels, block_map


def save_binary_block_events(
    block_events: np.ndarray,
    channels: Sequence[str],
    block_labels: Sequence[str],
    block_map: Dict[str, str],
    direction: str,
    output_root: Path,
) -> None:
    event_dir = output_root / "03_binary_events_blocks" / direction
    event_dir.mkdir(parents=True, exist_ok=True)

    for ci, channel in enumerate(channels):
        for bi, block_label in enumerate(block_labels):
            out_name = safe_name(f"EventsBlock_{direction}_Channel_{channel}_{block_label}.txt")
            np.savetxt(event_dir / out_name, block_events[ci, bi][None, :], fmt="%d", delimiter=",")

    map_rows = [
        {"block": label, "scales": block_map.get(label, "")}
        for label in block_labels
    ]
    save_csv(output_root / "03_binary_events_blocks" / direction / "scale_blocks.csv", map_rows)


def contingency_2x2(a: np.ndarray, b: np.ndarray) -> Tuple[int, int, int, int]:
    x = np.asarray(a).astype(bool).ravel()
    y = np.asarray(b).astype(bool).ravel()
    if x.shape != y.shape:
        raise ValueError(f"Ряды должны иметь одинаковую длину: {x.shape} != {y.shape}")

    n11 = int(np.sum(x & y))
    n10 = int(np.sum(x & ~y))
    n01 = int(np.sum(~x & y))
    n00 = int(np.sum(~x & ~y))
    return n11, n10, n01, n00


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


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


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def chi_square_p_fallback(n11: int, n10: int, n01: int, n00: int) -> float:
    # Для 2x2 chi-square statistic, p = survival chi-square(df=1).
    n = n11 + n10 + n01 + n00
    row1 = n11 + n10
    row2 = n01 + n00
    col1 = n11 + n01
    col2 = n10 + n00
    den = row1 * row2 * col1 * col2
    if den == 0:
        return 1.0
    stat = n * (n11 * n00 - n10 * n01) ** 2 / den
    # chi-square df=1 survival = 2 * (1 - Phi(sqrt(stat)))
    return max(0.0, min(1.0, 2.0 * (1.0 - normal_cdf(math.sqrt(stat)))))


def binomial_two_sided_p_fallback(k: int, n: int, p: float) -> float:
    if n <= 0:
        return 1.0
    if p <= 0.0:
        return 1.0 if k == 0 else 0.0
    if p >= 1.0:
        return 1.0 if k == n else 0.0

    # Простая точная двухсторонняя оценка: суммируем вероятности исходов
    # не более вероятных, чем наблюдаемый k.
    probs = []
    for i in range(n + 1):
        prob = math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
        probs.append(prob)
    observed = probs[k]
    return max(0.0, min(1.0, sum(prob for prob in probs if prob <= observed + 1e-15)))


def binomial_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0

    if SCIPY_AVAILABLE and beta is not None:
        low = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
        high = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
        return low, high

    # Wilson fallback
    z = 1.959963984540054
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def sync_statistics(a: np.ndarray, b: np.ndarray) -> Dict[str, float | int]:
    n11, n10, n01, n00 = contingency_2x2(a, b)
    n = n11 + n10 + n01 + n00

    jaccard = safe_div(n11, n11 + n10 + n01)
    dice = safe_div(2 * n11, 2 * n11 + n10 + n01)

    phi_den = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    phi = safe_div(n11 * n00 - n10 * n01, phi_den)

    odds_ratio = safe_div((n11 + 0.5) * (n00 + 0.5), (n10 + 0.5) * (n01 + 0.5))

    table = np.array([[n11, n10], [n01, n00]], dtype=np.int64)

    if SCIPY_AVAILABLE:
        try:
            _, chi2_p, _, _ = chi2_contingency(table, correction=False)
            chi2_p = float(chi2_p)
        except Exception:
            chi2_p = 1.0

        try:
            _, fisher_p = fisher_exact(table, alternative="two-sided")
            fisher_p = float(fisher_p)
        except Exception:
            fisher_p = 1.0
    else:
        chi2_p = chi_square_p_fallback(n11, n10, n01, n00)
        fisher_p = 1.0

    discordant = n10 + n01
    if discordant == 0:
        mcnemar_p = 1.0
    else:
        stat = (abs(n10 - n01) - 1.0) ** 2 / discordant
        if SCIPY_AVAILABLE and chi2 is not None:
            mcnemar_p = float(chi2.sf(stat, 1))
        else:
            mcnemar_p = max(0.0, min(1.0, 2.0 * (1.0 - normal_cdf(math.sqrt(stat)))))

    # Биномиальная проверка - среди событий A проверяем число совпадений с B
    # относительно общей вероятности события B в окне.
    trials = n11 + n10
    p_expected = safe_div(n11 + n01, n)
    if trials == 0:
        binomial_p = 1.0
    else:
        if SCIPY_AVAILABLE and binomtest is not None:
            binomial_p = float(binomtest(n11, trials, p_expected, alternative="two-sided").pvalue)
        else:
            binomial_p = binomial_two_sided_p_fallback(n11, trials, p_expected)

    ci_low, ci_high = binomial_ci(n11, n)

    return {
        "n": n,
        "n11_both_1": n11,
        "n10_only_a": n10,
        "n01_only_b": n01,
        "n00_both_0": n00,
        "jaccard": jaccard,
        "dice": dice,
        "phi": phi,
        "odds_ratio": odds_ratio,
        "chi2_p": chi2_p,
        "fisher_p": fisher_p,
        "mcnemar_p": mcnemar_p,
        "binomial_p": binomial_p,
        "n11_ratio": safe_div(n11, n),
        "n11_ci_low": ci_low,
        "n11_ci_high": ci_high,
    }


def lagged_windows(a_full: np.ndarray, b_full: np.ndarray, start: int, end: int, lag: int) -> Tuple[np.ndarray, np.ndarray]:
    length = len(a_full)
    a_start = start
    a_end = end
    b_start = start + lag
    b_end = end + lag

    if b_start < 0:
        shift = -b_start
        a_start += shift
        b_start = 0

    if b_end > length:
        shift = b_end - length
        a_end -= shift
        b_end = length

    if a_end <= a_start or b_end <= b_start:
        return np.array([], dtype=np.uint8), np.array([], dtype=np.uint8)

    return a_full[a_start:a_end], b_full[b_start:b_end]


def make_windows(length: int, window_size: int, step: int) -> Iterable[Tuple[int, int]]:
    if window_size <= 0 or step <= 0:
        raise ValueError("window_size и step должны быть положительными")
    if window_size > length:
        yield 0, length
        return
    for start in range(0, length - window_size + 1, step):
        yield start, start + window_size


def generate_pairs_for_model(
    model: str,
    channels: Sequence[str],
    bands: Sequence[str],
    lag: int,
) -> List[PairSpec]:
    pairs: List[PairSpec] = []

    if model == "A":
        for ch in channels:
            for b1, b2 in itertools.combinations(bands, 2):
                pairs.append(PairSpec("A", ch, b1, ch, b2, 0))

    elif model == "B":
        for band in bands:
            for c1, c2 in itertools.combinations(channels, 2):
                pairs.append(PairSpec("B", c1, band, c2, band, 0))

    elif model == "V":
        for ch in channels:
            for b1, b2 in itertools.combinations(bands, 2):
                pairs.append(PairSpec("V", ch, b1, ch, b2, lag))

    elif model == "G":
        for c1, c2 in itertools.permutations(channels, 2):
            for b1 in bands:
                for b2 in bands:
                    if b1 == b2:
                        continue
                    pairs.append(PairSpec("G", c1, b1, c2, b2, lag))
    else:
        raise ValueError(f"Неизвестная модель: {model}")

    return pairs


def analyze_models_for_direction(
    events: np.ndarray,
    channels: Sequence[str],
    bands: Sequence[str],
    direction: str,
    window_size: int,
    step: int,
    lag: int,
    band_type: str = "scale",
    band_map: Optional[Dict[str, str]] = None,
) -> List[Dict[str, object]]:
    ch_index = {ch: i for i, ch in enumerate(channels)}
    band_index = {band: i for i, band in enumerate(bands)}
    length = events.shape[2]
    axis_name = "X" if direction == "rows" else "Y"

    rows: List[Dict[str, object]] = []

    for model in ["A", "B", "V", "G"]:
        pairs = generate_pairs_for_model(model, channels, bands, lag)
        for start, end in make_windows(length, window_size, step):
            for pair in pairs:
                a_full = events[ch_index[pair.channel_a], band_index[pair.band_a]]
                b_full = events[ch_index[pair.channel_b], band_index[pair.band_b]]

                a_win, b_win = lagged_windows(a_full, b_full, start, end, pair.lag)
                if len(a_win) == 0:
                    continue

                stats = sync_statistics(a_win, b_win)
                row = {
                    "direction": direction,
                    "axis": axis_name,
                    "model": model,
                    "band_type": band_type,
                    "start": start,
                    "end": end,
                    "window_size": window_size,
                    "step": step,
                    "actual_length": len(a_win),
                    "lag": pair.lag,
                    "channel_a": pair.channel_a,
                    "band_a": pair.band_a,
                    "band_a_scales": band_map.get(pair.band_a, pair.band_a) if band_map else pair.band_a,
                    "channel_b": pair.channel_b,
                    "band_b": pair.band_b,
                    "band_b_scales": band_map.get(pair.band_b, pair.band_b) if band_map else pair.band_b,
                }
                row.update(stats)
                rows.append(row)

    return rows


def run_lengths(flags: Sequence[bool]) -> List[int]:
    lengths: List[int] = []
    current = 0
    for flag in flags:
        if flag:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def compute_significant_runs(
    rows: Sequence[Dict[str, object]],
    alpha: float,
    p_metrics: Sequence[str] = ("chi2_p", "fisher_p", "mcnemar_p", "binomial_p"),
) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[object, ...], List[Dict[str, object]]] = defaultdict(list)

    for row in rows:
        key = (
            row.get("direction"),
            row.get("axis"),
            row.get("model"),
            row.get("band_type"),
            row.get("lag"),
            row.get("channel_a"),
            row.get("band_a"),
            row.get("band_a_scales"),
            row.get("channel_b"),
            row.get("band_b"),
            row.get("band_b_scales"),
        )
        grouped[key].append(row)

    result: List[Dict[str, object]] = []

    for key, group in grouped.items():
        group_sorted = sorted(group, key=lambda r: (parse_int(r.get("start")), parse_int(r.get("end"))))

        for p_metric in p_metrics:
            flags = []
            for row in group_sorted:
                p_value = parse_float(row.get(p_metric))
                flags.append((not math.isnan(p_value)) and p_value < alpha)

            lengths = run_lengths(flags)
            total_windows = len(flags)
            significant_windows = int(sum(flags))

            result.append({
                "direction": key[0],
                "axis": key[1],
                "model": key[2],
                "band_type": key[3],
                "lag": key[4],
                "channel_a": key[5],
                "band_a": key[6],
                "band_a_scales": key[7],
                "channel_b": key[8],
                "band_b": key[9],
                "band_b_scales": key[10],
                "p_metric": p_metric,
                "alpha": alpha,
                "total_windows": total_windows,
                "significant_windows": significant_windows,
                "significant_ratio": safe_div(significant_windows, total_windows),
                "max_significant_run": max(lengths) if lengths else 0,
                "mean_significant_run": float(np.mean(lengths)) if lengths else 0.0,
                "number_of_runs": len(lengths),
            })

    return result


def save_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Постобработка: мощности, кросс-спектры, модели синхронизации, статистика."
    )
    parser.add_argument("task_folder", help="Папка задачи, внутри которой лежат Scale_* с txt коэффициентами")
    parser.add_argument("--output", default="WaveletSyncResults", help="Имя папки результатов")
    parser.add_argument("--window-size", type=int, default=100, help="Размер скользящего окна")
    parser.add_argument("--step", type=int, default=25, help="Шаг скользящего окна")
    parser.add_argument("--lag", type=int, default=5, help="Сдвиг для моделей В и Г")
    parser.add_argument("--peak-percentile", type=float, default=95.0, help="Процентиль порога для выделения пиков")
    parser.add_argument("--scale-block-size", type=int, default=3, help="Количество соседних масштабов в одном частотном блоке")
    parser.add_argument("--alpha", type=float, default=0.05, help="Уровень значимости для критерия серий")

    args = parser.parse_args()

    task_folder = Path(args.task_folder)
    if not task_folder.exists():
        raise FileNotFoundError(f"Папка не найдена: {task_folder}")

    output_root = task_folder / args.output
    output_root.mkdir(parents=True, exist_ok=True)

    print("Этап 0. Поиск файлов коэффициентов...")
    files = find_wavelet_files(task_folder)
    if not files:
        print("Не найдено файлов вида Расчет_вейвлетов_*.txt")
        return

    directions = sorted({f.direction for f in files})
    scales = sorted({f.scale_label for f in files}, key=float)
    channels = sorted({f.channel for f in files}, key=lambda c: CHANNEL_ORDER.get(c, 99))

    print(f"Найдено файлов: {len(files)}")
    print(f"Направления: {directions}")
    print(f"Масштабы: {scales}")
    print(f"Каналы: {channels}")

    print("Этап 1. Расчёт мощностей P = W^2...")
    powers = compute_power_spectra(files, output_root)
    print(f"Сохранено мощностей: {len(powers)}")

    print("Этап 2. Расчёт кросс-спектров C = P1 * P2...")
    cross_count = compute_and_save_cross_spectra(powers, directions, scales, channels, output_root)
    print(f"Сохранено кросс-спектров: {cross_count}")

    print("Этап 3. Выделение бинарных событий пиков...")
    all_rows: List[Dict[str, object]] = []

    for direction in directions:
        events, used_channels, used_scales = build_event_cube(
            powers=powers,
            direction=direction,
            scales=scales,
            channels=channels,
            peak_percentile=args.peak_percentile,
        )
        save_binary_events(events, used_channels, used_scales, direction, output_root)
        print(
            f"  {direction}: events shape = {events.shape} "
            f"(channels, scales, spatial_time)"
        )

        print(f"Этап 4. Модели синхронизации и статистика для {direction}...")
        rows = analyze_models_for_direction(
            events=events,
            channels=used_channels,
            bands=used_scales,
            direction=direction,
            window_size=args.window_size,
            step=args.step,
            lag=args.lag,
            band_type="scale",
        )

        block_events, block_labels, block_map = build_block_event_cube(
            events=events,
            scales=used_scales,
            block_size=args.scale_block_size,
        )

        if block_events.size != 0 and block_labels:
            save_binary_block_events(
                block_events=block_events,
                channels=used_channels,
                block_labels=block_labels,
                block_map=block_map,
                direction=direction,
                output_root=output_root,
            )
            block_rows = analyze_models_for_direction(
                events=block_events,
                channels=used_channels,
                bands=block_labels,
                direction=direction,
                window_size=args.window_size,
                step=args.step,
                lag=args.lag,
                band_type="block",
                band_map=block_map,
            )
            rows.extend(block_rows)
            print(f"  блоки масштабов для {direction}: {block_map}")
            print(f"  строк статистики по блокам для {direction}: {len(block_rows)}")

        all_rows.extend(rows)

        save_csv(
            output_root / "04_synchronization_statistics" / f"sync_statistics_{direction}.csv",
            rows,
        )

        run_rows_direction = compute_significant_runs(rows, alpha=args.alpha)
        save_csv(
            output_root / "04_synchronization_statistics" / f"sync_significant_runs_{direction}.csv",
            run_rows_direction,
        )

        print(f"  строк статистики для {direction}: {len(rows)}")
        print(f"  строк критерия серий для {direction}: {len(run_rows_direction)}")

    save_csv(output_root / "04_synchronization_statistics" / "sync_statistics_all.csv", all_rows)

    run_rows_all = compute_significant_runs(all_rows, alpha=args.alpha)
    save_csv(output_root / "04_synchronization_statistics" / "sync_significant_runs_all.csv", run_rows_all)

    summary = [
        "Готово.",
        f"Файлов коэффициентов: {len(files)}",
        f"Мощностей W^2: {len(powers)}",
        f"Кросс-спектров P1*P2: {cross_count}",
        f"Строк статистики: {len(all_rows)}",
        f"window_size: {args.window_size}",
        f"step: {args.step}",
        f"lag: {args.lag}",
        f"peak_percentile: {args.peak_percentile}",
        f"scale_block_size: {args.scale_block_size}",
        f"alpha: {args.alpha}",
        f"significant_run_rows: {len(run_rows_all)}",
        f"scipy_available: {SCIPY_AVAILABLE}",
        "",
        "Структура результатов:",
        "01_power_spectra — мощности W^2",
        "02_cross_spectra — произведения спектров P1*P2",
        "03_binary_events — бинарные ряды событий пиков",
        "04_synchronization_statistics — CSV со статистиками моделей A/Б/В/Г",
        "sync_significant_runs_all.csv — критерий серий значимых окон",
        "03_binary_events_blocks — бинарные события после сжатия масштабов в блоки",
    ]
    (output_root / "summary.txt").write_text("\n".join(summary), encoding="utf-8")

    print("Готово.")
    print(f"Папка результатов: {output_root}")
    print(f"Общая таблица статистики: {output_root / '04_synchronization_statistics' / 'sync_statistics_all.csv'}")


if __name__ == "__main__":
    main()