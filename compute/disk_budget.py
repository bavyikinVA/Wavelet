"""Preflight capacity checks for long-running scientific queues.

Estimates are intentionally labelled as forecasts.  They are not disk quotas:
outputs (especially peaks/KNN) depend on image content and optional exports.
Atomic writes and ENOSPC handling remain authoritative during execution.
"""
from __future__ import annotations

import errno
import shutil
from pathlib import Path

MIB = 1024 ** 2
# Keep journal/report space available when a long queue is almost full.
SAFETY_RESERVE = 64 * MIB


def _existing_ancestor(folder):
    path = Path(folder).expanduser().resolve()
    while not path.exists():
        parent = path.parent
        if path == parent:
            raise FileNotFoundError(f'Невозможно определить файловую систему: {folder}')
        path = parent
    return path


def disk_free_bytes(folder):
    return int(shutil.disk_usage(_existing_ancestor(folder)).free)


def _image_shape(task, row=None):
    image = getattr(task, 'original_image', None)
    if image is not None:
        return tuple(image.shape[:2])
    from PIL import Image
    candidates = [
        (row or {}).get('input_path'),
        getattr(task, '_input_snapshot_path', None),
        getattr(task, 'image_path', None),
    ]
    for filename in candidates:
        if not filename:
            continue
        path = Path(filename)
        if not path.is_file():
            continue
        try:
            with Image.open(path) as original:
                return original.height, original.width
        except OSError as error:
            if error.errno in {errno.ENOSPC, errno.EROFS, getattr(errno, 'EDQUOT', errno.ENOSPC)}:
                raise
            # A corrupt or unreadable snapshot must be diagnosed by the
            # checksum-aware input loader, not by the capacity estimator.
            continue
    return None


def forecast_task_bytes(task, row=None):
    """Return a *rough* disk forecast or None for an input without dimensions.

    Calibrated for uncompressed float32/complex64 CWT and sparse point/knn
    artifacts.  The forecast is NOT a guaranteed upper or lower bound; export
    density can vary by orders of magnitude with the image and settings.
    """
    shape = _image_shape(task, row)
    if shape is None:
        return None
    height, width = shape
    pixels = max(0, int(height) * int(width))
    scales = max(1, len(getattr(task, 'scales', ())))
    channel_codes = task.analysis_channel_codes_for_settings()
    channels = max(1, len(channel_codes))
    if getattr(task, 'analysis_mode', '1d') == '2d':
        angles = max(1, len(getattr(task, 'orientations', ())))
        cwt = pixels * scales * channels * angles * 8  # complex64
        requested_points = False
        requested_knn = False
    else:
        directions = int(bool(getattr(task, 'process_rows', True))) + int(
            bool(getattr(task, 'process_columns', True)))
        cwt = pixels * scales * channels * max(1, directions) * 4  # float32
        plan = task.resolve_pipeline()
        requested_points = bool(plan.extrema or plan.envelopes)
        requested_knn = bool(plan.knn)
    # Calibrated against two 1920x2560 RGB/4-scale/2-direction GPU runs
    # with distance=1, prominence=0, KNN k=5 (2026-11-03): point NPZ
    # 800.8–905.8 MiB and KNN NPZ 1480.0–1931.5 MiB, against 450 MiB
    # CWT each.  Coefficients are a mathematical size; the other factors are
    # *heuristics*, not upper bounds for arbitrary image textures.
    point_bytes = int(cwt * 2.1) if requested_points else 0
    # KNN NPZ grows with the number of links; interpolate fixed per-point
    # features and k-dependent neighbour arrays (reference case: k=5).
    k = max(1, int(getattr(task, 'k_neighbors', 5)))
    knn_factor = 4.5 * (0.30 + 0.70 * k / 5.0)
    knn_bytes = int(cwt * knn_factor) if requested_knn else 0
    extra = 0
    if getattr(task, 'output_wavelet_text', False):
        extra += cwt * 3  # text values are larger than binary float32
    if getattr(task, 'output_wavelet_image', False):
        extra += cwt
    if requested_points:
        if getattr(task, 'output_extremes_text', False):
            extra += point_bytes * 2
        if getattr(task, 'output_envelopes_text', False):
            extra += point_bytes * 2
        if (getattr(task, 'output_extremes_image', False)
                or getattr(task, 'output_envelopes_image', False)):
            extra += cwt // 2
    if requested_knn:
        if getattr(task, 'output_knn_text', False):
            extra += knn_bytes * 2
        if getattr(task, 'output_knn_image', False):
            extra += cwt // 2

    stored = int(cwt + point_bytes + knn_bytes + extra)
    # atomic_* writers temporarily require one extra artifact on the same
    # filesystem.  The true largest NPZ/PNG is data-dependent.  Model it as
    # a fraction of sparse scientific output, or one float32 CWT file.
    single_coefficient = pixels * (8 if getattr(task, 'analysis_mode', '1d') == '2d' else 4)
    temporary = max(single_coefficient, int(0.06 * (point_bytes + knn_bytes)))
    return dict(
        image_shape=(height, width),
        coefficient_bytes=int(cwt),
        forecast_bytes=stored,
        temporary_bytes=int(temporary),
        working_bytes=stored + int(temporary),
        multiplier=stored / max(1, cwt),
    )


def check_task_capacity(task, folder, *, row=None, reserve_bytes=SAFETY_RESERVE,
                        free_bytes=None, log=None):
    """Prevent starting known-impossible work, without treating a forecast as a limit.

    The guaranteed binary CWT output is a *lower bound* for a new run.  The
    point/KNN volume is content-dependent, so an estimate larger than free
    space produces a warning, not a false-positive failure.  OS ENOSPC still
    causes a global, durably journaled queue stop.
    """
    free = disk_free_bytes(folder) if free_bytes is None else int(free_bytes)
    forecast = forecast_task_bytes(task, row)
    # Keep enough space for the journal and the report even for unknown images.
    required = reserve_bytes + (forecast['coefficient_bytes'] if forecast else 0)
    if free < required:
        predicted = (f' и как минимум {forecast["coefficient_bytes"] / MIB:.1f} МиБ на CWT'
                     if forecast else '')
        raise OSError(
            errno.ENOSPC,
            f'Недостаточно места для задачи «{task.task_name}»: доступно '
            f'{free / MIB:.1f} МиБ, требуется резерв {reserve_bytes / MIB:.1f} МиБ'
            f'{predicted}. Освободите место или выберите другой диск.',
        )
    if forecast and forecast['working_bytes'] + reserve_bytes > free and log:
        log(f'Диск: {task.task_name}: ориентировочно потребуется '
            f'{forecast["forecast_bytes"] / MIB:.1f} МиБ файлов + '
            f'{forecast["temporary_bytes"] / MIB:.1f} МиБ для атомарной записи; '
            f'свободно {free / MIB:.1f} МиБ; при нехватке места очередь остановится.')
    return dict(free_bytes=free, reserve_bytes=reserve_bytes, **(forecast or {}))


def check_snapshot_capacity(task, folder, *, reserve_bytes=SAFETY_RESERVE):
    """The cropped PNG is never larger than a modest uncompressed RGB bound.

    This check is done before snapshotting the whole queue: otherwise hundreds
    of images may fill the disk before the scientific work even begins.
    """
    shape = _image_shape(task)
    free = disk_free_bytes(folder)
    estimated = (int(shape[0]) * int(shape[1]) * 3 + MIB) if shape else 0
    if free < estimated + reserve_bytes:
        raise OSError(errno.ENOSPC,
                      f'Недостаточно места для сохранения входа «{task.task_name}» '
                      f'(свободно {free / MIB:.1f} МиБ).')
    return free


def queue_disk_forecast(tasks, root):
    """Informational total, never used as a hard quota for the whole queue."""
    values = [forecast_task_bytes(task) for task in tasks]
    recognized = [value for value in values if value is not None]
    return dict(free_bytes=disk_free_bytes(root),
                forecast_bytes=sum(value['forecast_bytes'] for value in recognized),
                temporary_bytes=max((value['temporary_bytes'] for value in recognized), default=0),
                recognized=len(recognized), unknown=len(values)-len(recognized))
