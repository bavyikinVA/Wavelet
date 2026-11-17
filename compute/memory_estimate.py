"""Stage-aware *planning* estimate, not a guaranteed peak process RSS.

The 1-D runner holds the CWT coefficients in float32 but processes individual
channel/direction transforms and point/KNN datasets sequentially.  In
particular, the total number of KNN neighbours written over an entire run is
not a simultaneous allocation.  The previous dense-point ``pixels * scales *
channels * k * 48`` term was therefore misleading by an order of magnitude.

These estimates describe working arrays.  StageProfile supplies measured RSS
from real executions; it is deliberately kept separate from this prediction.
"""
import math

import numpy as np


_F32 = 4
_MIB = 1024 ** 2
_GIB = 1024 ** 3


def _shape(task):
    image = getattr(task, 'original_image', None)
    if image is not None:
        return image.shape[:2]
    # Input arrays may already have been released. Reading only the PNG header
    # is much cheaper and does not bring a full image into RAM.
    from compute.disk_budget import _image_shape
    return _image_shape(task)


def _task_working_bytes(task, h, w):
    n = int(h) * int(w)
    scales = len(task.scales)
    channels = max(1, len(task.analysis_channel_codes_for_settings()))
    if task.analysis_mode == '2d':
        radius = max(2, math.ceil(4 * max(task.scales) * max(1., task.morlet_anisotropy)))
        fft_pixels = (h + 2 * radius) * (w + 2 * radius)
        # One scale/orientation/channel at a time; do not multiply by angles.
        work = n * 32 + fft_pixels * 128 + (2 * radius + 1) ** 2 * 96
        return int(work), int(work)

    directions = int(bool(task.process_rows)) + int(bool(task.process_columns))
    directions = max(1, directions)
    plan = task.resolve_pipeline()

    # Both requested directions remain in task.result until the point stage.
    cwt_resident = n * scales * channels * directions * _F32
    # GPU path temporarily constructs one channel's coefficient array and
    # compacts the per-signal arrays (CuPy batches the GPU work, not the full
    # NumPy output); include a centering/float32 preparation allowance.
    one_channel = n * scales * _F32
    cwt_peak = cwt_resident + 2 * one_channel + n * (channels * _F32 + 8)

    # Extrema/envelopes and KNN visit a single dataset at a time.  Scratch
    # space depends on the number and topology of points, so use a per-image
    # allowance instead of allocating all k-neighbour tables simultaneously.
    points_peak = cwt_resident
    if plan.extrema:
        points_peak += n * 32
    if plan.envelopes:
        points_peak += n * 8
    if plan.knn or plan.ml:
        points_peak += n * (8 + max(1, int(task.k_neighbors)))  # one compact KNN group
    if plan.synchronization:
        points_peak += h * h * 16

    # A modest headroom for NumPy temporaries and allocator fragmentation;
    # not an upper bound (pathological point patterns can exceed this).
    cpu_work = math.ceil(max(cwt_peak, points_peak) * 1.10)

    # Retain the separate historical GPU working-array indicator, NOT a claim
    # about peak VRAM or CUDA context/pool memory.  2D similarly returns its
    # existing padded-FFT workspace proxy.
    gpu_work = n * scales * 8 * channels + n * 32
    return int(cpu_work), int(gpu_work)


def estimate_memory(tasks):
    """Return loaded source bytes, largest stage workspace, and GPU proxy.

    Workspaces are a maximum, not a sum, because the queue is sequential. The
    ``total`` legacy field excludes the Python/CUDA/GUI process baseline; the
    label combines the *current measured RSS* with the additional workspace.
    """
    inputs, peaks, gpu_peaks = 0, [], []
    seen = set()
    for task in tasks:
        for name in ('original_image', 'data', 'data_copy', 'analysis_data', 'result'):
            values = getattr(task, name, None)
            values = values if isinstance(values, (list, tuple)) else [values]
            for value in values:
                if isinstance(value, np.ndarray):
                    base = value
                    while isinstance(base.base, np.ndarray):
                        base = base.base
                    if id(base) not in seen:
                        inputs += base.nbytes
                        seen.add(id(base))
        shape = _shape(task)
        if shape is None or not len(task.scales):
            continue
        work, gpu = _task_working_bytes(task, int(shape[0]), int(shape[1]))
        peaks.append(work)
        gpu_peaks.append(gpu)
    return dict(inputs=int(inputs), additional=max(peaks, default=0),
                total=int(inputs) + max(peaks, default=0),
                gpu=max(gpu_peaks, default=0))


def estimate_details(tasks, disk_root=None):
    """Structured estimate for compact UI cards and reports."""
    size = estimate_memory(tasks)
    from compute.stage_profiling import current_rss_bytes
    current_rss = current_rss_bytes()

    details = {
        'inputs': size['inputs'],
        'additional': size['additional'],
        'total': size['total'],
        'gpu': size['gpu'],
        'current_rss': current_rss,
        'ram_total_gib': (current_rss + size['additional']) / _GIB if current_rss is not None else size['total'] / _GIB,
        'ram_current_gib': current_rss / _GIB if current_rss is not None else None,
        'ram_additional_gib': size['additional'] / _GIB,
        'gpu_gib': size['gpu'] / _GIB,
        'disk_forecast_gib': None,
        'disk_temporary_gib': None,
        'disk_free_gib': None,
        'notes': {
            'ram': 'Оценка для последовательных этапов, а не сумма всех буферов.',
            'gpu': 'Рабочие массивы без пика VRAM и CUDA-пула.',
            'disk': 'Объём зависит от числа точек и выбранного экспорта.',
            'summary': 'Ориентировочно; реальные пики RSS/VRAM могут быть выше.',
        },
    }
    if disk_root and tasks:
        from compute.disk_budget import forecast_task_bytes, disk_free_bytes
        try:
            estimates = [forecast_task_bytes(task) for task in tasks]
            if all(entry is not None for entry in estimates):
                forecast = sum(entry['forecast_bytes'] for entry in estimates)
                temporary = max((entry['temporary_bytes'] for entry in estimates), default=0)
                free = disk_free_bytes(disk_root)
                details['disk_forecast_gib'] = forecast / _GIB
                details['disk_temporary_gib'] = temporary / _GIB
                details['disk_free_gib'] = free / _GIB
        except (OSError, ValueError, TypeError):
            pass  # Capacity is checked authoritatively before writing.
    return details


def estimate_label(tasks, disk_root=None):
    details = estimate_details(tasks, disk_root)
    if details['current_rss'] is None:
        ram = (f'RAM: рабочие массивы до ≈ {details["ram_total_gib"]:.2f} ГиБ '
               '(без памяти интерфейса и библиотек).')
    else:
        ram = (f'RAM: ориентир RSS ≈ {details["ram_total_gib"]:.2f} ГиБ '
               f'(сейчас {details["ram_current_gib"]:.2f} + этапы до ≈ '
               f'{details["ram_additional_gib"]:.2f} ГиБ).')
    message = (ram + '\n'
               f'GPU: рабочие массивы ≈ {details["gpu_gib"]:.2f} ГиБ '
               '(не пик VRAM с CUDA-пулом).\n'
               'Ориентировочно; реальные пики RSS/VRAM могут быть выше.')
    if details['disk_forecast_gib'] is not None:
        message += (f'\nДиск: файлы ≈ {details["disk_forecast_gib"]:.2f} ГиБ + '
                    f'временный запас ≈ {details["disk_temporary_gib"]:.2f} ГиБ; '
                    f'свободно ≈ {details["disk_free_gib"]:.2f} ГиБ. '
                    'Объём зависит от числа точек и экспорта.')
    return message
