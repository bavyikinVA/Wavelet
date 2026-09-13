"""Transparent planning estimate, not a promise of the process's peak RSS."""
import math
import numpy as np


def estimate_memory(tasks):
    inputs, peaks, gpu_peaks = 0, [], []
    seen = set()
    for task in tasks:
        for name in ('original_image', 'data', 'data_copy', 'result'):
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
        image = getattr(task, 'original_image', None)
        if image is None or not len(task.scales):
            continue
        h, w = image.shape[:2]
        n, scales = h*w, len(task.scales)
        if task.analysis_mode == '2d':
            # The application processes one scale/orientation/channel at a time.
            radius = max(2, math.ceil(4 * max(task.scales) * max(1., task.morlet_anisotropy)))
            fft_pixels = (h + 2*radius) * (w + 2*radius)
            work = n*32 + fft_pixels*128 + (2*radius+1)**2*96
            gpu_peaks.append(work)
        else:
            directions = int(task.process_rows) + int(task.process_columns)
            work = n*scales*8*(3*directions + 3)  # retained channels plus worker/result copies
            gpu_peaks.append(n*scales*8*3 + n*32)
            plan = task.resolve_pipeline()
            if plan.extrema:
                work += n*64
            if plan.knn or plan.ml:
                # Conservative dense-point scenario; Python containers vary.
                work += n*scales*3*max(1, task.k_neighbors)*48
            if plan.synchronization:
                work += h*h*16
        peaks.append(int(work))
    return dict(inputs=int(inputs), additional=max(peaks, default=0),
                total=int(inputs)+max(peaks, default=0), gpu=max(gpu_peaks, default=0))


def estimate_label(tasks):
    size = estimate_memory(tasks)
    gib = 1024**3
    return (f'Память: RAM ≈ {size["total"]/gib:.2f} ГиБ · '
            f'массивы GPU ≈ {size["gpu"]/gib:.2f} ГиБ.\n'
            'Предварительная оценка; пиковое потребление может быть выше.')
