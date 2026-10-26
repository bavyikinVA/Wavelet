"""Shared categorical labels/colors; no Tk dependencies."""
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import hsv_to_rgb


def class_name(label: int) -> str:
    return 'Шум' if int(label) == -1 else f'Кластер {int(label)}'


def categorical_ml_rgba(labels):
    """Noise is always gray, integer cluster ids map to a discrete palette."""
    numbers = np.asarray(labels, dtype=np.int64)
    out = np.empty((len(numbers), 4), dtype=np.float32)
    out[:] = (0.48, 0.48, 0.48, 1.0)
    positive = numbers >= 0
    if not np.any(positive):
        return out
    labels_positive = numbers[positive]
    colors = np.empty((len(labels_positive), 4), dtype=np.float32)
    tab20 = np.asarray(colormaps['tab20'](np.arange(20)), dtype=np.float32)
    initial = labels_positive < 20
    if np.any(initial):
        colors[initial] = tab20[labels_positive[initial]]
    if np.any(~initial):
        # Beyond tab20, assign deterministic, label-dependent hues instead
        # of repeating the first 20 swatches for every additional cluster.
        extra = labels_positive[~initial].astype(float)
        hsv = np.column_stack(((extra * 0.618033988749895) % 1.,
                               np.full(len(extra), .72),
                               np.full(len(extra), .94)))
        colors[~initial, :3] = hsv_to_rgb(hsv)
        colors[~initial, 3] = 1.0
    out[positive] = colors
    return out
