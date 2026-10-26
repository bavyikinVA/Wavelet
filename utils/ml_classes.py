"""Presentation of saved ML labels; never recompute scientific classifications."""
import numpy as np
from matplotlib.colors import to_rgba, to_hex
from matplotlib import colormaps
from ml.color_scheme import class_name, categorical_ml_rgba

DBSCAN_STATES = ('плотная структура', 'неустойчивая область', 'устойчивая аномалия')
STATE_COLORS = ('#50bfa5', '#edbd58', '#ef7085')


def class_styles(data):
    labels = data.get('class_labels')
    if labels is None:
        return {}
    unique, counts = np.unique(labels, return_counts=True)
    counts = dict(zip(unique.tolist(), counts.tolist()))
    states = data.get('class_kind') == 'dbscan' and any(isinstance(k, str) for k in counts)
    keys = list(DBSCAN_STATES) if states else sorted(counts)
    keys += [k for k in counts if k not in keys]
    result = {}
    for index, key in enumerate(keys):
        color = STATE_COLORS[index] if states and index < 3 else to_hex(categorical_ml_rgba([int(key)])[0])
        name = str(key).capitalize() if states else class_name(int(key))
        result[key] = dict(name=name, color=color, visible=True, count=counts.get(key, 0))
    return result


def visible_points(data, styles, indices):
    """Subset labels and coordinates with the same indices, including empty views."""
    labels = data.get('class_labels')
    if labels is None or not styles:
        return indices, None
    keys = sorted(styles)
    selected = np.asarray(labels)[indices]
    codes = np.searchsorted(np.asarray(keys), selected)
    enabled = np.array([styles[key]['visible'] for key in keys], dtype=bool)
    palette = np.array([to_rgba(styles[key]['color']) for key in keys], dtype=float)
    mask = enabled[codes]
    return indices[mask], palette[codes[mask]].reshape(-1, 4)
