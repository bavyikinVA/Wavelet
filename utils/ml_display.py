"""Shared, GUI-independent ML appearance policy for base data and overlays.

Coordinates, saved labels and colours always share one set of point indices.
This module does not reclassify DBSCAN scores or mutate exported numerical data.
"""
from dataclasses import dataclass
from math import ceil

from matplotlib.colors import to_rgba
import numpy as np

from utils.ml_classes import class_styles, visible_points


@dataclass(frozen=True)
class LegendEntry:
    label: str
    color: str


def snapshot_styles(styles):
    """JSON-compatible display state; key representations survive Tk snapshots."""
    return [{'key': str(key), 'visible': bool(value['visible']),
             'color': str(value['color'])} for key, value in styles.items()]


def apply_style_snapshot(styles, snapshot):
    """Only apply presentation fields to classes actually present in saved ML data."""
    if not isinstance(snapshot, (list, tuple)):
        return styles
    by_key = {str(entry.get('key')): entry for entry in snapshot
              if isinstance(entry, dict) and 'key' in entry}
    for key, style in styles.items():
        saved = by_key.get(str(key))
        if saved is None:
            continue
        if 'visible' in saved:
            style['visible'] = bool(saved['visible'])
        if 'color' in saved:
            # Fail closed on malformed colours, without damaging class mapping.
            try:
                to_rgba(saved['color'])
            except (TypeError, ValueError):
                continue
            style['color'] = saved['color']
    return styles


def display_subset(data, styles, indices):
    """Single rendering contract for ML base and overlay, also used in tests."""
    indices = np.asarray(indices, dtype=np.intp)
    if styles:
        return visible_points(data, styles, indices)
    scores = data.get('colors')
    if scores is not None:
        return indices, np.asarray(scores)[indices]
    return indices, None


def render_primary_ml_points(scatter, data, styles):
    """Apply saved ML colours to the primary Matplotlib PathCollection.

    `PathCollection.autoscale()` operates on the scalar colour array; calling
    it after `set_array(None)` for explicit RGBA classes raises TypeError in
    Matplotlib. The viewer already determines image geometry from data['shape']
    and sets axes limits explicitly, so the collection must not autoscale.
    """
    points = np.asarray(data['points'])
    indices, colours = display_subset(data, styles, np.arange(len(points)))
    scatter.set_offsets(points[indices, :2])
    if colours is not None and np.asarray(colours).ndim == 2:
        scatter.set_array(None)
        scatter.set_facecolors(colours)
    else:
        scatter.set_array(np.asarray(colours if colours is not None else np.zeros(len(indices))))
        if data.get('continuous'):
            scatter.set_cmap('inferno')
            scatter.set_clim(0.0, 1.0)
    return indices


def compact_result_name(item):
    algorithm = item.get('algorithm') or 'ML'
    params = item.get('ml_parameters') or {}
    if algorithm == 'K-means' and params.get('n_clusters') is not None:
        algorithm += f" k={params['n_clusters']}"
    if algorithm == 'DBSCAN':
        if params.get('eps') is not None:
            algorithm += f" ε={params['eps']}"
        if params.get('min_samples') is not None:
            algorithm += f" min_samples={params['min_samples']}"
    fields = ([str(item['task_name'])] if item.get('task_name') else []) + [algorithm]
    if item.get('channel'):
        fields.append(str(item['channel']))
    if item.get('scale') is not None:
        fields.append(f"s={item['scale']}")
    # A short suffix distinguishes repeated operations with the same ML params.
    if item.get('ml_run'):
        fields.append(str(item['ml_run']).rstrip('/').split('/')[-1])
    return ' · '.join(fields)


def legend_entries(layer_specs):
    """Build unambiguous legend labels for all visible layers and classes.

    layer_specs: iterable of (item, styles, visible, fallback_color).
    """
    visible_layers = [(i, s, c) for i, s, enabled, c in layer_specs if enabled]
    multi = len(visible_layers) > 1
    result = []
    for item, styles, color in visible_layers:
        prefix = compact_result_name(item) if multi else ''
        if styles:
            for style in styles.values():
                if not style.get('visible') or not style.get('count'):
                    continue
                label = f"{prefix} · {style['name']}" if multi else style['name']
                result.append(LegendEntry(label, style['color']))
        elif item.get('category') == 'ML-кластеры':
            label = (prefix + ' · ' if multi else '') + 'Оценка аномальности'
            result.append(LegendEntry(label, '#edbd58'))
        else:
            result.append(LegendEntry(prefix or item.get('category', 'Слой'), color))
    return result


def compact_legend(entries, *, limit=12):
    """Never silently drop classes from the on-screen legend."""
    if len(entries) <= limit:
        return list(entries)
    return list(entries[:limit]) + [LegendEntry(
        f'Ещё {len(entries)-limit} классов — см. панель «Слои»', '#b1bbc5')]


def export_legend_columns(total, *, max_label_length=0, figure_width=10.0):
    """Select columns by label length *and* number of classes.

    A long "task + ML-run + class" caption must not overlap its neighbour.
    Estimate at 7 pt text and a marker/column gap; the final visual contract
    is verified on the actual Windows export.
    """
    if not total:
        return 1
    preferred = min(6, max(1, ceil(total / 5)))
    if not max_label_length:
        return preferred
    estimated_column_inches = max_label_length * .062 + .65
    available_inches = max(1.0, float(figure_width) - 1.0)
    capacity = max(1, int(available_inches / estimated_column_inches))
    return min(preferred, capacity)
