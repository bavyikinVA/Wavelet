"""Presentation-only catalogue metadata for the overlay explorer.

All labels are obtained from saved artifact metadata. File names are never
presented in normal cards. They remain accessible via details() for diagnostics.
"""
from datetime import datetime
import math
from pathlib import Path
import re

GROUPS = ('Вейвлеты', 'Экстремумы', 'Огибающие', 'KNN', 'ML-кластеры')
GROUP_LABELS = {'Вейвлеты': 'CWT', 'ML-кластеры': 'ML'}
ALGORITHMS = ('K-means', 'DBSCAN')
DIRECTIONS = {'row': 'CWT по строкам', 'rows': 'CWT по строкам',
              'col': 'CWT по столбцам', 'columns': 'CWT по столбцам'}
AXES = {'row': 'строкам', 'col': 'столбцам'}
KINDS = {'max': 'Максимумы', 'min': 'Минимумы',
         'upper': 'Верхняя огибающая', 'lower': 'Нижняя огибающая'}
CATEGORY_COLORS = {
    'Вейвлеты': '#a593e9', 'Экстремумы': '#eaa078',
    'Огибающие': '#ddbd77', 'KNN': '#83c6a7',
    'ML-кластеры': '#87b9ec', 'K-means': '#87b9ec', 'DBSCAN': '#d9a1e8',
}


def result_id(item):
    """Unique identity of a stored artifact, including distinct ML reruns."""
    return str(Path(item['path']).resolve())


def task_id(item):
    return str(item.get('task_id') or item.get('task_folder') or 'Задача')


def direction(item):
    key = item.get('cwt_direction') or item.get('direction')
    return DIRECTIONS.get(key, 'Без направления')


def point_scope(item):
    kind = item.get('point_kind')
    axis = AXES.get(item.get('feature_axis'))
    if kind in KINDS:
        return f'{KINDS[kind]} по {axis}' if axis else KINDS[kind]
    return f'По {axis}' if axis else None


def _bool_text(value):
    if isinstance(value, str):
        value = value.strip().casefold() in {'true', '1', 'yes', 'да'}
    return 'Да' if value else 'Нет'


def title(item):
    """User-facing title: scale and channel, without filesystem identifiers."""
    fields = []
    if item.get('scale') is not None:
        raw_scale = item['scale']
        try:
            number = float(raw_scale)
            scale = f'{number:g}' if math.isfinite(number) else str(raw_scale)
        except (TypeError, ValueError, OverflowError):
            scale = str(raw_scale)
        fields.append(f'Масштаб {scale}')
    if item.get('channel'):
        fields.append(str(item['channel']))
    if fields:
        return ' · '.join(fields)
    return GROUP_LABELS.get(item.get('category'), item.get('category') or 'Результат')


def algorithm_line(item):
    """The ML parameters that distinguish variants must be visible on a card."""
    if item.get('category') != 'ML-кластеры':
        return GROUP_LABELS.get(item.get('category'), item.get('category') or 'Результат')
    params = item.get('ml_parameters') or {}
    algorithm = item.get('algorithm') or 'ML'
    parts = [algorithm]
    if algorithm == 'K-means' and params.get('n_clusters') is not None:
        parts.append(f"k={params['n_clusters']}")
    elif algorithm == 'DBSCAN':
        if params.get('eps') is not None:
            parts.append(f"ε={params['eps']}")
        if params.get('min_samples') is not None:
            parts.append(f"min_samples={params['min_samples']}")
    return ' · '.join(str(p) for p in parts)


def process_line(item):
    return ' · '.join(label for label in (point_scope(item), direction(item))
                      if label and label != 'Без направления')


def parameter_line(item):
    params = item.get('ml_parameters') or {}
    if item.get('category') != 'ML-кластеры':
        return ''
    fields = []
    if params.get('feature_set') is not None:
        fields.append(f"Признаки: {params['feature_set']}")
    if 'standardize' in params and _bool_text(params['standardize']) == 'Да':
        fields.append('Стандартизация')
    # Random state is kept in diagnostic details, not in the selection card.
    return ' · '.join(fields)


def _run_stamp(item):
    """Human-readable date/time without exposing ML's technical folder stem."""
    run = str(item.get('ml_run') or '')
    # Example: cwtrow_axisrow_g_s5_max/20_10_2026_21_33_07_951759
    for segment in reversed(run.replace('\\', '/').split('/')):
        match = re.search(r'(\d{2})_(\d{2})_(\d{4})_(\d{2})_(\d{2})(?:_(\d{2}))?', segment)
        if not match:
            continue
        try:
            day, month, year, hour, minute = map(int, match.groups()[:5])
            datetime(year, month, day, hour, minute)
        except ValueError:
            continue
        return f'{day:02d}.{month:02d}.{year} · {hour:02d}:{minute:02d}'
    return ''


def card_footer(item):
    task = str(item.get('task_name') or task_id(item))
    stamp = _run_stamp(item)
    return f'{task} · {stamp}' if stamp else task


def category_color(item):
    key = item.get('algorithm') if item.get('category') == 'ML-кластеры' else item.get('category')
    return CATEGORY_COLORS.get(key, '#87b9ec')


def card_title(item):
    return title(item)


def card_meta(item):
    return algorithm_line(item)


def card_badges(item):
    return [x for x in (process_line(item), parameter_line(item)) if x]


def subtitle(item):
    return ' · '.join(x for x in (algorithm_line(item), process_line(item)) if x)


def details(item):
    params = item.get('ml_parameters') or {}
    fields = [title(item), f"s={item['scale']}" if item.get('scale') is not None else None,
            algorithm_line(item), process_line(item), parameter_line(item), card_footer(item)]
    for key, label in [('n_clusters', 'Кластеров'), ('eps', 'ε'),
                       ('min_samples', 'Мин. соседей'), ('random_state', 'Random state'),
                       ('feature_set', 'Признаки'), ('standardize', 'Стандартизация')]:
        if key in params:
            value = _bool_text(params[key]) if key == 'standardize' else params[key]
            fields.append(f'{label}: {value}')
    fields.append('Файл: ' + Path(item['path']).name)
    return '\n'.join(str(value) for value in fields if value)


def search_results(items, scope=None, query='', order='scale'):
    """AND search over human labels and exact saved metadata."""
    terms = query.casefold().replace('ё', 'е').split()
    matches = []
    for item in items:
        if scope is not None and not scope(item):
            continue
        searchable = ' '.join(str(x) for x in (
            title(item), f"s={item['scale']}" if item.get('scale') is not None else None,
            algorithm_line(item), process_line(item), parameter_line(item),
            card_footer(item), item.get('task_name'), task_id(item),
            item.get('algorithm'), item.get('scale'), item.get('channel'),
            item.get('direction'), item.get('cwt_direction'), item.get('feature_axis'),
            item.get('point_kind'), item.get('ml_run'), item.get('display_label'),
            Path(item['path']).stem, item.get('ml_parameters')
        ) if x is not None).casefold().replace('ё', 'е')
        if all(term in searchable for term in terms):
            matches.append(item)

    def scale_key(item):
        try:
            value = float(item.get('scale'))
            return value if math.isfinite(value) else float('inf')
        except (TypeError, ValueError, OverflowError):
            return float('inf')

    if order == 'recent':
        def modified(item):
            try:
                return Path(item['path']).stat().st_mtime_ns
            except OSError:
                return 0
        matches.sort(key=lambda it: (modified(it), result_id(it)), reverse=True)
    elif order == 'scale_desc':
        matches.sort(key=lambda it: (math.isinf(scale_key(it)), -scale_key(it),
                                     str(it.get('channel') or ''), result_id(it)))
    elif order == 'channel':
        matches.sort(key=lambda it: (str(it.get('channel') or ''), scale_key(it), result_id(it)))
    elif order == 'name':
        matches.sort(key=lambda it: (title(it).casefold(), result_id(it)))
    else:
        matches.sort(key=lambda it: (scale_key(it), str(it.get('channel') or ''), result_id(it)))
    return matches


def category_tree(items):
    """task -> result category -> ML algorithm; omit empty nodes."""
    tree = {}
    for item in items:
        if item.get('category') not in GROUPS:
            continue
        task = task_id(item)
        task_node = tree.setdefault(task, {'name': item.get('task_name') or task, 'groups': {}})
        group = item['category']
        entry = task_node['groups'].setdefault(group, {'count': 0, 'algorithms': {}})
        entry['count'] += 1
        if group == 'ML-кластеры':
            algorithm = item.get('algorithm') or 'Другой алгоритм'
            entry['algorithms'][algorithm] = entry['algorithms'].get(algorithm, 0) + 1
    return tree
