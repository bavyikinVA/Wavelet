from __future__ import annotations
from copy import deepcopy
from itertools import product
import math
from .validation import validate_ml_parameters


def _tokens(value):
    if isinstance(value, (tuple, list, set)):
        return [str(x).strip() for x in value]
    if value is None:
        return ['*']
    return [x.strip() for x in str(value).split(',') if x.strip()]


def _matches(value, selected):
    choices = {x.lower() for x in _tokens(selected)}
    if choices & {'*', 'all', 'все'}:
        return True
    normalized = str(value).lower()
    aliases = {
        'rows': 'row', 'строки': 'row', 'по строкам': 'row',
        'columns': 'col', 'столбцы': 'col', 'по столбцам': 'col',
        'красный': 'r', 'зелёный': 'g', 'зеленый': 'g', 'синий': 'b',
        'red': 'r', 'green': 'g', 'blue': 'b',
    }
    if aliases.get(normalized, normalized) in {aliases.get(x, x) for x in choices}:
        return True
    try:
        return any(math.isfinite(float(x)) and float(x) == float(value) for x in choices)
    except (TypeError, ValueError):
        return False


def _numbers(value, cast, name, *, minimum=0, strict=False):
    result = []
    for token in _tokens(value):
        try:
            number = cast(token)
        except (ValueError, TypeError) as exc:
            raise ValueError(f'{name}: неверное значение {token!r}') from exc
        if not math.isfinite(number) or (number <= minimum if strict else number < minimum):
            raise ValueError(f'{name}: значение {number} вне допустимого диапазона')
        if number not in result:
            result.append(number)
    if not result:
        raise ValueError(f'{name}: требуется хотя бы одно значение')
    return result


def default_configuration(algorithm='kmeans'):
    return {
        'algorithm': algorithm, 'channels': '*', 'scales': '*', 'directions': '*',
        'point_types': '*', 'k_values': '5', 'eps_values': '0.8',
        'min_samples_values': '5', 'feature_set': 'knn_raw', 'standardize': True,
        'random_state': 42, 'point_filter': 'all',
        'dbscan_batch_size': 512,
    }


def expand_ml_plan(plan, datasets, max_operations=5000):
    operations = []
    for index, raw in enumerate(plan):
        cfg = deepcopy(raw)
        cfg.pop('rerun_selected', None)
        cfg.pop('_execution_mode', None)
        for deprecated in ('dbscan_tile_size', 'dbscan_tile_overlap', 'dbscan_max_points_per_tile'):
            cfg.pop(deprecated, None)
        cfg['feature_set'] = 'knn_raw'
        algorithm = str(cfg.get('algorithm', 'kmeans')).lower()
        if algorithm not in {'kmeans', 'dbscan'}:
            raise ValueError(f'ML-карточка {index+1}: неизвестный алгоритм {algorithm}')
        available = []
        explicit_key = cfg.get('dataset_key')
        for option in datasets:
            direction, channel, scale, point_type = option['key']
            if explicit_key and tuple(explicit_key) != tuple(option['key']):
                continue
            if not (_matches(channel, cfg.get('channels', '*'))
                    and _matches(f'{float(scale):g}', cfg.get('scales', '*'))
                    and _matches(direction, cfg.get('directions', '*'))
                    and _matches(point_type, cfg.get('point_types', '*'))):
                continue
            available.append(option)
        if not available:
            raise ValueError(f'ML-карточка {index+1}: нет KNN-наборов для выбранных фильтров')
        if algorithm == 'kmeans':
            variants = [(k, None, None) for k in _numbers(
                cfg.get('k_values', cfg.get('n_clusters', 5)), int, 'k', minimum=2)]
        else:
            eps_values = _numbers(cfg.get('eps_values', cfg.get('eps', 0.8)), float,
                                  'eps', minimum=0, strict=True)
            min_values = _numbers(cfg.get('min_samples_values', cfg.get('min_samples', 5)),
                                  int, 'min_samples', minimum=1)
            variants = [(None, eps, mins) for eps, mins in product(eps_values, min_values)]
        for opt in available:
            for k, eps, mins in variants:
                op = dict(cfg)
                op.update(algorithm=algorithm, dataset_key=list(opt['key']),
                          n_clusters=k if k is not None else cfg.get('n_clusters', 5),
                          eps=eps if eps is not None else cfg.get('eps', 0.8),
                          min_samples=mins if mins is not None else cfg.get('min_samples', 5),
                          card_index=index + 1)
                try:
                    op = validate_ml_parameters(op, opt.get('count'))
                except ValueError as error:
                    raise ValueError(f"ML-карточка {index + 1}, набор {opt['key']}: {error}") from error
                operations.append(op)
                if len(operations) > max_operations:
                    raise ValueError(f'ML-план содержит больше {max_operations} операций; сузьте фильтры')
    return operations


def expand_task_ml(task):
    if not getattr(task, 'ml_plan', []):
        return []
    from pathlib import Path
    from .clustering import knn_dataset_options, select_knn_dataset
    from compute.knn.result_format import is_disk_knn_payload
    operations = expand_ml_plan(task.ml_plan, knn_dataset_options(task.knn_results))
    checked = set()
    for operation in operations:
        key = tuple(operation['dataset_key'])
        if key in checked:
            continue
        selected = select_knn_dataset(task.knn_results, key)
        for payload in selected.values():
            if is_disk_knn_payload(payload) and not Path(payload['path']).is_file():
                raise ValueError(
                    f"ML-карточка {operation['card_index']}, набор {key}: "
                    f"файл KNN отсутствует: {payload['path']}")
        checked.add(key)
    return operations


def preview_task_ml_operations(task, max_operations=5000):
    """Expand a task plan before KNN exists, using task input settings."""
    plan = getattr(task, 'ml_plan', [])
    if not plan:
        return []
    if task.analysis_mode != '1d' or not task.resolve_pipeline().knn:
        raise ValueError('Пакетный ML требует 1D-конвейер с включённым KNN')
    choices = task_source_choices(task)
    dataset_options = [{'key': (d, c, float(scale), pt), 'count': None}
                       for d, _ in choices['directions']
                       for c, _ in choices['channels']
                       for scale, _ in choices['scales']
                       for pt, _ in choices['point_types']]
    return expand_ml_plan(plan, dataset_options, max_operations=max_operations)


def task_source_choices(task):
    """User-visible ML source choices derived from this task's pending 1D plan.

    Works before any KNN result has been calculated.  Internal identifiers match
    those used by the KNN result/catalog; only the UI labels are localized.
    """
    representation = getattr(task, 'channel_representation', 'rgb')
    if representation == 'gram_schmidt':
        channels = [('gs1', 'GS1'), ('gs2', 'GS2'), ('gs3', 'GS3')]
    elif representation == 'grayscale':
        channels = [('gray', 'Оттенки серого')]
    elif getattr(task, 'rgb_channel_mode', 'all') == 'single':
        code = str(getattr(task, 'rgb_single_channel', 'R')).lower()
        rgb_names = {'r': 'Красный', 'g': 'Зелёный', 'b': 'Синий'}
        channels = [(code, rgb_names.get(code, code))]
    else:
        channels = [('r', 'Красный'), ('g', 'Зелёный'), ('b', 'Синий')]

    # ProcessingTask.scales is normally a NumPy array.  Never use `or []` here:
    # NumPy refuses implicit truth-value conversion for arrays with >1 element.
    raw_scales = getattr(task, 'scales', None)
    if raw_scales is None:
        raw_scales = ()
    unique_scales = sorted({float(scale) for scale in raw_scales})
    scales = [(f'{scale:g}', f'{scale:g}') for scale in unique_scales]
    directions = ([('row', 'По строкам')] if getattr(task, 'process_rows', False) else []) + (
        [('col', 'По столбцам')] if getattr(task, 'process_columns', False) else [])
    types = []
    if getattr(task, 'find_maxima', False):
        types.append(('max_by_row', 'Максимум по строкам'))
    if getattr(task, 'find_minima', False):
        types.append(('min_by_row', 'Минимум по строкам'))
    if getattr(task, 'find_maxima', False):
        types.append(('max_by_column', 'Максимум по столбцам'))
    if getattr(task, 'find_minima', False):
        types.append(('min_by_column', 'Минимум по столбцам'))
    return {'channels': channels, 'scales': scales,
            'directions': directions, 'point_types': types}
