"""Numerical result discovery and lossless coefficient storage for the viewer."""
from pathlib import Path
import re
import hashlib
from functools import lru_cache
from decimal import Decimal
import numpy as np
from result_naming import number_from_token


def save_coefficient_preview(folder, name, coefficients):
    directory = Path(folder) / 'previews'
    directory.mkdir(exist_ok=True, parents=True)
    # Keep every coefficient and its original dtype. Downsampling here used to
    # discard narrow extrema and phase detail before the viewer even loaded it.
    array = np.asarray(coefficients)
    path = directory / (name + '.npy')
    np.save(path, array, allow_pickle=False)
    return path


def category(path):
    name = str(path).lower()
    basename = Path(path).name.lower()
    if basename in {'image.png', 'изображение.png'}:
        return 'Источник'
    if basename.startswith('source_channel_'):
        return 'Исходные данные'
    if basename.startswith('centering_mean_'):
        return 'Служебные данные'
    if basename == 'run_log.txt':
        return 'Журнал'
    if basename.startswith('ml_'):
        return 'Статистика' if '_summary_' in basename else 'ML-кластеры'
    if basename.startswith('sync_'):
        return 'Синхронизация'
    if basename.startswith('stat_'):
        return 'Статистика'
    if basename.startswith('knn_'):
        return 'KNN'
    if basename.startswith('env_'):
        return 'Огибающие'
    if basename.startswith('ext_'):
        return 'Экстремумы'
    if basename.startswith(('cwt1d_', 'cwt2d_')) or '/previews/' in f'/{name}':
        return 'Вейвлеты'
    if 'характеристики_кластер' in basename:
        return 'Статистика'
    if 'кластер' in basename or basename.startswith('ml_'):
        return 'ML-кластеры'
    if 'синх' in name or 'sync' in name:
        return 'Синхронизация'
    if 'гист' in name or 'hist' in name or 'статист' in name:
        return 'Статистика'
    if 'knn' in name:
        return 'KNN'
    if 'огиба' in name or 'envelope' in name:
        return 'Огибающие'
    if 'сырые' in name or 'экстрем' in name:
        return 'Экстремумы'
    if 'morlet' in name or 'вейвлет' in name or 'wavelet' in name or 'предпросмотр' in name:
        return 'Вейвлеты'
    return 'Другие файлы'



def _source_image(root):
    for name in ("image.png", "Изображение.png"):
        path = Path(root) / name
        if path.is_file():
            return path
    return None


@lru_cache(maxsize=128)
def _file_fingerprint(filename, modified, size):
    digest = hashlib.sha1()
    with open(filename, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint(root):
    path = _source_image(root)
    if path is None:
        return None
    stat = path.stat()
    return (stat.st_size, _file_fingerprint(str(path), stat.st_mtime_ns, stat.st_size))


def _artifact(path, root, label_prefix=""):
    relative = path.relative_to(root)
    label = str(relative)
    parameters = result_parameters(relative)
    display_label = result_display_label(relative, parameters)
    if label_prefix:
        label = f"{label_prefix} / {label}"
        display_label = f"{label_prefix} · {display_label}"
    return {
        "path": str(path), "label": label,
        "display_label": display_label,
        "category": category(relative), **parameters,
    }


def _related_ml_artifacts(root):
    """ML outputs from sibling run folders that use the exact same source image."""
    parent = root.parent
    if not parent.is_dir():
        return []
    fingerprint = source_fingerprint(root)
    if fingerprint is None:
        return []
    related = []
    for sibling in parent.iterdir():
        if not sibling.is_dir() or sibling == root:
            continue
        try:
            if source_fingerprint(sibling) != fingerprint:
                continue
        except OSError:
            continue
        for path in sibling.rglob('*'):
            if not path.is_file() or path.suffix.lower() not in {'.csv', '.npy', '.npz'}:
                continue
            rel_name = str(path.relative_to(sibling)).casefold()
            if 'ml_кластеризация' not in rel_name and 'ml_' not in path.name.casefold():
                continue
            item = _artifact(path, sibling, label_prefix=f"ML · {sibling.name}")
            if item['category'] in {'ML-кластеры', 'Статистика'}:
                related.append(item)
    return related

def discover_results(folder):
    root = Path(folder) if folder else None
    if root is None or not root.is_dir():
        return []
    local = [
        _artifact(path, root)
        for path in root.rglob('*') if path.is_file()
        and (path.name.casefold() in {'image.png', 'изображение.png'}
             or (path.suffix.lower() in {'.npy', '.npz', '.csv', '.txt'}
                 and not path.stem.casefold().startswith('график_расчетов')))
    ]
    seen = {item['path'] for item in local}
    related = [item for item in _related_ml_artifacts(root) if item['path'] not in seen]
    return sorted(local + related,
        key=lambda item: (item['category'] != 'Источник', item['category'],
                          Path(item['path']).suffix.lower() not in {'.png', '.npy'}, item['label']))


def result_parameters(path):
    """Read parameters from the application's current and legacy export names.

    Unknown values remain absent, never inferred from an arbitrary number.
    """
    name = str(path).replace('\\', '/').casefold()
    basename = Path(name).name
    number = r'([-+]?\d+(?:[.,]\d+)?(?:e[-+]?\d+)?)'
    def numeric(pattern):
        match = re.search(pattern + number, name)
        if not match:
            return None
        value = Decimal(match[1].replace(',', '.'))
        text = format(value, 'f')
        return text.rstrip('0').rstrip('.') if '.' in text else text
    channel = None
    # Full Russian names, including the spelling variants of green.
    for token, label in [('красный', 'Красный'), ('зелёный', 'Зелёный'),
                         ('зеленый', 'Зелёный'), ('синий', 'Синий'),
                         ('channel_red', 'Красный'), ('channel_green', 'Зелёный'),
                         ('channel_blue', 'Синий')]:
        if token in name:
            channel = label
    channel_match = re.search(r'(?:^|_)(r|g|b)(?:_|\.|$)', basename)
    if channel_match:
        channel = {'r': 'Красный', 'g': 'Зелёный', 'b': 'Синий'}[
            channel_match[1]
        ]

    def token_number(prefix):
        match = re.search(
            rf'(?:^|_){prefix}(m?\d+(?:p\d+)?(?:em?\d+|e\d+)?)(?:_|\.|$)',
            basename,
        )
        return number_from_token(match[1]) if match else None

    current_scale = token_number('s')
    current_angle = token_number('a')
    cwt_match = (
        re.search(r'^cwt1d_(row|col)_', basename)
        or re.search(r'_cwt(row|col)_', basename)
        or re.search(r'^centering_mean_(row|col)_', basename)
    )
    cwt_direction = cwt_match[1] if cwt_match else None
    feature_match = re.search(r'_axis(row|col)_', basename)
    feature_axis = feature_match[1] if feature_match else None
    kind_match = re.search(r'_(max|min|upper|lower)(?:_|\.|$)', basename)
    point_kind = kind_match[1] if kind_match else None
    point_type = None
    if feature_axis and point_kind in {'max', 'min'}:
        suffix = 'row' if feature_axis == 'row' else 'column'
        point_type = f'{point_kind}_by_{suffix}'
    if point_type is None:
        point_type = next((token for token in (
            'max_by_row', 'min_by_row', 'max_by_column', 'min_by_column'
        ) if token in name), None)

    artifact = next((prefix for prefix in (
        'source_channel', 'centering_mean', 'run_log',
        'cwt1d', 'cwt2d', 'ext', 'env', 'knn', 'stat', 'sync', 'ml'
    ) if basename.startswith(prefix + '_') or basename.startswith(prefix + '.')), None)
    family = re.sub(r'(?:масштаб|scale)[_ ]*' + number, 'scale_#', name)
    family = re.sub(r'(?:angle|угол)[_ ]*' + number, 'angle_#', family)
    family = re.sub(r'(?:^|_)s(?:m?\d+(?:p\d+)?(?:em?\d+|e\d+)?)', '_s#', family)
    family = re.sub(r'(?:^|_)a(?:m?\d+(?:p\d+)?(?:em?\d+|e\d+)?)', '_a#', family)
    family = re.sub(r'красный|зелёный|зеленый|синий', 'channel_#', family)
    family = re.sub(r'(?<=_)(r|g|b)(?=_)', 'channel_#', family)
    algorithm = ('DBSCAN' if 'dbscan' in name else
                 'K-means' if 'kmeans' in name or 'k-means' in name else None)
    legacy_direction = (
        'rows' if name.endswith('_rows.csv') or '_rows/' in name or 'knn_str_' in name
        else 'columns' if name.endswith('_columns.csv') or '_columns/' in name or 'knn_col_' in name
        else None
    )
    direction = cwt_direction or legacy_direction
    coordinate_match = re.search(r'_(x|y)(\d+)(?:_|\.|$)', basename)
    quantity = next((value for value in ('magnitude', 'phase', 'complex')
                     if f'_{value}' in basename), None)
    return dict(
        scale=current_scale or numeric(r'(?:масштаб|scale)[_ ]*'),
        orientation=current_angle or numeric(r'(?:angle|угол)[_ ]*'),
        channel=channel,
        family=family,
        algorithm=algorithm,
        artifact=artifact,
        point_type=point_type,
        point_kind=point_kind,
        cwt_direction=cwt_direction,
        feature_axis=feature_axis,
        direction=direction,
        coordinate=(
            (coordinate_match[1], coordinate_match[2])
            if coordinate_match else None
        ),
        quantity=quantity,
        file_format=Path(basename).suffix.lstrip('.').upper(),
        mode=(
            '2D' if artifact == 'cwt2d' or '2d' in name
            else '1D' if artifact in {'cwt1d', 'ext', 'env', 'knn', 'stat', 'sync'}
            or any(s in name for s in ('1d', 'вейвлет', 'str_', 'scale_'))
            else None
        ),
    )


def result_display_label(path, parameters=None):
    """Return a readable Russian label while keeping the real path untouched."""
    parameters = parameters or result_parameters(path)
    basename = Path(path).name
    category_name = category(path)
    if category_name == 'Источник':
        return 'Исходное изображение'

    artifact = parameters.get('artifact')
    labels = {
        'source_channel': 'Исходный цветовой канал',
        'centering_mean': 'Средние значения центрирования',
        'run_log': 'Журнал запуска',
        'cwt1d': '1D CWT', 'cwt2d': '2D CWT Морле',
        'ext': 'Экстремумы', 'env': 'Огибающие', 'knn': 'KNN',
        'stat': 'Статистика', 'sync': 'Синхронизация строк',
        'ml': 'ML-результат',
    }
    parts = [labels.get(artifact, category_name)]
    cwt_direction = parameters.get('cwt_direction')
    if cwt_direction:
        parts.append(
            'CWT по строкам' if cwt_direction == 'row'
            else 'CWT по столбцам'
        )
    feature_axis = parameters.get('feature_axis')
    if feature_axis:
        parts.append(
            'поиск по строкам' if feature_axis == 'row'
            else 'поиск по столбцам'
        )
    if parameters.get('channel'):
        parts.append(parameters['channel'])
    if parameters.get('scale') is not None:
        parts.append(f"масштаб {parameters['scale']}")
    if parameters.get('orientation') is not None:
        parts.append(f"ориентация {parameters['orientation']}°")
    kind_labels = {
        'max': 'максимумы', 'min': 'минимумы',
        'upper': 'верхняя огибающая', 'lower': 'нижняя огибающая',
    }
    if parameters.get('point_kind'):
        parts.append(kind_labels.get(
            parameters['point_kind'], parameters['point_kind']
        ))
    if parameters.get('algorithm'):
        parts.append(parameters['algorithm'])
    if parameters.get('coordinate'):
        axis, value = parameters['coordinate']
        parts.append(f"{axis}={value}")
    quantity_labels = {
        'magnitude': 'модуль', 'phase': 'фаза',
        'complex': 'комплексные коэффициенты',
    }
    if parameters.get('quantity'):
        parts.append(quantity_labels.get(
            parameters['quantity'], parameters['quantity']
        ))
    if artifact == 'stat' and '_block' in basename:
        parts.append('блоки масштабов')
    if artifact == 'sync':
        parts.append('метрики пар' if '_pairs' in basename else 'матрица')
    if artifact == 'ml' and '_summary_' in basename:
        parts.append('сводная статистика')
    if parameters.get('file_format'):
        parts.append(parameters['file_format'])
    if artifact is None:
        parts.append(basename)
    return ' · '.join(parts)
