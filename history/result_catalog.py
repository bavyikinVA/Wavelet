"""Numerical result discovery and lossless coefficient storage for the viewer."""
from pathlib import Path
import re
import hashlib
from functools import lru_cache
from decimal import Decimal
import numpy as np


def save_coefficient_preview(folder, name, coefficients):
    directory = Path(folder) / 'Предпросмотр'
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
    if label_prefix:
        label = f"{label_prefix} / {label}"
    return {"path": str(path), "label": label,
            "category": category(relative), **result_parameters(relative)}


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
    family = re.sub(r'(?:масштаб|scale)[_ ]*' + number, 'scale_#', name)
    family = re.sub(r'(?:angle|угол)[_ ]*' + number, 'angle_#', family)
    family = re.sub(r'красный|зелёный|зеленый|синий', 'channel_#', family)
    algorithm = ('DBSCAN' if 'dbscan' in name else
                 'K-means' if 'kmeans' in name or 'k-means' in name else None)
    point_type = next((token for token in ('max_by_row', 'min_by_row', 'max_by_column', 'min_by_column')
                       if token in name), None)
    direction = ('rows' if name.endswith('_rows.csv') or '_rows/' in name or 'knn_str_' in name else
                 'columns' if name.endswith('_columns.csv') or '_columns/' in name or 'knn_col_' in name else None)
    return dict(scale=numeric(r'(?:масштаб|scale)[_ ]*'),
                orientation=numeric(r'(?:angle|угол)[_ ]*'), channel=channel, family=family,
                algorithm=algorithm, point_type=point_type, direction=direction,
                mode='2D' if '2d' in name else '1D' if any(s in name for s in ('1d', 'вейвлет', 'str_', 'scale_')) else None)
