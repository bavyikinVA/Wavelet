"""Numerical result discovery and lossless coefficient storage for the viewer."""
from pathlib import Path
import re
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
    if Path(path).name.lower() in {'image.png', 'изображение.png'}:
        return 'Источник'
    if 'кластер' in name or 'ml_' in name:
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


def discover_results(folder):
    root = Path(folder) if folder else None
    if root is None or not root.is_dir():
        return []
    return sorted(
        [{'path': str(path), 'label': str(path.relative_to(root)),
          'category': category(path.relative_to(root)), **result_parameters(path.relative_to(root))}
         for path in root.rglob('*') if path.is_file()
         and (path.name.casefold() in {'image.png', 'изображение.png'}
              or (path.suffix.lower() in {'.npy', '.npz', '.csv', '.txt'}
                  and not path.stem.casefold().startswith('график_расчетов')))],
        key=lambda item: (item['category'] != 'Источник', item['category'],
                          Path(item['path']).suffix.lower() not in {'.png', '.npy'}, item['label'])
    )


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
    return dict(scale=numeric(r'(?:масштаб|scale)[_ ]*'),
                orientation=numeric(r'(?:angle|угол)[_ ]*'), channel=channel, family=family,
                mode='2D' if '2d' in name else '1D' if any(s in name for s in ('1d', 'вейвлет', 'str_', 'scale_')) else None)
