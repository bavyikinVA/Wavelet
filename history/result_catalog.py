"""Disk-backed result discovery and bounded numerical previews."""
from pathlib import Path
import numpy as np


def save_coefficient_preview(folder, name, coefficients):
    directory = Path(folder) / 'Предпросмотр'
    directory.mkdir(exist_ok=True, parents=True)
    array = np.asarray(coefficients)
    step = max(1, int(np.ceil(max(array.shape) / 256)))
    dtype = np.complex64 if np.iscomplexobj(array) else np.float32
    path = directory / (name + '.npy')
    np.save(path, array[::step, ::step].astype(dtype))
    return path


def category(path):
    name = str(path).lower()
    if 'изображение.png' == Path(path).name.lower():
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
    if 'morlet' in name or 'вейвлет' in name or 'предпросмотр' in name:
        return 'Вейвлеты'
    return 'Другие файлы'


def discover_results(folder):
    root = Path(folder) if folder else None
    if root is None or not root.is_dir():
        return []
    return sorted(
        [{'path': str(path), 'label': str(path.relative_to(root)),
          'category': category(path.relative_to(root))}
         for path in root.rglob('*') if path.is_file()
         and path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.npy', '.npz', '.csv', '.txt'}],
        key=lambda item: (item['category'] != 'Источник', item['category'],
                          Path(item['path']).suffix.lower() not in {'.png', '.npy'}, item['label'])
    )
