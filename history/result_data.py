"""Read native numerical artifacts without GUI dependencies."""
import csv
import hashlib
from functools import lru_cache
from history.array_cache import byte_cache
from pathlib import Path
import numpy as np
from PIL import Image



@lru_cache(maxsize=128)
def _source_hash(filename, modified, size):
    digest = hashlib.sha1()
    with open(filename, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_identity(source, fallback):
    if not source.is_file():
        return ("folder", str(Path(fallback).resolve()))
    stat = source.stat()
    return ("sha1", stat.st_size, _source_hash(str(source), stat.st_mtime_ns, stat.st_size))

def load_result(item, folder):
    path = Path(item['path'])
    # Prefer numerical data over a PNG containing baked axes/labels.
    if path.suffix.lower() == '.png' and item['category'] != 'Источник':
        candidates = [path.with_suffix(s) for s in ('.npz', '.npy', '.txt', '.csv')]
        if 'morlet' in path.stem.lower():
            candidates.extend([path.with_name(path.stem + '_complex.npy'),
                               path.with_name(path.stem + '_magnitude.csv'),
                               Path(folder) / 'Предпросмотр' / (path.stem + '.npy')])
        if 'кластеры_изображение' in path.stem:
            candidates.insert(0, path.parent / 'кластеры.csv')
        path = next((p for p in candidates if p.is_file()), path)
    stat = path.stat()
    source = Path(folder) / 'image.png'
    if not source.is_file():
        source = Path(folder) / 'Изображение.png'  # Existing run archives.
    shape = None
    if source.is_file():
        with Image.open(source) as image:
            shape = (image.height, image.width)
    data = dict(_read(str(path), stat.st_mtime_ns, stat.st_size, item['category'], shape))
    name = path.stem.casefold()
    data['quantity'] = ('phase' if name.endswith('_phase') else
                        'magnitude' if name.endswith('_magnitude') else 'coefficient')
    # A run-local source identity prevents same-size, unrelated layers matching.
    # Content identity lets sibling ML runs of the same research image be
    # compared even though each run owns its own copied image.png.
    data['source_id'] = _source_identity(source, folder)
    return data


def _map(array, shape=None, spatial=True):
    h, w = array.shape[:2]
    return dict(kind='map', array=np.array(array, copy=True),
                shape=shape or (h, w), spatial=spatial)


@byte_cache()
def _read(filename, modified, size, category, shape):
    path = Path(filename)
    suffix = path.suffix.lower()
    if suffix == '.npz':
        with np.load(path, allow_pickle=False) as payload:
            if 'matrix' in payload:
                return _map(payload['matrix'], spatial=False)
            if 'series' in payload:
                return dict(kind='series', array=payload['series'].copy(),
                            labels=payload['labels'].tolist(),
                            ticks=payload['ticks'].tolist() if 'ticks' in payload else None, spatial=False)
            points = payload['points'].copy()
            shape = tuple(payload['shape'])
            return dict(kind='points', points=points, shape=shape, spatial=True)
    if suffix == '.npy':
        mapped = np.load(path, mmap_mode='r', allow_pickle=False)
        try:
            if mapped.ndim != 2:
                raise ValueError(f'Ожидается матрица 2D, получено {mapped.shape}')
            return _map(mapped, shape if category != 'Синхронизация' else None,
                        spatial=category != 'Синхронизация')
        finally:
            mapped._mmap.close()
    if suffix in ('.png', '.jpg', '.jpeg'):
        with Image.open(path) as image:
            original = (image.height, image.width)
            array = np.array(image.convert('RGB'))
        return dict(kind='image', array=array, shape=original, spatial=category == 'Источник',
                    legacy=category != 'Источник')
    with path.open(encoding='utf-8-sig', errors='replace') as stream:
        first = stream.readline()
    if category == 'Вейвлеты':
        array = np.loadtxt(path, delimiter=',' if ',' in first else None, ndmin=2)
        return _map(array, shape)
    if category in ('Экстремумы', 'Огибающие'):
        points = np.loadtxt(path, delimiter=',', ndmin=2)
        if points.shape[1] == 2:
            return dict(kind='points', points=points, shape=shape or (int(points[:, 1].max())+1, int(points[:, 0].max())+1), spatial=shape is not None)
    if suffix == '.csv' or category == 'KNN':
        with path.open(encoding='utf-8-sig', errors='replace') as stream:
            rows = list(csv.DictReader(line for line in stream if line.strip() and not line.startswith('#')))
        if rows and ('X' in rows[0] or 'x' in rows[0]):
            x, y = ('X', 'Y') if 'X' in rows[0] else ('x', 'y')
            points = np.array([[float(r[x]), float(r[y])] for r in rows])
            continuous = 'Аномальность DBSCAN' in rows[0]
            colors = np.array([float(r.get('Аномальность DBSCAN', r.get('Кластер', 0))) for r in rows])
            edges = []
            if category == 'KNN' and 'point_id' in rows[0]:
                coordinates = {r['point_id']: p for r, p in zip(rows, points)}
                edges = [[coordinates[r['point_id']], coordinates[r['neighbor_id']]] for r in rows if r['neighbor_id'] in coordinates]
            return dict(kind='points', points=points, colors=colors, edges=edges, continuous=continuous,
                        shape=shape or (int(points[:, 1].max())+1, int(points[:, 0].max())+1), spatial=shape is not None)
        if rows and list(rows[0])[0] == 'row' and category == 'Синхронизация':
            return _map(np.array([[float(v) for k, v in r.items() if k != 'row'] for r in rows]), spatial=False)
        if rows and category == 'Статистика':
            if 'block_label' in rows[0] and 'upper_envelope_maxima_sum' in rows[0]:
                return dict(kind='series', array=np.array([[i, float(r['upper_envelope_maxima_sum'])]
                            for i, r in enumerate(rows)]), labels=['Блок', 'Количество максимумов'],
                            ticks=[r['block_label'] for r in rows], spatial=False)
            # Row ids and block size are metadata, not measured series.
            keys = [k for k in rows[0] if k not in ('row_index', 'block_size', 'scales')]
            try:
                array = np.array([[float(r[k]) for k in keys] for r in rows])
                return dict(kind='series', array=array, labels=keys, spatial=False)
            except (ValueError, TypeError):
                pass
    with path.open(encoding='utf-8-sig', errors='replace') as stream:
        lines = [stream.readline(2000) for _ in range(150)]
    return dict(kind='text', text=''.join(lines), spatial=False)
