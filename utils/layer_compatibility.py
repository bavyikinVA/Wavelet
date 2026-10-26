"""Conservative source and coordinate checks for cross-task spatial overlays."""
from pathlib import Path
from PIL import Image
from history.result_catalog import source_fingerprint


def image_shape(task_folder):
    root = Path(task_folder)
    for name in ('image.png', 'Изображение.png'):
        path = root / name
        if path.is_file():
            with Image.open(path) as image:
                return (image.height, image.width)
    return None


def preflight(item, base_item, base_data):
    """Return (is_compatible, explanation) before loading heavy numerical arrays.

    Comparing the image's exact bytes is deliberately conservative: different
    encodings of the same pixels are not assumed equivalent.  We cannot prove
    arbitrary image registration or pixel transforms from an equal shape alone.
    """
    if not base_data or not base_data.get('spatial') or not base_item:
        return False, 'Для слоёв сначала выберите пространственный основной результат'
    if item.get('category') == 'Вейвлеты':
        return False, 'Наложение числовых карт CWT пока не поддерживается'
    if item.get('category') not in {'Экстремумы', 'Огибающие', 'KNN', 'ML-кластеры'}:
        return False, 'Этот тип данных не поддерживается для наложения'
    if result_is_base(item, base_item):
        return False, 'Этот результат уже используется как основа'
    if not Path(item['path']).is_file():
        return False, 'Файл результата отсутствует'
    origin = item.get('task_folder')
    base_origin = base_item.get('task_folder')
    if not origin or not base_origin:
        return False, 'Неизвестно происхождение результата'
    try:
        own_shape = image_shape(origin)
        base_shape = image_shape(base_origin)
        if own_shape is None or base_shape is None:
            return False, 'Нет исходного изображения для подтверждения координат'
        if own_shape != base_shape or tuple(base_data.get('shape', ())) != own_shape:
            return False, 'Размеры исходного изображения и координатной карты различаются'
        if source_fingerprint(origin) != source_fingerprint(base_origin):
            return False, 'Исходные изображения различаются'
        # All standard pipeline outputs use original-image pixel coordinates.
        # Unrecognized external-coordinate artifacts must not be silently aligned.
        if base_item.get('coordinate_frame', 'source_pixels') != 'source_pixels':
            return False, 'Система координат основы не подтверждена'
        if item.get('coordinate_frame', 'source_pixels') != 'source_pixels':
            return False, 'Система координат результата не подтверждена'
        return True, ''
    except (OSError, ValueError) as error:
        return False, f'Не удалось проверить происхождение: {error}'


def result_is_base(item, base_item):
    return bool(base_item and Path(item['path']).resolve() == Path(base_item['path']).resolve())


def check_loaded(data, base_data):
    """Check exact provenance, shape and point-frame after the worker has loaded data."""
    if data.get('kind') != 'points' or not data.get('spatial'):
        return False, 'Данные не содержат пространственных точек для наложения'
    if tuple(data.get('shape', ())) != tuple(base_data.get('shape', ())):
        return False, 'Размеры координат результата не совпадают с основой'
    source = data.get('source_id')
    base_source = base_data.get('source_id')
    if not (isinstance(source, tuple) and source[0] == 'sha1' and source == base_source):
        return False, 'Не удалось подтвердить общий источник изображения'
    points = data.get('points')
    if points is None or getattr(points, 'ndim', 0) != 2 or points.shape[1] < 2:
        return False, 'Некорректная матрица координат точек'
    # The data loader currently exports spatial coordinates in source pixels;
    # reject coordinates beyond the source rectangle (including negative ones).
    import numpy as np
    xy = points[:, :2]
    h, w = base_data['shape']
    if not np.isfinite(xy).all():
        return False, 'Есть нечисловые координаты точек'
    if len(xy) and ((xy[:, 0] < 0).any() or (xy[:, 0] >= w).any()
                    or (xy[:, 1] < 0).any() or (xy[:, 1] >= h).any()):
        return False, 'Координаты точек выходят за размеры изображения'
    return True, ''
