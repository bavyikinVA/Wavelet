"""Операции обрезки в пиксельных координатах оригинала; не зависят от GUI."""

import math
from PIL import Image, ImageDraw


def rectangle_crop(image: Image.Image, coords):
    """Обрезать по рамке (x1,y1,x2,y2), ограниченной границами изображения.

    Возвращает новый объект Image. Пограничные координаты округляются наружу,
    чтобы область не теряла последний пиксель из-за масштабирования предпросмотра.
    """
    if image is None or len(coords) != 4 or not all(math.isfinite(v) for v in coords):
        raise ValueError('Некорректная прямоугольная область')
    x1, x2 = sorted((coords[0], coords[2]))
    y1, y2 = sorted((coords[1], coords[3]))
    bounds = (max(0, math.floor(x1)), max(0, math.floor(y1)),
              min(image.width, math.ceil(x2)), min(image.height, math.ceil(y2)))
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise ValueError('Выделение должно находиться внутри изображения')
    return image.crop(bounds)


def polygon_crop(image: Image.Image, points, *, background='black'):
    """Вырезать ограничивающий прямоугольник полигона с RGB-фоном снаружи.

    Прозрачность намеренно не используется: вычислительный конвейер работает с
    RGB-массивами. background='black' или 'white' определяет маскирующие пиксели.
    """
    if image is None or len(points) < 3:
        raise ValueError('Для полигона требуются минимум три вершины')
    if background not in ('black', 'white'):
        raise ValueError('Неизвестный фон полигона')
    if not all(len(p) == 2 and all(math.isfinite(v) for v in p) for p in points):
        raise ValueError('Некорректные координаты полигона')
    if any(not (0 <= x < image.width and 0 <= y < image.height) for x, y in points):
        raise ValueError('Вершины полигона должны находиться внутри изображения')
    twice_area = abs(sum(
        points[i][0] * points[(i + 1) % len(points)][1]
        - points[(i + 1) % len(points)][0] * points[i][1]
        for i in range(len(points))
    ))
    if twice_area < 2:
        raise ValueError('Полигон должен ограничивать ненулевую область')

    # Строим маску только в пределах полигона, не создавая промежуточных
    # полноразмерных RGBA/RGB-изображений (важно для научных снимков).
    left = max(0, math.floor(min(p[0] for p in points)))
    top = max(0, math.floor(min(p[1] for p in points)))
    right = min(image.width, math.floor(max(p[0] for p in points)) + 1)
    bottom = min(image.height, math.floor(max(p[1] for p in points)) + 1)
    width, height = right - left, bottom - top
    if width <= 0 or height <= 0:
        raise ValueError('Полигон не содержит пикселей')
    mask = Image.new('L', (width, height), 0)
    ImageDraw.Draw(mask).polygon([(x - left, y - top) for x, y in points], fill=255)
    if mask.getbbox() is None:
        raise ValueError('Полигон не содержит пикселей')
    crop = image.crop((left, top, right, bottom)).convert('RGB')
    result = Image.new('RGB', (width, height), background)
    result.paste(crop, (0, 0), mask)
    return result
