"""Изотропное 2D CWT Морле — непрерывное среднее по всем направлениям.

Математическое ядро перенесено из корневого ``morlet2d_short.py``:
    psi_s(r) = exp(-r²/(2s²)) * [J0(omega0*r/s) - C_s] / s,
где C_s — дискретное среднее J0 с гауссовыми весами.

Здесь нет дискретного списка ориентаций, анизотропии и комплексной фазы:
угловое среднее плоской комплексной волны является действительной J0.
CPU и GPU используют *одно и то же* дискретное ядро; результат — float32.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy.signal import fftconvolve
from scipy.special import j0


def morlet2d_kernel(scale: float, omega0: float = 6.0) -> np.ndarray:
    """Действительное изотропное ядро, как в morlet2d_short.py (float64).

    Дискретная поправка обеспечивает ``sum(kernel) ≈ 0``. Размер ядра —
    ``(2 * ceil(4 * scale) + 1)²``; нормировка строго ``1 / scale``.
    """
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Масштаб должен быть конечным положительным числом")
    if not np.isfinite(omega0) or omega0 <= 0:
        raise ValueError("Центральная частота должна быть конечным положительным числом")

    radius = int(math.ceil(4 * scale))
    coords = np.arange(-radius, radius + 1, dtype=np.float64)
    x, y = np.meshgrid(coords, coords)
    r = np.hypot(x, y)

    gaussian = np.exp(-(r ** 2) / (2 * scale ** 2))
    carrier = j0(omega0 * r / scale)
    correction = np.sum(gaussian * carrier) / np.sum(gaussian)
    return gaussian * (carrier - correction) / scale


def _validate_image_and_scales(image, scales):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2 or min(image.shape) == 0:
        raise ValueError("Для 2D CWT ожидается непустой двумерный канал изображения")
    if not np.all(np.isfinite(image)):
        raise ValueError("Изображение содержит неконечные значения")
    try:
        scales = [float(value) for value in scales]
    except (TypeError, ValueError) as error:
        raise ValueError("Масштабы 2D CWT должны быть числами") from error
    if not scales:
        raise ValueError("Задайте хотя бы один масштаб 2D CWT")
    if any(not np.isfinite(scale) or scale <= 0 for scale in scales):
        raise ValueError("Масштабы 2D CWT должны быть конечными положительными числами")
    return image, scales


def cwt2d_morlet_cpu(
        image: np.ndarray,
        scales: Iterable[float],
        omega0: float = 6.0) -> np.ndarray:
    """2D FFT-свёртка на CPU; возвращает ``(n_scales, H, W)`` float32.

    В отличие от прежнего ориентированного CWT, *не* вычитаем среднее
    изображения: это соответствует исходному эксперименту morlet2d_short.py.
    Используется нулевое продолжение за границами (fftconvolve, same).
    """
    image, scales = _validate_image_and_scales(image, scales)
    result = np.empty((len(scales), *image.shape), dtype=np.float32)
    for index, scale in enumerate(scales):
        kernel = morlet2d_kernel(scale, omega0).astype(np.float32)
        result[index] = fftconvolve(image, kernel[::-1, ::-1], mode="same")
    return result


def cwt2d_morlet_gpu(
        image: np.ndarray,
        scales: Iterable[float],
        omega0: float = 6.0) -> np.ndarray:
    """То же 2D CWT на CuPy; возвращает NumPy ``(n_scales,H,W)`` float32."""
    image, scales = _validate_image_and_scales(image, scales)
    import cupy as cp
    from cupyx.scipy.signal import fftconvolve as gpu_fftconvolve

    image_gpu = cp.asarray(image)
    result = np.empty((len(scales), *image.shape), dtype=np.float32)
    try:
        for index, scale in enumerate(scales):
            kernel_gpu = cp.asarray(morlet2d_kernel(scale, omega0).astype(np.float32))
            coefficients = gpu_fftconvolve(image_gpu, kernel_gpu[::-1, ::-1], mode="same")
            result[index] = cp.asnumpy(coefficients).astype(np.float32, copy=False)
            del coefficients, kernel_gpu
    finally:
        del image_gpu
        cp.get_default_memory_pool().free_all_blocks()
    return result
