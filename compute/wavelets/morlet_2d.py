"""Настоящее двумерное непрерывное преобразование Морле."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy.signal import fftconvolve


def morlet_kernel_2d(
        scale: float,
        angle_deg: float,
        omega0: float = 6.0,
        anisotropy: float = 1.0,
        truncate: float = 4.0) -> np.ndarray:
    """Построить комплексный повёрнутый 2D-вейвлет Морле.

    Нормировка ``1 / scale`` соответствует L2-нормировке в двумерном
    пространстве. Из осциллирующей части вычитается поправка
    ``exp(-omega0**2 / 2)``, обеспечивающая практически нулевое среднее.
    """
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Масштаб должен быть положительным")
    if not np.isfinite(omega0) or omega0 <= 0:
        raise ValueError("Центральная частота должна быть положительной")
    if not np.isfinite(anisotropy) or anisotropy <= 0:
        raise ValueError("Анизотропность должна быть положительной")

    radius = max(2, int(math.ceil(truncate * scale * max(1.0, anisotropy))))
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    x, y = np.meshgrid(coordinates, coordinates)

    angle = np.deg2rad(angle_deg)
    x_rot = x * np.cos(angle) + y * np.sin(angle)
    y_rot = -x * np.sin(angle) + y * np.cos(angle)

    x_scaled = x_rot / scale
    y_scaled = y_rot / (scale * anisotropy)
    gaussian = np.exp(-0.5 * (x_scaled ** 2 + y_scaled ** 2))
    correction = np.exp(-0.5 * omega0 ** 2)
    carrier = np.exp(1j * omega0 * x_scaled) - correction
    kernel = carrier * gaussian / scale

    # Дискретизация и усечение немного нарушают нулевое среднее.
    kernel -= kernel.mean()
    norm = np.sqrt(np.sum(np.abs(kernel) ** 2))
    if norm > 0:
        kernel /= norm
    return kernel.astype(np.complex64)


def cwt2d_morlet_cpu(
        image: np.ndarray,
        scales: Iterable[float],
        orientations: Iterable[float],
        omega0: float = 6.0,
        anisotropy: float = 1.0) -> np.ndarray:
    """Вычислить 2D CWT на CPU.

    Возвращаемая форма: ``(число масштабов, число ориентаций, H, W)``.
    """
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("Для 2D CWT ожидается двумерный канал изображения")

    scales = [float(value) for value in scales]
    orientations = [float(value) for value in orientations]
    result = np.empty(
        (len(scales), len(orientations), image.shape[0], image.shape[1]),
        dtype=np.complex64
    )

    centered = image - float(image.mean())
    for scale_index, scale in enumerate(scales):
        for angle_index, angle in enumerate(orientations):
            kernel = morlet_kernel_2d(scale, angle, omega0, anisotropy)
            result[scale_index, angle_index] = fftconvolve(
                centered,
                np.conjugate(kernel[::-1, ::-1]),
                mode="same"
            ).astype(np.complex64)
    return result


def cwt2d_morlet_gpu(
        image: np.ndarray,
        scales: Iterable[float],
        orientations: Iterable[float],
        omega0: float = 6.0,
        anisotropy: float = 1.0) -> np.ndarray:
    """Вычислить 2D CWT на GPU через CuPy с возвратом NumPy-массива."""
    import cupy as cp
    from cupyx.scipy.signal import fftconvolve as gpu_fftconvolve

    image_gpu = cp.asarray(np.asarray(image, dtype=np.float32))
    image_gpu = image_gpu - cp.mean(image_gpu)
    scales = [float(value) for value in scales]
    orientations = [float(value) for value in orientations]
    result = np.empty(
        (len(scales), len(orientations), image.shape[0], image.shape[1]),
        dtype=np.complex64
    )

    for scale_index, scale in enumerate(scales):
        for angle_index, angle in enumerate(orientations):
            kernel = morlet_kernel_2d(scale, angle, omega0, anisotropy)
            kernel_gpu = cp.asarray(np.conjugate(kernel[::-1, ::-1]))
            coefficients = gpu_fftconvolve(image_gpu, kernel_gpu, mode="same")
            result[scale_index, angle_index] = cp.asnumpy(coefficients)

    cp.get_default_memory_pool().free_all_blocks()
    return result
