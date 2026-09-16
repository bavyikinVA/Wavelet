"""
Двумерное непрерывное вейвлет-преобразование изображения (2D CWT)
с комплексным материнским вейвлетом Морле.

Пример запуска из корня проекта:

    python morlet_2d_analysis.py --image "C:\\Images\\sample.png"

Пример с собственными масштабами и углами:

    python morlet_2d_analysis.py ^
        --image "C:\\Images\\sample.png" ^
        --scales 4,8,16,32 ^
        --angles 0,30,60,90,120,150 ^
        --channel all ^
        --save-arrays

Результаты автоматически сохраняются вне проекта:

    C:\\Users\\<имя>\\Downloads\\Morlet_2D_Results\\<изображение>_<дата>\\
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import matplotlib
import numpy as np

from result_naming import dated_folder_name
from scipy.signal import fftconvolve

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# -----------------------------------------------------------------------------
# Настройки анализа
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalysisConfig:
    """Все параметры одного запуска двумерного анализа Морле."""

    image_path: Path
    scales: tuple[float, ...]
    angles_degrees: tuple[float, ...]
    channel: str
    omega0: float
    truncate: float
    save_arrays: bool
    save_3d: bool
    surface_max_points: int
    output_root: Path


# -----------------------------------------------------------------------------
# Разбор параметров командной строки
# -----------------------------------------------------------------------------


def parse_number_list(value: str, parameter_name: str) -> tuple[float, ...]:
    """
    Преобразует строку вида ``4,8,16`` в кортеж чисел.

    Дополнительно поддерживается диапазон ``start:end:step``:
    ``4:16:4`` превращается в ``(4, 8, 12, 16)``.
    Правая граница включается, если она достижима с указанным шагом.
    """

    value = value.strip()
    if not value:
        raise argparse.ArgumentTypeError(f"Параметр {parameter_name} пуст")

    try:
        if ":" in value:
            parts = [float(part.strip()) for part in value.split(":")]
            if len(parts) != 3:
                raise ValueError

            start, end, step = parts
            if step <= 0 or end < start:
                raise ValueError

            count = int(math.floor((end - start) / step + 1e-12)) + 1
            numbers = tuple(start + index * step for index in range(count))
        else:
            numbers = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Неверный формат {parameter_name}: {value!r}. "
            "Используйте, например, 4,8,16 или 4:16:4"
        ) from error

    if not numbers:
        raise argparse.ArgumentTypeError(f"Не заданы значения {parameter_name}")

    return numbers


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Вычисление комплексного двумерного CWT изображения "
            "с материнским вейвлетом Морле"
        )
    )
    parser.add_argument(
        "--image",
        required=True,
        type=Path,
        help="путь к исходному изображению",
    )
    parser.add_argument(
        "--scales",
        default="4,8,16,32",
        help="масштабы: 4,8,16,32 или диапазон 4:32:4",
    )
    parser.add_argument(
        "--angles",
        default="0,30,60,90,120,150",
        help="углы в градусах: 0,30,60,90,120,150",
    )
    parser.add_argument(
        "--channel",
        choices=("gray", "red", "green", "blue", "all"),
        default="gray",
        help="анализируемый канал; all означает отдельный анализ R, G и B",
    )
    parser.add_argument(
        "--omega0",
        type=float,
        default=6.0,
        help="центральная частота Морле (по умолчанию 6.0)",
    )
    parser.add_argument(
        "--truncate",
        type=float,
        default=4.0,
        help="радиус ядра в масштабах сигма (по умолчанию 4.0)",
    )
    parser.add_argument(
        "--save-arrays",
        action="store_true",
        help="дополнительно сохранять числовые коэффициенты в NPZ/NPY",
    )
    parser.add_argument(
        "--no-3d",
        action="store_true",
        help="не создавать трёхмерные графики мощности",
    )
    parser.add_argument(
        "--surface-max-points",
        type=int,
        default=180,
        help=(
            "максимальное число точек 3D-поверхности вдоль каждой оси; "
            "исходные числовые данные при этом не изменяются (по умолчанию 180)"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path.home() / "Downloads" / "Morlet_2D_Results",
        help=(
            "корневая папка результатов; по умолчанию "
            "Загрузки/Morlet_2D_Results"
        ),
    )
    return parser


def config_from_arguments() -> AnalysisConfig:
    args = build_argument_parser().parse_args()

    scales = parse_number_list(args.scales, "масштабов")
    angles = parse_number_list(args.angles, "углов")

    if any(scale <= 0 for scale in scales):
        raise ValueError("Все масштабы должны быть больше нуля")
    if any(angle < 0 or angle >= 180 for angle in angles):
        raise ValueError("Углы должны находиться в диапазоне [0, 180)")
    if args.omega0 <= 0:
        raise ValueError("omega0 должен быть больше нуля")
    if args.truncate < 2:
        raise ValueError("truncate должен быть не меньше 2")
    if args.surface_max_points < 20:
        raise ValueError("surface-max-points должен быть не меньше 20")

    return AnalysisConfig(
        image_path=args.image.expanduser().resolve(),
        scales=tuple(scales),
        angles_degrees=tuple(angles),
        channel=args.channel,
        omega0=args.omega0,
        truncate=args.truncate,
        save_arrays=args.save_arrays,
        save_3d=not args.no_3d,
        surface_max_points=args.surface_max_points,
        output_root=args.output_root.expanduser().resolve(),
    )


# -----------------------------------------------------------------------------
# Загрузка и подготовка изображения
# -----------------------------------------------------------------------------


def read_image_unicode(path: Path) -> np.ndarray:
    """Загружает изображение, включая пути с кириллицей в Windows."""

    if not path.is_file():
        raise FileNotFoundError(f"Изображение не найдено: {path}")

    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV не смог прочитать изображение: {path}")

    return image


def select_channels(
    image_bgr: np.ndarray,
    channel_mode: str,
) -> dict[str, np.ndarray]:
    """Возвращает выбранные каналы как матрицы float32 в диапазоне [0, 1]."""

    blue, green, red = cv2.split(image_bgr)

    channels = {
        "red": red,
        "green": green,
        "blue": blue,
        "gray": cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY),
    }

    selected_names = ("red", "green", "blue") if channel_mode == "all" else (channel_mode,)

    return {
        name: channels[name].astype(np.float32) / 255.0
        for name in selected_names
    }


# -----------------------------------------------------------------------------
# Математика двумерного вейвлета Морле
# -----------------------------------------------------------------------------


def create_morlet_2d_kernel(
    scale: float,
    angle_radians: float,
    omega0: float = 6.0,
    truncate: float = 4.0,
) -> np.ndarray:
    r"""
    Создаёт комплексное двумерное ядро Морле.

    Используемая материнская функция:

        psi(x, y) = C * [exp(i * omega0 * x') - exp(-omega0^2 / 2)]
                    * exp(-(x'^2 + y'^2) / 2)

    где (x', y') — повёрнутые и разделённые на масштаб координаты.

    Вычитание ``exp(-omega0^2 / 2)`` обеспечивает нулевое среднее
    непрерывного вейвлета. После обрезания ядра дополнительно вычитается
    его небольшое дискретное среднее. Затем выполняется L2-нормировка,
    чтобы отклики разных масштабов можно было сравнивать.
    """

    radius = int(math.ceil(truncate * scale))
    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]

    # Масштабирование координат. Больший scale создаёт более широкое ядро.
    x = x.astype(np.float64) / scale
    y = y.astype(np.float64) / scale

    cos_angle = math.cos(angle_radians)
    sin_angle = math.sin(angle_radians)

    # R(-theta): поворот системы координат относительно изображения.
    x_rotated = x * cos_angle + y * sin_angle
    y_rotated = -x * sin_angle + y * cos_angle

    gaussian_window = np.exp(-0.5 * (x_rotated**2 + y_rotated**2))
    admissibility_correction = math.exp(-0.5 * omega0**2)
    complex_oscillation = (
        np.exp(1j * omega0 * x_rotated) - admissibility_correction
    )

    kernel = complex_oscillation * gaussian_window

    # Конечное окно слегка нарушает нулевое среднее — исправляем это численно.
    kernel -= np.mean(kernel)

    norm = np.sqrt(np.sum(np.abs(kernel) ** 2))
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Не удалось нормировать ядро Морле")

    return (kernel / norm).astype(np.complex64)


def cwt2_morlet_single(
    image: np.ndarray,
    scale: float,
    angle_radians: float,
    omega0: float,
    truncate: float,
) -> np.ndarray:
    """
    Вычисляет 2D CWT для одного масштаба и одного угла.

    Преобразование является корреляцией изображения с вейвлетом.
    Корреляция вычисляется через быстрое двумерное преобразование Фурье.
    Перед расчётом изображение симметрично продолжается, чтобы уменьшить
    ложные отклики около его границ.
    """

    kernel = create_morlet_2d_kernel(
        scale=scale,
        angle_radians=angle_radians,
        omega0=omega0,
        truncate=truncate,
    )

    pad_y = kernel.shape[0] // 2
    pad_x = kernel.shape[1] // 2
    padded_image = np.pad(
        image,
        ((pad_y, pad_y), (pad_x, pad_x)),
        mode="reflect",
    )

    # fftconvolve выполняет свёртку. Для получения корреляции комплексное
    # ядро нужно сопрячь и развернуть по обеим координатам.
    correlation_kernel = np.conj(kernel[::-1, ::-1])
    padded_coefficients = fftconvolve(
        padded_image,
        correlation_kernel,
        mode="same",
    )

    coefficients = padded_coefficients[
        pad_y : pad_y + image.shape[0],
        pad_x : pad_x + image.shape[1],
    ]
    return coefficients.astype(np.complex64, copy=False)


def iterate_coefficients(
    image: np.ndarray,
    config: AnalysisConfig,
) -> Iterator[tuple[float, float, np.ndarray]]:
    """
    Последовательно выдаёт результат для каждой пары масштаб–угол.

    Генератор не создаёт общий массив формы
    ``[масштаб, угол, высота, ширина]`` и поэтому заметно экономит память.
    """

    centered_image = image - np.mean(image, dtype=np.float64)

    for scale in config.scales:
        for angle_degrees in config.angles_degrees:
            coefficients = cwt2_morlet_single(
                image=centered_image,
                scale=scale,
                angle_radians=math.radians(angle_degrees),
                omega0=config.omega0,
                truncate=config.truncate,
            )
            yield scale, angle_degrees, coefficients


# -----------------------------------------------------------------------------
# Сохранение результатов
# -----------------------------------------------------------------------------


def number_for_filename(value: float) -> str:
    """Преобразует число в безопасную и компактную часть имени файла."""

    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}".replace(".", "p")


def save_heatmap(
    matrix: np.ndarray,
    path: Path,
    title: str,
    colorbar_label: str,
    cmap: str,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Сохраняет двумерную матрицу как PNG с цветовой шкалой."""

    figure, axis = plt.subplots(figsize=(10, 7))
    image = axis.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_title(title)
    axis.set_xlabel("X, пиксели")
    axis.set_ylabel("Y, пиксели")
    figure.colorbar(image, ax=axis, label=colorbar_label)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def robust_positive_limit(matrix: np.ndarray) -> float | None:
    """Возвращает 99.5-й процентиль для читаемой визуализации выбросов."""

    finite_values = matrix[np.isfinite(matrix)]
    if finite_values.size == 0:
        return None

    limit = float(np.percentile(finite_values, 99.5))
    return limit if limit > 0 else None


def save_power_surface_3d(
    power: np.ndarray,
    path: Path,
    title: str,
    max_points: int,
) -> None:
    """
    Сохраняет пространственную 3D-поверхность в координатах X–Y–мощность.

    Матрица коэффициентов может содержать миллионы пикселей. Matplotlib не
    требуется рисовать каждый из них, поэтому для графика выбирается
    равномерная сетка не более ``max_points × max_points``. Это прореживание
    касается только PNG-визуализации и никак не меняет расчёт или NPZ-файлы.
    """

    height, width = power.shape
    step_y = max(1, math.ceil(height / max_points))
    step_x = max(1, math.ceil(width / max_points))

    sampled_power = power[::step_y, ::step_x].astype(np.float64, copy=False)
    x_coordinates = np.arange(0, width, step_x)
    y_coordinates = np.arange(0, height, step_y)
    x_grid, y_grid = np.meshgrid(x_coordinates, y_coordinates)

    # Очень редкие большие пики могут сделать остальную поверхность почти
    # плоской. Ограничиваем только отображаемую высоту 99.5-м процентилем.
    display_limit = robust_positive_limit(sampled_power)
    displayed_power = (
        np.minimum(sampled_power, display_limit)
        if display_limit is not None
        else sampled_power
    )

    figure = plt.figure(figsize=(12, 8))
    axis = figure.add_subplot(111, projection="3d")
    surface = axis.plot_surface(
        x_grid,
        y_grid,
        displayed_power,
        cmap="inferno",
        linewidth=0,
        antialiased=True,
        rcount=min(max_points, displayed_power.shape[0]),
        ccount=min(max_points, displayed_power.shape[1]),
    )
    axis.set_title(title)
    axis.set_xlabel("X, пиксели")
    axis.set_ylabel("Y, пиксели")
    axis.set_zlabel("Мощность |W|²")
    axis.view_init(elev=35, azim=-55)
    axis.invert_yaxis()
    figure.colorbar(surface, ax=axis, shrink=0.65, pad=0.1, label="|W|²")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_scale_angle_surface_3d(
    mean_power: np.ndarray,
    scales: Sequence[float],
    angles_degrees: Sequence[float],
    path: Path,
    channel_name: str,
) -> None:
    """
    Строит общую 3D-схему «угол–масштаб–средняя мощность».

    Такой график показывает, структуры какого размера и направления сильнее
    всего представлены во всём изображении. Для локального положения этих
    структур нужно смотреть отдельные пространственные 3D-поверхности.
    """

    angle_grid, scale_grid = np.meshgrid(
        np.asarray(angles_degrees, dtype=np.float64),
        np.asarray(scales, dtype=np.float64),
    )

    figure = plt.figure(figsize=(11, 8))
    axis = figure.add_subplot(111, projection="3d")

    if len(scales) >= 2 and len(angles_degrees) >= 2:
        graph = axis.plot_surface(
            angle_grid,
            scale_grid,
            mean_power,
            cmap="viridis",
            linewidth=0.35,
            edgecolor="black",
            antialiased=True,
        )
    else:
        graph = axis.scatter(
            angle_grid.ravel(),
            scale_grid.ravel(),
            mean_power.ravel(),
            c=mean_power.ravel(),
            cmap="viridis",
            s=70,
        )

    axis.set_title(
        f"Масштабно-угловой спектр 2D Морле: канал={channel_name}"
    )
    axis.set_xlabel("Угол, градусы")
    axis.set_ylabel("Масштаб, пиксели")
    axis.set_zlabel("Средняя мощность")
    axis.view_init(elev=30, azim=-55)
    figure.colorbar(
        graph,
        ax=axis,
        shrink=0.65,
        pad=0.1,
        label="Средняя |W|²",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_angle_result(
    coefficients: np.ndarray,
    result_directory: Path,
    channel_name: str,
    scale: float,
    angle_degrees: float,
    save_arrays: bool,
    save_3d: bool,
    surface_max_points: int,
) -> np.ndarray:
    """Сохраняет амплитуду, мощность, фазу и, при запросе, коэффициенты."""

    result_directory.mkdir(parents=True, exist_ok=True)

    amplitude = np.abs(coefficients)
    power = amplitude**2
    phase = np.angle(coefficients)
    description = (
        f"канал={channel_name}, масштаб={scale:g}, угол={angle_degrees:g}°"
    )

    save_heatmap(
        amplitude,
        result_directory / "amplitude.png",
        f"Амплитуда 2D Морле: {description}",
        "|W|",
        "viridis",
        vmin=0,
        vmax=robust_positive_limit(amplitude),
    )
    save_heatmap(
        power,
        result_directory / "power.png",
        f"Мощность 2D Морле: {description}",
        "|W|²",
        "inferno",
        vmin=0,
        vmax=robust_positive_limit(power),
    )
    save_heatmap(
        phase,
        result_directory / "phase.png",
        f"Фаза 2D Морле: {description}",
        "фаза, рад",
        "twilight",
        vmin=-math.pi,
        vmax=math.pi,
    )

    if save_3d:
        save_power_surface_3d(
            power=power,
            path=result_directory / "power_3d.png",
            title=f"3D-мощность 2D Морле: {description}",
            max_points=surface_max_points,
        )

    if save_arrays:
        np.savez_compressed(
            result_directory / "coefficients.npz",
            coefficients=coefficients,
            amplitude=amplitude.astype(np.float32),
            power=power.astype(np.float32),
            phase=phase.astype(np.float32),
        )

    return power


def save_scale_summary(
    maximum_power: np.ndarray,
    dominant_angles: np.ndarray,
    output_directory: Path,
    channel_name: str,
    scale: float,
    save_arrays: bool,
) -> None:
    """Сохраняет максимальный отклик и доминирующий угол для масштаба."""

    summary_directory = output_directory / "summary"
    summary_directory.mkdir(parents=True, exist_ok=True)

    save_heatmap(
        maximum_power,
        summary_directory / "maximum_power.png",
        f"Максимальная мощность: канал={channel_name}, масштаб={scale:g}",
        "max |W|²",
        "inferno",
        vmin=0,
        vmax=robust_positive_limit(maximum_power),
    )
    save_heatmap(
        dominant_angles,
        summary_directory / "dominant_angle.png",
        f"Доминирующий угол: канал={channel_name}, масштаб={scale:g}",
        "угол, градусы",
        "hsv",
        vmin=0,
        vmax=180,
    )

    if save_arrays:
        np.save(summary_directory / "maximum_power.npy", maximum_power)
        np.save(summary_directory / "dominant_angle_degrees.npy", dominant_angles)


def create_run_directory(config: AnalysisConfig) -> Path:
    """Создаёт уникальную папку запуска в Загрузках."""

    timestamp = dated_folder_name(datetime.now())
    image_name = config.image_path.stem
    run_directory = config.output_root / f"{image_name}_{timestamp}"
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_directory


# -----------------------------------------------------------------------------
# Полный сценарий анализа
# -----------------------------------------------------------------------------


def analyze_channel(
    image: np.ndarray,
    channel_name: str,
    config: AnalysisConfig,
    run_directory: Path,
) -> None:
    """Вычисляет и сохраняет все масштабы и углы одного канала."""

    channel_directory = run_directory / f"channel_{channel_name}"
    mean_power_by_scale_and_angle = np.zeros(
        (len(config.scales), len(config.angles_degrees)),
        dtype=np.float64,
    )

    # Обрабатываем по одному масштабу. В памяти находятся только текущие
    # коэффициенты и две итоговые карты масштаба.
    for scale_index, scale in enumerate(config.scales):
        scale_name = number_for_filename(scale)
        scale_directory = channel_directory / f"scale_{scale_name}"
        maximum_power: np.ndarray | None = None
        dominant_angles = np.zeros(image.shape, dtype=np.float32)

        for current_scale, angle_degrees, coefficients in iterate_coefficients(
            image,
            AnalysisConfig(
                image_path=config.image_path,
                scales=(scale,),
                angles_degrees=config.angles_degrees,
                channel=config.channel,
                omega0=config.omega0,
                truncate=config.truncate,
                save_arrays=config.save_arrays,
                save_3d=config.save_3d,
                surface_max_points=config.surface_max_points,
                output_root=config.output_root,
            ),
        ):
            angle_name = number_for_filename(angle_degrees)
            angle_directory = scale_directory / f"angle_{angle_name}_deg"

            print(
                f"  Канал {channel_name}: масштаб {current_scale:g}, "
                f"угол {angle_degrees:g}°"
            )
            power = save_angle_result(
                coefficients=coefficients,
                result_directory=angle_directory,
                channel_name=channel_name,
                scale=current_scale,
                angle_degrees=angle_degrees,
                save_arrays=config.save_arrays,
                save_3d=config.save_3d,
                surface_max_points=config.surface_max_points,
            )

            angle_index = config.angles_degrees.index(angle_degrees)
            mean_power_by_scale_and_angle[scale_index, angle_index] = float(
                np.mean(power, dtype=np.float64)
            )

            if maximum_power is None:
                maximum_power = power.astype(np.float32, copy=True)
                dominant_angles.fill(angle_degrees)
            else:
                stronger = power > maximum_power
                maximum_power[stronger] = power[stronger]
                dominant_angles[stronger] = angle_degrees

        if maximum_power is None:
            raise RuntimeError("Не получены коэффициенты ни для одного угла")

        save_scale_summary(
            maximum_power=maximum_power,
            dominant_angles=dominant_angles,
            output_directory=scale_directory,
            channel_name=channel_name,
            scale=scale,
            save_arrays=config.save_arrays,
        )

    if config.save_3d:
        save_scale_angle_surface_3d(
            mean_power=mean_power_by_scale_and_angle,
            scales=config.scales,
            angles_degrees=config.angles_degrees,
            path=channel_directory / "scale_angle_mean_power_3d.png",
            channel_name=channel_name,
        )

    if config.save_arrays:
        np.save(
            channel_directory / "scale_angle_mean_power.npy",
            mean_power_by_scale_and_angle,
        )


def main() -> None:
    config = config_from_arguments()
    image_bgr = read_image_unicode(config.image_path)
    channels = select_channels(image_bgr, config.channel)
    run_directory = create_run_directory(config)

    print("Двумерный анализ Морле запущен")
    print(f"Изображение: {config.image_path}")
    print(f"Размер: {image_bgr.shape[1]} x {image_bgr.shape[0]} пикселей")
    print(f"Результаты: {run_directory}")

    for channel_name, channel_image in channels.items():
        analyze_channel(
            image=channel_image,
            channel_name=channel_name,
            config=config,
            run_directory=run_directory,
        )

    print("Анализ завершён")
    print(f"Все файлы сохранены в: {run_directory}")


if __name__ == "__main__":
    main()
