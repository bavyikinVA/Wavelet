from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from scipy.signal import fftconvolve
from scipy.special import j0

# Параметры эксперимента
TEST_DIR = Path(r"C:\Users\bavyk\PycharmProjects\Wavelets\test_images\morlet_tests")
SCALE = 15
OMEGA0 = 6.0


def morlet2d_kernel(scale, omega0=6.0):
    """Изотропный 2D Morlet — угловое среднее."""

    if scale <= 0:
        raise ValueError("Масштаб должен быть положительным")

    radius = int(np.ceil(4 * scale))
    coords = np.arange(-radius, radius + 1)

    x, y = np.meshgrid(coords, coords)
    r = np.hypot(x, y)

    gaussian = np.exp(-r**2 / (2 * scale**2))
    carrier = j0(omega0 * r / scale)

    correction = np.sum(gaussian * carrier) / np.sum(gaussian)
    kernel = gaussian * (carrier - correction) / scale

    return kernel


def cwt2d(image, scale, omega0=6.0):
    """Полное двумерное преобразование Морле."""

    kernel = morlet2d_kernel(scale, omega0)

    coefficients = fftconvolve(
        image,
        np.conj(kernel[::-1, ::-1]),
        mode="same"
    )

    return coefficients


def load_image_grayscale(path: Path):
    with Image.open(path) as img:
        image = np.asarray(img.convert("L"), dtype=np.float64) / 255.0
    return image


def save_pair_figure(image, coefficients, image_name, output_path):
    max_abs = max(float(np.max(np.abs(coefficients))), 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    fig.suptitle(
        f"{image_name} | 2D isotropic Morlet | scale={SCALE}, omega0={OMEGA0}",
        fontsize=12
    )

    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Оригинал")
    axes[0].set_axis_off()

    im = axes[1].imshow(
        coefficients,
        cmap="RdBu_r",
        vmin=-max_abs,
        vmax=max_abs
    )
    axes[1].set_title("Результат 2D Morlet")
    axes[1].set_axis_off()

    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    if not TEST_DIR.is_dir():
        raise FileNotFoundError(f"Папка с тестами не найдена: {TEST_DIR}")

    image_paths = sorted(
        path for path in TEST_DIR.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    )

    if not image_paths:
        raise FileNotFoundError(f"В папке нет PNG-изображений: {TEST_DIR}")

    output_dir = TEST_DIR.parent / "morlet_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Найдено изображений: {len(image_paths)}")
    print(f"Результаты будут сохранены в: {output_dir}\n")

    for index, path in enumerate(image_paths, start=1):
        print(f"[{index}/{len(image_paths)}] Обрабатывается: {path.name}")

        image = load_image_grayscale(path)
        coefficients = cwt2d(image, scale=SCALE, omega0=OMEGA0)

        pair_output_path = output_dir / f"{path.stem}_pair.png"
        save_pair_figure(image, coefficients, path.name, pair_output_path)

        max_abs = float(np.max(np.abs(coefficients)))
        print(f"    Сохранён файл: {pair_output_path.name}")
        print(f"    Размер: {image.shape[1]}x{image.shape[0]} | max|W|={max_abs:.6g}\n")

    print("Готово. Для каждого теста сохранён только один файл-пара: оригинал + результат.")


if __name__ == "__main__":
    main()