"""2D Morlet CWT: directional kernels and their isotropic angular mean.

Dependencies: numpy, scipy, matplotlib
Run: python morlet2d_rings.py

The isotropic construction is the *angular average of 2D Morlet wavelets*,
not a claim that a single directional Morlet produces concentric rings.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve
from scipy.special import j0


def _grid(scale: float, support: float = 4.0):
    if scale <= 0:
        raise ValueError("scale must be > 0")
    radius = int(np.ceil(support * scale))
    coord = np.arange(-radius, radius + 1, dtype=np.float64)
    x, y = np.meshgrid(coord, coord, indexing="xy")
    envelope = np.exp(-(x * x + y * y) / (2.0 * scale * scale))
    return x, y, envelope


def morlet2d_directional(scale: float, theta: float, omega0: float = 6.0):
    """Complex 2D Morlet at angle theta (in radians); L2 scale 1/scale.

    Subtract the envelope-weighted discrete mean to guarantee zero DC for
    the finite sampled kernel. Continuous theory gives exp(-omega0**2/2).
    """
    x, y, envelope = _grid(scale)
    phase = (omega0 / scale) * (x * np.cos(theta) + y * np.sin(theta))
    carrier = np.exp(1j * phase)
    dc_correction = np.sum(envelope * carrier) / np.sum(envelope)
    return (envelope * (carrier - dc_correction)) / scale


def morlet2d_isotropic(scale: float, omega0: float = 6.0):
    """Exact angular average of complex directional Morlet (continuum).

    Angle integration gives Bessel J0(omega0 * r / scale), so this is a
    *real, isotropic* wavelet with alternating circular lobes.
    """
    x, y, envelope = _grid(scale)
    r = np.hypot(x, y)
    carrier = j0((omega0 / scale) * r)
    dc_correction = np.sum(envelope * carrier) / np.sum(envelope)
    return (envelope * (carrier - dc_correction)) / scale


def morlet2d_angular_average(scale: float, omega0: float = 6.0,
                             n_angles: int = 64):
    """Numerical approximation by averaging over angles [0, 2*pi)."""
    if n_angles < 1:
        raise ValueError("n_angles must be positive")
    accum = np.zeros_like(morlet2d_directional(scale, 0.0, omega0))
    for theta in np.linspace(0, 2 * np.pi, n_angles, endpoint=False):
        accum += morlet2d_directional(scale, theta, omega0)
    return accum / n_angles


def cwt2d(image: np.ndarray, wavelet: np.ndarray) -> np.ndarray:
    """Single-scale 2D continuous wavelet coefficients, zero-padded edges.

    W(a,b) = integral image(x) * conj(wavelet(x-b)) dx.
    fftconvolve + reversed-conjugate kernel computes correlation.
    """
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("image must be a 2D array")
    if not np.isfinite(image).all():
        raise ValueError("image contains NaN or infinity")
    return fftconvolve(image, np.conj(wavelet[::-1, ::-1]), mode="same")


def main():
    size, scale, omega0, n_angles = 257, 15.0, 6.0, 64
    delta = np.zeros((size, size), dtype=np.float64)
    center = size // 2
    delta[center, center] = 1.0

    k0 = morlet2d_directional(scale, theta=0.0, omega0=omega0)
    k45 = morlet2d_directional(scale, theta=np.pi / 4, omega0=omega0)
    k_average = morlet2d_angular_average(scale, omega0, n_angles)
    k_iso = morlet2d_isotropic(scale, omega0)

    directional_0 = cwt2d(delta, k0)
    directional_45 = cwt2d(delta, k45)
    numerical_average = cwt2d(delta, k_average)
    isotropic = cwt2d(delta, k_iso)

    # Delta gives back the wavelet shape, and numerical angle integration
    # must agree with the closed form containing Bessel J0.
    max_error = np.max(np.abs(numerical_average - isotropic))
    assert max_error < 1e-8, f"Angular average differs from J0 by {max_error}"
    assert abs(np.sum(k_iso)) < 1e-10
    assert abs(np.sum(k0)) < 1e-10
    print(f"Angular average vs J0: max absolute error = {max_error:.3e}")
    print(f"Isotropic kernel discrete sum = {np.sum(k_iso):.3e}")

    # Crop around the delta so that the rings remain clearly visible.
    crop = 64
    sl = np.s_[center-crop:center+crop+1, center-crop:center+crop+1]
    extent = [-crop - 0.5, crop + 0.5, crop + 0.5, -crop - 0.5]
    maxabs = np.max(np.abs(isotropic))

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    panels = [
        (delta[sl], "Input: central delta", "gray", None, None),
        (directional_0[sl].real, "Directional Morlet: 0 deg (real)",
         "RdBu_r", -maxabs, maxabs),
        (directional_45[sl].real, "Directional Morlet: 45 deg (real)",
         "RdBu_r", -maxabs, maxabs),
        (numerical_average[sl].real,
         f"Average over {n_angles} directions (real)", "RdBu_r", -maxabs, maxabs),
        (isotropic[sl].real, "Analytic isotropic Morlet (Bessel J0)",
         "RdBu_r", -maxabs, maxabs),
        (np.abs(isotropic[sl]), "Absolute value of isotropic response",
         "magma", 0, maxabs),
    ]

    for ax, (values, title, cmap, vmin, vmax) in zip(axes.flat, panels):
        im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax,
                       extent=extent, interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x (pixels relative to center)")
        ax.set_ylabel("y (pixels relative to center)")
        fig.colorbar(im, ax=ax, shrink=0.77)

    fig.suptitle(f"2D Morlet on an impulse | scale={scale:g}, omega0={omega0:g}",
                 fontsize=15)
    out_png = Path(__file__).with_name("morlet2d_rings.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    # Save coefficients for further investigation in Wavelets Analysis Studio.
    out_npz = Path(__file__).with_name("morlet2d_rings.npz")
    np.savez_compressed(out_npz, delta=delta,
                        directional_0=directional_0,
                        directional_45=directional_45,
                        angular_mean=numerical_average,
                        isotropic=isotropic,
                        scale=np.array(scale), omega0=np.array(omega0))
    print("Saved:", out_png)
    print("Saved:", out_npz)


if __name__ == "__main__":
    main()
