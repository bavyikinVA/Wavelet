import numpy as np


def validate_scales(scales):
    """Return a finite positive 1D scale vector, or a user-facing error."""
    try:
        values = np.asarray(scales, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Масштабы должны быть числами") from error
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Задайте непустой одномерный список масштабов")
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Все масштабы должны быть конечными числами больше нуля")
    return values
