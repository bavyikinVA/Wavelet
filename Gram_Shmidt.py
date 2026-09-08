import numpy as np


# Numerical independence threshold: sin(angle), invariant to RGB brightness.
MIN_COLOR_SINE = 1e-6


def _unit_color(value):
    try:
        color = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Цвет должен содержать три числовых RGB-компонента") from error
    if color.shape != (3,) or not np.isfinite(color).all():
        raise ValueError("Цвет должен содержать три конечных RGB-компонента")
    magnitude = np.max(np.abs(color))
    if magnitude == 0:
        raise ValueError("Черный цвет не задает направление. Выберите ненулевой цвет")
    # Scaling first avoids overflow/underflow in the norm.
    scaled = color / magnitude
    return scaled / np.linalg.norm(scaled)


def change_channels(v1, v2, data):
    def gram_schmidt(v1_, v2_):
        v1_normalize = _unit_color(v1_)
        v2_ = _unit_color(v2_)

        v2_proj = np.dot(v2_, v1_normalize) * v1_normalize
        v2_orth = v2_ - v2_proj

        sine = np.linalg.norm(v2_orth)
        if sine <= MIN_COLOR_SINE:
            raise ValueError(
                "Выбранные цвета совпадают или слишком близки по направлению. "
                "Выберите другой цветовой оттенок, а не только другую яркость"
            )
        v2_orth_normalize = v2_orth / sine

        return v1_normalize, v2_orth_normalize

    def cross_product(v1_, v2_):
        v3 = np.cross(v1_, v2_)
        v3_normalize = v3 / np.linalg.norm(v3)
        return v3_normalize

    data = np.array(data, dtype=np.float64)

    # Perform Gram-Schmidt
    v1_norm, v2_orth_norm = gram_schmidt(v1, v2)
    v3_norm = cross_product(v1_norm, v2_orth_norm)

    # Transformation matrix
    p = np.array([v1_norm, v2_orth_norm, v3_norm]).T
    # Transform image data
    data_transposed = np.transpose(data, (2, 0, 1))
    data_new = p @ data_transposed
    data_new = np.transpose(data_new, (1, 2, 0))

    return data_new
