import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

def get_row_envelopes(coefs, max_points, min_points):
    """Построение огибающих и нахождение их экстремумов для всех строк изображения"""
    num_rows = coefs.shape[0]
    upper_envelopes = np.zeros_like(coefs)
    lower_envelopes = np.zeros_like(coefs)
    upper_max_points = []
    lower_min_points = []

    for row in range(num_rows):
        row_data = coefs[row, :]
        # print(f"Row {row}, data shape: {row_data.shape}, min: {row_data.min()}, max: {row_data.max()}")  # Отладка

        # координаты экстремумов для текущей строки
        row_max_points = np.array([x for x, y in max_points if y == row])
        row_min_points = np.array([x for x, y in min_points if y == row])
        # print(f"Row {row}, max_points: {len(row_max_points)}, min_points: {len(row_min_points)}")  # Отладка

        # значения в точках экстремумов
        if len(row_max_points) > 0:
            max_values = row_data[row_max_points]
        else:
            max_values = np.array([])
            upper_envelopes[row, :] = row_data
            # print(f"Row {row}, max_values empty")

        if len(row_min_points) > 0:
            min_values = row_data[row_min_points]
        else:
            min_values = np.array([])
            lower_envelopes[row, :] = row_data
            # print(f"Row {row}, min_values empty")

        t = np.arange(len(row_data))

        if len(row_max_points) > 3:
            f_max = interp1d(row_max_points, max_values, kind='linear',
                             fill_value="extrapolate", bounds_error=False)
            upper_envelopes[row, :] = f_max(t)
        # else: уже обработано выше

        if len(row_min_points) > 3:
            f_min = interp1d(row_min_points, min_values, kind='linear',
                             fill_value="extrapolate", bounds_error=False)
            lower_envelopes[row, :] = f_min(t)
        # else: уже обработано выше

        # экстремумы огибающих
        upper_max_idx, _ = find_peaks(upper_envelopes[row, :])
        lower_min_idx, _ = find_peaks(-lower_envelopes[row, :])
        # print(f"Row {row}, upper_max_idx: {len(upper_max_idx)}, lower_min_idx: {len(lower_min_idx)}")  # Отладка

        upper_max_points.extend([(x, row) for x in upper_max_idx])
        lower_min_points.extend([(x, row) for x in lower_min_idx])

    return upper_max_points, lower_min_points


def get_column_envelopes(coefs, max_points, min_points):
    """Построение огибающих и нахождение их экстремумов для всех столбцов изображения"""
    num_cols = coefs.shape[1]
    upper_envelopes = np.zeros_like(coefs)
    lower_envelopes = np.zeros_like(coefs)
    upper_max_points = []
    lower_min_points = []

    for col in range(num_cols):
        col_data = coefs[:, col]
        # print(f"Col {col}, data shape: {col_data.shape}, min: {col_data.min()}, max: {col_data.max()}")  # Отладка

        # координаты экстремумов для текущего столбца
        col_max_points = np.array([y for x, y in max_points if x == col])
        col_min_points = np.array([y for x, y in min_points if x == col])
        # print(f"Col {col}, max_points: {len(col_max_points)}, min_points: {len(col_min_points)}")  # Отладка

        # значения в точках экстремумов
        if len(col_max_points) > 0:
            max_values = col_data[col_max_points]
        else:
            max_values = np.array([])
            upper_envelopes[:, col] = col_data
            # print(f"Col {col}, max_values empty")

        if len(col_min_points) > 0:
            min_values = col_data[col_min_points]
        else:
            min_values = np.array([])
            lower_envelopes[:, col] = col_data
            # print(f"Col {col}, min_values empty")

        t = np.arange(len(col_data))

        if len(col_max_points) > 3:
            f_max = interp1d(col_max_points, max_values, kind='linear',
                             fill_value="extrapolate", bounds_error=False)
            upper_envelopes[:, col] = f_max(t)
        # else: уже обработано выше

        if len(col_min_points) > 3:
            f_min = interp1d(col_min_points, min_values, kind='linear',
                             fill_value="extrapolate", bounds_error=False)
            lower_envelopes[:, col] = f_min(t)
        # else: уже обработано выше

        # экстремумы огибающих
        upper_max_idx, _ = find_peaks(upper_envelopes[:, col])
        lower_min_idx, _ = find_peaks(-lower_envelopes[:, col])
        # print(f"Col {col}, upper_max_idx: {len(upper_max_idx)}, lower_min_idx: {len(lower_min_idx)}")  # Отладка

        # координаты экстремумов в формате (x, y)
        upper_max_points.extend([(col, y) for y in upper_max_idx])
        lower_min_points.extend([(col, y) for y in lower_min_idx])

    return upper_max_points, lower_min_points