import numpy as np
import matplotlib.pyplot as plt
from compute.extremes.extremes_finder import ExtremesFinder
from compute.extremes.interpolator import Interpolator
from compute.backend import ComputeBackend
from compute.wavelets.gpu_processor import GPUWaveletProcessor

backend = ComputeBackend()
gpu_processor = GPUWaveletProcessor()
backend_info = backend.get_backend_info()
print(f"Вычислительный бэкенд: {backend_info['device_name']}")

file = open(r"C:\Users\bavyk\Downloads\ВП_20_03_2026_00_18\Задача 1 20_03_2026_00_19\Scale_40\Расчет_вейвлетов_построчно_Масштаб_40_Красный.txt").readlines()
coords = []
for row in file:
    coords.append([np.float32(x) for x in row.split(',')])

coefs_2d = np.array(coords)
coefs_2d = np.round(coefs_2d, decimals=3)

coefs_2d, pmaxr, pmaxc, pminr, pminc = ExtremesFinder.find_extremes_gpu(
        coefs_2d,
        row_var=True,
        col_var=True,
        max_var=True,
        min_var=True)

interpolator = Interpolator(gpu_backend=backend)
upper_max_row_points, lower_min_row_points = interpolator.get_envelopes(
        coefs_2d,
        pmaxc,
        pminc,
        direction='row')

points = np.array(upper_max_row_points)
x = points[:, 0]
y = points[:, 1]

plt.figure(figsize=(12, 8))
plt.scatter(x, y, s=4)
plt.gca().invert_yaxis()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("upper_max_row_points")
plt.grid(True)
plt.show()

# формируем список точек для KNN
knn_extremes = [(int(px), int(py)) for px, py in zip(x, y)]
from compute.knn.knn_cpu import process_extremes_with_knn

# куда сохранять результаты
scale_folder = r"C:\Users\bavyk\Downloads\knn_results"

# количество соседей
knn_var = 5

# сохранять ли текст / изображение
knn_bool_text_var = True
knn_bool_image_var = False

extreme_dict = {
    'scale': 40,              # масштаб
    'type_data': 0,           # 0 = Str (построчно)
    'channel': 0,             # 0 = Red
    'max_by_row': upper_max_row_points,
    'max_by_column': [],
    'min_by_row': lower_min_row_points,
    'min_by_column': []
}

use_gpu_knn = True
from compute.knn.knn_cpu import process_extremes_with_knn

process_extremes_with_knn(
    extreme_dict=extreme_dict,
    scale_folder_path=scale_folder,
    k=5,
    original_image=None,
    print_text_var=True,
    print_image_var=False,
    use_gpu=True,
    progress_callback=None,
    log_callback=print)