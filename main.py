import datetime
import os
import time
import math
import tkinter as tk
from multiprocessing import Pool, freeze_support
from tkinter import filedialog
from tkinter import messagebox as mb
import customtkinter as ctk
import cv2
import matplotlib.pyplot as plt
import numpy as np
from numba import jit, prange

import points
import interpol
from Gram_Shmidt import change_channels
from image_cropper_app import run_cropper
from pipette import run_pipette


@jit(nopython=True)
def morlet_wavelet_single_scale(data, scale, j):
    w0 = 0.0
    for k in range(len(data)):
        t = (k - j) / scale
        w0 += data[k] * 0.75 * math.exp(-(t * t) / 2) * math.cos(2 * math.pi * t)
    return w0 / math.sqrt(scale)


@jit(nopython=True, parallel=True)
def morlet_wavelet(data, scales):
    coef = np.zeros((len(scales), len(data)))
    for i in prange(len(scales)):
        for j in range(len(data)):
            coef[i,j] = morlet_wavelet_single_scale(data, scales[i], j)
    return coef


@jit(nopython=True)
def apply_gaussian_edge_filter(channel_data, scales, wavelet_lengths):
    """
    Применяет гауссов фильтр только к краевым зонам, сохраняя внутреннюю часть неизменной

    Параметры:
    channel_data - массив вейвлет-коэффициентов (scales, rows, cols)
    scales - список масштабов
    wavelet_lengths - список длин вейвлетов для каждого масштаба
    """
    num_scales = len(scales)
    rows, cols = channel_data.shape[1], channel_data.shape[2]

    for scale_idx in range(num_scales):
        edge_size = wavelet_lengths[scale_idx]

        # Создаем гауссов фильтр (только возрастающую часть)
        x = np.linspace(-3, 0, edge_size)
        gaussian = np.exp(-(x ** 2) / 2)
        gaussian = (gaussian - gaussian[0]) / (gaussian[-1] - gaussian[0])  # Нормализуем от 0 до 1

        # Применяем фильтр к началу каждой строки
        for row in range(rows):
            for col in range(edge_size):
                channel_data[scale_idx, row, col] *= gaussian[col]

        # Применяем фильтр к концу каждой строки (зеркально)
        for row in range(rows):
            for col in range(cols - edge_size, cols):
                idx = cols - col - 1
                channel_data[scale_idx, row, col] *= gaussian[idx]

    return channel_data


def process_row(args):
    row_data, scales, scales_size = args
    return morlet_wavelet(row_data, scales)


def process_channel(data, scales):
    rows = data.shape[0]
    cols = data.shape[1]
    scales_size = len(scales)
    result = np.zeros((rows, scales_size, cols))

    print("start_morlet")

    with Pool() as pool:
        args = [(data[i], scales, scales_size) for i in range(rows)]
        results = pool.map(process_row, args)

    for i, res in enumerate(results):
        result[i] = res

    return result


class ImageProcessor:
    def __init__(self):
        self.image_path = ""
        self.folder_path = ""
        self.original_image = None
        self.data = []
        self.data_copy = []
        self.num_scale = 0
        self.scales = np.array([])
        self.result = []
        self.current_image_path = None
        self.color1 = []
        self.color2 = []

    def load_image(self, master_window=None):
        self.image_path = run_cropper(master_window)
        if self.image_path:
            print('Selected image file converted to:', self.image_path)
            image = cv2.imread(self.image_path)
            self.original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            b, g, r = cv2.split(image)
            self.data = [r, g, b]
            self.data_copy = [channel.copy() for channel in self.data]
            return True
        else:
            print('Image conversion failed')
            return False

    def pipette_channel(self):
        self.color1, self.color2 = run_pipette(master=None, image_path=self.image_path)

    def save_orig_channels_txt(self, print_channels_txt):
        if print_channels_txt:
            colors = ['Red', 'Green', 'Blue']
            for channel in range(len(self.data_copy)):
                filename = f"Исходный_цветовой_канал_{colors[channel]}.txt"
                array_2d = self.data_copy[channel]
                file_path = os.path.join(self.folder_path, filename)
                np.savetxt(file_path, array_2d, fmt='%d', delimiter=",")
                print(f"Saved to file: {file_path}")

    def gram_shmidt_transform(self):
        self.data = change_channels(self.color1, self.color2, self.data)
        print("Использовано преобразование Грамма-Шмидта")

    def load_scales(self, start, end, step):
        self.scales = np.arange(start=start, stop=end + 1, step=step)
        self.num_scale = self.scales.shape[0]

    def load_scales_from_file(self, filename):
        with open(filename, 'r') as file_of_scales:
            for line in file_of_scales:
                numbers = [np.double(x) for x in line.split()]
                self.scales = np.append(self.scales, numbers)
        self.scales = np.array(self.scales)
        self.num_scale = len(self.scales)

    def create_scale_folder(self, scale):
        scale_folder_path = os.path.join(self.folder_path, f"Scale_{scale}")
        os.makedirs(scale_folder_path, exist_ok=True)
        return scale_folder_path

    def wavelets(self, type_data, data_3_channel):
        """ type_data - флажок для транспонирования матриц
        (0 - построчно, 1 - транспонированный по столбцам) """
        t_compute_wavelet_start = time.time()

        if type_data == 0:
            print("Обработка изображения НВП построчно.")
        else:
            print("Обработка изображения НВП по столбцам.")

        num_channels = 3
        num_rows = self.data[0].shape[0]
        num_cols = self.data[0].shape[1]

        print(f"Масштабы: {self.scales}")
        print(f"Кол-во масштабов: {self.num_scale}")
        print(f"Кол-во строк: {num_rows}")
        print(f"Кол-во столбцов: {num_cols}")

        for channel in range(num_channels):
            data_channel = self.data[channel].astype(np.float64)
            if type_data == 1:
                data_channel = np.transpose(data_channel)

            file_mean_path = os.path.join(self.folder_path, f'mean_to_rows_by_channel_{channel}.txt')
            with open(file_mean_path, 'w') as file:
                for row in data_channel:
                    mean = np.mean(row)
                    file.write(str(mean) + "\n")
                    row -= mean

            data_channel_after = process_channel(data_channel, self.scales)

            wavelet_lengths = [int(7 * scale) for scale in self.scales]
            data_channel_after = apply_gaussian_edge_filter(data_channel_after, self.scales, wavelet_lengths)

            print(data_channel_after.shape)
            data_channel_after_transposed = np.transpose(data_channel_after, (1, 0, 2))
            data_3_channel[channel] = data_channel_after_transposed  # [channels, scales, rows, cols]

        print(f"Вейвлет-преобразование завершено за {time.time() - t_compute_wavelet_start:.2f} секунд.")
        return data_3_channel

    def compute_wavelets(self, info_out):
        data_3_channels = np.zeros((3, self.num_scale, self.data[0].shape[0], self.data[0].shape[1]))
        data_3_channels = self.wavelets(0, data_3_channels)
        self.result.append(data_3_channels)
        self.save_print_wavelets(0, info_out)

        data_3_channels_tr = np.zeros((3, self.num_scale, self.data[0].shape[1], self.data[0].shape[0]))
        data_3_channels_tr = self.wavelets(1, data_3_channels_tr)
        data_3_channels_tr = np.transpose(data_3_channels_tr, (0, 1, 3, 2))
        self.result.append(data_3_channels_tr)
        self.save_print_wavelets(1, info_out)

        for i, result in enumerate(self.result):
            print(f"Размерность элемента {i} в self.result: {result.shape}")

    def save_print_wavelets(self, type_data, info_out):
        colors = ['Красный', 'Зелёный', 'Синий']
        if type_data == 0:
            type_matrix = 0
            type_matrix_str = "построчно"
        else:
            type_matrix = 1
            type_matrix_str = "по_столбцам"

        for channel in range(3):
            for scale in range(self.num_scale):
                scale_folder_path = self.create_scale_folder(self.scales[scale])
                array_2d = self.result[type_matrix][channel][scale]  # Результаты для текущего масштаба
                if info_out == 0 or info_out == 10:  # Сохранение текстовых файлов при 0 и 10
                    filename = f"Расчет_вейвлетов_{type_matrix_str}_Масштаб_{self.scales[scale]}_{colors[channel]}.txt"
                    file_path = os.path.join(scale_folder_path, filename)
                    np.savetxt(file_path, array_2d, fmt='%.3f', delimiter=",")
                    print(f"Сохранено в файл: {file_path}")

                if info_out == 0 or info_out == 1:  # Сохранение графиков при 0 и 1
                    plt.figure()
                    plt.imshow(array_2d, cmap='viridis')
                    plt.title(f'Wavelets: Scale = {self.scales[scale]}, Channel = {colors[channel]}')
                    plt.colorbar()
                    plt.savefig(os.path.join(scale_folder_path,
                                f'График_расчетов_В_П_{type_matrix_str}_Масштаб_{self.scales[scale]}_{colors[channel]}.png'),
                                dpi=300, bbox_inches='tight')
                    plt.close()

    def create_downloads_folder(self, folder_name):
        downloads_path = os.path.join(os.path.expanduser('~'), 'Downloads')
        self.folder_path = os.path.join(downloads_path, folder_name)
        os.makedirs(self.folder_path, exist_ok=True)
        print(f"Папка '{folder_name}' создана в папке 'Загрузки'.")
        print(f"Путь к папке: {self.folder_path}")

    @staticmethod
    def delete_edge_points(coefs_2d, scale, type_wavelet):
        wavelet_length = 7 * scale
        rows, cols = coefs_2d.shape

        mask = np.ones((rows, cols), dtype=np.float32)
        if type_wavelet == 0:
            # обнуление краевых зон по горизонтали (столбцы слева и справа)
            if int(wavelet_length) < cols // 2:
                mask[:, :int(wavelet_length)] = 0  # Левый край
                mask[:, cols - int(wavelet_length):] = 0  # Правый край

        # обнуление краевых зон по вертикали (строки сверху и снизу)
        if type_wavelet == 1:
            if int(wavelet_length) < rows // 2:
                mask[:int(wavelet_length), :] = 0  # Верхний край
                mask[rows - int(wavelet_length):, :] = 0  # Нижний край

        return coefs_2d * mask  # Применяем маску

    def find_extremes(self, coefs, scale, row_var, col_var, max_var, min_var, type_wavelet):
        # удаляем коэффициенты с краевыми эффектами
        # coefs = self.delete_edge_points(np.array(coefs, dtype=np.float32), scale, type_wavelet)

        points_max_by_row = []
        points_min_by_row = []
        points_max_by_column = []
        points_min_by_column = []

        # Экстремумы построчно
        if row_var and (max_var or min_var):
            left = coefs[:, :-2]
            center = coefs[:, 1:-1]
            right = coefs[:, 2:]

            if max_var:
                max_mask = (center > left) & (center > right)
                max_coords = np.where(max_mask)
                points_max_by_row = [[x + 1, y] for y, x in zip(max_coords[0], max_coords[1])]

            if min_var:
                min_mask = (center < left) & (center < right)
                min_coords = np.where(min_mask)
                points_min_by_row = [[x + 1, y] for y, x in zip(min_coords[0], min_coords[1])]

        # Экстремумы по столбцам
        if col_var and (max_var or min_var):
            up = coefs[:-2, :]
            center = coefs[1:-1, :]
            down = coefs[2:, :]

            if max_var:
                max_mask = (center > up) & (center > down)
                max_coords = np.where(max_mask)
                points_max_by_column = [[x, y + 1] for y, x in zip(max_coords[0], max_coords[1])]

            if min_var:
                min_mask = (center < up) & (center < down)
                min_coords = np.where(min_mask)
                points_min_by_column = [[x, y + 1] for y, x in zip(min_coords[0], min_coords[1])]

        return coefs, points_max_by_row, points_max_by_column, points_min_by_row, points_min_by_column


    def compute_points(self, row_var, col_var, max_var, min_var,
                       knn_var, knn_bool_text_var, knn_bool_image_var, print_text_var, print_graphic, pipette_state):
        extremes = []
        for type_data in range(2):
            channels_to_process = [0] if pipette_state == 'normal' else range(3)
            for channel in channels_to_process:
                for scale in range(self.num_scale):
                    coefs_2d = self.result[type_data][channel][scale]
                    coefs_2d = np.round(coefs_2d, decimals=3)
                    coefs_2d, pmaxr, pmaxc, pminr, pminc = self.find_extremes(
                        coefs_2d, self.scales[scale],
                        row_var, col_var, max_var, min_var, type_data)

                    colors = ['Красный', 'Зелёный', 'Синий']
                    type_matrix_str = "Str" if type_data == 0 else "Tr"

                    upper_max_row_points, lower_min_row_points = interpol.get_row_envelopes(coefs_2d, pmaxr, pminr)
                    upper_max_col_points, lower_min_col_points = interpol.get_column_envelopes(coefs_2d, pmaxc, pminc)

                    # массив для выгрузки массивов точек экстремумов
                    extremes_to_process = []
                    # массив названий заголовков файлов
                    titles = []

                    if max_var:
                        if row_var:
                            # extremes_to_process.append(pmaxr)
                            extremes_to_process.append(upper_max_row_points)
                            titles.append(
                                f"{type_matrix_str}_Точки_максимума_по_строкам_масштаб_{self.scales[scale]}_{colors[channel]}")
                        if col_var:
                            # extremes_to_process.append(pmaxc)
                            extremes_to_process.append(upper_max_col_points)
                            titles.append(
                                f"{type_matrix_str}_Точки_максимума_по_cтолбцам_масштаб_{self.scales[scale]}_{colors[channel]}")
                    if min_var:
                        if row_var:
                            # extremes_to_process.append(pminr)
                            extremes_to_process.append(lower_min_row_points)
                            titles.append(
                                f"{type_matrix_str}_Точки_минимума_по_строкам_масштаб_{self.scales[scale]}_{colors[channel]}")
                        if col_var:
                            # extremes_to_process.append(pminc)
                            extremes_to_process.append(lower_min_col_points)
                            titles.append(
                                f"{type_matrix_str}_Точки_минимума_по_cтолбцам_масштаб_{self.scales[scale]}_{colors[channel]}")

                    scale_folder = self.find_scale_folder(self.scales[scale])
                    for i, p in enumerate(extremes_to_process):
                        if len(p) > 0:
                            if print_text_var:
                                self.save_extremes_to_file(scale_folder, titles[i], p)
                            if print_graphic:
                                self.graphic(scale_folder, titles[i], p, coefs_2d.shape)

                    # словарь с отфильтрованными экстремумами
                    knn_extremes = {
                        'type_data': type_data,
                        'channel': channel,
                        'scale': self.scales[scale],
                        'max_by_row': upper_max_row_points if (row_var and max_var) else [],
                        'max_by_column': upper_max_col_points if (col_var and max_var) else [],
                        'min_by_row': lower_min_row_points if (row_var and min_var) else [],
                        'min_by_column': lower_min_col_points if (col_var and min_var) else []
                    }
                    extremes.append(knn_extremes)

                    if knn_bool_text_var or knn_bool_image_var:
                        points.process_extremes_with_knn(knn_extremes, scale_folder, knn_var,
                                                         self.original_image, knn_bool_text_var, knn_bool_image_var)

        return extremes

    @staticmethod
    def save_extremes_to_file(path, title, local_points):
        if not local_points:
            print(f"Нет точек для сохранения в {title}")
            return

        file_path = os.path.join(path, f"{title}.txt")
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                for point in local_points:
                    file.write(f"{point[0]}, {point[1]}\n")
            print(f"Файл сохранён: {file_path}")
        except Exception as e:
            print(f"Ошибка при сохранении файла {file_path}: {str(e)}")

    @staticmethod
    def graphic(path, title, points_local, original_img_shape):
        if not points_local:
            print(f"Нет точек для отображения: {title}")
            return

        plt.figure(figsize=(10, 10))

        data = np.array(points_local)
        x = data[:, 0]
        y = data[:, 1]

        # оси с сохранением пропорций
        ax = plt.gca()

        if original_img_shape is not None:
            height, width = original_img_shape[:2]
            ax.set_xlim(0, width)
            ax.set_ylim(height, 0)  # инвертируем ось Y
            ax.set_aspect('equal')  # фиксируем соотношение сторон 1:1

        # рисуем точки
        plt.scatter(x, y, s=1, alpha=0.6)
        plt.title(title)

        plt.grid(True)
        plt.xlabel('X (пиксели)')
        plt.ylabel('Y (пиксели)')

        filename = os.path.join(path, f"{title}.png")
        plt.savefig(filename, bbox_inches='tight', dpi=96)
        plt.close()
        print(f"График сохранён: {filename}")

    def find_scale_folder(self, scale):
        scale_folder_name = f"Scale_{scale}"
        scale_folder_path = os.path.join(self.folder_path, scale_folder_name)
        if os.path.exists(scale_folder_path) and os.path.isdir(scale_folder_path):
            return scale_folder_path + "\\"
        else:
            print(f"Directory {scale_folder_path} is not found")
            return None

    def compute(self, wp_var1, wp_var2, print_channels_txt_var, row_var, col_var, max_var, min_var,
                p_ex_var1, p_ex_var2, k_neighbors, knn_bool_text_var, knn_bool_image_var,  pipette_state):
        current_date = datetime.datetime.now()
        date_str = current_date.strftime("%d_%m_%Y_%H_%M_%S")
        folder_name = f"Вейвлет_преобразования_{date_str}"
        self.create_downloads_folder(folder_name)
        self.save_orig_channels_txt(print_channels_txt_var)
        if (wp_var1.get() is True) and (wp_var2.get() is True):
            self.compute_wavelets(0)
        if (wp_var1.get() is True) and (wp_var2.get() is False):
            self.compute_wavelets(1)
        if (wp_var1.get() is False) and (wp_var2.get() is True):
            self.compute_wavelets(10)
        if (wp_var1.get() is False) and (wp_var2.get() is False):
            self.compute_wavelets(11)

        if p_ex_var1.get() or p_ex_var2.get():
            self.compute_points(row_var, col_var, max_var, min_var,
                                k_neighbors, knn_bool_text_var.get(), knn_bool_image_var.get(), p_ex_var1.get(), p_ex_var2.get(), pipette_state)


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("1000x700")
        self.title("Wavelets")
        self.resizable(False, False)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.image_processor = ImageProcessor()

        self.label_load = ctk.CTkLabel(self, text="Загрузить и изменить изображение")
        self.label_load.grid(row=0, column=0, padx=20, sticky="w")
        self.label_load.configure(font=ctk.CTkFont(size=14, weight="bold"))
        self.load_button = ctk.CTkButton(self, text="Загрузить",
                                         command=self.load_image_callback)
        self.load_button.configure(font=ctk.CTkFont(size=18), width=200, height=50)
        self.load_button.grid(row=1, column=0, padx=20, pady=10, sticky="w")
        self.print_load_image = ctk.CTkLabel(self, text="")
        self.print_load_image.grid(row=2, column=0, padx=20, pady=10, sticky="w")

        self.label_col_channel = ctk.CTkLabel(self, text="Выбрать канал изображения")
        self.label_col_channel.grid(row=3, column=0, padx=20, pady=10, sticky="w")
        self.label_col_channel.configure(font=ctk.CTkFont(size=14, weight="bold"))

        self.pipette_button = ctk.CTkButton(self, text="Пипетка", command=self.pipette_channel)
        self.pipette_button.grid(row=4, column=0, padx=20, sticky="w")
        self.data = tk.StringVar()

        self.gram_shmidt_label = ctk.CTkLabel(self, text="Преобразование Грамма-Шмидта")
        self.gram_shmidt_label.grid(row=5, column=0, padx=20, pady=10, sticky="w")
        self.gram_shmidt_label.configure(font=ctk.CTkFont(size=14, weight="bold"))

        self.gram_shmidt_button = ctk.CTkButton(self, text='Выполнить', command=self.gramm_shmidt_transform)
        self.gram_shmidt_button.grid(row=5, column=1, padx=20, sticky="w")

        self.label_scales = ctk.CTkLabel(self, text="Масштабы")
        self.label_scales.configure(font=ctk.CTkFont(size=14, weight="bold"))
        self.label_scales.grid(row=6, column=0, sticky="w", padx=20, pady=10)
        self.label_start = ctk.CTkLabel(self, text="От:")
        self.label_start.grid(row=7, column=0, padx=20, sticky="w")
        self.entry_start = ctk.CTkEntry(self)
        self.entry_start.grid(row=7, column=0, padx=45, sticky="w")
        self.label_end = ctk.CTkLabel(self, text="До:")
        self.label_end.grid(row=8, column=0, padx=20, sticky="w")
        self.entry_end = ctk.CTkEntry(self)
        self.entry_end.grid(row=8, column=0, padx=45, sticky="w")
        self.label_step = ctk.CTkLabel(self, text="Шаг:")
        self.label_step.grid(row=9, column=0, padx=20, sticky="w")
        self.entry_step = ctk.CTkEntry(self)
        self.entry_step.grid(row=9, column=0, padx=45, sticky="w")
        self.button_save_scales = ctk.CTkButton(self, text="Сохранить значения", command=self.load_scales)
        self.button_save_scales.grid(row=10, column=0, padx=45, sticky="w")

        self.label_custom_scale = ctk.CTkLabel(self, text="")
        self.label_custom_scale.grid(row=11, column=0, sticky="w")
        self.button_load_scales_file = ctk.CTkButton(self, text="Загрузить из файла",
                                                     command=self.load_scales_from_file)
        self.button_load_scales_file.grid(row=12, column=0, padx=45, sticky="w")

        self.label_t_extr = ctk.CTkLabel(self, text="Точки экстремумов:")
        self.label_t_extr.grid(row=13, column=0, padx=20, pady=10, sticky="w")
        self.label_t_extr.configure(font=ctk.CTkFont(size=14, weight="bold"))

        self.row_var = tk.BooleanVar(value=False)
        self.col_var = tk.BooleanVar(value=False)
        self.max_var = tk.BooleanVar(value=False)
        self.min_var = tk.BooleanVar(value=False)

        self.row_checkbox = ctk.CTkCheckBox(self, text="По строкам", variable=self.row_var, onvalue=True,
                                            offvalue=False)
        self.row_checkbox.grid(row=16, column=0, padx=20, sticky="w")
        self.col_checkbox = ctk.CTkCheckBox(self, text="По столбцам", variable=self.col_var, onvalue=True,
                                            offvalue=False)
        self.col_checkbox.grid(row=17, column=0, padx=20, sticky="w")
        self.max_checkbox = ctk.CTkCheckBox(self, text="Максимум", variable=self.max_var, onvalue=True, offvalue=False)
        self.max_checkbox.grid(row=16, column=1, padx=20, sticky="w")
        self.min_checkbox = ctk.CTkCheckBox(self, text="Минимум", variable=self.min_var, onvalue=True, offvalue=False)
        self.min_checkbox.grid(row=17, column=1, padx=20, sticky="w")

        self.num_near_point = ctk.CTkLabel(self, text="Количество ближайших точек")
        self.num_near_point.configure(font=ctk.CTkFont(size=14, weight="bold"))
        self.num_near_point.grid(row=19, column=0, padx=20, pady=10, sticky="w")
        self.knn_text_var = tk.StringVar()
        self.knn_text_var.set("n (ex. 5)")
        self.entry_near_point = ctk.CTkEntry(self, textvariable=self.knn_text_var)
        self.entry_near_point.grid(row=19, column=1, sticky="w")
        self.entry_near_point.configure(text_color="gray")
        self.entry_near_point.bind("<Button-1>", self.on_entry_click)

        # output gui
        self.output_label = ctk.CTkLabel(self, text="Вывод вычислений")
        self.output_label.configure(font=ctk.CTkFont(size=16, weight="bold"))
        self.output_label.grid(row=0, column=3, padx=20, sticky="w")

        self.wp_var1 = tk.BooleanVar(value=False)
        self.wp_var2 = tk.BooleanVar(value=False)
        self.p_ex_var1 = tk.BooleanVar(value=False)
        self.p_ex_var2 = tk.BooleanVar(value=False)
        self.knn_bool_text_var = tk.BooleanVar(value=False)
        self.knn_bool_image_var = tk.BooleanVar(value=False)

        self.wp_label = ctk.CTkLabel(self, text="Вейвлет преобразование (Морле)")
        self.wp_label.grid(row=1, column=3, padx=20, sticky="w")
        self.wp_label.configure(font=ctk.CTkFont(size=14, weight="bold"))
        self.wp1_checkbox = ctk.CTkCheckBox(self, text="Вывести изображением",
                                            variable=self.wp_var1)
        self.wp1_checkbox.grid(row=2, column=3, padx=40, sticky="w")
        self.wp2_checkbox = ctk.CTkCheckBox(self, text="Вывести текстовым файлом",
                                            variable=self.wp_var2)
        self.wp2_checkbox.grid(row=3, column=3, padx=40, sticky="w")

        self.p_ex_label = ctk.CTkLabel(self, text="Точки экстремума:")
        self.p_ex_label.configure(font=ctk.CTkFont(size=14, weight="bold"))
        self.p_ex_label.grid(row=5, column=3, padx=20, sticky="w")
        self.p_ex1_checkbox = ctk.CTkCheckBox(self, text="Вывести текстовым файлом",
                                              variable=self.p_ex_var1)
        self.p_ex1_checkbox.grid(row=6, column=3, padx=40, sticky="w")
        self.p_ex2_checkbox = ctk.CTkCheckBox(self, text="Вывести изображением",
                                              variable=self.p_ex_var2)
        self.p_ex2_checkbox.grid(row=7, column=3, padx=40, sticky="w")

        self.knn_label = ctk.CTkLabel(self, text="Расчет k-ближайших соседей точек экстремумов")
        self.knn_label.configure(font=ctk.CTkFont(size=14, weight="bold"))
        self.knn_label.grid(row=8, column=3, padx=20, sticky="w")
        self.knn_text_checkbox = ctk.CTkCheckBox(self, text="Вывести текстовым файлом",
                                                   variable=self.knn_bool_text_var)
        self.knn_text_checkbox.grid(row=9, column=3, padx=40, sticky="w")
        self.knn_image_checkbox = ctk.CTkCheckBox(self, text="Вывести изображением",
                                                   variable=self.knn_bool_image_var)
        self.knn_image_checkbox.grid(row=10, column=3, padx=40, sticky="w")



        self.image_channel_label = ctk.CTkLabel(self, text="Промежуточные вычисления")
        self.image_channel_label.configure(font=ctk.CTkFont(size=14, weight="bold"))
        self.image_channel_label.grid(row=11, column=3, padx=20, sticky="w")
        self.print_channels_txt_var = tk.BooleanVar(value=False)
        self.print_channels_txt_checkbox = ctk.CTkCheckBox(self,
                                                           text="Исходные матрицы RGB текстовым файлом",
                                                           variable=self.print_channels_txt_var)
        self.print_channels_txt_checkbox.grid(row=12, column=3, padx=40, sticky="w")

        self.app_start_button = ctk.CTkButton(self, text="Вычислить", command=self.compute)
        self.app_start_button.grid(row=16, column=3, sticky="s")
        self.app_start_button.configure(width=200, height=70, border_width=4, border_color="black", font=ctk.CTkFont(size=20))

    def load_image_callback(self):
        if self.image_processor.load_image(self):
            self.load_button.configure(text="Загружено", text_color="black", fg_color="white", border_color="black",
                                       border_width=2)
            text = "Выбранный файл:\n" + self.image_processor.image_path
            self.print_load_image.configure(text=text)
        else:
            self.load_button.configure(text="Ошибка", fg_color="red")
            self.print_load_image.configure(text="No file selected")

    def pipette_channel(self):
        self.image_processor.pipette_channel()
        self.pipette_button.configure(state='disabled', text_color="black", fg_color="white", border_color="black",
                                      border_width=2)

    def gramm_shmidt_transform(self):
        self.image_processor.gram_shmidt_transform()
        self.gram_shmidt_button.configure(state='disabled', text_color="black", fg_color="white", border_color="black",
                                          border_width=2)

    def load_scales(self):
        start = int(self.entry_start.get())
        end = int(self.entry_end.get())
        step = int(self.entry_step.get())
        self.image_processor.load_scales(start, end, step)
        self.button_save_scales.configure(text="Сохранено", text_color="black", fg_color="white", border_color="black",
                                          border_width=2)
        self.button_load_scales_file.configure(state='disabled')

    def load_scales_from_file(self):
        self.entry_start.configure(state='disabled')
        self.entry_step.configure(state='disabled')
        self.entry_end.configure(state='disabled')
        filetypes = (
            ('Text files', '*.txt'),
            ('All files', '*.*')
        )
        filename = tk.filedialog.askopenfilename(
            title='Open an image file',
            initialdir='/',
            filetypes=filetypes)
        self.image_processor.load_scales_from_file(filename)

        if self.image_processor.num_scale <= 7:
            self.label_custom_scale.configure(text=str(self.image_processor.scales))

        self.button_load_scales_file.configure(text="Загружено", text_color="black", fg_color="white",
                                               border_color="black", border_width=2)
        self.button_save_scales.configure(text="Сохранено", text_color="black", fg_color="white", border_color="black",
                                          border_width=2)

    def on_entry_click(self, event=None):
        self.entry_near_point.delete(0, ctk.END)
        self.entry_near_point.configure(text_color="black")

    def compute(self):
        timer = time.time()
        # сбор параметров для передачи в метод compute()
        wp_var1 = self.wp_var1
        wp_var2 = self.wp_var2
        print_channels_txt_var = self.print_channels_txt_var.get()
        p_ex_var1 = self.p_ex_var1
        p_ex_var2 = self.p_ex_var2
        k = int(self.knn_text_var.get()) if self.knn_text_var.get().isdigit() else 5
        pipette_button_state = self.pipette_button.cget('state')

        try:
            self.image_processor.compute(wp_var1, wp_var2, print_channels_txt_var,
                                         self.row_var.get(), self.col_var.get(),
                                         self.max_var.get(), self.min_var.get(),
                                         p_ex_var1, p_ex_var2, k_neighbors=k,
                                         knn_bool_text_var=self.knn_bool_text_var,
                                         knn_bool_image_var=self.knn_bool_image_var,
                                         pipette_state=pipette_button_state)

            msg_box = tk.Toplevel()
            msg_box.title("Вычисления завершены")
            msg_box.geometry("500x200")
            msg_box.resizable(True, True)

            # центрируем окно относительно главного окна
            x = self.winfo_x() + (self.winfo_width() // 2) - 200
            y = self.winfo_y() + (self.winfo_height() // 2) - 75
            msg_box.geometry(f"+{x}+{y}")

            # Текст сообщения
            label = ctk.CTkLabel(msg_box,
                                 text=f"Вычисления выполнены успешно."
                                      f"\nВсе файлы сохранены в папку:"
                                      f"\n{self.image_processor.folder_path}"
                                      f"\nПотрачено времени: {format_time(time.time()-timer)}"
                                 )
            label.pack(pady=10)

            button_frame = ctk.CTkFrame(msg_box)
            button_frame.pack(pady=10)

            restart_btn = ctk.CTkButton(button_frame, text="Перезапустить",
                                        command=lambda: [msg_box.destroy(), self.restart_app()])
            restart_btn.pack(side=tk.LEFT, padx=10)

            close_btn = ctk.CTkButton(button_frame, text="Закрыть",
                                      command=lambda: [msg_box.destroy(), self.destroy()])
            close_btn.pack(side=tk.LEFT, padx=10)

        except Exception as e:
            mb.showerror("Ошибка", f"Произошла ошибка:\n{str(e)}")

    def restart_app(self):
        self.destroy()
        new_app = App()
        new_app.mainloop()

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours} ч {minutes:02d} м {seconds:02d} с"

if __name__ == '__main__':
    freeze_support()  # for multiprocess
    app = App()
    app.mainloop()