import datetime
import math
import os
import time
import tkinter as tk
from multiprocessing import Pool, freeze_support
from tkinter import filedialog
from tkinter import messagebox as mb

import customtkinter as ctk
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from numba import jit

import points
from Gram_Shmidt import change_channels
from image_cropper_app import run_cropper
from pipette import pipette


# Morlet wavelet transform functions
@jit(nopython=True)
def morlet_wavelet_single_scale(data, data_size, scale, j):
    w0 = 0
    for k in range(data_size):
        t = (k - j) / scale
        w0 += data[k] * 0.75 * math.exp(-(t * t) / 2) * math.cos(2 * math.pi * t)
    return w0 / math.sqrt(scale)


def morlet_wavelet(data, data_size, scales, weight_size):
    coef = np.zeros((weight_size, data_size))
    for i in range(weight_size):
        for j in range(data_size):
            coef[i][j] = morlet_wavelet_single_scale(data, data_size, scales[i], j)
    return coef


def process_row(args):
    row_data, scales, scales_size = args
    return morlet_wavelet(row_data, len(row_data), scales, scales_size)


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

    @staticmethod
    def convert_to_png(image_file_path):
        png_file_path = os.path.splitext(image_file_path)[0] + '.png'
        try:
            img = Image.open(image_file_path)
            img_converted = img.convert("RGB")
            img_converted.save(png_file_path, "PNG")
            print(f"Image saved as {png_file_path}")
            return png_file_path
        except Exception as e:
            print(f"Error: {e}")
            return None

    def load_image(self, master_window=None):  # Добавьте параметр master_window
        filename = run_cropper(master_window)  # Передаем корневое окно
        if filename:
            self.image_path = self.convert_to_png(filename)
            if self.image_path:
                print('Selected image file converted to:', self.image_path)
                # split image channels to rgb
                image = cv2.imread(self.image_path)
                self.original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                b, g, r = cv2.split(image)
                self.data = [r, g, b]
                self.data_copy = self.data
                return True
            else:
                print('Image conversion failed')
                return False
        else:
            print('No file selected')
            return False

    def pipette_channel(self):
        pipette(self.image_path)

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
        with open("color_tone.txt", 'r') as file:
            content = file.readlines()
            color1 = np.array(content[0].strip('\n').split(' '), dtype=np.int16)
            color2 = np.array(content[1].strip('\n').split(' '), dtype=np.int16)

        # color1 = np.array([54, 28, 99], dtype=np.int16)  # blue
        # color2 = np.array([150, 122, 147], dtype=np.int16)  # pink
        self.data = change_channels(color1, color2, self.data)
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
                    np.savetxt(file_path, array_2d, fmt='%.15f', delimiter=",")
                    print(f"Сохранено в файл: {file_path}")

                if info_out == 0 or info_out == 1:  # Сохранение графиков при 0 и 1
                    plt.figure()
                    plt.imshow(array_2d, cmap='grey')
                    plt.title(f'Wavelets: Scale = {self.scales[scale]}, Channel = {colors[channel]}')
                    plt.colorbar()
                    plt.savefig(os.path.join(scale_folder_path,
                                             f'График_расчетов_В_П_{type_matrix_str}_Масштаб_{self.scales[scale]}_{colors[channel]}.png'))
                    plt.close()

    def create_downloads_folder(self, folder_name):
        downloads_path = os.path.join(os.path.expanduser('~'), 'Downloads')
        self.folder_path = os.path.join(downloads_path, folder_name)
        os.makedirs(self.folder_path, exist_ok=True)
        print(f"Папка '{folder_name}' создана в папке 'Загрузки'.")
        print(f"Путь к папке: {self.folder_path}")

    @staticmethod
    def find_extremes(coefs, row_var, col_var, max_var, min_var):
        points_max_by_row = []
        points_min_by_row = []
        points_max_by_column = []
        points_min_by_column = []

        if row_var:
            for i in range(len(coefs)):
                row = coefs[i]

                if max_var:
                    max_indices = np.where(row == row.max())
                    max_indices = [max_indices[0][0], i]
                    points_max_by_row.append(max_indices)

                if min_var:
                    min_indices = np.where(row == row.min())
                    min_indices = [min_indices[0][0], i]
                    points_min_by_row.append(min_indices)

        # Обработка столбцов
        if col_var:
            for i in range(len(coefs[0])):
                column = coefs[:, i]

                if max_var:
                    max_indices = np.where(column == column.max())
                    max_indices = [i, max_indices[0][0]]
                    points_max_by_column.append(max_indices)

                if min_var:
                    min_indices = np.where(column == column.min())
                    min_indices = [i, min_indices[0][0]]
                    points_min_by_column.append(min_indices)
        return points_max_by_row, points_max_by_column, points_min_by_row, points_min_by_column


    def compute_points(self, row_var, col_var, max_var, min_var, knn_var):
        extremes = []
        for type_data in range(2):
            for channel in range(3):
                for scale in range(self.num_scale):
                    pmaxr, pmaxc, pminr, pminc = self.find_extremes(
                        self.result[type_data][channel][scale], row_var, col_var, max_var, min_var)
                    small_extremes = {
                        'type_data': type_data,
                        'channel': channel,
                        'scale': self.scales[scale],
                        'max_by_row': pmaxr,
                        'max_by_column': pmaxc,
                        'min_by_row': pminr,
                        'min_by_column': pminc
                    }

                    extremes.append(small_extremes)
                    colors = ['Красный', 'Зелёный', 'Синий']
                    if type_data == 0:
                        type_matrix_str = "Str_"
                    else:
                        type_matrix_str = "Tr_"

                    titles = [f"{type_matrix_str}_Точки_максимума_по_строкам_масштаб_{self.scales[scale]}_{colors[channel]}",
                              f"{type_matrix_str}_Точки_максимума_по_cтолбцам_масштаб_{self.scales[scale]}_{colors[channel]}",
                              f"{type_matrix_str}_Точки_минимума_по_строкам_масштаб_{self.scales[scale]}_{colors[channel]}",
                              f"{type_matrix_str}_Точки_минимума_по_cтолбцам_масштаб_{self.scales[scale]}_{colors[channel]}"]

                    i = 0
                    scale_folder = self.find_scale_folder(self.scales[scale])
                    for p in [pmaxr, pmaxc, pminr, pminc]:
                        self.graphic(scale_folder, titles[i], p)
                        i += 1
                    points.process_extremes_with_knn(small_extremes, scale_folder, knn_var, self.original_image)
        return extremes

    @staticmethod
    def graphic(path, title, points_local):
        data = np.array(points_local)
        x = data[:, 0]
        y = data[:, 1]
        plt.figure(figsize=(7, 8))
        plt.title(title)
        plt.scatter(x, y, s=1)
        plt.gca().invert_yaxis()
        name = path + title + '.png'
        plt.savefig(name)
        print(f"File {name} saved.")
        plt.close()
        return

    def find_scale_folder(self, scale):
        scale_folder_name = f"Scale_{scale}"
        scale_folder_path = os.path.join(self.folder_path, scale_folder_name)
        if os.path.exists(scale_folder_path) and os.path.isdir(scale_folder_path):
            return scale_folder_path + "\\"
        else:
            print(f"Directory {scale_folder_path} is not found")
            return None

    def compute(self, wp_var1, wp_var2, print_channels_txt_var, row_var, col_var, max_var, min_var, p_ex_var1, p_ex_var2, k_neighbors=5):
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

        if (p_ex_var1.get() is True) and (p_ex_var2.get() is True):
            self.compute_points(row_var, col_var, max_var, min_var, k_neighbors)


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("1000x800")
        self.title("Wavelets and other")
        self.resizable(False, False)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.image_processor = ImageProcessor()

        self.label_load = ctk.CTkLabel(self, text="Загрузить и изменить изображение")
        self.label_load.grid(row=0, column=0, padx=20, sticky="w")
        self.label_load.configure(font=ctk.CTkFont(size=14, weight="bold"))
        self.load_button = ctk.CTkButton(self, text="Загрузить",
                                         command=self.load_image_callback)
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
        self.label_scales.configure(font=ctk.CTkFont(size=16, weight="bold"))
        self.label_scales.grid(row=6, column=0, sticky="w", padx=15, pady=10)
        self.label_start = ctk.CTkLabel(self, text="От:")
        self.label_start.grid(row=7, column=0, padx=15, sticky="w")
        self.entry_start = ctk.CTkEntry(self)
        self.entry_start.grid(row=7, column=0, padx=45, sticky="w")
        self.label_end = ctk.CTkLabel(self, text="До:")
        self.label_end.grid(row=8, column=0, padx=15, sticky="w")
        self.entry_end = ctk.CTkEntry(self)
        self.entry_end.grid(row=8, column=0, padx=45, sticky="w")
        self.label_step = ctk.CTkLabel(self, text="Шаг:")
        self.label_step.grid(row=9, column=0, padx=15, sticky="w")
        self.entry_step = ctk.CTkEntry(self)
        self.entry_step.grid(row=9, column=0, padx=45, sticky="w")
        self.button_save_scales = ctk.CTkButton(self, text="Сохранить значения", command=self.load_scales)
        self.button_save_scales.grid(row=10, column=0, padx=45, sticky="w")

        self.label_custom_scale = ctk.CTkLabel(self, text="")
        self.label_custom_scale.grid(row=11, column=0, sticky="w")
        self.button_load_scales_file = ctk.CTkButton(self, text="Загрузить из файла",
                                                     command=self.load_scales_from_file)
        self.button_load_scales_file.grid(row=12, column=0, padx=45, pady=10, sticky="w")

        self.label_t_extr = ctk.CTkLabel(self, text="Точки экстремумов:")
        self.label_t_extr.grid(row=13, column=0, padx=20, pady=10, sticky="w")
        self.label_t_extr.configure(font=ctk.CTkFont(size=16, weight="bold"))

        self.row_var = tk.BooleanVar(value=False)
        self.col_var = tk.BooleanVar(value=False)
        self.max_var = tk.BooleanVar(value=False)
        self.min_var = tk.BooleanVar(value=False)
        self.plane_var = tk.BooleanVar(value=False)

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
        self.dist_angle_var = tk.BooleanVar(value=False)

        self.wp_label = ctk.CTkLabel(self, text="Вейвлет преобразование (Морле)")
        self.wp_label.grid(row=1, column=3, padx=20, sticky="w")
        self.wp1_checkbox = ctk.CTkCheckBox(self, text="Вывести изображением",
                                            variable=self.wp_var1)
        self.wp1_checkbox.grid(row=2, column=3, padx=40, sticky="w")
        self.wp2_checkbox = ctk.CTkCheckBox(self, text="Вывести текстовым файлом",
                                            variable=self.wp_var2)
        self.wp2_checkbox.grid(row=3, column=3, padx=40, sticky="w")

        self.p_ex_label = ctk.CTkLabel(self, text="Точки экстремума:")
        self.p_ex_label.grid(row=5, column=3, padx=20, sticky="w")
        self.p_ex1_checkbox = ctk.CTkCheckBox(self, text="Вывести текстовым файлом",
                                              variable=self.p_ex_var1)
        self.p_ex1_checkbox.grid(row=6, column=3, padx=40, sticky="w")
        self.p_ex2_checkbox = ctk.CTkCheckBox(self, text="Вывести изображением",
                                              variable=self.p_ex_var2)
        self.p_ex2_checkbox.grid(row=7, column=3, padx=40, sticky="w")

        self.dist_angle_label = ctk.CTkLabel(self, text="Расчет k-ближайших соседей точек экстремумов")
        self.dist_angle_label.grid(row=8, column=3, padx=20, sticky="w")
        self.dist_angle_checkbox = ctk.CTkCheckBox(self, text="Вывести текстовым файлом",
                                                   variable=self.dist_angle_var)
        self.dist_angle_checkbox.grid(row=9, column=3, padx=40, sticky="w")

        self.image_channel_label = ctk.CTkLabel(self, text="Промежуточные вычисления")
        self.image_channel_label.grid(row=10, column=3, padx=20, sticky="w")
        self.print_channels_txt_var = tk.BooleanVar(value=False)
        self.print_channels_txt_checkbox = ctk.CTkCheckBox(self,
                                                           text="Исходные матрицы каналов rgb виде текстового файла",
                                                           variable=self.print_channels_txt_var)
        self.print_channels_txt_checkbox.grid(row=11, column=3, padx=40, sticky="w")

        self.app_start_button = ctk.CTkButton(self, text="Вычислить", command=self.compute)
        self.app_start_button.grid(row=16, column=3, sticky="s")
        self.app_start_button.configure(width=200, height=50, border_width=3, border_color="black")

    def load_image_callback(self):
        if self.image_processor.load_image(self):  # Передаем self (корневое окно) как master_window
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
        # Сбор параметров для передачи в метод compute()
        wp_var1 = self.wp_var1
        wp_var2 = self.wp_var2
        print_channels_txt_var = self.print_channels_txt_var.get()
        p_ex_var1 = self.p_ex_var1
        p_ex_var2 = self.p_ex_var2
        k = int(self.knn_text_var.get()) if self.knn_text_var.get().isdigit() else 5

        try:
            self.image_processor.compute(wp_var1, wp_var2, print_channels_txt_var,
                                         self.row_var, self.col_var, self.max_var, self.min_var,
                                         p_ex_var1, p_ex_var2, k_neighbors=k)

            msg_box = tk.Toplevel()
            msg_box.title("Вычисления завершены")
            msg_box.geometry("500x500")
            msg_box.resizable(False, False)

            # Центрируем окно относительно главного окна
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
        """Перезапуск приложения"""
        self.destroy()
        # Создаем новое окно приложения
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