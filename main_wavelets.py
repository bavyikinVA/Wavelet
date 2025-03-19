"""
import ctypes
import datetime
import math
import os
import sys
import time
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox as mb
import customtkinter as ctk
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from Gram_Shmidt import change_channels
from nalozhenie_tochek import nalozhenie
from pipette import pipette


class ImageProcessor:
    def __init__(self):
        self.image_path = ""
        self.folder_path = ""
        self.data = []
        self.data_copy = []
        self.num_scale = 0
        self.scales = np.array([], dtype=ctypes.c_double)
        self.result = []
        self.points_max_by_row = []
        self.points_min_by_row = []
        self.points_max_by_column = []
        self.points_min_by_column = []
        self.extremum_row_min_array = []
        self.extremum_row_max_array = []
        self.extremum_col_min_array = []
        self.extremum_col_max_array = []
        self.extremum_plane_max_array = []


    def find_max_points_by_rows(self, coefs):
        for i in range(len(coefs)):
            row = coefs[i]
            max = np.where(row == row.max())
            max = [max[0][0], i]
            self.points_max_by_row.append(max)
        return self.points_max_by_row

    def find_min_points_by_rows(self, coefs):
        for i in range(len(coefs)):
            row = coefs[i]
            min = np.where(row == row.min())
            min = [min[0][0], i]
            self.points_min_by_row.append(min)
        return self.points_min_by_row

    def find_max_points_by_column(self, coefs):
        for i in range(len(coefs[0])):
            column = coefs[:, i]
            max = np.where(column == column.max())
            max = [i, max[0][0]]
            self.points_max_by_column.append(max)
        return self.points_max_by_column

    def find_min_points_by_column(self, coefs):
        for i in range(len(coefs[0])):
            column = coefs[:, i]
            min = np.where(column == column.min())
            min = [i, min[0][0]]
            self.points_min_by_column.append(min)
        return self.points_min_by_column

    def find_max_points_by_plane(self, coefs):
        max_points = []
        for i in range(1, len(coefs) - 1):
            for j in range(1, len(coefs[0]) - 1):
                if (coefs[i, j] >= coefs[i - 1, j] and coefs[i, j] >= coefs[i + 1, j] and
                        coefs[i, j] >= coefs[i, j - 1] and coefs[i, j] >= coefs[i, j + 1]):
                    max_points.append([i, j])
        return max_points

    def find_extremum(self, row_var, col_var, max_var, min_var, p_ex_var1, p_ex_var2, plane_var):
        def process_extremum(chan, sc, matr, title):
            scale_folder_path = self.create_scale_folder(self.scales[sc])
            if p_ex_var1.get() is True:
                self.save_extremum_txt(matr, title, self.scales[sc], scale_folder_path, chan)
            if p_ex_var2.get() is True:
                self.save_extremum_picture(matr, title, self.scales[sc], scale_folder_path, chan)

        for channel in range(0, 3):
            for scale in range(self.num_scale):
                if row_var.get() is True:
                    if max_var.get() is True:
                        matrix = self.find_max_points_by_rows(self.result[channel][scale])
                        self.extremum_row_max_array.append(matrix)
                        process_extremum(channel, scale, matrix, '(максимум_по_строкам)')
                    if min_var.get() is True:
                        matrix = self.find_min_points_by_rows(self.result[channel][scale])
                        self.extremum_row_min_array.append(matrix)
                        process_extremum(channel, scale, matrix, '(минимум_по_строкам)')
                if col_var.get() is True:
                    if max_var.get() is True:
                        matrix = self.find_max_points_by_column(self.result[channel][scale])
                        self.extremum_col_max_array.append(matrix)
                        process_extremum(channel, scale, matrix, '(максимум_по_столбцам)')
                    if min_var.get() is True:
                        matrix = self.find_min_points_by_column(self.result[channel][scale])
                        self.extremum_col_min_array.append(matrix)
                        process_extremum(channel, scale, matrix, '(минимум_по_столбцам)')
                if plane_var.get() is True:
                    if max_var.get() is True:
                        matrix = self.find_max_points_by_plane(self.result[channel][scale])
                        self.extremum_plane_max_array.append(matrix)
                        process_extremum(channel, scale, matrix, '(максимум_по_плоскости)')

    def save_extremum_txt(self, matrix, title, scale, scale_folder_path, channel):
        main_title = 'Точки_экстремумов_' + title + '_Масштаб_' + str(scale)
        matrix_str = '\n'.join(' '.join(map(str, row)) for row in matrix)
        file_path = os.path.join(scale_folder_path, main_title + '.txt')
        import_path = os.path.join(scale_folder_path, main_title)
        with open(file_path, "w") as file:
            file.write(matrix_str)
        nalozhenie(self.image_path, file_path, import_path, channel)


    
    @staticmethod
    def save_extremum_picture(matrix, title, scale, scale_folder_path, channel, image_path):
        dat = np.array(matrix)
        main_title = 'Точки_экстремумов_' + title + '_Масштаб_' + str(scale) + f'_{channel}'
        x = dat[:, 0]
        y = dat[:, 1]

        # Загружаем изображение для получения его размеров
        img = mpimg.imread(image_path)
        height, width = img.shape[:2]

        fig, ax = plt.subplots(figsize=(width / 100, height / 100))  # Устанавливаем размеры фигуры соответственно исходному изображению
        ax.set_aspect('equal')  # Устанавливаем одинаковое соотношение сторон
        ax.set_title(main_title)
        ax.scatter(x, y, alpha=0.5, marker=".")
        ax.invert_yaxis()  # Переворачиваем ось Y

        name = main_title.replace(' ', '_') + '.png'
        file_path = os.path.join(scale_folder_path, name)
        plt.savefig(file_path)
        plt.close()
    

    @staticmethod
    def save_extremum_picture(matrix, title, scale, scale_folder_path, channel):
        dat = np.array(matrix)
        main_title = 'Точки_экстремумов_' + title + '_Масштаб_' + str(scale) + f'_{channel}'
        x = dat[:, 0]
        y = dat[:, 1]

        plt.figure(figsize=(7, 8))
        plt.title(main_title)
        plt.scatter(x, y, alpha=0.5, marker=".")
        plt.gca().invert_yaxis()

        name = main_title.replace(' ', '_') + '.png'
        file_path = os.path.join(scale_folder_path, name)
        plt.savefig(file_path)
        plt.close()

    def compute_dist_angle(self, points, text, scale, entry_near_point):
        def distance(p1, p2):
            return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        def sort(point, points):
            l = list(points)
            l.sort(key=lambda coord: distance(point, coord))
            points = np.array(l)
            dist = [round(distance(point, coord), 2) for coord in points]
            return points, dist

        def angle(P1, P2):
            P1_v = [P1[0], P1[1] + 1]
            ang1 = math.atan2(P1_v[1] - P1[1], P1_v[0] - P1[0])
            ang2 = math.atan2(P2[1] - P1[1], P2[0] - P1[0])
            return np.rad2deg((ang1 - ang2) % (2 * np.pi))

        n = int(entry_near_point.get())
        result_dist_angle = []
        for i in range(len(points)):
            p0 = points[i]
            points_sort, dist = sort(p0, points)
            angles = [round(angle(p0, p_i)) for p_i in points_sort[1:n + 1]]
            result_dist_angle.append([p0, dist[1:n + 1], angles, points_sort[1:n + 1]])
        scale_folder_path = self.create_scale_folder(scale)
        file_name = text + f'Масштаб_{scale}.txt'
        file_path = os.path.join(scale_folder_path, file_name)
        with open(file_path, "w") as file:
            for value in result_dist_angle:
                line = f"point{value[0]}\t:  dist = {value[1]}\t angle = {value[2]}, points \n {value[3]}\n"
                file.write(line)
            file.close()

    def create_downloads_folder(self, folder_name):
        downloads_path = os.path.join(os.path.expanduser('~'), 'Downloads')
        self.folder_path = os.path.join(downloads_path, folder_name)
        os.makedirs(self.folder_path, exist_ok=True)
        print(f"Папка '{folder_name}' создана в папке 'Загрузки'.")
        print(f"Путь к папке: {self.folder_path}")

    def compute(self, row_var, col_var, max_var, min_var, p_ex_var1, p_ex_var2, dist_angle_var, entry_near_point,
                plane_var, print_channels_txt_var):
        current_date = datetime.datetime.now()
        date_str = current_date.strftime("%d_%m_%Y_%H_%M_%S")
        folder_name = f"Вейвлет_преобразования_{date_str}"
        self.create_downloads_folder(folder_name)
        self.save_orig_channels_txt(print_channels_txt_var)
        if (p_ex_var1.get() is True) and (p_ex_var2.get() is True):
            self.compute_wavelets(0, row_var, col_var, max_var, min_var, p_ex_var1, p_ex_var2, dist_angle_var)
        if (p_ex_var1.get() is True) and (p_ex_var2.get() is False):
            self.compute_wavelets(1, row_var, col_var, max_var, min_var, p_ex_var1, p_ex_var2, dist_angle_var)
        if (p_ex_var1.get() is False) and (p_ex_var2.get() is True):
            self.compute_wavelets(10, row_var, col_var, max_var, min_var, p_ex_var1, p_ex_var2, dist_angle_var)
        if (p_ex_var1.get() is False) and (p_ex_var2.get() is False):
            self.compute_wavelets(11, row_var, col_var, max_var, min_var, p_ex_var1, p_ex_var2, dist_angle_var)

        self.find_extremum(row_var, col_var, max_var, min_var, p_ex_var1, p_ex_var2, plane_var)

        if dist_angle_var.get() is True:
            for scale in range(self.num_scale):
                if len(self.extremum_col_min_array) > 0:
                    self.compute_dist_angle(self.extremum_col_min_array[scale],
                                            text="Расстояния_и_углы_минимальные_по_столбцам_",
                                            scale=self.scales[scale], entry_near_point=entry_near_point)
                if len(self.extremum_col_max_array) > 0:
                    self.compute_dist_angle(self.extremum_col_max_array[scale],
                                            text="Расстояния_и_углы_максимальные_по_столбцам_",
                                            scale=self.scales[scale], entry_near_point=entry_near_point)
                if len(self.extremum_row_min_array) > 0:
                    self.compute_dist_angle(self.extremum_row_min_array[scale],
                                            text="Расстояния_и_углы_минимальные_по_строкам_", scale=self.scales[scale],
                                            entry_near_point=entry_near_point)
                if len(self.extremum_row_max_array) > 0:
                    self.compute_dist_angle(self.extremum_row_max_array[scale],
                                            text="Расстояния_и_углы_максимальные_по_строкам_", scale=self.scales[scale],
                                            entry_near_point=entry_near_point)
                if len(self.extremum_plane_max_array) > 0:
                    self.compute_dist_angle(self.extremum_plane_max_array[scale],
                                            text="Расстояния_и_углы_максимальные_по_плоскости_",
                                            scale=self.scales[scale],
                                            entry_near_point=entry_near_point)
        mb.showinfo(title="Информация", message="Вычисления выполнены успешно. \n"
                                                "Все файлы сохранены в папку: \n" + self.folder_path + ".")
        sys.exit(0)


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("950x750")
        self.title("Wavelets and other")
        self.resizable(False, False)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.image_processor = ImageProcessor()

        self.label_load = ctk.CTkLabel(self, text="Загрузить изображение")
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

        self.row_var = tk.BooleanVar(value=True)
        self.col_var = tk.BooleanVar(value=True)
        self.max_var = tk.BooleanVar(value=True)
        self.min_var = tk.BooleanVar(value=True)
        self.plane_var = tk.BooleanVar(value=True)

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
        self.plane_checkbox = ctk.CTkCheckBox(self, text="По плоскости", variable=self.plane_var, onvalue=True,
                                              offvalue=False)
        self.plane_checkbox.grid(row=18, column=0, padx=20, pady=20, sticky="w")

        self.num_near_point = ctk.CTkLabel(self, text="Введите количество ближайших точек")
        self.num_near_point.grid(row=19, column=0, padx=20, pady=10, sticky="w")
        self.entry_var = tk.StringVar()
        self.entry_var.set("n (ex. 5)")
        self.entry_near_point = ctk.CTkEntry(self, textvariable=self.entry_var)
        self.entry_near_point.grid(row=19, column=1, sticky="w")
        self.entry_near_point.configure(text_color="gray")
        self.entry_near_point.bind("<Button-1>", self.on_entry_click)

        # output gui
        self.output_label = ctk.CTkLabel(self, text="Вывод вычислений")
        self.output_label.configure(font=ctk.CTkFont(size=16, weight="bold"))
        self.output_label.grid(row=0, column=3, padx=20, sticky="w")

        self.wp_var1 = tk.BooleanVar(value=True)
        self.wp_var2 = tk.BooleanVar(value=True)
        self.p_ex_var1 = tk.BooleanVar(value=True)
        self.p_ex_var2 = tk.BooleanVar(value=True)
        self.dist_angle_var = tk.BooleanVar(value=True)

        self.wp_label = ctk.CTkLabel(self, text="1) Вейвлет преобразование (Морле)")
        self.wp_label.grid(row=1, column=3, padx=20, sticky="w")
        self.wp1_checkbox = ctk.CTkCheckBox(self, text="В виде .txt файла (по 1 .txt на каждый масштаб)",
                                            variable=self.wp_var1)
        self.wp1_checkbox.grid(row=2, column=3, padx=40, sticky="w")
        self.wp2_checkbox = ctk.CTkCheckBox(self, text="В виде изображений (по 1 .jpg на каждый масштаб)",
                                            variable=self.wp_var2)
        self.wp2_checkbox.grid(row=3, column=3, padx=40, sticky="w")

        self.p_ex_label = ctk.CTkLabel(self, text="2) Точки экстремума:")
        self.p_ex_label.grid(row=5, column=3, padx=20, sticky="w")
        self.p_ex1_checkbox = ctk.CTkCheckBox(self, text="В виде .txt файла (по 1 .txt на каждый масштаб)",
                                              variable=self.p_ex_var1)
        self.p_ex1_checkbox.grid(row=6, column=3, padx=40, sticky="w")
        self.p_ex2_checkbox = ctk.CTkCheckBox(self, text="В виде изображений (по 1 .jpg на каждый масштаб)",
                                              variable=self.p_ex_var2)
        self.p_ex2_checkbox.grid(row=7, column=3, padx=40, sticky="w")

        self.dist_angle_label = ctk.CTkLabel(self, text="3) Алгоритм нахождения расстояний и углов")
        self.dist_angle_label.grid(row=8, column=3, padx=20, sticky="w")
        self.dist_angle_checkbox = ctk.CTkCheckBox(self, text="В виде .txt файла (по 1 .txt на каждый масштаб)",
                                                   variable=self.dist_angle_var)
        self.dist_angle_checkbox.grid(row=9, column=3, padx=40, sticky="w")

        self.dist_angle_label = ctk.CTkLabel(self, text="4) Вывод промежуточных вычислений")
        self.dist_angle_label.grid(row=10, column=3, padx=20, sticky="w")
        self.print_channels_txt_var = tk.BooleanVar(value=True)
        self.print_channels_txt_checkbox = ctk.CTkCheckBox(self, text="Исходные зн-я каналов rgb изображения виде .txt файла",
                                              variable=self.print_channels_txt_var)
        self.print_channels_txt_checkbox.grid(row=11, column=3, padx=40, sticky="w")


        self.app_start_button = ctk.CTkButton(self, text="Вычислить", command=self.compute)
        self.app_start_button.grid(row=16, column=3, sticky="s")
        self.app_start_button.configure(width=200, height=50, border_width=3, border_color="black")

    def load_image_callback(self):
        if self.image_processor.load_image():
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

    def on_entry_click(self, event):
        self.entry_near_point.delete(0, ctk.END)

    def compute(self):
        self.image_processor.compute(self.row_var, self.col_var, self.max_var, self.min_var, self.p_ex_var1,
                                     self.p_ex_var2, self.dist_angle_var, self.entry_near_point, self.plane_var, self.print_channels_txt_var)


app = App()
app.mainloop()
"""