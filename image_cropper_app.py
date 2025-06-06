import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import os

global drawing, points, img, img_copy


class ImageCropperApp(ctk.CTkToplevel):  # Изменяем наследование на CTkToplevel
    def __init__(self, master=None):
        super().__init__(master)
        self.title("Advanced Image Cropper")
        self.geometry("1000x700")

        # Блокируем взаимодействие с родительским окном
        self.grab_set()
        self.focus_set()

        # Переменные состояния
        self.image = None
        self.original_image = None
        self.tk_image = None
        self.original_file_path = None
        self.rect_coords = None
        self.rect_id = None

        self.create_widgets()

    def create_widgets(self):
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.button_frame = ctk.CTkFrame(self.main_frame)
        self.button_frame.pack(fill="x", pady=5)

        self.btn_open = ctk.CTkButton(self.button_frame, text="Open Image", command=self.open_image)
        self.btn_open.pack(side="left", padx=5)

        self.btn_rect_crop = ctk.CTkButton(self.button_frame, text="Rectangle Crop", command=self.set_rectangle_mode)
        self.btn_rect_crop.pack(side="left", padx=5)
        self.btn_rect_crop.configure(state="disabled")

        self.btn_poly_crop = ctk.CTkButton(self.button_frame, text="Polygon Crop", command=self.start_polygon_crop)
        self.btn_poly_crop.pack(side="left", padx=5)
        self.btn_poly_crop.configure(state="disabled")

        self.btn_reset = ctk.CTkButton(self.button_frame, text="Reset", command=self.reset_image)
        self.btn_reset.pack(side="left", padx=5)
        self.btn_reset.configure(state="disabled")

        self.btn_save = ctk.CTkButton(self.button_frame, text="Save & Exit", command=self.save_and_exit)
        self.btn_save.pack(side="right", padx=5)
        self.btn_save.configure(state="disabled")

        # Фрейм для canvas с прокруткой
        self.canvas_frame = ctk.CTkFrame(self.main_frame)
        self.canvas_frame.pack(fill="both", expand=True)

        # Canvas с прокруткой
        self.canvas = ctk.CTkCanvas(self.canvas_frame, bg="gray20", cursor="cross")
        self.canvas.pack(side="left", fill="both", expand=True)

        # Вертикальная прокрутка
        self.scroll_y = ctk.CTkScrollbar(self.canvas_frame, orientation="vertical", command=self.canvas.yview)
        self.scroll_y.pack(side="right", fill="y")
        self.canvas.configure(yscrollcommand=self.scroll_y.set)

        # Горизонтальная прокрутка
        self.scroll_x = ctk.CTkScrollbar(self.main_frame, orientation="horizontal", command=self.canvas.xview)
        self.scroll_x.pack(fill="x")
        self.canvas.configure(xscrollcommand=self.scroll_x.set)

        # Привязка событий мыши
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)

    def start_polygon_crop(self):
        if not self.original_file_path:
            return

        # Сохраняем текущее состояние окна
        self.withdraw()
        self.polygon_crop(self.original_file_path)
        self.deiconify()

    def polygon_crop(self, image_path):
        global drawing, points, img, img_copy
        drawing = False
        points = []
        img = cv2.imread(image_path)
        img_copy = img.copy()

        # Создаем панель для инструкций
        instruction_panel = np.zeros((60, img.shape[1], 3), dtype=np.uint8)
        instructions = [
            "Instructions:",
            "LKM - Draw, 'C' - crop",
            "'R' - reset, 'Q' - exit"
        ]

        # Добавляем инструкции на панель
        y = 15
        for i, line in enumerate(instructions):
            cv2.putText(instruction_panel, line, (10, y + i * 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        def mouse_callback(event, x, y, flags, param):
            global drawing, points, img_copy

            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                points = [(x, y)]
                img_copy = img.copy()
                img_copy = np.vstack([img_copy, instruction_panel])  # Добавляем инструкции
                cv2.circle(img_copy, (x, y), 3, (0, 255, 0), -1)
                cv2.imshow("Polygon Cropper", img_copy)

            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    points.append((x, y))
                    if len(points) > 1:
                        img_copy = img.copy()
                        cv2.polylines(img_copy, [np.array(points)], False, (0, 255, 0), 2)
                        img_copy = np.vstack([img_copy, instruction_panel])  # Добавляем инструкции
                        cv2.imshow("Polygon Cropper", img_copy)

            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                if len(points) > 2:
                    img_copy = img.copy()
                    cv2.polylines(img_copy, [np.array(points)], True, (0, 255, 0), 2)
                    img_copy = np.vstack([img_copy, instruction_panel])  # Добавляем инструкции
                    cv2.imshow("Polygon Cropper", img_copy)

        cv2.namedWindow("Polygon Cropper")
        cv2.setMouseCallback("Polygon Cropper", mouse_callback)

        # Первоначальное отображение изображения с инструкциями
        img_copy = np.vstack([img_copy, instruction_panel])
        cv2.imshow("Polygon Cropper", img_copy)

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord("c"):  # Обрезать изображение
                if len(points) > 2:
                    # Создаем маску
                    mask = np.zeros(img.shape[:2], dtype=np.uint8)
                    pts = np.array([points], dtype=np.int32)
                    cv2.fillPoly(mask, pts, 255)

                    # Создаем белый фон
                    white_bg = np.ones_like(img) * 255

                    # Копируем только выбранную область
                    result = white_bg.copy()
                    result[mask == 255] = img[mask == 255]

                    # Конвертируем обратно в PIL Image
                    self.image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                    self.update_canvas()
                    self.btn_save.configure(state="normal")

                    cv2.destroyAllWindows()
                    break

            elif key == ord("r"):  # Сбросить контур
                points.clear()
                img_copy = img.copy()
                img_copy = np.vstack([img_copy, instruction_panel])  # Добавляем инструкции
                cv2.imshow("Polygon Cropper", img_copy)
                print("Контур сброшен")

            elif key == ord("q"):  # Выход
                cv2.destroyAllWindows()
                break

    def open_image(self):
        file_types = [("Image files", "*.png *.jpg *.jpeg *.bmp")]
        file_path = filedialog.askopenfilename(filetypes=file_types)

        if not file_path:
            return

        try:
            self.original_image = Image.open(file_path)
            self.image = self.original_image.copy()
            self.original_file_path = file_path
            self.update_canvas()
            self.btn_rect_crop.configure(state="normal")
            self.btn_poly_crop.configure(state="normal")
            self.btn_reset.configure(state="normal")
        except Exception as e:
            print(f"Error loading image: {e}")

    def update_canvas(self):
        if self.image:
            self.tk_image = ImageTk.PhotoImage(self.image)
            self.canvas.delete("all")
            self.canvas.config(
                scrollregion=(0, 0, self.image.width, self.image.height),
                width=min(900, self.image.width),
                height=min(600, self.image.height)
            )
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
            self.canvas.xview_moveto(0.5)
            self.canvas.yview_moveto(0.5)

    def set_rectangle_mode(self):
        self.reset_selection()
        self.canvas.config(cursor="cross")

    def on_button_press(self, event):
        if not self.image:
            return

        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)

        self.rect_coords = [x, y, x, y]
        self.rect_id = self.canvas.create_rectangle(
            x, y, x, y, outline="red", width=2, dash=(5, 5))

    def on_move_press(self, event):
        if not self.image or not self.rect_id:
            return

        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)

        self.rect_coords[2] = x
        self.rect_coords[3] = y
        self.canvas.coords(self.rect_id, *self.rect_coords)

    def on_button_release(self, event):
        if not self.image or not self.rect_id:
            return

        # Проверяем, что область выделения достаточно большая
        if abs(self.rect_coords[2] - self.rect_coords[0]) > 10 and abs(self.rect_coords[3] - self.rect_coords[1]) > 10:
            # Применяем обрезку сразу после выделения
            self.apply_rectangle_crop()

    def apply_rectangle_crop(self):
        try:
            x1, y1, x2, y2 = self.rect_coords
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(self.image.width, x2), min(self.image.height, y2)

            self.image = self.image.crop((x1, y1, x2, y2))
            self.update_canvas()
            self.reset_selection()
            self.btn_save.configure(state="normal")
        except Exception as e:
            print(f"Error cropping image: {e}")

    def on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def reset_image(self):
        if self.original_image:
            self.image = self.original_image.copy()
            self.update_canvas()
            self.reset_selection()
            self.btn_save.configure(state="disabled")

    def reset_selection(self):
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None
            self.rect_coords = None

    def save_and_exit(self):
        if not self.image:
            return

        file_path = self.get_cropped_filename()

        if file_path:
            try:
                self.image.save(file_path)
                with open("image_paths.txt", "w") as f:
                    f.write(file_path)
                self.destroy()  # Закрываем окно обрезки
                return file_path
            except Exception as e:
                print(f"Error saving image: {e}")
                return None

    def get_cropped_filename(self):
        """ Добавление 'cropped_image' перед расширением"""
        if hasattr(self, 'original_file_path'):
            path, ext = os.path.splitext(self.original_file_path)
            return f"{path}_cropped_image{ext}"


def run_cropper(master=None):
    if master is None:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        cropper_window = ImageCropperApp(root)
    else:
        cropper_window = ImageCropperApp(master)

    # Ждем, пока окно не закроется
    cropper_window.wait_window()

    # После закрытия окна читаем путь из файла
    try:
        with open("image_paths.txt", "r") as f:
            return f.readline().strip('\n')
    except FileNotFoundError:
        return None