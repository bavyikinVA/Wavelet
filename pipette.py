import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import os

class PipetteApp(ctk.CTkToplevel):
    def __init__(self, master=None, image_path=None):
        super().__init__(master)
        self.color1 = None
        self.color2 = None
        self.original_img = None
        self.title("Pipette Tool")
        self.geometry("1000x700")
        self.grab_set()
        self.focus_set()
        self.image_path = image_path
        self.click_count = 0
        self.colors = []
        self.img_tk = None

        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # фрейм для кнопок
        self.button_frame = ctk.CTkFrame(self.main_frame)
        self.button_frame.pack(fill="x", pady=5)

        self.btn_start_choice = ctk.CTkButton(self.button_frame, text="Выбрать цвет", command=self.start_color_choice)
        self.btn_start_choice.pack(side="left", padx=5)

        self.btn_reset = ctk.CTkButton(self.button_frame, text="Сбросить", command=self.reset_colors)
        self.btn_reset.pack(side="left", padx=5)
        self.btn_reset.configure(state="disabled")

        self.btn_save = ctk.CTkButton(self.button_frame, text="Сохранить и выйти", command=self.save_and_exit)
        self.btn_save.pack(side="right", padx=5)
        self.btn_save.configure(state="disabled")

        self.canvas_frame = ctk.CTkFrame(self.main_frame)
        self.canvas_frame.pack(fill="both", expand=True)

        self.canvas = ctk.CTkCanvas(self.canvas_frame, bg="gray20", cursor="cross")
        self.canvas.pack(side="left", fill="both", expand=True)

        # вертикальная прокрутка
        self.scroll_y = ctk.CTkScrollbar(self.canvas_frame, orientation="vertical", command=self.canvas.yview)
        self.scroll_y.pack(side="right", fill="y")
        self.canvas.configure(yscrollcommand=self.scroll_y.set)

        # горизонтальная прокрутка
        self.scroll_x = ctk.CTkScrollbar(self.main_frame, orientation="horizontal", command=self.canvas.xview)
        self.scroll_x.pack(fill="x")
        self.canvas.configure(xscrollcommand=self.scroll_x.set)

        # вывод цветов
        self.color_frame = ctk.CTkFrame(self.main_frame)
        self.color_frame.pack(pady=10)

        self.color_canvases = []
        self.rgb_labels = []
        for i in range(2):
            frame = ctk.CTkFrame(self.color_frame)
            frame.pack(side=ctk.LEFT, padx=20, pady=10)

            canvas = ctk.CTkCanvas(frame, width=100, height=100)
            canvas.pack()
            self.color_canvases.append(canvas)

            label = ctk.CTkLabel(frame, text="RGB: ---")
            label.pack()
            self.rgb_labels.append(label)

        self.load_image()

    def load_image(self):
        if self.image_path and os.path.exists(self.image_path):
            self.original_img = Image.open(self.image_path)
            self.img_tk = ImageTk.PhotoImage(self.original_img, master=self)
            self.canvas.create_image(0, 0, anchor="nw", image=self.img_tk)
            self.canvas.configure(scrollregion=(0, 0, self.original_img.width, self.original_img.height))

    def start_color_choice(self):
        self.canvas.bind("<Button-1>", self.get_pixel_rgb)
        self.btn_start_choice.configure(state="disabled")
        self.btn_reset.configure(state="normal")

    def get_pixel_rgb(self, event):
        if not hasattr(self, 'original_img'):
            return

        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)

        if 0 <= x < self.original_img.width and 0 <= y < self.original_img.height:
            r, g, b = self.original_img.convert('RGB').getpixel((x, y))
            self.click_count += 1
            self.colors.append((r, g, b))

            if self.click_count <= 2:
                self.update_color_squares(r, g, b, self.click_count)

            if self.click_count == 2:
                self.canvas.unbind("<Button-1>")
                self.btn_save.configure(state="normal")
                self.btn_reset.configure(state="normal")
                self.btn_start_choice.configure(state="disabled")

    def update_color_squares(self, r, g, b, count):
        self.color_canvases[count - 1].create_rectangle(0, 0, 100, 100, fill=f'#{r:02x}{g:02x}{b:02x}')
        self.rgb_labels[count - 1].configure(text=f'RGB: {r}, {g}, {b}')

    def reset_colors(self):
        self.click_count = 0
        self.colors = []
        self.canvas.bind("<Button-1>", self.get_pixel_rgb)
        for canvas in self.color_canvases:
            canvas.delete("all")
        for label in self.rgb_labels:
            label.configure(text="RGB: ---")
        self.btn_save.configure(state="disabled")
        self.btn_reset.configure(state="disabled")
        self.btn_start_choice.configure(state="normal")

    def save_and_exit(self):
        if len(self.colors) == 2:
            self.color1 = np.array(self.colors[0], dtype=np.int16)
            self.color2 = np.array(self.colors[1], dtype=np.int16)
            self.destroy()
        else:
            self.color1, self.color2 = None, None
            self.destroy()

    def get_colors(self):
        return self.color1, self.color2

def run_pipette(master=None, image_path=None):
    if master is None:
        root = tk.Tk()
        root.withdraw()
        pipette_window = PipetteApp(root, image_path)
    else:
        pipette_window = PipetteApp(master, image_path)

    pipette_window.wait_window()
    return pipette_window.get_colors()