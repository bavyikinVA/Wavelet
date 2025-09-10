import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk


class ImageCropper:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Cropper")
        self.root.geometry("800x600")

        # Настройка темы
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        # Главный фрейм
        self.main_frame = ctk.CTkFrame(root)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Фрейм для кнопок
        self.button_frame = ctk.CTkFrame(self.main_frame)
        self.button_frame.pack(fill="x", pady=5)

        # Кнопки
        self.btn_open = ctk.CTkButton(self.button_frame, text="Open Image", command=self.open_image)
        self.btn_open.pack(side="left", padx=5)

        self.btn_crop = ctk.CTkButton(self.button_frame, text="Crop Image", command=self.crop_image)
        self.btn_crop.pack(side="left", padx=5)
        self.btn_crop.configure(state="disabled")

        self.btn_save = ctk.CTkButton(self.button_frame, text="Save Image", command=self.save_image)
        self.btn_save.pack(side="left", padx=5)
        self.btn_save.configure(state="disabled")

        self.btn_reset = ctk.CTkButton(self.button_frame, text="Reset", command=self.reset_image)
        self.btn_reset.pack(side="left", padx=5)
        self.btn_reset.configure(state="disabled")

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

        # Привязка колеса мыши для прокрутки
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)

        # Переменные для изображения
        self.image = None
        self.tk_image = None
        self.original_image = None
        self.start_x = None
        self.start_y = None
        self.rect = None
        self.crop_rect = None

        # Привязка событий мыши
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def open_image(self):
        file_types = [("Image files", "*.png *.jpg *.jpeg *.bmp")]
        file_path = filedialog.askopenfilename(filetypes=file_types)

        if not file_path:
            return

        try:
            self.original_image = Image.open(file_path)
            self.image = self.original_image.copy()
            self.update_canvas()
            self.btn_reset.configure(state="normal")
            self.btn_crop.configure(state="normal")
        except Exception as e:
            print(f"Error loading image: {e}")

    def update_canvas(self):
        if self.image:
            self.tk_image = ImageTk.PhotoImage(self.image)
            self.canvas.delete("all")
            self.canvas.config(
                scrollregion=(0, 0, self.image.width, self.image.height),
                width=min(800, self.image.width),
                height=min(500, self.image.height)
            )
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

            # Центрируем изображение
            self.canvas.xview_moveto(0.5)
            self.canvas.yview_moveto(0.5)

    def on_button_press(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)

        if not self.rect:
            self.rect = self.canvas.create_rectangle(
                self.start_x, self.start_y,
                self.start_x, self.start_y,
                outline='red', width=2, dash=(5, 5))

    def on_move_press(self, event):
        if self.rect:
            x = self.canvas.canvasx(event.x)
            y = self.canvas.canvasy(event.y)
            self.canvas.coords(
                self.rect,
                self.start_x, self.start_y,
                x, y)

    def on_button_release(self, event):
        if self.rect:
            x = self.canvas.canvasx(event.x)
            y = self.canvas.canvasy(event.y)

            self.crop_rect = (
                min(self.start_x, x),
                min(self.start_y, y),
                max(self.start_x, x),
                max(self.start_y, y)
            )

            # Проверяем, что область выделения достаточно большая
            if abs(self.crop_rect[2] - self.crop_rect[0]) > 10 and abs(self.crop_rect[3] - self.crop_rect[1]) > 10:
                self.btn_save.configure(state="normal")
            else:
                self.canvas.delete(self.rect)
                self.rect = None

    def crop_image(self):
        if self.image and self.crop_rect:
            try:
                self.image = self.image.crop(self.crop_rect)
                self.update_canvas()
                self.btn_save.configure(state="normal")
                self.canvas.delete(self.rect)
                self.rect = None
                self.crop_rect = None
            except Exception as e:
                print(f"Error cropping image: {e}")

    def reset_image(self):
        if self.original_image:
            self.image = self.original_image.copy()
            self.update_canvas()
            if self.rect:
                self.canvas.delete(self.rect)
                self.rect = None
            self.crop_rect = None
            self.btn_save.configure(state="disabled")

    def save_image(self):
        if not self.image:
            return

        file_types = [
            ("PNG", "*.png"),
            ("JPEG", "*.jpg"),
            ("All files", "*.*")
        ]

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=file_types,
            initialfile="cropped_image.png")

        if file_path:
            try:
                self.image.save(file_path)
            except Exception as e:
                print(f"Error saving image: {e}")


if __name__ == "__main__":
    root = ctk.CTk()
    app = ImageCropper(root)
    root.mainloop()