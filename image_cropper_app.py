import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
import os

class ImageCropperApp(ctk.CTkToplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.master_app = master  # Сохраняем ссылку на главное приложение
        if hasattr(master, 'register_child_window'):
            master.register_child_window(self)

        self.title("Advanced Image Cropper")
        self.geometry("1000x700")
        self.grab_set()
        self.focus_set()

        self.image = None
        self.original_image = None
        self.tk_image = None
        self.original_file_path = None
        self.rect_coords = None
        self.rect_id = None
        self.polygon_points = []
        self.polygon_mode = False
        self.polygon_lines = []

        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.button_frame = ctk.CTkFrame(self.main_frame)
        self.button_frame.pack(fill="x", pady=5)

        self.btn_open = ctk.CTkButton(self.button_frame, text="Open Image", command=self.open_image)
        self.btn_open.pack(side="left", padx=5)

        self.btn_rect_crop = ctk.CTkButton(self.button_frame, text="Rectangle Crop", command=self.set_rectangle_mode)
        self.btn_rect_crop.pack(side="left", padx=5)
        self.btn_rect_crop.configure(state="disabled")

        self.btn_poly_crop = ctk.CTkButton(self.button_frame, text="Polygon Crop", command=self.set_polygon_mode)
        self.btn_poly_crop.pack(side="left", padx=5)
        self.btn_poly_crop.configure(state="disabled")

        self.btn_reset = ctk.CTkButton(self.button_frame, text="Reset", command=self.reset_image)
        self.btn_reset.pack(side="left", padx=5)
        self.btn_reset.configure(state="disabled")

        self.btn_save = ctk.CTkButton(self.button_frame, text="Save & Exit", command=self.save_and_exit)
        self.btn_save.pack(side="right", padx=5)
        self.btn_save.configure(state="disabled")

        self.canvas_frame = ctk.CTkFrame(self.main_frame)
        self.canvas_frame.pack(fill="both", expand=True)

        self.canvas = ctk.CTkCanvas(self.canvas_frame, bg="gray20", cursor="cross")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.scroll_y = ctk.CTkScrollbar(self.canvas_frame, orientation="vertical", command=self.canvas.yview)
        self.scroll_y.pack(side="right", fill="y")
        self.canvas.configure(yscrollcommand=self.scroll_y.set)

        self.scroll_x = ctk.CTkScrollbar(self.main_frame, orientation="horizontal", command=self.canvas.xview)
        self.scroll_x.pack(fill="x")
        self.canvas.configure(xscrollcommand=self.scroll_x.set)

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)

        self.polygon_control_frame = ctk.CTkFrame(self.main_frame)
        self.polygon_control_frame.pack(fill="x", pady=5)

        self.btn_poly_confirm = ctk.CTkButton(
            self.polygon_control_frame,
            text="Confirm Polygon",
            command=self.apply_polygon_crop,
            state="disabled"
        )
        self.btn_poly_confirm.pack(side="left", padx=5)

        self.btn_poly_reset = ctk.CTkButton(
            self.polygon_control_frame,
            text="Reset Polygon",
            command=self.reset_polygon,
            state="disabled"
        )
        self.btn_poly_reset.pack(side="left", padx=5)

        self.btn_poly_cancel = ctk.CTkButton(
            self.polygon_control_frame,
            text="Cancel",
            command=self.cancel_polygon_mode,
            state="disabled"
        )
        self.btn_poly_cancel.pack(side="left", padx=5)

        self.polygon_control_frame.pack_forget()

    def set_polygon_mode(self):
        self.reset_selection()
        self.polygon_mode = True
        self.canvas.config(cursor="cross")
        self.polygon_control_frame.pack(fill="x", pady=5)
        self.btn_poly_confirm.configure(state="disabled")
        self.btn_poly_reset.configure(state="normal")
        self.btn_poly_cancel.configure(state="normal")

        self.btn_open.configure(state="disabled")
        self.btn_rect_crop.configure(state="disabled")
        self.btn_poly_crop.configure(state="disabled")
        self.btn_reset.configure(state="disabled")
        self.btn_save.configure(state="disabled")

    def cancel_polygon_mode(self):
        self.polygon_mode = False
        self.reset_polygon()
        self.polygon_control_frame.pack_forget()

        self.btn_open.configure(state="normal")
        self.btn_rect_crop.configure(state="normal")
        self.btn_poly_crop.configure(state="normal")
        self.btn_reset.configure(state="normal")
        self.btn_save.configure(state="normal")

    def reset_polygon(self):
        self.polygon_points = []
        for line in self.polygon_lines:
            self.canvas.delete(line)
        self.polygon_lines = []
        self.btn_poly_confirm.configure(state="disabled")

    def on_button_press(self, event):
        if not self.image:
            return

        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)

        if self.polygon_mode:
            self.polygon_points.append((x, y))

            # Draw point
            point_size = 3
            self.canvas.create_oval(
                x - point_size, y - point_size,
                x + point_size, y + point_size,
                fill="green", outline="green"
            )

            if len(self.polygon_points) > 1:
                line = self.canvas.create_line(
                    self.polygon_points[-2][0], self.polygon_points[-2][1],
                    self.polygon_points[-1][0], self.polygon_points[-1][1],
                    fill="green", width=2
                )
                self.polygon_lines.append(line)

            if len(self.polygon_points) >= 3:
                self.btn_poly_confirm.configure(state="normal")
        else:
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

        if abs(self.rect_coords[2] - self.rect_coords[0]) > 10 and abs(self.rect_coords[3] - self.rect_coords[1]) > 10:
            self.apply_rectangle_crop()

    def apply_polygon_crop(self):
        if len(self.polygon_points) < 3:
            return

        try:
            mask = Image.new('L', (self.image.width, self.image.height), 0)
            draw = ImageDraw.Draw(mask)
            draw.polygon(self.polygon_points, fill=255)

            result = Image.new('RGBA', (self.image.width, self.image.height), (0, 0, 0, 0))
            result.paste(self.image, (0, 0), mask)

            min_x = min(p[0] for p in self.polygon_points)
            max_x = max(p[0] for p in self.polygon_points)
            min_y = min(p[1] for p in self.polygon_points)
            max_y = max(p[1] for p in self.polygon_points)

            result = result.crop((min_x, min_y, max_x, max_y))

            self.image = result
            self.update_canvas()
            self.cancel_polygon_mode()
            self.btn_save.configure(state="normal")
        except Exception as e:
            print(f"Error cropping image: {e}")

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

    def open_image(self):
        file_types = [("Image files", "*.png *.jpg *.jpeg *.bmp")]
        file_path = filedialog.askopenfilename(filetypes=file_types)

        if not file_path:
            return

        try:
            file_path = self.convert_to_png(file_path)
            self.original_image = Image.open(file_path)
            self.image = self.original_image.copy()
            self.original_file_path = file_path
            self.update_canvas()
            self.btn_rect_crop.configure(state="normal")
            self.btn_poly_crop.configure(state="normal")
            self.btn_reset.configure(state="normal")
            self.btn_save.configure(state="normal")
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
        self.cancel_polygon_mode()
        self.reset_selection()
        self.canvas.config(cursor="cross")

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

    def safe_destroy(self):
        """Безопасное уничтожение окна"""
        try:
            if hasattr(self, 'master_app') and hasattr(self.master_app, 'unregister_child_window'):
                self.master_app.unregister_child_window(self)
            self.grab_release()
            self.destroy()
        except tk.TclError:
            pass

    def save_and_exit(self):
        if not self.image:
            return None

        file_path = self.get_cropped_filename()

        if not file_path:
            return None

        try:
            self.image.save(file_path)
            self.safe_destroy()
            return file_path
        except Exception as e:
            print(f"Error saving image: {e}")
            return None

    def get_cropped_filename(self):
        if not hasattr(self, 'original_file_path'):
            return None

        base_path, ext = os.path.splitext(self.original_file_path)
        filename = os.path.basename(base_path).lower()

        if "cropped" in filename or "cropped_image" in filename:
            return self.original_file_path
        else:
            return f"{base_path}_cropped{ext}"


def run_cropper(master=None):
    try:
        if master is None:
            root = tk.Tk()
            root.withdraw()
            cropper_window = ImageCropperApp(root)
        else:
            cropper_window = ImageCropperApp(master)

        cropper_window.wait_window()
        result = cropper_window.save_and_exit()
        return result
    except Exception as e:
        print(f"Error in image cropper: {e}")
        return None
