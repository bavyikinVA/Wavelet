"""Диалог загрузки и обрезки изображений для Wavelets Analysis Studio.

Публичный контракт: run_cropper(master) -> (PIL.Image RGB, source_path) | None.
Исходное изображение не перезаписывается; изменения только в памяти.
"""

import math
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

import customtkinter as ctk
from PIL import Image, ImageOps, ImageTk

from .operations import polygon_crop, rectangle_crop
from .window_geometry import place_centered


class ImageCropperApp(ctk.CTkToplevel):
    # Prevent a nested update() in CustomTkinter Windows titlebar setup.
    _deactivate_windows_window_header_manipulation = True
    MAX_UNDO_STEPS = 4
    MAX_UNDO_BYTES = 96 * 1024 * 1024
    MAX_PREVIEW_PIXELS = 16_000_000

    def __init__(self, master=None):
        super().__init__(master)
        self.withdraw()
        self.master_app = master
        self._closed = False
        self._resize_after = None
        self._undo_stack = []
        self.result = None
        self.image = None
        self.original_image = None
        self.original_file_path = None
        self.tk_image = None
        self.rect_coords = None
        self.rect_id = None
        self.polygon_points = []
        self.polygon_lines = []
        self.polygon_mode = False
        self._scale = 1.0
        self._fit_view = True
        self._origin = (0.0, 0.0)

        self.title('Загрузка и обрезка изображения')
        self.minsize(480, 360)
        self.protocol('WM_DELETE_WINDOW', self.safe_destroy)
        from utils.rendering import enable_dark_titlebar
        self.bind('<Map>', lambda event: enable_dark_titlebar(self)
                  if event.widget is self else None, add='+')

        self._build_ui()
        self._refresh_controls()
        place_centered(self, master, width=1100, height=760)
        if hasattr(master, 'register_child_window'):
            master.register_child_window(self)
        if master is not None and master.winfo_viewable():
            self.transient(master)
        self.deiconify()
        # Never grab an unmapped window; X11 rejects it. Deferring also avoids
        # running callbacks while CTkToplevel.__init__ is incomplete.
        self.after_idle(self._activate_modal)

    def _build_ui(self):
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill='both', expand=True, padx=12, pady=12)

        self.button_frame = ctk.CTkFrame(self.main_frame, fg_color='transparent')
        self.button_frame.pack(fill='x', pady=(0, 10))
        self.btn_open = ctk.CTkButton(self.button_frame, text='Открыть изображение', width=150,
                                     command=self.open_image)
        self.btn_open.pack(side='left', padx=(0, 6))
        self.btn_rect_crop = ctk.CTkButton(self.button_frame, text='Прямоугольник', width=135,
                                          command=self.set_rectangle_mode)
        self.btn_rect_crop.pack(side='left', padx=3)
        self.btn_poly_crop = ctk.CTkButton(self.button_frame, text='Полигон', width=110,
                                          command=self.set_polygon_mode)
        self.btn_poly_crop.pack(side='left', padx=3)

        self.filename_label = ctk.CTkLabel(self.main_frame, text='Изображение не загружено',
                                          anchor='w')
        self.filename_label.pack(fill='x', padx=4)

        self.canvas_frame = ctk.CTkFrame(self.main_frame)
        self.canvas_frame.pack(fill='both', expand=True, pady=6)
        self.canvas = tk.Canvas(self.canvas_frame, bg='#202124', highlightthickness=0,
                                cursor='cross')
        self.canvas.pack(side='left', fill='both', expand=True)
        self.scroll_y = ctk.CTkScrollbar(self.canvas_frame, orientation='vertical',
                                        command=self._scroll_y)
        self.scroll_y.pack(side='right', fill='y')
        self.scroll_x = ctk.CTkScrollbar(self.main_frame, orientation='horizontal',
                                        command=self._scroll_x)
        self.scroll_x.pack(fill='x')
        self.canvas.configure(yscrollcommand=self.scroll_y.set, xscrollcommand=self.scroll_x.set)

        self.canvas.bind('<ButtonPress-1>', self.on_button_press)
        self.canvas.bind('<B1-Motion>', self.on_move_press)
        self.canvas.bind('<ButtonRelease-1>', self.on_button_release)
        self.canvas.bind('<ButtonPress-2>', lambda e: self.canvas.scan_mark(e.x, e.y))
        self.canvas.bind('<B2-Motion>', lambda e: self.canvas.scan_dragto(e.x, e.y, gain=1))
        self.canvas.bind('<Button-3>', self._undo_polygon_event)
        self.canvas.bind('<MouseWheel>', self.on_mousewheel)
        self.canvas.bind('<Button-4>', self.on_mousewheel)
        self.canvas.bind('<Button-5>', self.on_mousewheel)
        self.canvas.bind('<Configure>', self._on_canvas_resize)

        self.tool_frame = ctk.CTkFrame(self.main_frame, fg_color='transparent')
        self.tool_frame.pack(fill='x', pady=(4, 2))
        self.btn_zoom_out = ctk.CTkButton(self.tool_frame, text='−', width=36,
                                         command=lambda: self.zoom(1 / 1.25))
        self.btn_zoom_out.pack(side='left', padx=(0, 3))
        self.btn_zoom_in = ctk.CTkButton(self.tool_frame, text='+', width=36,
                                        command=lambda: self.zoom(1.25))
        self.btn_zoom_in.pack(side='left', padx=3)
        self.btn_zoom_fit = ctk.CTkButton(self.tool_frame, text='Вписать в окно', width=120,
                                         command=self.fit_image)
        self.btn_zoom_fit.pack(side='left', padx=5)
        self.zoom_label = ctk.CTkLabel(self.tool_frame, text='100%', width=65)
        self.zoom_label.pack(side='left', padx=5)
        self.bg_label = ctk.CTkLabel(self.tool_frame, text='Фон полигона:')
        self.bg_label.pack(side='right', padx=(4, 2))
        self.polygon_bg = ctk.CTkOptionMenu(self.tool_frame,
                                           values=['Чёрный', 'Белый'], width=95)
        self.polygon_bg.set('Чёрный')
        self.polygon_bg.pack(side='right', padx=(0, 4))

        self.polygon_control_frame = ctk.CTkFrame(self.main_frame, fg_color='transparent')
        self.btn_poly_confirm = ctk.CTkButton(self.polygon_control_frame,
                                              text='Обрезать по контуру',
                                              command=self.apply_polygon_crop, width=125)
        self.btn_poly_confirm.pack(side='left', padx=(0, 6))
        self.btn_poly_reset = ctk.CTkButton(self.polygon_control_frame,
                                            text='Очистить контур', command=self.reset_polygon, width=110)
        self.btn_poly_reset.pack(side='left', padx=6)
        self.btn_poly_back = ctk.CTkButton(self.polygon_control_frame,
                                           text='Убрать вершину', command=self.undo_polygon_point, width=105)
        self.btn_poly_back.pack(side='left', padx=6)
        self.btn_poly_cancel = ctk.CTkButton(self.polygon_control_frame, text='Отмена',
                                             width=80, command=self.cancel_polygon_mode)
        self.btn_poly_cancel.pack(side='right')

        self.action_frame = ctk.CTkFrame(self.main_frame, fg_color='transparent')
        self.action_frame.pack(fill='x', pady=(6, 0))
        self.btn_confirm_rect = ctk.CTkButton(self.action_frame, text='Обрезать выделенное',
                                             command=self.apply_rectangle_crop, width=170)
        self.btn_confirm_rect.pack(side='left', padx=(0, 4))
        self.btn_clear_selection = ctk.CTkButton(self.action_frame, text='Снять выделение',
                                                 command=self.reset_selection, width=135)
        self.btn_clear_selection.pack(side='left', padx=4)
        self.btn_reset = ctk.CTkButton(self.action_frame, text='К исходному',
                                      command=self.reset_image, width=115)
        self.btn_reset.pack(side='left', padx=4)
        self.final_frame = ctk.CTkFrame(self.main_frame, fg_color='transparent')
        self.final_frame.pack(fill='x', pady=(6, 0))
        self.btn_undo = ctk.CTkButton(self.final_frame, text='Отменить обрезку',
                                     command=self.undo_crop, width=120)
        self.btn_undo.pack(side='left', padx=(0, 6))
        self.btn_cancel = ctk.CTkButton(self.final_frame, text='Закрыть без изменений',
                                       command=self.safe_destroy, width=135, fg_color='#565960',
                                       hover_color='#45474c')
        self.btn_cancel.pack(side='right')
        self.btn_save = ctk.CTkButton(self.final_frame, text='Использовать изображение',
                                     width=170, command=self.save_and_exit)
        self.btn_save.pack(side='right', padx=(0, 6))

        self.hint_label = ctk.CTkLabel(self.main_frame, anchor='w',
                                       text='Откройте изображение для начала работы.')
        self.hint_label.pack(fill='x', padx=4, pady=(5, 0))
        self.bind('<Escape>', self._on_escape)
        self.bind('<Control-z>', self._on_undo_key)
        self.bind('<Control-Z>', self._on_undo_key)
        self.bind('<Return>', self._on_enter)

    def _activate_modal(self):
        if self._closed:
            return
        try:
            if self.winfo_viewable():
                self.grab_set()
                self.focus_set()
            else:
                self.after(30, self._activate_modal)
        except tk.TclError:
            pass

    def _on_canvas_resize(self, _event=None):
        if self._closed:
            return
        if self._resize_after is not None:
            try:
                self.after_cancel(self._resize_after)
            except tk.TclError:
                pass
        self._resize_after = self.after(90, self._redraw_after_resize)

    def _redraw_after_resize(self):
        self._resize_after = None
        if not self._closed:
            self.update_canvas()

    def _refresh_controls(self):
        loaded = self.image is not None
        for btn in (self.btn_rect_crop, self.btn_poly_crop, self.btn_reset, self.btn_save,
                    self.btn_zoom_in, self.btn_zoom_out, self.btn_zoom_fit):
            btn.configure(state='normal' if loaded else 'disabled')
        self.btn_open.configure(state='normal')
        self.btn_undo.configure(state='normal' if self._undo_stack else 'disabled')
        selection = self.rect_coords is not None and not self.polygon_mode
        self.btn_confirm_rect.configure(state='normal' if selection else 'disabled')
        self.btn_clear_selection.configure(state='normal' if selection else 'disabled')
        self.btn_poly_confirm.configure(state='normal' if len(self.polygon_points) >= 3 else 'disabled')
        self.btn_poly_reset.configure(state='normal' if self.polygon_points else 'disabled')
        self.btn_poly_back.configure(state='normal' if self.polygon_points else 'disabled')
        if self.polygon_mode:
            self.hint_label.configure(text='Полигон: кликайте по вершинам; ПКМ — убрать последнюю. Вне контура — выбранный фон.')
        elif selection:
            self.hint_label.configure(text='Рамка выделена. Нажмите «Обрезать выделенное» или Escape для отмены.')
        elif loaded:
            self.hint_label.configure(text='Потяните мышью для прямоугольной обрезки. Ctrl + колесо — масштаб, колесо — прокрутка.')
        else:
            self.hint_label.configure(text='Откройте изображение для начала работы.')
        if self.polygon_mode:
            if not self.polygon_control_frame.winfo_manager():
                self.polygon_control_frame.pack(fill='x', pady=(4, 0), before=self.action_frame)
        else:
            self.polygon_control_frame.pack_forget()
        base = ('#3a7ebf', '#1f538d')
        self.btn_rect_crop.configure(fg_color='#216e79' if loaded and not self.polygon_mode else base)
        self.btn_poly_crop.configure(fg_color='#216e79' if loaded and self.polygon_mode else base)

    def open_image(self):
        file_types = [('Изображения', '*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp'),
                      ('Все файлы', '*.*')]
        file_path = filedialog.askopenfilename(parent=self, title='Открыть изображение',
                                               filetypes=file_types)
        if not file_path:
            return
        try:
            with Image.open(file_path) as source:
                converted = ImageOps.exif_transpose(source).convert('RGB')
                converted.load()
        except Exception as exc:
            messagebox.showerror('Ошибка загрузки', f'Не удалось открыть изображение:\n{exc}', parent=self)
            return
        # Only replace state when decoding actually succeeded.
        self.original_image = converted
        self.image = converted.copy()
        self.original_file_path = file_path
        self._undo_stack.clear()
        self.polygon_mode = False
        self.polygon_points.clear()
        self.rect_coords = None
        self._fit_view = True
        self._refresh_filename()
        self.update_canvas()
        self._refresh_controls()

    def _refresh_filename(self):
        if self.image is not None and self.original_file_path:
            self.filename_label.configure(
                text=f'{Path(self.original_file_path).name}  •  {self.image.width} × {self.image.height} пикс.')

    def _display_scale(self):
        if self.image is None:
            return 1.0
        if self._fit_view:
            w = max(1, self.canvas.winfo_width() - 16)
            h = max(1, self.canvas.winfo_height() - 16)
            return min(1.0, w / self.image.width, h / self.image.height)
        return self._scale

    def update_canvas(self):
        self.canvas.delete('all')
        if self.image is None:
            self.canvas.configure(scrollregion=(0, 0, 1, 1))
            self.tk_image = None
            return
        self._scale = self._display_scale()
        display_w = max(1, round(self.image.width * self._scale))
        display_h = max(1, round(self.image.height * self._scale))
        visible_w = max(1, self.canvas.winfo_width())
        visible_h = max(1, self.canvas.winfo_height())
        total_w = max(visible_w, display_w + 12)
        total_h = max(visible_h, display_h + 12)
        self._origin = ((total_w - display_w) / 2 if display_w <= visible_w else 0,
                        (total_h - display_h) / 2 if display_h <= visible_h else 0)
        preview = self.image.resize((display_w, display_h), Image.Resampling.BILINEAR)
        self.tk_image = ImageTk.PhotoImage(preview, master=self)
        self.canvas.create_image(*self._origin, image=self.tk_image, anchor='nw', tags='image')
        self.canvas.configure(scrollregion=(0, 0, total_w, total_h))
        self.zoom_label.configure(text=f'{self._scale:.0%}')
        self._redraw_selection()

    def _to_image(self, event, *, allow_edge=False):
        if self.image is None:
            return None
        x = (self.canvas.canvasx(event.x) - self._origin[0]) / self._scale
        y = (self.canvas.canvasy(event.y) - self._origin[1]) / self._scale
        if allow_edge:
            x = min(max(x, 0), self.image.width)
            y = min(max(y, 0), self.image.height)
        elif not (0 <= x < self.image.width and 0 <= y < self.image.height):
            return None
        return x, y

    def _to_canvas(self, x, y):
        return self._origin[0] + x * self._scale, self._origin[1] + y * self._scale

    def _redraw_selection(self):
        self.canvas.delete('selection')
        self.rect_id = None
        self.polygon_lines = []
        if self.rect_coords is not None and not self.polygon_mode:
            x1, y1, x2, y2 = self.rect_coords
            self.rect_id = self.canvas.create_rectangle(*self._to_canvas(x1, y1),
                                                        *self._to_canvas(x2, y2),
                                                        outline='#ff7665', width=2, dash=(5, 4),
                                                        tags='selection')
        if self.polygon_mode and self.polygon_points:
            pts = [self._to_canvas(x, y) for x, y in self.polygon_points]
            for i, (x, y) in enumerate(pts):
                self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4,
                                        fill='#59e2aa', outline='#f5f5f5', tags='selection')
                if i:
                    line = self.canvas.create_line(*pts[i-1], x, y, fill='#59e2aa',
                                                   width=2, tags='selection')
                    self.polygon_lines.append(line)
            if len(pts) >= 3:
                self.polygon_lines.append(self.canvas.create_line(*pts[-1], *pts[0],
                                                                   fill='#59e2aa', width=2,
                                                                   dash=(4, 4), tags='selection'))

    def set_rectangle_mode(self):
        if self.image is None:
            return
        self.polygon_mode = False
        self.polygon_points.clear()
        self.rect_coords = None
        self._redraw_selection()
        self._refresh_controls()

    def set_polygon_mode(self):
        if self.image is None:
            return
        self.polygon_mode = True
        self.polygon_points.clear()
        self.rect_coords = None
        self._redraw_selection()
        self._refresh_controls()

    def cancel_polygon_mode(self):
        self.polygon_mode = False
        self.reset_polygon()
        self._refresh_controls()

    def reset_polygon(self):
        self.polygon_points.clear()
        self._redraw_selection()
        self._refresh_controls()

    def undo_polygon_point(self):
        if self.polygon_mode and self.polygon_points:
            self.polygon_points.pop()
            self._redraw_selection()
            self._refresh_controls()

    def _undo_polygon_event(self, _event):
        self.undo_polygon_point()

    def on_button_press(self, event):
        point = self._to_image(event)
        if point is None:
            return
        if self.polygon_mode:
            self.polygon_points.append(point)
        else:
            self.rect_coords = (point[0], point[1], point[0], point[1])
        self._redraw_selection()
        self._refresh_controls()

    def on_move_press(self, event):
        if self.image is None or self.polygon_mode or self.rect_coords is None:
            return
        x1, y1, _, _ = self.rect_coords
        x2, y2 = self._to_image(event, allow_edge=True)
        self.rect_coords = (x1, y1, x2, y2)
        self._redraw_selection()

    def on_button_release(self, event):
        if self.image is None or self.polygon_mode or self.rect_coords is None:
            return
        self.on_move_press(event)
        x1, y1, x2, y2 = self.rect_coords
        if abs(x2 - x1) < 2 or abs(y2 - y1) < 2:
            self.reset_selection()
        else:
            self._refresh_controls()

    def reset_selection(self):
        self.rect_coords = None
        self._redraw_selection()
        self._refresh_controls()

    def _remember_undo(self):
        if self.image is None:
            return
        snapshot = self.image.copy()
        self._undo_stack.append(snapshot)
        total = sum(img.width * img.height * 3 for img in self._undo_stack)
        while len(self._undo_stack) > self.MAX_UNDO_STEPS or (total > self.MAX_UNDO_BYTES and len(self._undo_stack) > 1):
            removed = self._undo_stack.pop(0)
            total -= removed.width * removed.height * 3

    def _apply_image(self, image):
        self._remember_undo()
        self.image = image
        self._refresh_filename()
        self.rect_coords = None
        self.polygon_points.clear()
        self.polygon_mode = False
        self._fit_view = True
        self.update_canvas()
        self._refresh_controls()

    def apply_rectangle_crop(self):
        if self.image is None or self.rect_coords is None:
            return
        try:
            new_image = rectangle_crop(self.image, self.rect_coords)
        except (ValueError, OSError) as exc:
            messagebox.showerror('Ошибка обрезки', str(exc), parent=self)
            return
        self._apply_image(new_image)

    def apply_polygon_crop(self):
        if self.image is None or len(self.polygon_points) < 3:
            return
        try:
            background = 'white' if self.polygon_bg.get() == 'Белый' else 'black'
            new_image = polygon_crop(self.image, self.polygon_points, background=background)
        except (ValueError, OSError) as exc:
            messagebox.showerror('Ошибка обрезки', str(exc), parent=self)
            return
        self._apply_image(new_image)

    def undo_crop(self):
        if not self._undo_stack:
            return
        self.image = self._undo_stack.pop()
        self._refresh_filename()
        self.polygon_mode = False
        self.polygon_points.clear()
        self.rect_coords = None
        self._fit_view = True
        self.update_canvas()
        self._refresh_controls()

    def reset_image(self):
        if self.original_image is None:
            return
        self._apply_image(self.original_image.copy())

    def fit_image(self):
        self._fit_view = True
        self.update_canvas()

    def zoom(self, multiplier):
        if self.image is None:
            return
        old_scale = self._display_scale()
        max_scale_by_pixels = math.sqrt(self.MAX_PREVIEW_PIXELS / (self.image.width * self.image.height))
        max_scale = max(0.02, min(4.0, max_scale_by_pixels))
        new_scale = max(0.02, min(max_scale, old_scale * multiplier))
        if abs(new_scale - old_scale) < 1e-9:
            return
        cx = (self.canvas.canvasx(self.canvas.winfo_width() / 2) - self._origin[0]) / old_scale
        cy = (self.canvas.canvasy(self.canvas.winfo_height() / 2) - self._origin[1]) / old_scale
        self._fit_view = False
        self._scale = new_scale
        self.update_canvas()
        # Restore approximate center of the source view after zooming.
        x_total, y_total = (max(1, float(v)) for v in self.canvas.cget('scrollregion').split()[2:])
        x_px = self._origin[0] + cx * new_scale - self.canvas.winfo_width() / 2
        y_px = self._origin[1] + cy * new_scale - self.canvas.winfo_height() / 2
        self.canvas.xview_moveto(max(0, x_px / x_total))
        self.canvas.yview_moveto(max(0, y_px / y_total))

    def on_mousewheel(self, event):
        direction = -1 if (getattr(event, 'num', None) == 4 or getattr(event, 'delta', 0) > 0) else 1
        state = getattr(event, 'state', 0)
        if state & 0x0004:
            self.zoom(1.25 if direction < 0 else 1 / 1.25)
        elif state & 0x0001:
            self.canvas.xview_scroll(direction * 3, 'units')
        else:
            self.canvas.yview_scroll(direction * 3, 'units')
        return 'break'

    def _scroll_y(self, *args):
        self.canvas.yview(*args)

    def _scroll_x(self, *args):
        self.canvas.xview(*args)

    def _on_escape(self, _event):
        if self.polygon_mode:
            self.cancel_polygon_mode()
        elif self.rect_coords is not None:
            self.reset_selection()
        else:
            self.safe_destroy()
        return 'break'

    def _on_undo_key(self, _event):
        if self.polygon_mode and self.polygon_points:
            self.undo_polygon_point()
        else:
            self.undo_crop()
        return 'break'

    def _on_enter(self, _event):
        if self.polygon_mode:
            self.apply_polygon_crop()
        elif self.rect_coords is not None:
            self.apply_rectangle_crop()
        return 'break'

    def safe_destroy(self):
        if self._closed:
            return
        self._closed = True
        if self._resize_after is not None:
            try:
                self.after_cancel(self._resize_after)
            except tk.TclError:
                pass
            self._resize_after = None
        if hasattr(self.master_app, 'unregister_child_window'):
            self.master_app.unregister_child_window(self)
        try:
            if self.grab_current() == self:
                self.grab_release()
        except tk.TclError:
            pass
        try:
            self.destroy()
        except tk.TclError:
            pass
        self.tk_image = None
        self._undo_stack.clear()

    def save_and_exit(self):
        if self.image is None:
            return None
        self.result = (self.image.convert('RGB'), self.original_file_path)
        self.safe_destroy()
        return self.result


def run_cropper(master=None):
    root = None
    try:
        if master is None:
            root = tk.Tk()
            root.withdraw()
        window = ImageCropperApp(master if master is not None else root)
        window.wait_window()
        return window.result
    finally:
        if root is not None:
            try:
                root.destroy()
            except tk.TclError:
                pass
