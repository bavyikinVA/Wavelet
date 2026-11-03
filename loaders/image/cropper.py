"""Компактный редактор загрузки и обрезки изображений.

Публичный контракт сохраняется: ``run_cropper(master)`` возвращает
``(PIL.Image в RGB, исходный путь)`` или ``None`` при закрытии.
Программа не изменяет исходные файлы на диске.
"""

import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
from PIL import Image, ImageOps, ImageTk

from .operations import polygon_crop, rectangle_crop
from loaders.window_geometry import place_centered
from .help_popup import CropperHelpOverlay


# Спокойные заливки вместо яркого свечения и обводки активных кнопок.
BLUE = '#256FA6'
BLUE_HOVER = '#1E6092'
NEUTRAL = '#343840'
NEUTRAL_HOVER = '#454B55'
BORDER = '#505660'
TEXT = '#F2F4F7'


@dataclass(frozen=True)
class _UndoState:
    # Ссылка на неизменяемый для редактора объект PIL Image. Масштабирование
    # и добавление вершин не копируют исходные мегапиксельные данные.
    image: Image.Image
    fit_view: bool
    scale: float
    polygon_mode: bool
    crop_active: bool
    polygon_points: tuple
    rect_coords: tuple | None
    xview: float
    yview: float


class ImageCropperApp(ctk.CTkToplevel):
    # CTk на Windows вызывает update() во время оформления заголовка.
    _deactivate_windows_window_header_manipulation = True
    MAX_PREVIEW_PIXELS = 16_000_000

    def __init__(self, master=None):
        super().__init__(master)
        self.withdraw()
        self.master_app = master
        self._closed = False
        self._resize_after = None
        self.result = None
        self.image = None
        self.original_image = None
        self.original_file_path = None
        self.tk_image = None
        self._image_origin = (0.0, 0.0)
        self._scale = 1.0
        self._fit_view = True
        self._image_item = None
        self.rect_coords = None
        self.rect_id = None
        self.polygon_points = []
        self.polygon_lines = []
        self.polygon_mode = False
        self.crop_active = False
        self._undo_stack = deque(maxlen=40)

        self.title('Image Cropper')
        self.minsize(580, 420)
        self.protocol('WM_DELETE_WINDOW', self.safe_destroy)
        from utils.rendering import enable_dark_titlebar
        self.bind('<Map>', lambda e: enable_dark_titlebar(self)
                  if e.widget is self else None, add='+')
        self._build_ui()
        self._refresh_controls()
        # Располагаем на рабочем мониторе (Windows) или virtual root (Linux).
        place_centered(self, master, width=1200, height=820, center_on_monitor=True)
        if hasattr(master, 'register_child_window'):
            master.register_child_window(self)
        self.deiconify()
        self.after_idle(self._activate_modal)

    def _build_ui(self):
        # Три части окна: тихая верхняя панель, рабочее поле, нижняя панель
        # режимов, которую можно скрыть одной кнопкой.
        self.main_frame = ctk.CTkFrame(self, fg_color='#25272B', corner_radius=0)
        self.main_frame.pack(fill='both', expand=True, padx=8, pady=8)
        self.button_frame = ctk.CTkFrame(self.main_frame, fg_color='transparent')
        self.button_frame.pack(fill='x', padx=6, pady=(4, 8))

        self.btn_open = ctk.CTkButton(
            self.button_frame, text='Загрузить', command=self.open_image,
            width=154, height=36, fg_color=NEUTRAL, hover_color=NEUTRAL_HOVER,
            text_color=TEXT, corner_radius=8,
        )
        self.btn_open.pack(side='left', padx=(0, 8))
        self.btn_crop = ctk.CTkButton(
            self.button_frame, text='Обрезать', command=self.toggle_crop_options,
            width=154, height=36, fg_color=NEUTRAL, hover_color=NEUTRAL_HOVER,
            text_color=TEXT, corner_radius=8,
        )
        self.btn_crop.pack(side='left')
        self.btn_save = ctk.CTkButton(
            self.button_frame, text='Продолжить', command=self.save_and_exit,
            width=160, height=36, fg_color=BLUE, hover_color=BLUE_HOVER,
            text_color=TEXT, corner_radius=8,
        )
        self.btn_save.pack(side='right')

        # Ровное поле без белой линии по границе изображения. Панель
        # прокрутки показывается только если изображение не помещается.
        self.canvas_frame = ctk.CTkFrame(
            self.main_frame, fg_color='#202225', corner_radius=8,
            border_width=1, border_color='#3B3E44',
        )
        self.canvas_frame.pack(fill='both', expand=True, padx=6)
        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(
            self.canvas_frame, bg='#202225', highlightthickness=0,
            borderwidth=0, cursor='arrow',
        )
        # tk.Canvas не поддерживает скругление углов. Одинаковый фон и
        # отступ внутрь CTkFrame сохраняют видимую скруглённую границу,
        # не позволяя прямоугольному Canvas перекрывать её углы.
        self.canvas.grid(row=0, column=0, sticky='nsew', padx=9, pady=9)
        self.scroll_y = ctk.CTkScrollbar(
            self.canvas_frame, orientation='vertical', command=self.canvas.yview,
            width=12,
        )
        self.scroll_x = ctk.CTkScrollbar(
            self.canvas_frame, orientation='horizontal', command=self.canvas.xview,
            height=12,
        )
        self.canvas.configure(
            yscrollcommand=self.scroll_y.set, xscrollcommand=self.scroll_x.set,
        )
        self.canvas.bind('<ButtonPress-1>', self.on_button_press)
        self.canvas.bind('<B1-Motion>', self.on_move_press)
        self.canvas.bind('<ButtonRelease-1>', self.on_button_release)
        self.canvas.bind('<Button-3>', self._undo_polygon_point)
        self.canvas.bind('<MouseWheel>', self.on_mousewheel)
        self.canvas.bind('<Button-4>', self.on_mousewheel)
        self.canvas.bind('<Button-5>', self.on_mousewheel)
        self.canvas.bind('<Configure>', self._on_canvas_resize)

        # Вариант А: нижняя компактная панель без вложенных светящихся карточек.
        self.crop_options_frame = ctk.CTkFrame(
            self.main_frame, fg_color='#2B2E33', corner_radius=8,
        )
        ctk.CTkLabel(
            self.crop_options_frame, text='Тип обрезки:', text_color='#D5DAE1',
        ).pack(side='left', padx=(12, 4))
        # Текстовый знак вопроса вместо мелкого контурного кружка: его
        # граница давала заметные пиксельные артефакты при масштабировании DPI.
        self.help_button = ctk.CTkButton(
            self.crop_options_frame, text='?', width=28, height=28,
            corner_radius=6, border_width=0,
            fg_color=NEUTRAL, hover_color=NEUTRAL_HOVER,
            text_color='#D8E6F5', font=ctk.CTkFont(size=15, weight='bold'),
            command=self.toggle_help,
        )
        self.help_button.pack(side='left', padx=(0, 10))
        self.btn_rect_crop = ctk.CTkButton(
            self.crop_options_frame, text='Прямоугольная',
            command=self.set_rectangle_mode, width=195, height=34, corner_radius=7,
        )
        self.btn_rect_crop.pack(side='left', padx=(0, 7), pady=7)
        self.btn_poly_crop = ctk.CTkButton(
            self.crop_options_frame, text='Полигональная',
            command=self.set_polygon_mode, width=147, height=34, corner_radius=7,
        )
        self.btn_poly_crop.pack(side='left', padx=(0, 7), pady=7)
        self.btn_reset = ctk.CTkButton(
            self.crop_options_frame, text='Сбросить',
            command=self.reset_image, width=102, height=34,
            corner_radius=7, fg_color=NEUTRAL, hover_color=NEUTRAL_HOVER,
            text_color=TEXT,
        )
        self.btn_reset.pack(side='left', pady=7)

        # Только для полигона: подтверждение контура и его очистка.
        self.polygon_control_frame = ctk.CTkFrame(
            self.main_frame, fg_color='transparent',
        )
        self.btn_poly_confirm = ctk.CTkButton(
            self.polygon_control_frame, text='Применить контур',
            command=self.apply_polygon_crop, width=150, height=32,
            corner_radius=7, fg_color=BLUE, hover_color=BLUE_HOVER,
        )
        self.btn_poly_confirm.pack(side='left', padx=(0, 6))
        self.btn_poly_reset = ctk.CTkButton(
            self.polygon_control_frame, text='Очистить контур',
            command=self.reset_polygon, width=140, height=32,
            corner_radius=7, fg_color=NEUTRAL, hover_color=NEUTRAL_HOVER,
        )
        self.btn_poly_reset.pack(side='left', padx=(0, 6))
        self.btn_poly_cancel = ctk.CTkButton(
            self.polygon_control_frame, text='Отмена', command=self.cancel_polygon_mode,
            width=86, height=32, corner_radius=7,
            fg_color=NEUTRAL, hover_color=NEUTRAL_HOVER,
        )
        self.btn_poly_cancel.pack(side='left')

        # Текст появляется в одной строке, не занимает отдельную панель.
        self.status_label = ctk.CTkLabel(
            self.main_frame, text='Загрузите изображение для начала работы',
            anchor='w', text_color='#A9B0B9', height=22,
        )
        self.status_label.pack(fill='x', padx=12, pady=(5, 1))
        # Создаём один раз и выводим поверх изображения только по запросу.
        self.help_overlay = CropperHelpOverlay(self.main_frame, on_close=self.hide_help)
        self.help_overlay.place_forget()
        self.bind('<ButtonPress-1>', self._dismiss_help_on_click, add='+')
        self.bind('<Escape>', self._on_escape)
        self.bind('<Control-Key-0>', lambda e: self.fit_image())
        self.bind('<Control-KeyPress-z>', self.undo, add='+')
        self.bind('<Control-KeyPress-Z>', self.undo, add='+')

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
        self._resize_after = self.after(75, self._redraw_after_resize)

    def _redraw_after_resize(self):
        self._resize_after = None
        if not self._closed:
            self.update_canvas()

    def _refresh_controls(self):
        loaded = self.image is not None
        self.btn_crop.configure(state='normal' if loaded else 'disabled')
        self.btn_save.configure(state='normal' if loaded else 'disabled')
        self.btn_reset.configure(state='normal' if loaded else 'disabled')
        self.btn_poly_confirm.configure(state='normal' if len(self.polygon_points) >= 3 else 'disabled')
        self.btn_poly_reset.configure(state='normal' if self.polygon_points else 'disabled')
        self.btn_crop.configure(
            fg_color=BLUE if self.crop_active else NEUTRAL,
            hover_color=BLUE_HOVER if self.crop_active else NEUTRAL_HOVER,
        )
        self._style_mode_button(self.btn_rect_crop, self.crop_active and not self.polygon_mode,
                                'Прямоугольная')
        self._style_mode_button(self.btn_poly_crop, self.crop_active and self.polygon_mode,
                                'Полигональная')
        self.canvas.configure(cursor='crosshair' if self.crop_active else 'arrow')
        if loaded:
            filename = Path(self.original_file_path).name
            hint = (f'{filename}  ·  {self.image.width} × {self.image.height} пикс.  ·  '
                    'Масштаб — Ctrl + колесо мыши')
        else:
            hint = 'Файл не загружен  ·  Масштаб — Ctrl + колесо мыши'
        self.status_label.configure(text=hint)

    def toggle_help(self):
        if self.help_overlay.winfo_manager():
            self.hide_help()
            return
        self.main_frame.update_idletasks()
        width = min(470, max(260, self.main_frame.winfo_width() - 16))
        height = min(385, max(230, self.main_frame.winfo_height() - 32))
        # Позиционируем над значком, сохраняя карточку внутри окна.
        x = self.help_button.winfo_rootx() - self.main_frame.winfo_rootx() - 46
        y = self.help_button.winfo_rooty() - self.main_frame.winfo_rooty() - height - 10
        x = max(8, min(x, self.main_frame.winfo_width() - width - 8))
        y = max(8, min(y, self.main_frame.winfo_height() - height - 8))
        # CustomTkinter запрещает width/height в .place(): размеры задаются
        # через .configure(), а позиция — отдельно.
        self.help_overlay.configure(width=width, height=height)
        self.help_overlay.place(x=x, y=y)
        self.help_overlay.lift()

    def hide_help(self):
        self.help_overlay.place_forget()

    def _dismiss_help_on_click(self, event):
        if not self.help_overlay.winfo_manager():
            return
        widget = event.widget
        while widget is not None:
            if widget is self.help_overlay or widget is self.help_button:
                return
            widget = getattr(widget, 'master', None)
        self.hide_help()

    def _remember(self, *, clear_rectangle=False):
        if self.image is None:
            return
        self._undo_stack.append(_UndoState(
            image=self.image,
            fit_view=self._fit_view,
            scale=self._scale,
            polygon_mode=self.polygon_mode,
            crop_active=self.crop_active,
            polygon_points=tuple(self.polygon_points),
            rect_coords=None if clear_rectangle else self.rect_coords,
            xview=self.canvas.xview()[0],
            yview=self.canvas.yview()[0],
        ))

    def undo(self, _event=None):
        """Отменяет последнее изменение масштаба, обрезки или точек."""
        if not self._undo_stack or self.image is None:
            return 'break'
        state = self._undo_stack.pop()
        self.image = state.image
        self._fit_view = state.fit_view
        self._scale = state.scale
        self.polygon_mode = state.polygon_mode
        self.crop_active = state.crop_active
        self.polygon_points = list(state.polygon_points)
        self.rect_coords = state.rect_coords
        if self.crop_active:
            if not self.crop_options_frame.winfo_manager():
                self.crop_options_frame.pack(fill='x', padx=6, pady=(7, 0), before=self.status_label)
        else:
            self.crop_options_frame.pack_forget()
        if self.crop_active and self.polygon_mode:
            if not self.polygon_control_frame.winfo_manager():
                self.polygon_control_frame.pack(fill='x', padx=6, pady=(5, 0), before=self.status_label)
        else:
            self.polygon_control_frame.pack_forget()
        self.hide_help()
        self.update_canvas()
        self.canvas.xview_moveto(state.xview)
        self.canvas.yview_moveto(state.yview)
        self._refresh_controls()
        return 'break'

    @staticmethod
    def _style_mode_button(button, selected, label):
        button.configure(
            text=label,
            fg_color=BLUE if selected else NEUTRAL,
            hover_color=BLUE_HOVER if selected else NEUTRAL_HOVER,
            border_width=0 if selected else 1,
            border_color=BLUE if selected else BORDER,
            text_color=TEXT,
        )

    def toggle_crop_options(self):
        if self.image is None:
            return
        if self.crop_active:
            if self.polygon_points or self.rect_coords:
                self._remember()
            self.hide_help()
            self.crop_active = False
            self.polygon_mode = False
            self.polygon_points.clear()
            self.rect_coords = None
            self._redraw_selection()
            self.crop_options_frame.pack_forget()
            self.polygon_control_frame.pack_forget()
        else:
            self.crop_active = True
            self.polygon_mode = False
            self.crop_options_frame.pack(fill='x', padx=6, pady=(7, 0), before=self.status_label)
        self._refresh_controls()

    def set_rectangle_mode(self):
        if self.image is None:
            return
        self.crop_active = True
        if self.polygon_points or self.rect_coords:
            self._remember()
        self.polygon_mode = False
        self.polygon_points.clear()
        self.rect_coords = None
        self._redraw_selection()
        self.polygon_control_frame.pack_forget()
        self._refresh_controls()

    def set_polygon_mode(self):
        if self.image is None:
            return
        self.crop_active = True
        if self.polygon_points or self.rect_coords:
            self._remember()
        self.polygon_mode = True
        self.polygon_points.clear()
        self.rect_coords = None
        self._redraw_selection()
        if not self.polygon_control_frame.winfo_manager():
            self.polygon_control_frame.pack(fill='x', padx=6, pady=(5, 0), before=self.status_label)
        self._refresh_controls()

    def cancel_polygon_mode(self):
        self.set_rectangle_mode()

    def reset_polygon(self):
        if self.polygon_points:
            self._remember()
        self.polygon_points.clear()
        self._redraw_selection()
        self._refresh_controls()

    def _undo_polygon_point(self, _event=None):
        if self.polygon_mode and self.polygon_points:
            self._remember()
            self.polygon_points.pop()
            self._redraw_selection()
            self._refresh_controls()
        return 'break'

    def open_image(self):
        types = [('Изображения', '*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp'),
                 ('Все файлы', '*.*')]
        path = filedialog.askopenfilename(parent=self, title='Загрузить изображение', filetypes=types)
        if not path:
            return
        try:
            with Image.open(path) as src:
                loaded = ImageOps.exif_transpose(src).convert('RGB')
                loaded.load()
        except Exception as exc:
            messagebox.showerror('Ошибка загрузки', f'Не удалось открыть изображение:\n{exc}', parent=self)
            return
        # Новая загрузка начинает новую историю — не смешиваем разные файлы.
        self._undo_stack.clear()
        self.hide_help()
        self.original_image = loaded
        self.image = loaded.copy()
        self.original_file_path = path
        self.polygon_points.clear()
        self.rect_coords = None
        self.polygon_mode = False
        self.crop_active = False
        self.crop_options_frame.pack_forget()
        self.polygon_control_frame.pack_forget()
        self._fit_view = True
        self.update_canvas()
        self._refresh_controls()

    def _display_scale(self):
        if self.image is None:
            return 1.0
        if self._fit_view:
            vw = max(1, self.canvas.winfo_width() - 20)
            vh = max(1, self.canvas.winfo_height() - 20)
            return min(1.0, vw / self.image.width, vh / self.image.height)
        return self._scale

    def update_canvas(self):
        self.canvas.delete('all')
        self._image_item = None
        self.rect_id = None
        if self.image is None:
            self.canvas.configure(scrollregion=(0, 0, 1, 1))
            self.tk_image = None
            self._show_scrollbars(False, False)
            return
        self._scale = self._display_scale()
        display_w = max(1, round(self.image.width * self._scale))
        display_h = max(1, round(self.image.height * self._scale))
        visible_w = max(1, self.canvas.winfo_width())
        visible_h = max(1, self.canvas.winfo_height())
        self._image_origin = (max(0, (visible_w - display_w) / 2),
                              max(0, (visible_h - display_h) / 2))
        # Рендерим предпросмотр, источник неизменён; координаты пересчитываем.
        preview = self.image.resize((display_w, display_h), Image.Resampling.BILINEAR)
        self.tk_image = ImageTk.PhotoImage(preview, master=self)
        self._image_item = self.canvas.create_image(
            *self._image_origin, anchor='nw', image=self.tk_image, tags='image',
        )
        self.canvas.configure(scrollregion=(0, 0,
                              max(visible_w, display_w), max(visible_h, display_h)))
        self._show_scrollbars(display_w > visible_w, display_h > visible_h)
        self._redraw_selection()

    def _show_scrollbars(self, horizontal, vertical):
        if horizontal:
            self.scroll_x.grid(row=1, column=0, sticky='ew', padx=9, pady=(0, 8))
        else:
            self.scroll_x.grid_remove()
            self.canvas.xview_moveto(0)
        if vertical:
            self.scroll_y.grid(row=0, column=1, sticky='ns', padx=(0, 8), pady=9)
        else:
            self.scroll_y.grid_remove()
            self.canvas.yview_moveto(0)

    def _to_image(self, event, *, clamp=False):
        if self.image is None:
            return None
        x = (self.canvas.canvasx(event.x) - self._image_origin[0]) / self._scale
        y = (self.canvas.canvasy(event.y) - self._image_origin[1]) / self._scale
        if clamp:
            return (min(max(x, 0), self.image.width),
                    min(max(y, 0), self.image.height))
        if not (0 <= x < self.image.width and 0 <= y < self.image.height):
            return None
        return x, y

    def _to_canvas(self, x, y):
        return self._image_origin[0] + x * self._scale, self._image_origin[1] + y * self._scale

    def _redraw_selection(self):
        self.canvas.delete('crop_overlay')
        self.rect_id = None
        self.polygon_lines = []
        if self.rect_coords and not self.polygon_mode:
            x1, y1, x2, y2 = self.rect_coords
            self.rect_id = self.canvas.create_rectangle(
                *self._to_canvas(x1, y1), *self._to_canvas(x2, y2),
                outline='#F2C879', width=2, dash=(5, 4), tags='crop_overlay',
            )
        if self.polygon_points and self.polygon_mode:
            pts = [self._to_canvas(x, y) for x, y in self.polygon_points]
            for n, (x, y) in enumerate(pts):
                self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3,
                                        fill='#F2C879', outline='#242628', tags='crop_overlay')
                if n:
                    line = self.canvas.create_line(*pts[n - 1], x, y,
                                                   fill='#F2C879', width=2, tags='crop_overlay')
                    self.polygon_lines.append(line)
            if len(pts) >= 3:
                line = self.canvas.create_line(*pts[-1], *pts[0], fill='#F2C879',
                                               dash=(4, 4), width=1, tags='crop_overlay')
                self.polygon_lines.append(line)

    def on_button_press(self, event):
        if self.image is None or not self.crop_active:
            return
        pt = self._to_image(event)
        if pt is None:
            return
        if self.polygon_mode:
            self._remember()
            self.polygon_points.append(pt)
            self._redraw_selection()
            self._refresh_controls()
        else:
            self.rect_coords = (pt[0], pt[1], pt[0], pt[1])
            self._redraw_selection()

    def on_move_press(self, event):
        if self.image is None or not self.crop_active or self.polygon_mode or self.rect_coords is None:
            return
        startx, starty, _, _ = self.rect_coords
        finish = self._to_image(event, clamp=True)
        self.rect_coords = (startx, starty, finish[0], finish[1])
        self._redraw_selection()

    def on_button_release(self, event):
        if self.image is None or not self.crop_active or self.polygon_mode or self.rect_coords is None:
            return
        self.on_move_press(event)
        x1, y1, x2, y2 = self.rect_coords
        # Минимальный размер — 10 физических пикселей исходника, как в старом кроппере.
        if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
            self.apply_rectangle_crop()
        else:
            self.reset_selection()

    def reset_selection(self):
        self.rect_coords = None
        self._redraw_selection()

    def apply_rectangle_crop(self):
        if self.image is None or self.rect_coords is None:
            return
        try:
            new_image = rectangle_crop(self.image, self.rect_coords)
        except (ValueError, OSError) as exc:
            messagebox.showerror('Ошибка обрезки', str(exc), parent=self)
            return
        self._remember(clear_rectangle=True)
        self.image = new_image
        self.rect_coords = None
        self._fit_view = True
        self.update_canvas()
        self._refresh_controls()

    def apply_polygon_crop(self):
        if self.image is None or len(self.polygon_points) < 3:
            return
        try:
            new_image = polygon_crop(self.image, self.polygon_points)
        except (ValueError, OSError) as exc:
            messagebox.showerror('Ошибка обрезки', str(exc), parent=self)
            return
        self._remember()
        self.image = new_image
        self.polygon_points.clear()
        self.polygon_mode = False
        self._fit_view = True
        self.polygon_control_frame.pack_forget()
        self.update_canvas()
        self._refresh_controls()

    def reset_image(self):
        if self.original_image is None:
            return
        self._remember()
        self.image = self.original_image.copy()
        self.polygon_points.clear()
        self.rect_coords = None
        self.polygon_mode = False
        self.polygon_control_frame.pack_forget()
        self._fit_view = True
        self.update_canvas()
        self._refresh_controls()

    def fit_image(self):
        if self.image is None:
            return 'break'
        if not self._fit_view:
            self._remember()
            self._fit_view = True
            self.update_canvas()
        return 'break'

    def on_mousewheel(self, event):
        if self.image is None:
            return
        direction = -1 if getattr(event, 'num', None) == 4 or getattr(event, 'delta', 0) > 0 else 1
        if getattr(event, 'state', 0) & 0x0004:  # Ctrl + колесо
            cap = math.sqrt(self.MAX_PREVIEW_PIXELS / (self.image.width * self.image.height))
            next_scale = max(0.05, min(4.0, cap, self._scale * (1.25 if direction < 0 else 0.8)))
            if not math.isclose(next_scale, self._scale, rel_tol=1e-12, abs_tol=1e-12):
                self._remember()
                self._fit_view = False
                self._scale = next_scale
                self.update_canvas()
        elif getattr(event, 'state', 0) & 0x0001:
            self.canvas.xview_scroll(direction * 3, 'units')
        else:
            self.canvas.yview_scroll(direction * 3, 'units')
        return 'break'

    def _on_escape(self, _event=None):
        if self.help_overlay.winfo_manager():
            self.hide_help()
        elif self.polygon_mode:
            self.cancel_polygon_mode()
        elif self.rect_coords is not None:
            self.reset_selection()
        elif self.crop_active:
            self.toggle_crop_options()
        else:
            self.safe_destroy()
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
