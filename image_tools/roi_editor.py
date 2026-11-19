"""Modal ROI editor: freehand lasso, brush/eraser and independent region cards.

All geometry is stored in ORIGINAL image pixel coordinates.  Canvas zoom,
scrolling and presentation-only style properties never modify scientific masks.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import colorchooser

import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk
from scipy.ndimage import binary_erosion

from image_tools.roi_model import (
    ROICollection,
    polygon_mask,
    draw_brush,
    regions_touched_by_stroke,
)
from utils.theme import AppTheme


class ROIEditor(ctk.CTkToplevel):
    _deactivate_windows_window_header_manipulation = True

    def __init__(self, master, image, regions=None):
        super().__init__(master)
        from utils.rendering import enable_dark_titlebar
        self.bind("<Map>", lambda e: enable_dark_titlebar(self) if e.widget is self else None, add="+")
        self.title("Выделение областей ROI")
        self.geometry("1180x760")
        self.minsize(850, 530)
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.image = Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB")
        self.regions = regions.copy() if regions is not None else ROICollection((self.image.height, self.image.width))
        if self.regions.shape != (self.image.height, self.image.width):
            raise ValueError("ROI не соответствует загруженному изображению")
        self.result = None
        self.mode = "lasso"
        self.active_id = self.regions.regions[-1].id if self.regions.regions else None
        self.brush_size = 14
        self.zoom = 1.0
        self._fitted = False
        self._drawing = []
        self._photo = None
        self._image_id = None
        self._image_x = 0.0
        self._image_y = 0.0
        self._cached_composite = None
        self._undo = []
        # Tk <Configure> can fire while widgets are still being constructed.
        # Ignore it until the status label and the entire editor are ready.
        self._ui_ready = False
        self._last_erase_point = None
        self._erase_undo_taken = False
        self._preview_id = None
        self._draft_id = None
        self._draft_coords = []
        self._display_source = None
        self._display_size = None
        self._region_refresh_job = None

        outer = ctk.CTkFrame(self, fg_color="#25272A", corner_radius=0)
        outer.pack(fill="both", expand=True, padx=8, pady=8)
        toolbar = ctk.CTkFrame(outer, fg_color="#303339", corner_radius=9)
        toolbar.pack(fill="x", padx=6, pady=(6, 5))
        self.mode_buttons = {}
        for mode, text in (("lasso", "Выделение"), ("brush", "Кисть"), ("eraser", "Ластик")):
            btn = ctk.CTkButton(toolbar, text=text, width=135,
                                command=lambda m=mode: self._set_mode(m))
            btn.pack(side="left", padx=4, pady=7)
            self.mode_buttons[mode] = btn
        ctk.CTkButton(toolbar, text="Отменить", width=95, fg_color="#44474D",
                      command=self._undo_action).pack(side="left", padx=(10, 4), pady=7)
        ctk.CTkButton(toolbar, text="Сбросить масштаб", width=140,
                      fg_color="#44474D", command=self._fit).pack(side="left", padx=4, pady=7)
        ctk.CTkButton(toolbar, text="Применить", width=130,
                      command=self._accept).pack(side="right", padx=7, pady=7)
        ctk.CTkButton(toolbar, text="Отмена", width=75, fg_color="#44474D",
                      command=self._cancel).pack(side="right", padx=3, pady=7)

        work = ctk.CTkFrame(outer, fg_color="transparent")
        work.pack(fill="both", expand=True, padx=6)
        work.grid_columnconfigure(0, weight=1)
        work.grid_rowconfigure(0, weight=1)
        drawing_frame = ctk.CTkFrame(work, fg_color="#1F2123", corner_radius=8)
        drawing_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        self.canvas = tk.Canvas(drawing_frame, bg="#1F2123", borderwidth=0,
                                highlightthickness=0, cursor="crosshair")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        drawing_frame.grid_rowconfigure(0, weight=1)
        drawing_frame.grid_columnconfigure(0, weight=1)
        vs = ctk.CTkScrollbar(drawing_frame, command=self.canvas.yview)
        hs = ctk.CTkScrollbar(drawing_frame, orientation="horizontal", command=self.canvas.xview)
        vs.grid(row=0, column=1, sticky="ns")
        hs.grid(row=1, column=0, sticky="ew")
        self.canvas.configure(xscrollcommand=hs.set, yscrollcommand=vs.set)
        self.canvas.bind("<Configure>", self._on_configure)
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.canvas.bind("<Control-MouseWheel>", self._zoom_wheel)
        self.canvas.bind("<MouseWheel>", self._scroll_wheel)
        self.canvas.bind("<ButtonPress-3>", lambda e: self.canvas.scan_mark(e.x, e.y))
        self.canvas.bind("<B3-Motion>", lambda e: self.canvas.scan_dragto(e.x, e.y, gain=1))
        self.canvas.bind("<Motion>", self._move_preview)
        self.canvas.bind("<Leave>", self._hide_preview)
        self.canvas.bind("<Button-4>", lambda e: self._zoom_step(1.15))
        self.canvas.bind("<Button-5>", lambda e: self._zoom_step(1 / 1.15))

        right = ctk.CTkFrame(work, width=290, fg_color="#303339", corner_radius=9)
        right.grid(row=0, column=1, sticky="ns")
        right.grid_propagate(False)
        self.regions_heading = ctk.CTkLabel(
            right, text="Выделенные области: 0",
            font=AppTheme.section_title_font(), anchor="w")
        self.regions_heading.pack(fill="x", padx=13, pady=(13, 8))
        self.card_container = ctk.CTkScrollableFrame(right, fg_color="transparent")
        self.card_container.pack(fill="both", expand=True, padx=7, pady=(0, 8))
        brush_controls = ctk.CTkFrame(right, fg_color="#393C41", corner_radius=7)
        brush_controls.pack(fill="x", padx=8, pady=(0, 8))
        self.brush_label = ctk.CTkLabel(brush_controls, text="Диаметр кисти / ластика · 14 пикс.", anchor="w")
        self.brush_label.pack(fill="x", padx=9, pady=(4, 0))
        brush_slider = ctk.CTkSlider(brush_controls, from_=2, to=100, number_of_steps=98,
                                     command=self._set_brush_size)
        brush_slider.pack(fill="x", padx=9, pady=(0, 8))
        brush_slider.set(self.brush_size)

        self.status = ctk.CTkLabel(outer, anchor="w", text_color=AppTheme.TEXT_SECONDARY,
                                   text="ЛКМ — выделение · Ctrl+колесо — масштаб · ПКМ — перемещение · Ctrl+Z — отмена")
        self.status.pack(fill="x", padx=12, pady=(2, 4))
        self.bind("<Control-z>", self._undo_shortcut, add="+")
        self.bind("<Control-Z>", self._undo_shortcut, add="+")
        self._ui_ready = True
        self._refresh_cards()
        self._set_mode("lasso")
        self.after_idle(self._fit)
        self.grab_set()
        self.focus_set()

    def _set_mode(self, mode):
        self.mode = mode
        for key, btn in self.mode_buttons.items():
            btn.configure(fg_color=AppTheme.PRIMARY if mode == key else "#454950")
        # The preview ring indicates brush width; crosshair is portable on Windows.
        self.canvas.configure(cursor="crosshair")
        self._hide_preview()
        if mode == "brush":
            self.status.configure(text="Кисть добавляет пиксели в область, выбранную кнопкой «Редактировать». "
                                       "Ctrl+Z — отмена")
        elif mode == "eraser":
            self.status.configure(text="Ластик удаляет всю область при касании её контура. "
                                       "ПКМ — перемещение · Ctrl+Z — отмена")
        else:
            self.status.configure(text="ЛКМ — выделение новой области · Ctrl+колесо — масштаб · "
                                       "ПКМ — перемещение · Ctrl+Z — отмена")

    def _set_brush_size(self, value):
        self.brush_size = int(round(value))
        self.brush_label.configure(text=f"Диаметр кисти / ластика · {self.brush_size} пикс.")

    def _on_configure(self, _event):
        if not self._ui_ready:
            return
        if not self._fitted:
            self._fit()
        else:
            self._redraw()

    def _fit(self):
        cw = max(100, self.canvas.winfo_width() - 25)
        ch = max(100, self.canvas.winfo_height() - 25)
        self.zoom = min(1.0, cw / self.image.width, ch / self.image.height)
        self._fitted = True
        self._redraw()
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

    def _zoom_wheel(self, event):
        self._zoom_step(1.15 if event.delta > 0 else 1 / 1.15)
        return "break"

    def _scroll_wheel(self, event):
        self.canvas.yview_scroll(-int(event.delta / 120), "units")
        return "break"

    def _zoom_step(self, factor):
        self.zoom = float(np.clip(self.zoom * factor, 0.1, 8.0))
        self._redraw()
        return "break"

    def _image_point(self, event):
        x = (self.canvas.canvasx(event.x) - self._image_x) / self.zoom
        y = (self.canvas.canvasy(event.y) - self._image_y) / self.zoom
        x = float(np.clip(x, 0, self.image.width - 1))
        y = float(np.clip(y, 0, self.image.height - 1))
        return x, y

    def _inside_image(self, event):
        x = self.canvas.canvasx(event.x) - self._image_x
        y = self.canvas.canvasy(event.y) - self._image_y
        return 0 <= x < self.image.width * self.zoom and 0 <= y < self.image.height * self.zoom

    def _move_preview(self, event):
        if self.mode == "lasso" or not self._inside_image(event):
            self._hide_preview()
            return
        x, y = self._image_point(event)
        cx = self._image_x + x * self.zoom
        cy = self._image_y + y * self.zoom
        radius = max(1.0, self.brush_size * self.zoom / 2)
        coords = (cx - radius, cy - radius, cx + radius, cy + radius)
        if self._preview_id is None:
            self._preview_id = self.canvas.create_oval(
                *coords, outline="#F0B55A", width=1, tags="brush_preview")
            self.canvas.tag_raise(self._preview_id)
        else:
            # Moving the preview must not change the canvas stacking order.
            # Reordering items on every mouse event is expensive in Tk.
            self.canvas.coords(self._preview_id, *coords)

    def _hide_preview(self, _event=None):
        if self._preview_id is not None:
            self.canvas.delete(self._preview_id)
            self._preview_id = None

    def _press(self, event):
        if not self._inside_image(event):
            return
        p = self._image_point(event)
        if self.mode == "eraser":
            self._last_erase_point = p
            self._erase_undo_taken = False
            self._erase_between(p, p)
            return
        self.canvas.delete("draft")
        self._draft_id = None
        self._draft_coords = []
        self._drawing = [p]
        if self.mode == "brush" and self.active_id is None:
            self.status.configure(text="Создайте область инструментом «Выделение», "
                                       "затем выберите её кнопкой «Редактировать»")
            self._drawing.clear()

    def _drag(self, event):
        if self.mode == "eraser":
            if self._last_erase_point is None or not self._inside_image(event):
                return
            p = self._image_point(event)
            self._erase_between(self._last_erase_point, p)
            self._last_erase_point = p
            self._move_preview(event)
            return
        if not self._drawing:
            return
        p = self._image_point(event)
        if abs(p[0] - self._drawing[-1][0]) + abs(p[1] - self._drawing[-1][1]) < 0.7:
            return
        last = self._drawing[-1]
        self._drawing.append(p)
        xy = (self._image_x + p[0] * self.zoom,
              self._image_y + p[1] * self.zoom)
        if self._draft_id is None or len(self._draft_coords) >= 128:
            # Keep short polyline segments, not hundreds of Canvas items nor
            # one ever-growing Canvas.coords() payload (quadratic work).
            self._draft_coords = [self._image_x + last[0] * self.zoom,
                                  self._image_y + last[1] * self.zoom]
            self._draft_coords.extend(xy)
            line_width = (max(1, round(self.brush_size * self.zoom))
                          if self.mode == "brush" else 2)
            self._draft_id = self.canvas.create_line(
                *self._draft_coords, fill="#F0B55A", width=line_width,
                tags="draft", capstyle="round")
            if self._preview_id is not None:
                self.canvas.tag_raise(self._preview_id)
        else:
            self._draft_coords.extend(xy)
            self.canvas.coords(self._draft_id, *self._draft_coords)

    def _erase_between(self, start, end):
        touched = regions_touched_by_stroke(
            self.regions, [start, end], self.brush_size, visible_only=True)
        if not touched:
            return
        if not self._erase_undo_taken:
            self._remember_undo()
            self._erase_undo_taken = True
        for region_id in touched:
            self.regions.remove(region_id)
        if self.regions.get(self.active_id) is None:
            self.active_id = self.regions.regions[-1].id if self.regions.regions else None
        self._cached_composite = None
        self._schedule_region_refresh()

    def _schedule_region_refresh(self):
        """Coalesce expensive CTk card reconstruction and PhotoImage updates."""
        if not getattr(self, "_ui_ready", False):
            # Headless/unit-test callers have no live Tk event loop.
            self._refresh_cards()
            self._redraw()
            return
        if self._region_refresh_job is None:
            self._region_refresh_job = self.after(35, self._commit_region_refresh)

    def _commit_region_refresh(self):
        self._region_refresh_job = None
        if not self._ui_ready:
            return
        self._refresh_cards()
        self._redraw()

    def _flush_region_refresh(self):
        job = self._region_refresh_job
        if job is not None:
            self.after_cancel(job)
            self._commit_region_refresh()

    def _release(self, _event):
        if self.mode == "eraser":
            self._last_erase_point = None
            self._erase_undo_taken = False
            self._flush_region_refresh()
            return
        if not self._drawing:
            return
        points = self._drawing
        self._drawing = []
        self.canvas.delete("draft")
        self._draft_id = None
        self._draft_coords = []
        if self.mode == "lasso":
            mask = polygon_mask(self.regions.shape, points)
            if not mask.any():
                self.status.configure(text="Обведите замкнутый контур ненулевой площади")
                return
            self._remember_undo()
            region = self.regions.new_region(mask)
            self.active_id = region.id
        else:
            region = self.regions.get(self.active_id)
            if region is None:
                return
            self._remember_undo()
            region.mask = draw_brush(region.mask, points, self.brush_size)
        self._cached_composite = None
        self._refresh_cards()
        self._redraw()

    def _remember_undo(self):
        self._undo.append(self.regions.copy())
        if len(self._undo) > 10:
            self._undo.pop(0)

    def _undo_shortcut(self, _event):
        self._undo_action()
        return "break"

    def _undo_action(self):
        if getattr(self, "_region_refresh_job", None) is not None:
            self._flush_region_refresh()
        if self._undo:
            self.regions = self._undo.pop()
            self.active_id = self.regions.regions[-1].id if self.regions.regions else None
            self._cached_composite = None
            self._refresh_cards()
            self._redraw()

    def _choose_color(self, region_id):
        region = self.regions.get(region_id)
        if region is None:
            return
        chosen = colorchooser.askcolor(color=region.color, parent=self)[1]
        if chosen and chosen != region.color:
            self._remember_undo()
            region.color = chosen
            self._cached_composite = None
            self._refresh_cards()
            self._redraw()

    def _opacity(self, region_id, value):
        region = self.regions.get(region_id)
        if region is not None:
            region.opacity = float(value)
            self._cached_composite = None
            self._redraw()

    def _visible(self, region_id, value):
        region = self.regions.get(region_id)
        if region is not None and region.visible != bool(value):
            self._remember_undo()
            region.visible = bool(value)
            self._cached_composite = None
            self._redraw()

    def _delete(self, region_id):
        if self.regions.get(region_id) is None:
            return
        self._remember_undo()
        self.regions.remove(region_id)
        self.active_id = self.regions.regions[-1].id if self.regions.regions else None
        self._cached_composite = None
        self._refresh_cards()
        self._redraw()

    def _refresh_cards(self):
        for child in self.card_container.winfo_children():
            child.destroy()
        self.regions_heading.configure(text=f"Выделенные области: {len(self.regions)}")
        for region in self.regions.regions:
            card = ctk.CTkFrame(self.card_container, fg_color="#3B3E44", corner_radius=7)
            card.pack(fill="x", pady=4)

            # Top row: compact checkbox + editable name + explicit delete button.
            # Using grid prevents the empty CTkCheckBox text from inserting a gap.
            top = ctk.CTkFrame(card, fg_color="transparent")
            top.pack(fill="x", padx=8, pady=(7, 0))
            top.grid_columnconfigure(1, weight=1)
            top.grid_columnconfigure(2, minsize=28)
            visible = ctk.BooleanVar(value=region.visible)
            ctk.CTkCheckBox(
                top, text="", variable=visible, width=21,
                checkbox_width=20, checkbox_height=20,
                command=lambda i=region.id, v=visible: self._visible(i, v.get()),
            ).grid(row=0, column=0, sticky="w", padx=(0, 2))
            entry = ctk.CTkEntry(top, height=27, font=AppTheme.caption_font())
            entry.insert(0, region.name)
            entry.grid(row=0, column=1, sticky="ew", padx=(0, 5))
            entry.bind("<FocusOut>", lambda _e, i=region.id, w=entry: self._rename(i, w.get()))
            ctk.CTkButton(
                top, text="×", width=28, height=28, corner_radius=6,
                fg_color="#2B2D31", hover_color="#555960",
                text_color="#F0F1F2", font=("Segoe UI", 16, "bold"),
                command=lambda i=region.id: self._delete(i),
            ).grid(row=0, column=2, sticky="e", padx=0, pady=0)

            # Presentation-only transparency does not alter the mask geometry.
            ctk.CTkLabel(
                card, text="Прозрачность", anchor="w",
                font=AppTheme.caption_font(), text_color=AppTheme.TEXT_SECONDARY,
            ).pack(fill="x", padx=9, pady=(7, 0))
            opacity_slider = ctk.CTkSlider(
                card, from_=0, to=1, number_of_steps=100,
                command=lambda value, i=region.id: self._opacity(i, value),
            )
            opacity_slider.pack(fill="x", padx=10, pady=(0, 7))
            opacity_slider.set(region.opacity)

            # Editing controls follow the slider, as in the result-layer cards.
            controls = ctk.CTkFrame(card, fg_color="transparent")
            controls.pack(fill="x", padx=8, pady=(0, 8))
            controls.grid_columnconfigure(0, weight=1)
            is_active = region.id == self.active_id
            ctk.CTkButton(
                controls, text="Редактируется" if is_active else "Редактировать",
                width=125, height=26,
                fg_color=AppTheme.PRIMARY if is_active else "#50545C",
                command=lambda i=region.id: self._select(i),
            ).grid(row=0, column=0, sticky="w")
            ctk.CTkLabel(
                controls, text="Цвет", anchor="e",
                font=AppTheme.caption_font(),
            ).grid(row=0, column=1, sticky="e", padx=(5, 5))
            ctk.CTkButton(
                controls, text="", width=26, height=24, corner_radius=4,
                fg_color=region.color, hover_color=region.color,
                command=lambda i=region.id: self._choose_color(i),
            ).grid(row=0, column=2, sticky="e")

    def _select(self, region_id):
        self.active_id = region_id
        self._refresh_cards()

    def _rename(self, region_id, value):
        region = self.regions.get(region_id)
        if region is not None:
            name = value.strip() or f"Область {region_id}"
            if region.name != name:
                self._remember_undo()
                region.name = name

    def _composite(self):
        if self._cached_composite is not None:
            return self._cached_composite
        image = np.asarray(self.image.convert("RGBA")).copy()
        for region in self.regions.regions:
            if not region.visible or not region.mask.any():
                continue
            outline = region.mask & ~binary_erosion(region.mask)
            rgb = tuple(int(region.color[i:i+2], 16) for i in (1, 3, 5))
            alpha = region.opacity
            if not outline.any():
                continue
            image[outline, :3] = np.clip(np.asarray(rgb) * alpha + image[outline, :3] * (1 - alpha), 0, 255).astype(np.uint8)
        self._cached_composite = Image.fromarray(image, "RGBA").convert("RGB")
        return self._cached_composite

    def _redraw(self):
        if not self._ui_ready:
            return
        if self.canvas.winfo_width() < 2 or self.canvas.winfo_height() < 2:
            return
        width = max(1, int(round(self.image.width * self.zoom)))
        height = max(1, int(round(self.image.height * self.zoom)))
        self._image_x = max(0, (self.canvas.winfo_width() - width) // 2)
        self._image_y = max(0, (self.canvas.winfo_height() - height) // 2)
        composite = self._composite()
        # Resize/PhotoImage are expensive, especially with multiple ROIs.
        # Reuse the displayed raster for layout-only configure events.
        if (self._photo is None or self._display_source is not composite or
                self._display_size != (width, height)):
            scaled = composite.resize((width, height), Image.Resampling.BILINEAR)
            self._photo = ImageTk.PhotoImage(scaled, master=self.canvas)
            self._display_source = composite
            self._display_size = (width, height)
        if self._image_id is None:
            self._image_id = self.canvas.create_image(self._image_x, self._image_y,
                                                      image=self._photo, anchor="nw")
        else:
            self.canvas.coords(self._image_id, self._image_x, self._image_y)
            self.canvas.itemconfigure(self._image_id, image=self._photo)
        self.canvas.tag_lower(self._image_id)
        self.canvas.configure(scrollregion=(0, 0, max(width, self.canvas.winfo_width()),
                                            max(height, self.canvas.winfo_height())))
        self.status.configure(text=(f"{self.image.width} × {self.image.height} пикс. · "
                                    f"Масштаб {self.zoom * 100:.0f}% · "
                                    f"{len(self.regions)} ROI · "
                                    "Ctrl+колесо — масштаб · ПКМ — перемещение · Ctrl+Z — отмена"))

    def _accept(self):
        self._cancel_pending_refresh()
        # Empty annotations are a valid result: they opt out of ROI processing.
        self.result = self.regions.copy()
        self.grab_release()
        self.destroy()

    def _cancel_pending_refresh(self):
        job = self._region_refresh_job
        if job is not None:
            self.after_cancel(job)
            self._region_refresh_job = None

    def _cancel(self):
        self._cancel_pending_refresh()
        self.result = None
        self.grab_release()
        self.destroy()


def run_roi_editor(master, image, existing=None):
    editor = ROIEditor(master, image, existing)
    master.wait_window(editor)
    return editor.result
