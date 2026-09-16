"""Stable embedded comparison canvases with asynchronous, cached data loading."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import tkinter as tk
import numpy as np
import customtkinter as ctk
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from history.result_catalog import discover_results
from history.result_data import load_result
from utils.theme import AppTheme
from utils.icon_button import IconButton
from utils.animation import animate_visibility
from utils.result_layers import LayerControls
from utils.scrollable_dropdown import ScrollableComboBox, ScrollableOptionMenu
from utils.raster_levels import RasterLevels


class ViewCanvas(FigureCanvasTkAgg):
    """One scheduled frame per burst; prepare geometry immediately before drawing."""
    def __init__(self, *args, prepare_frame, **kwargs):
        self.prepare_frame = prepare_frame
        super().__init__(*args, **kwargs)

    def draw_idle(self):
        if self._idle_draw_id is None:
            def frame():
                self._idle_draw_id = None
                self.draw()
            self._idle_draw_id = self.get_tk_widget().after(16, frame)

    def draw(self):
        if self._idle_draw_id is not None:
            self.get_tk_widget().after_cancel(self._idle_draw_id)
            self._idle_draw_id = None
        self.prepare_frame()
        super().draw()


class ArtifactPane(ctk.CTkFrame):
    def __init__(self, parent, title):
        super().__init__(parent, fg_color=AppTheme.NAV_BACKGROUND)
        self.items = {}
        self.array = None
        self.payload = None
        self.on_view_changed = None
        self.on_data_changed = None
        self.on_details_changed = None
        self.selected_item = None
        self._catalog_items = []
        self._extent = None
        self._rendering = False
        self._view_dirty = False
        self._raster = None
        self._raster_key = None
        self._raster_enabled = True
        self._request_key = None
        self._future = None
        self._poll_id = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self.folder = ''
        self.suspended = False
        self._pending_view = None
        self._natural_limits = None
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(4, weight=1)
        header = ctk.CTkFrame(self, fg_color='transparent')
        header.grid(row=0, column=0, sticky='ew', padx=10, pady=4)
        ctk.CTkLabel(header, text=title, font=AppTheme.section_title_font()).pack(side='left')
        self.details_visible = False
        IconButton(header, 'settings', 'Инструменты просмотра', self.toggle_details).pack(side='right')
        selection = self.selection = ctk.CTkFrame(self, fg_color='transparent')
        selection.grid_columnconfigure((0, 1, 2), weight=1)
        self.parameters = {}
        for column, (key, label) in enumerate([('scale', 'Масштаб, пикс.'),
                                               ('orientation', 'Ориентация, °'), ('channel', 'Канал')]):
            padding = (0, 8) if column < 2 else 0
            ctk.CTkLabel(selection, text=label, anchor='w', height=20).grid(row=0, column=column, sticky='w', padx=padding)
            menu = ScrollableComboBox(selection, values=['Все'], state='readonly', width=110,
                                   command=lambda value, k=key: self.filter_parameters(k))
            menu.set('Все')
            menu.configure(state='disabled')
            menu.grid(row=1, column=column, sticky='ew', padx=padding)
            self.parameters[key] = menu
        self.selector = ScrollableComboBox(selection, values=['Нет файлов'], width=577,
                                          state='readonly', command=self.show_selected)
        self.selector.grid(row=2, column=0, columnspan=3, sticky='ew', pady=(5, 0))
        self.filter_hint = ctk.CTkLabel(selection, text='', anchor='w', height=20,
                                       font=AppTheme.caption_font())
        self.filter_hint.grid(row=3, column=0, columnspan=3, sticky='w')
        self.controls = ctk.CTkFrame(self, fg_color='transparent', height=1)
        self.component = ScrollableOptionMenu(self.controls, values=['Модуль', 'Фаза', 'Действительная часть'],
                                         command=lambda value: self.render_payload())
        self.body = ctk.CTkFrame(self, fg_color='transparent')
        self.body.grid(row=4, column=0, sticky='nsew', padx=4)
        self.body.grid_columnconfigure(0, weight=1)
        self.body.grid_rowconfigure(0, weight=1)
        self.figure = Figure(figsize=(5, 4), facecolor=AppTheme.NAV_BACKGROUND)
        # Identical axes slots even when the source has no color scale.
        self.ax = self.figure.add_axes([.09, .10, .72, .84])
        self.cax = self.figure.add_axes([.85, .14, .028, .74])
        self.canvas = ViewCanvas(self.figure, master=self.body, prepare_frame=self._prepare_frame)
        self.canvas.get_tk_widget().configure(background=AppTheme.NAV_BACKGROUND, highlightthickness=0, height=180, width=180)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.controls, pack_toolbar=False)
        self.toolbar.pan()
        # Only the data axes participate in dragging; the colorbar is a scale.
        self.cax.set_navigate(False)
        for label, action in [('Сохранить', self.save_view), ('Сброс', self.reset_view)]:
            ctk.CTkButton(header, text=label, width=65, height=26,
                          fg_color='transparent', hover_color=AppTheme.NAV_HOVER,
                          command=action).pack(side='right', padx=1)
        self.table = ctk.CTkTextbox(self.body, font=AppTheme.monospace_font())
        self.info = ctk.CTkLabel(self, text='', anchor='w', height=22, font=AppTheme.caption_font())
        self.info.grid(row=5, column=0, sticky='ew', padx=10)
        self.range_info = ctk.CTkLabel(self, text='', anchor='w', height=20,
                                      font=AppTheme.caption_font())
        self.range_info.grid(row=6, column=0, sticky='ew', padx=10)
        self.result_caption = ctk.CTkLabel(self, text='', anchor='w', height=22,
                                          font=AppTheme.caption_font())
        self.result_caption.grid(row=7, column=0, sticky='ew', padx=10)
        self.image_artist = self.ax.imshow(np.zeros((2, 2)), cmap='viridis', interpolation='nearest')
        self.colorbar = self.figure.colorbar(self.image_artist, cax=self.cax)
        self.scatter = self.ax.scatter([], [], c=np.empty(0), s=3, cmap='tab20', linewidths=0)
        self.edges = LineCollection([], colors='#7e9aa5', linewidths=.4)
        self.ax.add_collection(self.edges)
        self.lines = []
        self.placeholder = self.ax.text(.5, .5, 'Выберите результат', ha='center', va='center',
                                         color=AppTheme.TEXT_SECONDARY, transform=self.ax.transAxes)
        self.image_artist.set_visible(False)
        self.cax.set_visible(False)
        self.ax.set_facecolor(AppTheme.NAV_BACKGROUND)
        self.ax.tick_params(colors=AppTheme.TEXT_SECONDARY)
        self.colorbar.ax.tick_params(colors=AppTheme.TEXT_SECONDARY)
        for spine in self.ax.spines.values():
            spine.set_color(AppTheme.BORDER)
        self.ax.callbacks.connect('xlim_changed', self.view_changed)
        self.ax.callbacks.connect('ylim_changed', self.view_changed)
        self.canvas.mpl_connect('motion_notify_event', self.hover)
        self.canvas.mpl_connect('scroll_event', self.scroll_zoom)
        self.layers = LayerControls(self)
        self.canvas.mpl_connect('resize_event', lambda event: self._queue_view_frame())
        self.bind('<Configure>', self._fit_selectors, add='+')

    def _fit_selectors(self, event):
        # CTkFrame.bind attaches to its canvas, which owns Configure events.
        if event.widget not in (self, self._canvas):
            return
        # All selector rows share their right edge; the layer action sits
        # outside that edge, with its own 8 px gap and 75 px button.
        width = min(610, max(260, event.width / self._get_widget_scaling() - 218))
        for menu in self.parameters.values():
            menu.configure(width=(width-16)/3)
        self.selector.configure(width=width)
        self.layers.selector.configure(width=width)

    def refresh_control_states(self):
        for menu in self.parameters.values():
            menu.configure(state='readonly' if len(menu.cget('values')) > 1 else 'disabled')
        self.selector.configure(state='readonly' if self.items else 'disabled')
        self.layers.selector.configure(state='readonly' if self.layers.items else 'disabled')
        complex_data = self.payload is not None and np.iscomplexobj(self.payload.get('array'))
        self.component.configure(state='normal' if complex_data else 'disabled')

    def set_items(self, items, prefer_source=False):
        self._catalog_items = items
        for key, menu in self.parameters.items():
            values = sorted({a[key] for a in items if a.get(key) is not None},
                            key=float if key != 'channel' else str)
            menu.configure(values=['Все'] + values, state='readonly')
            if menu.get() not in values:
                menu.set('Все')
            menu.configure(state='readonly' if values else 'disabled')
        self.filter_parameters(prefer_source=prefer_source)
        unavailable = [label for key, label in [('scale', 'масштаб'), ('orientation', 'ориентация'), ('channel', 'канал')]
                       if len(self.parameters[key].cget('values')) == 1]
        self.filter_hint.configure(text='Нет параметров в выбранных данных: ' + ', '.join(unavailable) if unavailable else '')

    def filter_parameters(self, changed=None, prefer_source=False):
        items = [a for a in self._catalog_items if all(menu.get() == 'Все' or a.get(key) == menu.get()
                  for key, menu in self.parameters.items())]
        previous = self.selector.get()
        previous_family = self.items.get(previous, {}).get('family')
        self.items = {item['label']: item for item in items}
        labels = list(self.items)
        self.selector.configure(values=labels or ['Нет файлов'], state='readonly')
        chosen = previous if previous in self.items else (labels[0] if labels else 'Нет файлов')
        if previous not in self.items and previous_family:
            chosen = next((label for label, item in self.items.items()
                           if item.get('family') == previous_family), chosen)
        if prefer_source:
            chosen = next((label for label, item in self.items.items() if item['category'] == 'Источник'), chosen)
        self.selector.set(chosen)
        self.selector.configure(state='readonly' if labels else 'disabled')
        self.show_selected()

    def show_selected(self, value=None):
        if self.suspended:
            return
        item = self.items.get(value or self.selector.get())
        self.selected_item = item
        if item:
            self.result_caption.configure(text=' · '.join(str(v) for v in (
                item['category'], item.get('channel'),
                f"Масштаб {item['scale']}" if item.get('scale') else None,
                f"Ориентация {item['orientation']}°" if item.get('orientation') else None,
                Path(item['path']).name) if v))
        else:
            self.result_caption.configure(text='Нет результата для выбранных фильтров')
        if item is None:
            self._request_key = None
            self._future = None
            self.payload = None
            self.render_payload()
            return
        path = Path(item['path'])
        try:
            key = (str(path), path.stat().st_mtime_ns)
        except OSError as error:
            self._future = None
            self._request_key = None
            self.payload = None
            self.render_payload()
            self.info.configure(text=f'Не удалось открыть: {error}')
            return
        if key == self._request_key:
            return
        self._request_key = key
        if self._future is not None:
            self._future.cancel()
        self.info.configure(text='Загрузка…')
        self._future = self._executor.submit(load_result, item, self.folder)
        if self._poll_id is None:
            self._poll_id = self.after(15, self._poll)

    def _poll(self):
        self._poll_id = None
        future = self._future
        if future is None:
            return
        if not future.done():
            self._poll_id = self.after(20, self._poll)
            return
        self._future = None
        try:
            self.payload = future.result()
            self.render_payload()
        except Exception as error:
            self.payload = None
            self.render_payload()
            self.info.configure(text=f'Не удалось открыть: {error}')

    def render_payload(self):
        self._raster = None
        self._raster_key = None
        self.image_artist.set_clip_path(None)
        self._natural_limits = None
        previous = self.normalized_view()
        data = self.payload
        self._rendering = True
        self.array = None
        self._extent = None
        self.image_artist.set_visible(False)
        self.scatter.set_visible(False)
        self.edges.set_visible(False)
        # Invisible artists still own their arrays when the result type changes.
        self.image_artist.set_data(np.zeros((2, 2)))
        self.scatter.set_offsets(np.empty((0, 2)))
        self.scatter.set_array(None)
        self.edges.set_segments([])
        for line in self.lines:
            line.remove()
        self.lines = []
        self.cax.set_visible(False)
        self.placeholder.set_visible(data is None)
        self.component.configure(state='disabled')
        self.component.pack_forget()
        self.table.grid_remove()
        self.canvas.get_tk_widget().grid()
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_axis_on()
        from matplotlib.ticker import AutoLocator, ScalarFormatter
        self.ax.xaxis.set_major_locator(AutoLocator())
        self.ax.xaxis.set_major_formatter(ScalarFormatter())
        if data is None:
            self.placeholder.set_text('Нет результатов для выбранного фильтра')
            self.info.configure(text='')
        elif data['kind'] == 'text':
            self.canvas.get_tk_widget().grid_remove()
            self.table.grid(row=0, column=0, sticky='nsew')
            self.table.configure(state='normal')
            self.table.delete('1.0', 'end')
            self.table.insert('1.0', data['text'])
            self.table.configure(state='disabled')
            self.info.configure(text='Первые 150 строк')
        elif data['kind'] == 'series':
            self.ax.set_aspect('auto')
            values = data['array']
            for i in range(1, values.shape[1]):
                self.lines.extend(self.ax.plot(values[:, 0], values[:, i], label=data['labels'][i]))
            self.ax.relim()
            self.ax.autoscale(enable=True)
            if data.get('ticks'):
                self.ax.set_xticks(values[:, 0], data['ticks'], rotation=45)
            self.info.configure(text='Статистика · ' + ', '.join(data['labels']))
        else:
            h, w = data['shape']
            extent = (-.5, w-.5, h-.5, -.5)
            if data.get('spatial'):
                self._extent = extent
            if data['kind'] in ('map', 'image'):
                array = data['array']
                if np.iscomplexobj(array):
                    self.component.configure(state='normal')
                    self.component.pack(side='left')
                    choice = self.component.get()
                    array = np.angle(array) if choice == 'Фаза' else (array.real if choice == 'Действительная часть' else np.abs(array))
                self.array = array
                self.image_artist.set_data(array)
                self.image_artist.set_extent(extent)
                self.image_artist.set_visible(True)
                if data['kind'] == 'map':
                    self.image_artist.set_cmap('twilight' if self.component.get() == 'Фаза' and np.iscomplexobj(data['array']) else 'viridis')
                    finite = array[np.isfinite(array)]
                    lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0., 1.)
                    self._natural_limits = (lo, hi)
                    from matplotlib.colors import Normalize
                    self.image_artist.set_norm(Normalize(lo, hi if hi > lo else lo+1.))
                    self.cax.set_visible(True)
                self.info.configure(text='Старый PNG без численных данных: синхронизация отключена' if data.get('legacy') else '')
            else:
                points = data['points']
                self.scatter.set_offsets(points[:, :2])
                self.scatter.set_array(np.asarray(data.get('colors', np.zeros(len(points)))))
                self.scatter.autoscale()
                self.scatter.set_visible(True)
                self.edges.set_segments(data.get('edges', []))
                self.edges.set_visible(True)
                self.info.configure(text=f'Точек: {len(points)}')
            self.ax.set_xlim(extent[:2])
            self.ax.set_ylim(extent[2:])
            if previous is not None and self._extent is not None:
                self.apply_view(previous)
        self.toolbar.update()
        if self._pending_view is not None and self._extent is not None:
            self.apply_view(self._pending_view)
            self._pending_view = None
        self.update_range_info()
        self._rendering = False
        self.view_changed()
        self.layers.redraw()
        if self.on_data_changed is not None:
            self.on_data_changed()
        self.canvas.draw_idle()

    def update_range_info(self):
        if self.image_artist.get_visible() and self.cax.get_visible():
            lo, hi = self.image_artist.get_clim()
            overview = self._raster_key and self._raster_key[0] > 0 and self.array is not None and self.array.ndim == 2
            suffix = ' · обзор по пиковым значениям' if overview else ''
            self.range_info.configure(text=f'Мин: {lo:.6g} Макс: {hi:.6g}{suffix}')
        else:
            self.range_info.configure(text='')

    def color_identity(self):
        if not self.payload or self.payload['kind'] != 'map' or self.array is None:
            return None
        item = self.selected_item or {}
        if item.get('category') != 'Вейвлеты':
            return None
        quantity = self.payload.get('quantity', 'coefficient')
        if np.iscomplexobj(self.payload['array']):
            quantity = {'Фаза': 'phase', 'Модуль': 'magnitude',
                        'Действительная часть': 'coefficient'}[self.component.get()]
        return (item.get('mode') or 'unknown', quantity)

    def natural_color_limits(self):
        if self.array is None:
            return (0., 1.)
        if self.color_identity() and self.color_identity()[1] == 'phase':
            return (-np.pi, np.pi)
        if self._natural_limits is not None:
            return self._natural_limits
        finite = self.array[np.isfinite(self.array)]
        lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0., 1.)
        self._natural_limits = (lo, hi)
        return self._natural_limits

    def capture_state(self):
        return dict(parameters={k: v.get() for k, v in self.parameters.items()},
                    selected=self.selector.get(), component=self.component.get(),
                    view=self.normalized_view(), details=self.details_visible,
                    layers=[(l['item']['label'], l['alpha'], l['visible'].get()) for l in self.layers.layers])

    def restore_state(self, state):
        self._pending_view = state['view']
        self.component.set(state['component'])
        for key, value in state['parameters'].items():
            if value in self.parameters[key].cget('values'):
                self.parameters[key].set(value)
        self.filter_parameters()
        if state['selected'] in self.items:
            self.selector.set(state['selected'])
            self.show_selected()
        if not self.suspended and self._future is None:
            self.render_payload()
        self.layers.clear()
        if not self.suspended:
            for label, alpha, visible in state['layers']:
                if label in self.layers.items:
                    self.layers.selector.set(label)
                    self.layers.add_selected()
                    layer = self.layers.layers[-1]
                    layer['alpha'] = alpha
                    layer['slider'].set(alpha)
                    layer['visible'].set(visible)
            if state['details'] != self.details_visible:
                self.toggle_details()

    def suspend(self):
        self.suspended = True
        if self._future is not None:
            self._future.cancel()
        self._future = None
        self._request_key = None
        self.layers.clear()
        self.payload = None
        self.render_payload()

    def normalized_view(self):
        if self._extent is None:
            return None
        left, right, bottom, top = self._extent
        return (tuple((x-left)/(right-left) for x in self.ax.get_xlim()),
                tuple((y-bottom)/(top-bottom) for y in self.ax.get_ylim()))

    def apply_view(self, view):
        if self._extent is None:
            return
        left, right, bottom, top = self._extent
        rendering = self._rendering
        self._rendering = True
        try:
            self.ax.set_xlim([left+x*(right-left) for x in view[0]])
            self.ax.set_ylim([bottom+y*(top-bottom) for y in view[1]])
        finally:
            self._rendering = rendering
        self.view_changed()
        self.canvas.draw_idle()

    def reset_view(self):
        if self._extent is not None:
            self.apply_view(((0, 1), (0, 1)))
        else:
            self.toolbar.home()

    def save_view(self):
        self._raster_enabled = False
        try:
            if self.array is not None and self.payload['kind'] in ('map', 'image'):
                h, w = self.payload['shape']
                self.image_artist.set_data(self.array)
                self.image_artist.set_extent((-.5, w-.5, h-.5, -.5))
                self.image_artist.set_visible(True)
            self.canvas.draw()
            self.toolbar.save_figure()
        finally:
            self._raster_enabled = True
            self._raster_key = None
            self.canvas.draw_idle()

    def view_changed(self, axis=None):
        if not self._rendering and hasattr(self, 'layers'):
            self._queue_view_frame()
        if not self._rendering and self.on_view_changed is not None:
            self.on_view_changed(self)

    def _queue_view_frame(self):
        self._view_dirty = True
        self.canvas.draw_idle()

    def _prepare_frame(self):
        self._prepare_raster()
        if self._view_dirty and hasattr(self, 'layers'):
            self._view_dirty = False
            self.layers.update_marker_sizes()

    def _prepare_raster(self):
        if not self._raster_enabled or self.array is None or self.payload['kind'] not in ('map', 'image'):
            return
        if self._raster is None:
            phase = (self.payload.get('quantity') == 'phase' or
                     (np.iscomplexobj(self.payload['array']) and self.component.get() == 'Фаза'))
            self._raster = RasterLevels(self.array, self.payload['shape'], reduce=not phase)
            from matplotlib.patches import Rectangle
            h, w = self.payload['shape']
            self.image_artist.set_clip_path(Rectangle((-.5, -.5), w, h, transform=self.ax.transData))
        self.ax.apply_aspect()
        result = self._raster.view(self.ax.get_xlim(), self.ax.get_ylim(),
                                   (self.ax.bbox.width, self.ax.bbox.height))
        if result is None:
            self.image_artist.set_visible(False)
            self._raster_key = None
            return
        array, extent, key = result
        if key != self._raster_key:
            self.image_artist.set_data(array)
            self.image_artist.set_extent(extent)
            self.image_artist.set_visible(True)
            self._raster_key = key
            self.update_range_info()

    def scroll_zoom(self, event):
        if not self.payload or self.payload['kind'] == 'text' or event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return
        if self.toolbar._nav_stack() is None:
            self.toolbar.push_current()
        factor = 1.2 ** (-event.step)
        self._rendering = True
        self.ax.set_xlim([event.xdata+(x-event.xdata)*factor for x in self.ax.get_xlim()])
        self.ax.set_ylim([event.ydata+(y-event.ydata)*factor for y in self.ax.get_ylim()])
        self._rendering = False
        self.view_changed()
        self.canvas.draw_idle()

    def toggle_details(self):
        self.details_visible = not self.details_visible
        animate_visibility(self.selection, self.details_visible,
                           lambda: self.selection.grid(row=1, column=0, sticky='w', padx=10, pady=3),
                           self.selection.grid_remove)
        animate_visibility(self.controls, self.details_visible,
                           lambda: self.controls.grid(row=2, column=0, sticky='w', padx=10, pady=3),
                           self.controls.grid_remove)
        animate_visibility(self.layers, self.details_visible,
                           lambda: self.layers.grid(row=3, column=0, sticky='w', padx=10, pady=3),
                           self.layers.grid_remove)
        if self.on_details_changed is not None:
            self.on_details_changed(self.details_visible)

    def hover(self, event):
        if self.array is None or event.inaxes is not self.ax or event.xdata is None or event.ydata is None or self._extent is None:
            return
        h, w = self.payload['shape']
        x = int((event.xdata+.5)*self.array.shape[1]/w)
        y = int((event.ydata+.5)*self.array.shape[0]/h)
        if 0 <= y < self.array.shape[0] and 0 <= x < self.array.shape[1]:
            value = self.array[y, x]
            text = np.array2string(value) if np.ndim(value) else f'{value:.8g}'
            self.info.configure(text=f'X={event.xdata:.1f}, Y={event.ydata:.1f} · {text}')

    def destroy(self):
        if self._poll_id:
            self.after_cancel(self._poll_id)
            self._poll_id = None
        self._executor.shutdown(wait=False, cancel_futures=True)
        for name in ('_idle_draw_id', '_event_loop_id'):
            identifier = getattr(self.canvas, name, None)
            if identifier:
                self.canvas.get_tk_widget().after_cancel(identifier)
                setattr(self.canvas, name, None)
        super().destroy()


class ResultsPanel(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, fg_color='transparent')
        self.folder = ''
        self.artifacts = []
        self._updating_items = False
        self._view_states = {}
        self._right_state = None
        self._pending_folder_notice = False
        heading = ctk.CTkFrame(self, fg_color='transparent')
        heading.pack(fill='x', padx=12, pady=4)
        self._settings_visible = False
        IconButton(heading, 'settings', 'Общие фильтры результатов', self.toggle_global_settings).pack(side='right')
        ctk.CTkLabel(heading, text='Результаты и сравнение', font=AppTheme.page_title_font()).pack(side='left')
        self.comparison_visible = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(heading, text='Сравнение B', variable=self.comparison_visible,
                        command=self.toggle_comparison).pack(side='right')
        self.summary = ctk.CTkLabel(self, text='Результаты появятся после расчета', anchor='w')
        self.summary.pack(fill='x', padx=12)
        self.folder_notice = ctk.CTkLabel(self, text='', anchor='w', height=20,
                                         font=AppTheme.caption_font())
        self.settings = ctk.CTkFrame(self, fg_color='transparent')
        controls = ctk.CTkFrame(self.settings, fg_color='transparent')
        controls.pack(fill='x', padx=12, pady=8)
        self.category = ScrollableOptionMenu(controls, values=['Все результаты'], command=self.filter,
                                         fg_color=AppTheme.NAV_BACKGROUND, button_color=AppTheme.NAV_BACKGROUND)
        self.category.pack(side='left', padx=(0, 8))
        self.category.set('Все результаты')
        IconButton(controls, 'refresh', 'Обновить список результатов',
                   lambda: self.load_folder(self.folder, force=True)).pack(side='right')
        self.linked = tk.BooleanVar(value=False)
        self._syncing = False
        self.comparison_controls = ctk.CTkFrame(self, fg_color='transparent')
        ctk.CTkCheckBox(self.comparison_controls, text='Связать масштаб и положение', variable=self.linked,
                        command=lambda: self.sync_view(self.left)).pack(anchor='w', padx=14, pady=4)
        color_controls = ctk.CTkFrame(self.comparison_controls, fg_color='transparent')
        color_controls.pack(fill='x', padx=14, pady=4)
        self.shared_color = tk.BooleanVar(value=False)
        self.fixed_color = tk.BooleanVar(value=False)
        self._color_limits = {}
        for label, variable in [('Общая шкала A/B', self.shared_color),
                                ('Зафиксировать диапазон', self.fixed_color)]:
            checkbox = ctk.CTkCheckBox(color_controls, text=label, variable=variable,
                                      command=self.reset_colors)
            checkbox.pack(side='left', padx=(0, 16))
            if variable is self.shared_color:
                self.shared_color_control = checkbox
        self.recalculate_control = ctk.CTkButton(color_controls, text='Пересчитать диапазон', width=160,
                      state='disabled', fg_color=AppTheme.NAV_BACKGROUND, command=self.reset_colors)
        self.recalculate_control.pack(side='left')
        self.color_info = ctk.CTkLabel(self.comparison_controls, text='', anchor='w', height=20,
                                       font=AppTheme.caption_font())
        self.color_info.pack(fill='x', padx=14)
        panes = self.panes = tk.PanedWindow(self, orient='horizontal', sashwidth=8,
                                           background=AppTheme.BORDER, borderwidth=0, opaqueresize=True)
        panes.pack(fill='both', expand=True)
        self.left = ArtifactPane(panes, 'Сравнение A')
        self.right = ArtifactPane(panes, 'Сравнение B')
        self.right.suspended = True
        panes.add(self.left, minsize=220, stretch='always')
        self.left.on_view_changed = self.sync_view
        self.right.on_view_changed = self.sync_view
        self.left.on_data_changed = self.update_colors
        self.right.on_data_changed = self.update_colors
        self._initial_split = False
        self._split_ratio = .5
        self._split_job = None
        panes.bind('<Map>', self.initialize_split)
        panes.bind('<Configure>', self.initialize_split)
        panes.bind('<ButtonRelease-1>', self.remember_split)

    def toggle_settings(self, visible):
        animate_visibility(self.settings, visible,
                           lambda: self.settings.pack(fill='x', before=self.panes),
                           self.settings.pack_forget)

    def toggle_global_settings(self):
        self._settings_visible = not self._settings_visible
        self.toggle_settings(self._settings_visible)

    def toggle_comparison(self):
        if self.comparison_visible.get():
            self.right.suspended = False
            if self._right_state:
                self.right.restore_state(self._right_state)
            else:
                self.right.show_selected()
            self.comparison_controls.pack(fill='x', before=self.panes)
            if len(self.panes.panes()) == 1:
                self.panes.add(self.right, minsize=220, stretch='always')
            self.initialize_split()
            self.sync_view(self.left)
        elif len(self.panes.panes()) == 2:
            self._right_state = self.right.capture_state()
            self.right.suspend()
            self.remember_split()
            self.panes.forget(self.right)
            self.comparison_controls.pack_forget()
        self.update_colors()

    def initialize_split(self, event=None):
        if self._split_job is not None:
            self.after_cancel(self._split_job)
        def apply():
            self._split_job = None
            width = self.panes.winfo_width()
            if width > 1 and len(self.panes.panes()) == 2:
                self.panes.sash_place(0, round(width*self._split_ratio), 0)
        self._split_job = self.after_idle(apply)

    def remember_split(self, event=None):
        width = self.panes.winfo_width()
        if width > 1 and len(self.panes.panes()) == 2:
            self._split_ratio = self.panes.sash_coord(0)[0]/width

    def reset_colors(self):
        self._color_limits.clear()
        self.update_colors()

    def update_colors(self):
        if self._updating_items:
            return
        comparison = self.comparison_visible.get()
        panes = (self.left, self.right) if comparison else (self.left,)
        identities = [p.color_identity() for p in panes]
        loading = any(p._future is not None for p in panes)
        compatible = (comparison and not loading and identities[0] is not None
                      and identities[0] == identities[1])
        self.shared_color_control.configure(state='normal' if compatible else 'disabled')
        fixed = comparison and self.fixed_color.get()
        self.recalculate_control.configure(
            state='normal' if fixed and not loading and any(identities) else 'disabled')
        # Do not freeze the previous file's limits while its replacement loads.
        if loading:
            from matplotlib.colors import Normalize
            for index, pane in enumerate(panes):
                identity = pane.color_identity()
                same_type = comparison and ('shared', identity) in self._color_limits
                key = ('shared', identity) if self.shared_color.get() and same_type else (index, identity)
                bounds = self._color_limits.get(key) if fixed else None
                if bounds is not None and pane._future is None:
                    pane.image_artist.set_norm(Normalize(*bounds))
                    pane.image_artist.set_cmap('twilight' if identity[1] == 'phase' else 'viridis')
                    pane.canvas.draw_idle()
                    pane.update_range_info()
            self.color_info.configure(text='Загрузка… фиксированный диапазон сохранён' if self.fixed_color.get()
                                       else 'Загрузка… общий диапазон будет пересчитан после загрузки')
            return
        common = self.shared_color.get() and compatible
        groups = [(('shared', identities[0]), panes)] if common else [
            ((index, identity), [pane]) for index, (pane, identity) in enumerate(zip(panes, identities))
            if identity is not None]
        descriptions = []
        for key, group in groups:
            limits = [p.natural_color_limits() for p in group]
            bounds = (min(v[0] for v in limits), max(v[1] for v in limits))
            if bounds[0] == bounds[1]:
                bounds = (bounds[0], bounds[1] + 1.)
            if fixed:
                bounds = self._color_limits.setdefault(key, bounds)
            for pane in group:
                # Set both bounds atomically: colorbar callbacks otherwise see
                # an invalid intermediate range when switching disjoint data.
                from matplotlib.colors import Normalize
                pane.image_artist.set_norm(Normalize(*bounds))
                pane.image_artist.set_cmap('twilight' if pane.color_identity()[1] == 'phase' else 'viridis')
                pane.canvas.draw_idle()
                pane.update_range_info()
            name = 'A/B' if common else 'A' if group[0] is self.left else 'B'
            descriptions.append(f'{name}: {bounds[0]:.6g} … {bounds[1]:.6g}')
        note = ' · диапазон зафиксирован' if fixed and groups else ''
        if comparison and not compatible:
            note += ' · общая шкала недоступна: нужны две карты вейвлетов одного типа и компоненты'
        self.color_info.configure(text='; '.join(descriptions) + note)
        if self._pending_folder_notice and fixed and groups:
            self.folder_notice.configure(text='Открыта другая папка. Зафиксирован новый диапазон: '
                                         + '; '.join(descriptions))
            self._pending_folder_notice = False

    def sync_view(self, source):
        if not self.comparison_visible.get() or not self.linked.get() or self._syncing:
            return
        target = self.right if source is self.left else self.left
        view = source.normalized_view()
        if view is None or target.normalized_view() is None:
            return
        self._syncing = True
        try:
            target.apply_view(view)
        finally:
            self._syncing = False

    def load_folder(self, folder, force=False):
        folder = str(folder or '')
        if folder == self.folder and not force:
            return
        changed = folder != self.folder
        if changed:
            if self.folder:
                self._view_states[self.folder] = (
                    self.left.capture_state(), self.right.capture_state() if not self.right.suspended else self._right_state,
                    self.category.get())
                if len(self._view_states) > 20:
                    del self._view_states[next(iter(self._view_states))]
            self._right_state = None
            self._color_limits.clear()
            self._pending_folder_notice = self.fixed_color.get()
            if self._pending_folder_notice:
                self.folder_notice.configure(text='Открыта другая папка. Старые границы сброшены; '
                                             'новые будут зафиксированы по картам сравнения.')
                self.folder_notice.pack(fill='x', padx=12, after=self.summary)
            else:
                self.folder_notice.pack_forget()
            for pane in (self.left, self.right):
                pane._pending_view = None
                pane.layers.clear()
                for menu in pane.parameters.values():
                    menu.configure(state='readonly')
                    menu.set('Все')
        self.folder = folder
        self.artifacts = discover_results(folder)
        categories = ['Все результаты'] + sorted({a['category'] for a in self.artifacts})
        self.category.configure(values=categories)
        if changed or self.category.get() not in categories:
            self.category.set('Все результаты')
        self.left.folder = self.right.folder = folder
        self.left.layers.set_items(self.artifacts)
        self.right.layers.set_items(self.artifacts)
        if changed or force:
            for pane in (self.left, self.right):
                pane._request_key = None
                if changed:
                    pane._extent = None
        self.filter(initial=changed)
        if changed and folder in self._view_states:
            left, right, category = self._view_states[folder]
            if category in categories:
                self.category.set(category)
                self.filter()
            self.left.restore_state(left)
            self._right_state = right
            if right and not self.right.suspended:
                self.right.restore_state(right)

    def filter(self, selected=None, initial=False):
        if selected is not None:
            self.category.set(selected)
        selected = self.category.get()
        items = [a for a in self.artifacts if selected == 'Все результаты' or a['category'] == selected]
        self.summary.configure(text=f'{Path(self.folder).name} · показано {len(items)} из {len(self.artifacts)}')
        self._updating_items = True
        try:
            self.left.set_items(items, prefer_source=initial)
            self.right.set_items([a for a in items if a['category'] != 'Источник'] or items if initial else items)
        finally:
            self._updating_items = False
        self.update_colors()

    def destroy(self):
        if self._split_job is not None:
            self.after_cancel(self._split_job)
            self._split_job = None
        super().destroy()
