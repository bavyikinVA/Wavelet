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


VIEWER_CORNER_RADIUS = 0  # Прямые углы: без просвечивающих скруглённых стыков
VIEWER_OUTER_PAD_X = 2
VIEWER_OUTER_PAD_Y = 2
VIEWER_BLOCK_GAP = 2


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


def _result_group(item):
    category = item.get('category', '')
    return {'Вейвлеты': 'CWT', 'Источник': 'Исходное изображение',
            'Исходные данные': 'Исходные данные',
            'ML-кластеры': 'ML'}.get(category, category or 'Прочее')


def _result_direction(item):
    value = item.get('cwt_direction') or item.get('direction')
    mapping = {'row': 'По строкам', 'rows': 'По строкам',
               'col': 'По столбцам', 'columns': 'По столбцам'}
    if value not in mapping:
        name = Path(item.get('path', '')).name.lower()
        if 'cwtrow' in name or 'cwt1d_row' in name:
            value = 'row'
        elif 'cwtcol' in name or 'cwt1d_col' in name:
            value = 'col'
    return mapping.get(value, 'Без направления')


class ArtifactPane(ctk.CTkFrame):
    def __init__(self, parent, title, settings_frames=None):
        super().__init__(parent, fg_color=AppTheme.NAV_BACKGROUND,
                         corner_radius=VIEWER_CORNER_RADIUS, border_width=1, border_color=AppTheme.BORDER)
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
        header.grid(row=0, column=0, sticky='ew', padx=8, pady=(3, 2))
        ctk.CTkLabel(header, text=title, font=AppTheme.section_title_font()).pack(side='left')
        self.details_visible = False
        # Settings live in the permanent side panel, not above the plot.
        selection = self.selection = ctk.CTkFrame(settings_frames[0] if settings_frames else self, fg_color='transparent')
        selection.grid_columnconfigure((0, 1), weight=1, uniform='filters')
        self.task_selector = ScrollableComboBox(
            selection, values=['Все задачи'], state='readonly', width=260,
            command=lambda _value: self.set_items(self._experiment_items))
        self.task_selector.set('Все задачи')
        ctk.CTkLabel(selection, text='Задача исследования', anchor='w', height=20).grid(
            row=0, column=0, columnspan=2, sticky='w')
        self.task_selector.grid(row=1, column=0, columnspan=2, sticky='ew', pady=(0, 6))
        self._experiment_items = []
        self.group_selector = ScrollableComboBox(
            selection, values=['Все типы'], state='readonly', width=260,
            command=lambda _value: self._on_group_changed())
        self.group_selector.set('Все типы')
        self.group_selector.grid(row=3, column=0, columnspan=2, sticky='ew', pady=(0, 6))
        ctk.CTkLabel(selection, text='Тип результата', anchor='w', height=20).grid(
            row=2, column=0, columnspan=2, sticky='w')
        self.direction_selector = ScrollableComboBox(
            selection, values=['Все направления'], state='readonly', width=260,
            command=lambda _value: self.filter_parameters())
        self.direction_selector.set('Все направления')
        ctk.CTkLabel(selection, text='Направление', anchor='w', height=20).grid(
            row=4, column=0, columnspan=2, sticky='w')
        self.direction_selector.grid(row=5, column=0, columnspan=2, sticky='ew', pady=(0, 6))
        self.parameters = {}
        self.parameter_labels = {}
        for column, key, label in [(0, 'scale', 'Масштаб, пикс.'),
                                    (1, 'channel', 'Канал')]:
            caption = ctk.CTkLabel(selection, text=label, anchor='w', height=20)
            caption.grid(row=6, column=column, sticky='w', padx=(0, 3) if column == 0 else (3, 0))
            self.parameter_labels[key] = caption
            menu = ScrollableComboBox(selection, values=['Все'], state='readonly', width=125,
                                      command=lambda value, k=key: self.filter_parameters(k))
            menu.set('Все')
            menu.configure(state='disabled')
            menu.grid(row=7, column=column, sticky='ew', padx=(0, 3) if column == 0 else (3, 0), pady=(0, 6))
            self.parameters[key] = menu
        # Retain the orientation filter for 2D results, below the compact pair.
        self.parameter_labels['orientation'] = ctk.CTkLabel(
            selection, text='Ориентация, °', anchor='w', height=20)
        self.parameter_labels['orientation'].grid(row=8, column=0, columnspan=2, sticky='w')
        self.parameters['orientation'] = ScrollableComboBox(
            selection, values=['Все'], state='readonly', width=260,
            command=lambda value: self.filter_parameters('orientation'))
        self.parameters['orientation'].set('Все')
        self.parameters['orientation'].configure(state='disabled')
        self.parameters['orientation'].grid(row=9, column=0, columnspan=2, sticky='ew', pady=(0, 6))
        ctk.CTkLabel(selection, text='Конкретный результат', anchor='w', height=20).grid(
            row=10, column=0, columnspan=2, sticky='w')
        self.selector = ScrollableComboBox(selection, values=['Нет файлов'], width=260,
                                          dropdown_wide=True, dropdown_search=True,
                                          state='readonly', command=self.show_selected)
        self.selector.grid(row=11, column=0, columnspan=2, sticky='ew', pady=(0, 2))
        self.filter_hint = ctk.CTkLabel(selection, text='', anchor='w', height=0,
                                       font=AppTheme.caption_font())
        self.filter_hint.grid_remove()
        self.controls = ctk.CTkFrame(settings_frames[0] if settings_frames else self, fg_color='transparent', height=1)
        self.component = ScrollableOptionMenu(self.controls, values=['Модуль', 'Фаза', 'Действительная часть'],
                                         command=lambda value: self.render_payload())
        self.body = ctk.CTkFrame(self, fg_color='transparent')
        self.body.grid(row=4, column=0, sticky='nsew', padx=2)
        self.body.grid_columnconfigure(0, weight=1)
        self.body.grid_rowconfigure(0, weight=1)
        self.figure = Figure(figsize=(5, 4), facecolor=AppTheme.NAV_BACKGROUND)
        # Identical axes slots even when the source has no color scale.
        self.ax = self.figure.add_axes([.09, .16, .72, .78])
        self.cax = self.figure.add_axes([.85, .20, .028, .68])
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
        self.result_caption.grid(row=7, column=0, sticky='ew', padx=10, pady=(0, 5))
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
        self.layers = LayerControls(self, master=settings_frames[1], ml_master=settings_frames[2]) if settings_frames else LayerControls(self)
        self.canvas.mpl_connect('resize_event', lambda event: self._queue_view_frame())
        if settings_frames:
            self.settings_frames = settings_frames
            self.selection.pack(fill='x', padx=7, pady=(4, 7))
            self.controls.pack(fill='x', padx=7, pady=4)
            self.layers.pack(fill='both', expand=True, padx=7, pady=4)
        else:
            self.bind('<Configure>', self._fit_selectors, add='+')

    def _adapt_result_filters(self, item):
        """Show only the filters meaningful for the selected artifact."""
        if not item:
            return
        # Keep scale/channel filters when this family supports them; orientation
        # is meaningful only for oriented 2-D wavelet outputs.
        family = item.get('family')
        related = [a for a in self._catalog_items
                   if a.get('family') == family] if family else [item]
        flags = {
            key: any(a.get(key) is not None for a in related)
            for key in ('scale', 'orientation', 'channel')
        }
        # For image originals do not leave unrelated analysis parameters visible.
        if item.get('category') == 'Источник':
            flags = dict.fromkeys(flags, False)
        for key, visible in flags.items():
            widgets = (self.parameter_labels[key], self.parameters[key])
            for widget in widgets:
                if visible:
                    widget.grid()
                else:
                    widget.grid_remove()
        self.filter_hint.configure(text='')

    def _fit_selectors(self, event):
        # CTkFrame.bind attaches to its canvas, which owns Configure events.
        if event.widget not in (self, self._canvas):
            return
        # All selector rows share their right edge; the layer action sits
        # outside that edge, with its own 8 px gap and 75 px button.
        width = min(610, max(260, event.width / self._get_widget_scaling() - 218))
        for key, menu in self.parameters.items():
            menu.configure(width=(width - 6) / 2 if key in ('scale', 'channel') else width)
        self.selector.configure(width=width)
        self.task_selector.configure(width=width)
        self.group_selector.configure(width=width)
        self.direction_selector.configure(width=width)
        self.layers.selector.configure(width=width)

    def refresh_control_states(self):
        for menu in self.parameters.values():
            menu.configure(state='readonly' if len(menu.cget('values')) > 1 else 'disabled')
        self.selector.configure(state='readonly' if self.items else 'disabled')
        self.layers.selector.configure(state='readonly' if self.layers.items else 'disabled')
        complex_data = self.payload is not None and np.iscomplexobj(self.payload.get('array'))
        self.component.configure(state='normal' if complex_data else 'disabled')

    def set_items(self, items, prefer_source=False):
        self._experiment_items = items
        tasks = list(dict.fromkeys(a.get('task_id', '') for a in items))
        self.task_selector.configure(values=['Все задачи'] + tasks,
                                     state='readonly' if len(tasks) > 1 else 'disabled')
        if self.task_selector.get() not in tasks:
            self.task_selector.set('Все задачи')
        selected_task = self.task_selector.get()
        items = [a for a in items if selected_task == 'Все задачи'
                 or a.get('task_id') == selected_task]
        self._catalog_items = items
        groups = sorted({_result_group(a) for a in items},
                        key=lambda g: (['Исходное изображение', 'CWT', 'Экстремумы',
                                        'Огибающие', 'KNN', 'ML'].index(g)
                                       if g in ['Исходное изображение', 'CWT', 'Экстремумы',
                                                'Огибающие', 'KNN', 'ML'] else 100, g))
        self.group_selector.configure(values=['Все типы'] + groups)
        if self.group_selector.get() not in groups:
            self.group_selector.set('Все типы')
        self._update_directions()
        self._populate_parameter_values(items, prefer_source=prefer_source)

    def _update_directions(self):
        selected = self.group_selector.get()
        relevant = [a for a in self._catalog_items
                    if selected == 'Все типы' or _result_group(a) == selected]
        directions = [value for value in ('По строкам', 'По столбцам', 'Без направления')
                      if any(_result_direction(a) == value for a in relevant)]
        current = self.direction_selector.get()
        self.direction_selector.configure(values=['Все направления'] + directions,
                                          state='readonly' if len(directions) > 1 else 'disabled')
        self.direction_selector.set(current if current in directions else 'Все направления')

    def _on_group_changed(self):
        self._update_directions()
        self.filter_parameters()

    def _populate_parameter_values(self, items, prefer_source=False):
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
        group = self.group_selector.get()
        direction = self.direction_selector.get()
        items = [a for a in self._catalog_items
                 if (group == 'Все типы' or _result_group(a) == group)
                 and (direction == 'Все направления' or _result_direction(a) == direction)
                 and all(menu.get() == 'Все' or a.get(key) == menu.get()
                         for key, menu in self.parameters.items())]
        previous = self.selector.get()
        previous_family = self.items.get(previous, {}).get('family')
        self.items = {}
        for item in items:
            category = _result_group(item)
            parts = [str(value) for value in (
                f"s={item['scale']}" if item.get('scale') else None,
                item.get('channel'),
                _result_direction(item) if _result_direction(item) != 'Без направления' else None,
                {'row': 'Экстремумы по строкам', 'col': 'Экстремумы по столбцам'}.get(item.get('feature_axis')),
                f"{item['orientation']}°" if item.get('orientation') else None,
                {'max': 'Максимумы', 'min': 'Минимумы', 'upper': 'Верхняя',
                 'lower': 'Нижняя'}.get(item.get('point_kind')),
                item.get('file_format')) if value]
            base = ' · '.join(parts) if parts else Path(item['path']).name
            if self.task_selector.get() == 'Все задачи':
                base = f"{item.get('task_name', 'Задача')} · {base}"
            if item.get('ml_run'):
                base = f"{base} · {item['ml_run']}"
            label = base
            counter = 2
            while label in self.items:
                label = f'{base} · {Path(item["path"]).stem} ({counter})'
                counter += 1
            self.items[label] = item
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
        if hasattr(self, 'settings_frames'):
            self._adapt_result_filters(item)
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
                if data['kind'] == 'map' or (array.ndim == 2 and np.issubdtype(array.dtype, np.number)):
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
        self._refresh_result_details()
        self._rendering = False
        self.view_changed()
        self.layers.redraw()
        if self.on_data_changed is not None:
            self.on_data_changed()
        self.canvas.draw_idle()

    def _refresh_result_details(self):
        """The selectors already show the selected result parameters."""
        return

    def update_range_info(self):
        if self.image_artist.get_visible() and self.cax.get_visible():
            lo, hi = self.image_artist.get_clim()
            overview = self._raster_key and self._raster_key[0] > 0 and self.array is not None and self.array.ndim == 2
            suffix = ' · обзор по пиковым значениям' if overview else ''
            self.range_info.configure(text=f'Мин: {lo:.6g} Макс: {hi:.6g}{suffix}')
        else:
            self.range_info.configure(text='')

    def color_identity(self):
        if not self.payload or self.array is None or not self.image_artist.get_visible():
            return None
        kind = self.payload.get('kind')
        if kind == 'image':
            if self.array.ndim != 2 or not np.issubdtype(self.array.dtype, np.number):
                return None  # RGB images do not share a scalar color normalization.
            return ('image', 'intensity')
        if kind != 'map' or (self.selected_item or {}).get('category') != 'Вейвлеты':
            return None
        item = self.selected_item or {}
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
        return dict(group=self.group_selector.get(), direction=self.direction_selector.get(),
                    parameters={k: v.get() for k, v in self.parameters.items()},
                    selected=self.selector.get(), component=self.component.get(),
                    view=self.normalized_view(), details=self.details_visible,
                    layers=[(l['item']['label'], l['alpha'], l['visible'].get()) for l in self.layers.layers])

    def restore_state(self, state):
        self._pending_view = state['view']
        self.component.set(state['component'])
        group = state.get('group', 'Все типы')
        self.group_selector.set(group if group in self.group_selector.cget('values') else 'Все типы')
        self._update_directions()
        direction = state.get('direction', 'Все направления')
        if direction in self.direction_selector.cget('values'):
            self.direction_selector.set(direction)
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
        # Kept for restoring existing saved view states. The sidebar is always
        # available independently of the chart layout.
        self.details_visible = not self.details_visible
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
        # A single compact header: experiment identity and optional B pane.
        heading = ctk.CTkFrame(self, fg_color=AppTheme.NAV_BACKGROUND,
                               corner_radius=VIEWER_CORNER_RADIUS, border_width=1,
                               border_color=AppTheme.BORDER)
        heading.pack(fill='x', padx=VIEWER_OUTER_PAD_X, pady=(VIEWER_OUTER_PAD_Y, VIEWER_BLOCK_GAP))
        self.summary = ctk.CTkLabel(heading, text='Эксперимент: не выбран',
                                    font=AppTheme.panel_title_font(), anchor='w')
        self.summary.pack(side='left', fill='x', expand=True, padx=(12, 8), pady=6)
        self.sidebar_reopen = ctk.CTkButton(
            heading, text='☷', width=30, height=26,
            fg_color='transparent', hover_color=AppTheme.NAV_HOVER,
            command=self.toggle_sidebar)
        self.sidebar_reopen.pack(side='right', padx=(0, 8))
        self.comparison_visible = tk.BooleanVar(value=False)
        self.comparison_visible_widget = ctk.CTkCheckBox(
            heading, text='Сравнение B', variable=self.comparison_visible,
            command=self.toggle_comparison)
        self.comparison_visible_widget.pack(side='right', padx=12, pady=4)
        self.linked = tk.BooleanVar(value=False)
        self._syncing = False
        self.comparison_controls = ctk.CTkFrame(self, fg_color=AppTheme.NAV_BACKGROUND,
                                                corner_radius=VIEWER_CORNER_RADIUS, border_width=1,
                                                border_color=AppTheme.BORDER)
        color_controls = ctk.CTkFrame(self.comparison_controls, fg_color='transparent')
        color_controls.pack(fill='x', padx=12, pady=4)
        ctk.CTkCheckBox(color_controls, text='Связать масштаб и положение', variable=self.linked,
                        command=lambda: self.sync_view(self.left)).pack(side='left', padx=(0, 26))
        self.shared_color = tk.BooleanVar(value=False)
        self.shared_color_control = ctk.CTkCheckBox(
            color_controls, text='Общая шкала A/B', variable=self.shared_color,
            command=self.update_colors)
        self.shared_color_control.pack(side='left', padx=(0, 26))
        self.palette_choice = ctk.StringVar(value='Авто: Magma / Seismic')
        ctk.CTkLabel(color_controls, text='Цветовая схема:').pack(side='left', padx=(0, 8))
        self.palette_selector = ctk.CTkOptionMenu(
            color_controls,
            values=['Авто: Magma / Seismic', 'Magma', 'Seismic', 'Viridis'],
            variable=self.palette_choice, width=190,
            command=lambda _choice: self.update_colors(),
        )
        self.palette_selector.pack(side='left')
        # Outer split makes the settings panel independently resizable.
        workspace = self.workspace_split = tk.PanedWindow(
            self, orient='horizontal', sashwidth=7, sashrelief='flat',
            background=AppTheme.BORDER, borderwidth=0, relief='flat',
            opaqueresize=True)
        workspace.pack(fill='both', expand=True, padx=VIEWER_OUTER_PAD_X,
                       pady=(0, VIEWER_OUTER_PAD_Y))
        panes = self.panes = tk.PanedWindow(workspace, orient='horizontal', sashwidth=7,
                                           background=AppTheme.BORDER, borderwidth=0,
                                           relief='flat', opaqueresize=True)
        self.sidebar = ctk.CTkFrame(workspace, fg_color=AppTheme.NAV_BACKGROUND,
                                    corner_radius=0, border_width=1, border_color=AppTheme.BORDER,
                                    width=335)
        workspace.add(panes, minsize=350, stretch='always')
        workspace.add(self.sidebar, minsize=260, width=350, stretch='never')
        self._sidebar_width = 350
        workspace.bind('<ButtonRelease-1>', self._remember_sidebar_width, add='+')
        self.sidebar.grid_columnconfigure(0, weight=1)
        self.sidebar.grid_rowconfigure(3, weight=1)
        sidebar_header = ctk.CTkFrame(self.sidebar, fg_color='transparent')
        sidebar_header.grid(row=0, column=0, sticky='ew', padx=11, pady=(10, 6))
        ctk.CTkLabel(sidebar_header, text='Параметры визуализации',
                     font=AppTheme.section_title_font()).pack(side='left')
        self._sidebar_open = True
        self.sidebar_toggle = ctk.CTkButton(
            sidebar_header, text='⮞', width=30, height=26,
            fg_color='transparent', hover_color=AppTheme.NAV_HOVER,
            command=self.toggle_sidebar)
        self.sidebar_toggle.pack(side='right')
        self.active_pane = tk.StringVar(value='A')
        self.pane_switch = ctk.CTkSegmentedButton(
            self.sidebar, values=['A', 'B'], variable=self.active_pane,
            command=lambda _value: self._show_sidebar_section())
        self.pane_switch.grid(row=1, column=0, sticky='ew', padx=10, pady=(0, 7))
        self.pane_switch.grid_remove()
        self.settings_section = tk.StringVar(value='Результат')
        self.section_switch = ctk.CTkSegmentedButton(
            self.sidebar, values=['Результат', 'Слои', 'ML'],
            variable=self.settings_section,
            command=lambda _value: self._show_sidebar_section())
        self.section_switch.grid(row=2, column=0, sticky='ew', padx=10, pady=(0, 8))
        self.settings_container = ctk.CTkFrame(self.sidebar, fg_color='transparent')
        self.settings_container.grid(row=3, column=0, sticky='nsew', padx=5, pady=(0, 8))
        self.settings_container.grid_rowconfigure(0, weight=1)
        self.settings_container.grid_columnconfigure(0, weight=1)
        self.settings_frames = {}
        for key in ('A', 'B'):
            frames = []
            for section in ('Результат', 'Слои', 'ML'):
                frame = ctk.CTkScrollableFrame(self.settings_container, fg_color='transparent',
                                               corner_radius=0)
                self.settings_frames[key, section] = frame
                frames.append(frame)
            self.settings_frames[key, 'tuple'] = tuple(frames)
        self.left = ArtifactPane(panes, 'Сравнение A', self.settings_frames['A', 'tuple'])
        self.right = ArtifactPane(panes, 'Сравнение B', self.settings_frames['B', 'tuple'])
        self.sidebar_reopen.pack_forget()
        self._show_sidebar_section()
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

    def _remember_sidebar_width(self, event=None):
        if self._sidebar_open and len(self.workspace_split.panes()) == 2:
            try:
                self._sidebar_width = max(260, self.workspace_split.winfo_width()
                                          - self.workspace_split.sash_coord(0)[0] - 7)
            except tk.TclError:
                pass

    def toggle_sidebar(self):
        """Collapse or restore settings without rebuilding any widgets."""
        if self._sidebar_open:
            self._remember_sidebar_width()
            self.workspace_split.forget(self.sidebar)
            self._sidebar_open = False
            self.sidebar_reopen.pack(side='right', padx=(0, 8), before=self.comparison_visible_widget)
        else:
            self.workspace_split.add(self.sidebar, minsize=260,
                                     width=self._sidebar_width, stretch='never')
            self._sidebar_open = True
            self.sidebar_reopen.pack_forget()

    def _show_sidebar_section(self):
        target = self.active_pane.get() if self.comparison_visible.get() else 'A'
        section = self.settings_section.get()
        for (pane, label), frame in self.settings_frames.items():
            if label == 'tuple':
                continue
            frame.grid_forget()
        self.settings_frames[target, section].grid(row=0, column=0, sticky='nsew')

    def toggle_comparison(self):
        if self.comparison_visible.get():
            self.right.suspended = False
            if self._right_state:
                self.right.restore_state(self._right_state)
            else:
                self.right.show_selected()
            self.comparison_controls.pack(fill='x', padx=VIEWER_OUTER_PAD_X, pady=(0, VIEWER_BLOCK_GAP), before=self.panes.master)
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
        if not self.comparison_visible.get():
            self.active_pane.set('A')
            self.pane_switch.grid_remove()
        else:
            self.pane_switch.grid()
        self._show_sidebar_section()
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

    def _palette_norm(self, pane, lo, hi):
        """One normalization policy for both panes when the shared scale is active."""
        from matplotlib.colors import Normalize, PowerNorm, TwoSlopeNorm
        if hi <= lo:
            hi = lo + 1.0
        identity = pane.color_identity()
        quantity = identity[1] if identity else ''
        # Phase is circular: a sequential or divergent palette would be misleading.
        if quantity == 'phase':
            return 'twilight', Normalize(-np.pi, np.pi)
        choice = self.palette_choice.get()
        if choice == 'Viridis':
            return 'viridis', Normalize(lo, hi)
        if choice == 'Magma':
            return ('magma', PowerNorm(gamma=0.5, vmin=max(0.0, lo), vmax=hi)
                    if lo >= 0 and hi > 0 else Normalize(lo, hi))
        if choice == 'Seismic':
            if lo < 0 < hi:
                return 'seismic', TwoSlopeNorm(vmin=lo, vcenter=0, vmax=hi)
            return 'seismic', Normalize(lo, hi)
        # Automatic: high-contrast sequential for magnitude and intensities;
        # blue-white-red with a physically meaningful zero for signed coefficients.
        if lo < 0 < hi:
            return 'seismic', TwoSlopeNorm(vmin=lo, vcenter=0, vmax=hi)
        return ('magma', PowerNorm(gamma=0.5, vmin=lo, vmax=hi)
                if lo >= 0 and hi > lo else Normalize(lo, hi))

    def update_colors(self):
        """Use exactly the same limits and normalization for compatible A/B maps."""
        if self._updating_items:
            return
        comparison = self.comparison_visible.get()
        panes = (self.left, self.right) if comparison else (self.left,)
        identities = [pane.color_identity() for pane in panes]
        loading = any(pane._future is not None for pane in panes)
        compatible = (comparison and not loading and identities[0] is not None
                      and identities[0] == identities[1])
        self.shared_color_control.configure(state='normal' if compatible else 'disabled')
        if loading:
            return
        if compatible and self.shared_color.get():
            limits = [pane.natural_color_limits() for pane in panes]
            bounds = (min(lo for lo, _ in limits), max(hi for _, hi in limits))
            ranges = [(pane, bounds) for pane in panes]
        else:
            ranges = [(pane, pane.natural_color_limits())
                      for pane, identity in zip(panes, identities) if identity is not None]
        for pane, (lo, hi) in ranges:
            cmap, norm = self._palette_norm(pane, lo, hi)
            pane.image_artist.set_norm(norm)
            pane.image_artist.set_cmap(cmap)
            pane.canvas.draw_idle()
            pane.update_range_info()

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
                    self.left.capture_state(), self.right.capture_state() if not self.right.suspended else self._right_state)
                if len(self._view_states) > 20:
                    del self._view_states[next(iter(self._view_states))]
            self._right_state = None
            for pane in (self.left, self.right):
                pane._pending_view = None
                pane.task_selector.set('Все задачи')
                pane.layers.clear()
                for menu in pane.parameters.values():
                    menu.configure(state='readonly')
                    menu.set('Все')
        self.folder = folder
        self.artifacts = discover_results(folder)
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
            left, right = self._view_states[folder]
            self.left.restore_state(left)
            self._right_state = right
            if right and not self.right.suspended:
                self.right.restore_state(right)

    def filter(self, selected=None, initial=False):
        # The global category filter was removed; each pane has its own selector.
        items = self.artifacts
        experiment_name = Path(self.folder).name if self.folder else 'не выбран'
        self.summary.configure(text=f'Эксперимент: {experiment_name}')
        self._updating_items = True
        try:
            self.left.set_items(items, prefer_source=initial)
            self.right.set_items([a for a in items if a['category'] != 'Источник'] or items
                                 if initial else items)
        finally:
            self._updating_items = False
        self.update_colors()

    def destroy(self):
        if self._split_job is not None:
            self.after_cancel(self._split_job)
            self._split_job = None
        super().destroy()
