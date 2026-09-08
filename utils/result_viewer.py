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


class ArtifactPane(ctk.CTkFrame):
    def __init__(self, parent, title):
        super().__init__(parent, fg_color=AppTheme.NAV_BACKGROUND)
        self.items = {}
        self.array = None
        self.payload = None
        self.on_view_changed = None
        self._extent = None
        self._rendering = False
        self._request_key = None
        self._future = None
        self._poll_id = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self.folder = ''
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)
        header = ctk.CTkFrame(self, fg_color='transparent')
        header.grid(row=0, column=0, sticky='ew', padx=10, pady=4)
        ctk.CTkLabel(header, text=title, font=AppTheme.section_title_font()).pack(side='left')
        self.details_visible = False
        IconButton(header, 'settings', 'Инструменты просмотра', self.toggle_details).pack(side='right')
        self.selector = ctk.CTkComboBox(self, values=['Нет файлов'], state='readonly', command=self.show_selected)
        self.selector.grid(row=1, column=0, sticky='ew', padx=10, pady=3)
        self.controls = ctk.CTkFrame(self, fg_color='transparent')
        self.component = ctk.CTkOptionMenu(self.controls, values=['Модуль', 'Фаза', 'Действительная часть'],
                                         command=lambda value: self.render_payload())
        self.component.pack(side='left', padx=3)
        self.body = ctk.CTkFrame(self, fg_color='transparent')
        self.body.grid(row=3, column=0, sticky='nsew', padx=4)
        self.body.grid_columnconfigure(0, weight=1)
        self.body.grid_rowconfigure(0, weight=1)
        self.figure = Figure(figsize=(5, 4), facecolor=AppTheme.NAV_BACKGROUND)
        # Identical axes slots even when the source has no color scale.
        self.ax = self.figure.add_axes([.09, .10, .72, .84])
        self.cax = self.figure.add_axes([.85, .14, .028, .74])
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.body)
        self.canvas.get_tk_widget().configure(background=AppTheme.NAV_BACKGROUND, highlightthickness=0, height=180, width=180)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.controls, pack_toolbar=False)
        for label, action in [('Сброс', self.reset_view), ('Сдвиг', self.toolbar.pan),
                              ('Зум', self.toolbar.zoom), ('Сохранить', self.toolbar.save_figure)]:
            ctk.CTkButton(self.controls, text=label, width=65, height=26,
                          fg_color='transparent', hover_color=AppTheme.NAV_HOVER,
                          command=action).pack(side='left', padx=1)
        self.table = ctk.CTkTextbox(self.body, font=AppTheme.monospace_font())
        self.info = ctk.CTkLabel(self, text='', anchor='w', height=22, font=AppTheme.caption_font())
        self.info.grid(row=4, column=0, sticky='ew', padx=10)
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

    def set_items(self, items, prefer_source=False):
        previous = self.selector.get()
        self.items = {item['label']: item for item in items}
        labels = list(self.items)
        self.selector.configure(values=labels or ['Нет файлов'], state='readonly' if labels else 'disabled')
        chosen = previous if previous in self.items else (labels[0] if labels else 'Нет файлов')
        if prefer_source:
            chosen = next((label for label, item in self.items.items() if item['category'] == 'Источник'), chosen)
        self.selector.set(chosen)
        self.show_selected()

    def show_selected(self, value=None):
        item = self.items.get(value or self.selector.get())
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
        previous = self.normalized_view()
        data = self.payload
        self._rendering = True
        self.array = None
        self._extent = None
        self.image_artist.set_visible(False)
        self.scatter.set_visible(False)
        self.edges.set_visible(False)
        for line in self.lines:
            line.remove()
        self.lines = []
        self.cax.set_visible(False)
        self.placeholder.set_visible(data is None)
        self.component.configure(state='disabled')
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
                    self.image_artist.set_clim(lo, hi if hi > lo else lo+1.)
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
        self._rendering = False
        self.view_changed()
        self.canvas.draw_idle()

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
        self.ax.set_xlim([left+x*(right-left) for x in view[0]])
        self.ax.set_ylim([bottom+y*(top-bottom) for y in view[1]])
        self.canvas.draw_idle()

    def reset_view(self):
        if self._extent is not None:
            self.apply_view(((0, 1), (0, 1)))
        else:
            self.toolbar.home()

    def view_changed(self, axis=None):
        if not self._rendering and self.on_view_changed is not None:
            self.on_view_changed(self)

    def scroll_zoom(self, event):
        if self._extent is None or event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return
        factor = 1.2 ** (-event.step)
        self._rendering = True
        self.ax.set_xlim([event.xdata+(x-event.xdata)*factor for x in self.ax.get_xlim()])
        self.ax.set_ylim([event.ydata+(y-event.ydata)*factor for y in self.ax.get_ylim()])
        self._rendering = False
        self.view_changed()
        self.canvas.draw_idle()

    def toggle_details(self):
        self.details_visible = not self.details_visible
        animate_visibility(self.controls, self.details_visible,
                           lambda: self.controls.grid(row=2, column=0, sticky='ew', padx=8, pady=3),
                           self.controls.grid_remove)

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
        self._executor.shutdown(wait=True, cancel_futures=True)
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
        self._filter_id = None
        ctk.CTkLabel(self, text='Результаты и сравнение', font=AppTheme.page_title_font()).pack(anchor='w', padx=12, pady=8)
        self.summary = ctk.CTkLabel(self, text='Результаты появятся после расчета', anchor='w')
        self.summary.pack(fill='x', padx=12)
        controls = ctk.CTkFrame(self, fg_color='transparent')
        controls.pack(fill='x', padx=12, pady=8)
        self.category = ctk.CTkOptionMenu(controls, values=['Все результаты'], command=self.filter,
                                         fg_color=AppTheme.NAV_BACKGROUND, button_color=AppTheme.NAV_BACKGROUND)
        self.category.pack(side='left', padx=(0, 8))
        self.search = ctk.CTkEntry(controls, placeholder_text='Поиск по масштабу, каналу или имени')
        self.search.pack(side='left', fill='x', expand=True, padx=8)
        self.search.bind('<KeyRelease>', self.schedule_filter)
        IconButton(controls, 'refresh', 'Обновить список результатов',
                   lambda: self.load_folder(self.folder, force=True)).pack(side='right')
        self.linked = tk.BooleanVar(value=False)
        self._syncing = False
        ctk.CTkCheckBox(self, text='Связать масштаб и положение', variable=self.linked,
                        command=lambda: self.sync_view(self.left)).pack(anchor='w', padx=14, pady=4)
        panes = self.panes = tk.PanedWindow(self, orient='horizontal', sashwidth=8,
                                           background=AppTheme.BORDER, borderwidth=0, opaqueresize=True)
        panes.pack(fill='both', expand=True)
        self.left = ArtifactPane(panes, 'Сравнение A')
        self.right = ArtifactPane(panes, 'Сравнение B')
        panes.add(self.left, minsize=220, stretch='always')
        panes.add(self.right, minsize=220, stretch='always')
        self.left.on_view_changed = self.sync_view
        self.right.on_view_changed = self.sync_view
        self._initial_split = False
        panes.bind('<Map>', self.initialize_split)

    def initialize_split(self, event=None):
        if not self._initial_split:
            self._initial_split = True
            self.panes.sash_place(0, self.panes.winfo_width()//2, 0)

    def sync_view(self, source):
        if not self.linked.get() or self._syncing:
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
        self.folder = folder
        self.artifacts = discover_results(folder)
        categories = ['Все результаты'] + sorted({a['category'] for a in self.artifacts})
        self.category.configure(values=categories)
        if changed or self.category.get() not in categories:
            self.category.set('Все результаты')
            self.search.delete(0, 'end')
        self.left.folder = self.right.folder = folder
        if changed or force:
            for pane in (self.left, self.right):
                pane._request_key = None
                if changed:
                    pane._extent = None
        self.filter(initial=changed)

    def schedule_filter(self, event=None):
        if self._filter_id:
            self.after_cancel(self._filter_id)
        self._filter_id = self.after(180, self.filter)

    def filter(self, selected=None, initial=False):
        if self._filter_id:
            self.after_cancel(self._filter_id)
            self._filter_id = None
        if selected is not None:
            self.category.set(selected)
        selected = self.category.get()
        query = self.search.get().casefold()
        items = [a for a in self.artifacts if (selected == 'Все результаты' or a['category'] == selected)
                 and query in a['label'].casefold()]
        self.summary.configure(text=f'{Path(self.folder).name} · показано {len(items)} из {len(self.artifacts)}')
        self.left.set_items(items, prefer_source=initial)
        self.right.set_items([a for a in items if a['category'] != 'Источник'] or items if initial else items)

    def destroy(self):
        if self._filter_id:
            self.after_cancel(self._filter_id)
            self._filter_id = None
        super().destroy()
