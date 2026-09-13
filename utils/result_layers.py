"""Explicit spatial overlays, loaded without blocking Tk."""
import tkinter as tk
import customtkinter as ctk
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from history.result_data import load_result
from utils.theme import AppTheme
from utils.scrollable_dropdown import ScrollableComboBox
from utils.viewport import PointIndex, clip_segments


class LayerControls(ctk.CTkFrame):
    CATEGORIES = {'Экстремумы', 'Огибающие', 'KNN', 'ML-кластеры'}
    COLORS = ['#ff615d', '#4ed4e9', '#ffcc57', '#d58bff', '#65dc8b']

    def __init__(self, pane):
        super().__init__(pane, fg_color='transparent')
        self.pane = pane
        self.layers = []
        self.items = {}
        self.job = None
        self.legend = None
        self.grid_columnconfigure(0, weight=1)
        selector_row = ctk.CTkFrame(self, fg_color='transparent')
        selector_row.grid(row=0, column=0, columnspan=2, sticky='w')
        self.selector = ScrollableComboBox(selector_row, values=['Нет слоёв'], state='readonly', width=577)
        self.selector.pack(side='left')
        ctk.CTkButton(selector_row, text='+ Слой', width=75, fg_color=AppTheme.NAV_HOVER,
                      command=self.add_selected).pack(side='left', padx=(8, 0))
        self.status = ctk.CTkLabel(self, text='Слои: точки, огибающие, KNN, кластеры', anchor='w', height=20)
        self.status.grid(row=1, column=0, columnspan=2, sticky='ew')

    def set_items(self, items):
        self.items = {a['label']: a for a in items if a['category'] in self.CATEGORIES}
        values = list(self.items)
        previous = self.selector.get()
        self.selector.configure(state='readonly', values=values or ['Нет слоёв'])
        self.selector.set(previous if previous in self.items else values[0] if values else 'Нет слоёв')
        self.selector.configure(state='readonly' if values else 'disabled')

    def add_selected(self):
        item = self.items.get(self.selector.get())
        if not item or any(layer['item']['path'] == item['path'] for layer in self.layers):
            return
        layer = dict(item=item, data=None, visible=tk.BooleanVar(value=True), alpha=.8,
                     color=self.COLORS[len(self.layers) % len(self.COLORS)], artists=[])
        layer['future'] = self.pane._executor.submit(load_result, item, self.pane.folder)
        row = layer['row'] = ctk.CTkFrame(self, fg_color='transparent')
        row.grid(row=len(self.layers)+2, column=0, columnspan=2, sticky='ew', pady=2)
        row.grid_columnconfigure(0, weight=1)
        ctk.CTkCheckBox(row, text=item['label'][-65:], variable=layer['visible'],
                        command=self.redraw).grid(row=0, column=0, sticky='w')
        slider = ctk.CTkSlider(row, from_=0, to=1, width=95,
                              command=lambda value: self.set_alpha(layer, value))
        slider.set(.8)
        layer['slider'] = slider
        slider.grid(row=0, column=1, padx=6)
        ctk.CTkButton(row, text='×', width=28, fg_color=AppTheme.NAV_HOVER,
                      command=lambda: self.remove(layer)).grid(row=0, column=2)
        self.layers.append(layer)
        self.status.configure(text='Загрузка слоя…')
        if self.job is None:
            self.job = self.after(20, self.poll)

    def set_alpha(self, layer, value):
        layer['alpha'] = float(value)
        for artist in layer['artists']:
            artist.set_alpha(layer['alpha'])
        self.pane.canvas.draw_idle()

    def poll(self):
        self.job = None
        changed = False
        for layer in list(self.layers):
            future = layer.get('future')
            if future is None or not future.done():
                continue
            layer['future'] = None
            changed = True
            try:
                data = future.result()
                if data['kind'] != 'points' or not data.get('spatial'):
                    raise ValueError('слой требует численных координат точек и размера исходника')
                layer['data'] = data
            except Exception as error:
                layer['error'] = str(error)
        if changed:
            self.redraw()
        if any(l.get('future') is not None for l in self.layers):
            self.job = self.after(30, self.poll)

    def redraw(self):
        if self.legend is not None:
            self.legend.remove()
            self.legend = None
        handles, messages = [], []
        pane = self.pane
        base_shape = tuple(pane.payload['shape']) if pane.payload and pane._extent is not None else None
        for index, layer in enumerate(self.layers):
            data = layer['data']
            compatible = (data is not None and base_shape == tuple(data['shape'])
                          and data.get('source_id') == (pane.payload or {}).get('source_id'))
            visible = bool(layer['visible'].get() and compatible)
            if compatible and not layer['artists']:
                points = np.asarray(data['points'])
                clustered = layer['item']['category'] == 'ML-кластеры' and 'colors' in data
                kwargs = dict(c=data['colors'], cmap='tab20') if clustered else dict(c=layer['color'])
                dots = pane.ax.scatter(points[:, 0], points[:, 1], s=12, linewidths=0, zorder=5+index, **kwargs)
                lines = LineCollection(data.get('edges', []), colors=layer['color'], linewidths=.7, zorder=4+index)
                pane.ax.add_collection(lines)
                layer['artists'] = [dots, lines]
            for artist in layer['artists']:
                artist.set_visible(visible)
                artist.set_alpha(layer['alpha'])
            if visible:
                item = layer['item']
                name = item['label'].casefold()
                details = [word for word in ('верхняя', 'нижняя', 'максимумы', 'минимумы') if word in name]
                category = 'Точки огибающей' if item['category'] == 'Огибающие' else item['category']
                label = f'{index+1}. {category} {" ".join(details)} · a={item.get("scale") or "—"} · {item.get("channel") or ""}'
                if layer['item']['category'] == 'ML-кластеры' and 'colors' in data:
                    dots = layer['artists'][0]
                    for cluster in np.unique(data['colors'])[:10]:
                        handles.append(Line2D([], [], marker='o', linestyle='', color=dots.cmap(dots.norm(cluster)),
                                              label=f'{label} · кластер {cluster:g}'))
                else:
                    handles.append(Line2D([], [], marker='o', linestyle='', color=layer['color'], label=label))
            if layer.get('error'):
                messages.append(layer['error'])
            elif data is not None and not compatible:
                messages.append('слой скрыт: нужны тот же источник и размеры пространственной карты')
            elif layer.get('future') is not None:
                messages.append('Загрузка слоя…')
        if handles:
            self.legend = pane.ax.legend(handles=handles, fontsize=6, loc='upper left',
                                         facecolor=AppTheme.NAV_BACKGROUND, labelcolor=AppTheme.TEXT_ON_DARK)
        self.status.configure(text='; '.join(dict.fromkeys(messages)) or f'Слоёв: {len(self.layers)} · ползунок: прозрачность')
        self.update_marker_sizes()
        pane.canvas.draw_idle()

    def update_marker_sizes(self):
        """Update visible geometry only, retaining full source data for export."""
        x0, x1 = sorted(self.pane.ax.get_xlim())
        y0, y1 = sorted(self.pane.ax.get_ylim())
        for layer in self.layers:
            if not layer['artists'] or not layer['artists'][0].get_visible():
                continue
            points = np.asarray(layer['data']['points'])
            bounds = (x0, x1, y0, y1)
            viewport = (*bounds, self.pane.ax.bbox.width, self.pane.ax.bbox.height)
            if layer.get('_viewport') == viewport:
                continue
            layer['_viewport'] = viewport
            if '_point_index' not in layer:
                layer['_point_index'] = PointIndex(points)
                layer['_segments'] = np.asarray(layer['data'].get('edges', []), dtype=float).reshape(-1, 2, 2)
                segments = layer['_segments']
                layer['_edge_min'] = segments.min(axis=1)
                layer['_edge_max'] = segments.max(axis=1)
            index = layer['_point_index']
            count = len(index.query(bounds))
            # Include marker footprints just outside the axes, not only centers.
            margin_x = (x1-x0)*8/max(1, self.pane.ax.bbox.width)
            margin_y = (y1-y0)*8/max(1, self.pane.ax.bbox.height)
            expanded = (x0-margin_x, x1+margin_x, y0-margin_y, y1+margin_y)
            indices = index.query(expanded)
            layer['artists'][0].set_offsets(points[indices, :2])
            if layer['item']['category'] == 'ML-кластеры' and 'colors' in layer['data']:
                layer['artists'][0].set_array(np.asarray(layer['data']['colors'])[indices])
            segments = layer['_segments']
            if len(segments):
                low, high = layer['_edge_min'], layer['_edge_max']
                visible = ((high[:, 0] >= expanded[0]) & (low[:, 0] <= expanded[1]) &
                           (high[:, 1] >= expanded[2]) & (low[:, 1] <= expanded[3]))
                # Rebuilding large LineCollections costs more than backend clipping.
                if np.count_nonzero(visible) < len(segments)*.25:
                    layer['artists'][1].set_segments(clip_segments(segments[visible], expanded))
                    layer['_edges_subset'] = True
                elif layer.get('_edges_subset', False):
                    layer['artists'][1].set_segments(segments)
                    layer['_edges_subset'] = False
            layer['artists'][0].set_sizes([max(.01, min(12., 5000/max(1, count)))])

    def remove(self, layer):
        if layer.get('future') is not None:
            layer['future'].cancel()
        for artist in layer['artists']:
            artist.remove()
        layer['row'].destroy()
        self.layers.remove(layer)
        for i, remaining in enumerate(self.layers):
            remaining['row'].grid_configure(row=i+2)
        self.redraw()

    def clear(self):
        for layer in list(self.layers):
            self.remove(layer)

    def destroy(self):
        if self.job is not None:
            self.after_cancel(self.job)
            self.job = None
        for layer in self.layers:
            if layer.get('future') is not None:
                layer['future'].cancel()
        super().destroy()
