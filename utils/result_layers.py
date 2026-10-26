"""Explicit spatial overlays, loaded without blocking Tk."""
import tkinter as tk
from pathlib import Path
from tkinter import colorchooser
import customtkinter as ctk
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from history.result_data import load_result
from utils.theme import AppTheme
from utils.scrollable_dropdown import ScrollableComboBox
from utils.viewport import PointIndex, clip_segments
from utils.ml_classes import class_styles, visible_points
from utils.layer_catalog import result_id


def _layer_group(item):
    return item.get('category') or 'Прочее'


def _layer_direction(item):
    value = item.get('cwt_direction') or item.get('direction')
    mapping = {'row': 'По строкам', 'rows': 'По строкам',
               'col': 'По столбцам', 'columns': 'По столбцам'}
    if value not in mapping:
        from pathlib import Path
        name = Path(item.get('path', '')).name.lower()
        if 'cwtrow' in name or 'cwt1d_row' in name:
            value = 'row'
        elif 'cwtcol' in name or 'cwt1d_col' in name:
            value = 'col'
    return mapping.get(value, 'Без направления')


class LayerControls(ctk.CTkFrame):
    CATEGORIES = {'Экстремумы', 'Огибающие', 'KNN', 'ML-кластеры'}
    COLORS = ['#ff615d', '#4ed4e9', '#ffcc57', '#d58bff', '#65dc8b']

    def __init__(self, pane, master=None, ml_master=None):
        super().__init__(master or pane, fg_color='transparent')
        self.pane = pane
        self.layers = []
        self.items = {}
        self.job = None
        self.legend = None
        self.base = None
        self._style_cache = {}
        self.grid_columnconfigure(0, weight=1)
        self._dialog = None
        self._picker_root = None
        self._dialog_state = {}
        selector_row = ctk.CTkFrame(self, fg_color='transparent')
        selector_row.grid(row=0, column=0, columnspan=2, sticky='ew')
        ctk.CTkLabel(selector_row, text='Тип слоя', anchor='w').pack(fill='x', pady=(0, 2))
        self.group_selector = ScrollableComboBox(
            selector_row, values=['Все типы'], state='readonly', width=250,
            command=lambda _value: self._on_group_changed())
        self.group_selector.set('Все типы')
        self.group_selector.pack(fill='x', pady=(0, 6))
        ctk.CTkLabel(selector_row, text='Направление', anchor='w').pack(fill='x', pady=(0, 2))
        self.direction_selector = ScrollableComboBox(
            selector_row, values=['Все направления'], state='readonly', width=250,
            command=lambda _value: self._apply_filters())
        self.direction_selector.set('Все направления')
        self.direction_selector.pack(fill='x', pady=(0, 6))
        # Two-step layer creation: keep the picker hidden until requested.
        self.legacy_add_button = ctk.CTkButton(
            selector_row, text='+ Добавить слой', height=30,
            fg_color=AppTheme.NAV_HOVER, command=self.open_layer_picker)
        self.legacy_add_button.pack(fill='x', pady=(5, 5))
        self.picker_frame = ctk.CTkFrame(selector_row, fg_color='transparent')
        ctk.CTkLabel(self.picker_frame, text='Выберите данные слоя', anchor='w').pack(
            fill='x', pady=(4, 2))
        self.selector = ScrollableComboBox(
            self.picker_frame, values=['Нет слоёв'], state='readonly', width=220 if master else 577,
            dropdown_wide=True, dropdown_search=True,
            command=self._on_layer_chosen)
        self.selector.pack(fill='x')
        self._picker_open = False
        self._source_items = []
        # Legacy comboboxes remain unmounted only for old state-restoration and
        # compatibility tests. The actual user-facing picker is a separate window.
        selector_row.grid_remove()
        self.add_button = ctk.CTkButton(
            self, text='+ Добавить слой', height=32,
            fg_color=AppTheme.PRIMARY, command=self.open_layer_picker)
        self.add_button.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 7))
        self.status = ctk.CTkLabel(self, text='Добавленные слои: 0', anchor='w', height=20)
        self.status.grid(row=2, column=0, columnspan=2, sticky='ew')

    @staticmethod
    def _display_label(item):
        if item.get('category') != 'ML-кластеры':
            return (item.get('display_label') or item['label'])[-120:]
        algorithm = item.get('algorithm') or ('DBSCAN' if 'dbscan' in item['label'].casefold() else 'ML')
        channel = item.get('channel') or 'канал —'
        scale = item.get('scale') or '—'
        point_type = item.get('point_type') or 'точки'
        direction = {
            'row': 'CWT по строкам', 'rows': 'CWT по строкам',
            'col': 'CWT по столбцам', 'columns': 'CWT по столбцам',
        }.get(item.get('direction'), item.get('direction') or 'направление —')
        point_type = {
            'max_by_row': 'максимумы по строкам',
            'min_by_row': 'минимумы по строкам',
            'max_by_column': 'максимумы по столбцам',
            'min_by_column': 'минимумы по столбцам',
        }.get(point_type, point_type)
        return f'[{algorithm}] {channel} · масштаб {scale} · {point_type} · {direction}'

    def set_items(self, items):
        self._source_items = [item for item in items if item['category'] in self.CATEGORIES]
        groups = [g for g in ('Экстремумы', 'Огибающие', 'KNN', 'ML-кластеры')
                  if any(_layer_group(a) == g for a in self._source_items)]
        self.group_selector.configure(values=['Все типы'] + groups)
        if self.group_selector.get() not in groups:
            self.group_selector.set('Все типы')
        self._update_directions()
        self._apply_filters()

    def _update_directions(self):
        group = self.group_selector.get()
        relevant = [a for a in self._source_items
                    if group == 'Все типы' or _layer_group(a) == group]
        directions = [d for d in ('По строкам', 'По столбцам', 'Без направления')
                      if any(_layer_direction(a) == d for a in relevant)]
        previous = self.direction_selector.get()
        self.direction_selector.configure(values=['Все направления'] + directions,
                                          state='readonly' if len(directions) > 1 else 'disabled')
        self.direction_selector.set(previous if previous in directions else 'Все направления')

    def _on_group_changed(self):
        self._update_directions()
        self._apply_filters()

    def _apply_filters(self):
        filtered = []
        base = self.pane.selected_item or {}
        for item in self._source_items:
            if not self.pane.payload or not self.pane.payload.get('spatial'):
                continue
            if item.get('task_folder') != base.get('task_folder'):
                continue
            if item.get('path') == base.get('path'):
                continue
            if any(l['item']['path'] == item['path'] for l in self.layers):
                continue
            if (self.group_selector.get() != 'Все типы' and
                    _layer_group(item) != self.group_selector.get()):
                continue
            if (self.direction_selector.get() != 'Все направления' and
                    _layer_direction(item) != self.direction_selector.get()):
                continue
            filtered.append(item)
        filtered.sort(key=lambda item: (
            item['category'] != 'ML-кластеры',
            item.get('algorithm') or '', item.get('channel') or '', item.get('point_type') or '',
            item.get('direction') or '', float(item.get('scale') or 0), item['label']))
        self.items = {}
        for item in filtered:
            # The full semantic provenance is needed: two CWT axes can have
            # independent envelope/point results with identical scales/kinds.
            cwt = _layer_direction(item)
            axis = {'row': 'поиск по строкам', 'col': 'поиск по столбцам'}.get(
                item.get('feature_axis'))
            kind = {'max': 'Максимумы', 'min': 'Минимумы',
                    'upper': 'Верхняя', 'lower': 'Нижняя'}.get(item.get('point_kind'))
            parts = [item.get('task_name'), item.get('category'),
                     f"s={item['scale']}" if item.get('scale') else None,
                     item.get('channel'), cwt if cwt != 'Без направления' else None,
                     axis, kind]
            if item['category'] == 'ML-кластеры':
                parts.extend([item.get('algorithm'), item.get('ml_run')])
            base = ' · '.join(str(part) for part in parts if part)
            label = base
            counter = 2
            while label in self.items:
                # Only true clashes get a file-id suffix, not axis variants.
                label = f'{base} · {Path(item["path"]).stem} #{counter}'
                counter += 1
            self.items[label] = item
        values = list(self.items)
        previous = self.selector.get()
        self.selector.configure(state='readonly', values=values or ['Нет слоёв'])
        self.selector.set(previous if previous in self.items else values[0] if values else 'Нет слоёв')
        self.selector.configure(state='readonly' if values else 'disabled')

    def add_item(self, item):
        if any(layer['item'].get('path') == item['path'] for layer in self.layers):
            self.status.configure(text='Этот ML-слой уже добавлен')
            return
        self._append_layer(item)
        self.status.configure(text='Загрузка ML-слоя…')
        if self.job is None:
            self.job = self.after(20, self.poll)

    @staticmethod
    def _compact_layer_name(item):
        """Keep cards readable at the minimum sidebar width."""
        category = (item.get('algorithm') or 'ML') if item.get('category') == 'ML-кластеры' else item.get('category', 'Слой')
        channel = item.get('channel') or ''
        scale = item.get('scale')
        kind = item.get('point_kind') or ''
        original = (item.get('display_label') or item.get('label') or '').lower()
        if kind in ('upper', 'lower') or category == 'Огибающие':
            detail = 'Верхняя' if kind == 'upper' or 'верхняя' in original else 'Нижняя' if kind == 'lower' or 'нижняя' in original else ''
        elif kind in ('max', 'min') or category == 'Экстремумы':
            detail = 'Максимумы' if kind == 'max' or 'максимум' in original else 'Минимумы' if kind == 'min' or 'минимум' in original else ''
        else:
            detail = ''
        pieces = [category, detail]
        if scale not in (None, ''):
            pieces.append(f's={scale}')
        if channel:
            pieces.append(str(channel))
        name = ' · '.join(piece for piece in pieces if piece)
        return name or (item.get('display_label') or item.get('label') or 'Слой')[:65]

    def _append_layer(self, item, data=None, *, label=None, base=False):
        layer = dict(item=item, data=data, visible=tk.BooleanVar(value=True), alpha=.8,
                     marker_scale=1.0, color=self.COLORS[len(self.layers) % len(self.COLORS)], artists=[])
        if data is None:
            layer['future'] = self.pane._executor.submit(load_result, item, self.pane.folder)
        else:
            layer['future'] = None
        row = layer['row'] = ctk.CTkFrame(
            self, fg_color=AppTheme.NAV_HOVER, corner_radius=4)
        row.grid(row=3 if base else len(self.layers)+4, column=0, columnspan=2,
                 sticky='ew', pady=(4, 2), padx=1)
        row.grid_columnconfigure(1, weight=1)
        layer['base'] = base
        row_text = ('Основной результат · ' if base else '') + self._compact_layer_name(item)
        layer['full_label'] = label or self._display_label(item)
        # No long label is allowed to expand the panel's requested width.
        visible_box = ctk.CTkCheckBox(
            row, text='', width=25, variable=layer['visible'],
            command=self.redraw)
        visible_box.grid(row=0, column=0, padx=(7, 2), pady=(7, 3), sticky='w')
        name_label = ctk.CTkLabel(
            row, text=row_text, anchor='w', justify='left',
            font=AppTheme.body_font(), wraplength=148)
        name_label.grid(row=0, column=1, sticky='ew', padx=(0, 3), pady=(6, 3))
        layer['name_label'] = name_label
        if not base:
            ctk.CTkButton(row, text='×', width=25, height=25,
                fg_color=AppTheme.NAV_BACKGROUND, command=lambda: self.remove(layer)).grid(
                row=0, column=2, padx=(2, 7), pady=(5, 2))
        if item['category'] != 'ML-кластеры':
            swatch = ctk.CTkButton(row, text='', width=26, height=22, corner_radius=4,
                fg_color=layer['color'], hover_color=layer['color'], command=lambda: self.choose_color(layer))
            swatch.grid(row=1, column=0, sticky='w', padx=7, pady=4)
            layer['swatch'] = swatch
            ctk.CTkLabel(row, text='Цвет точек', font=AppTheme.caption_font()).grid(row=1, column=1, sticky='w')
        layer['class_frame'] = ctk.CTkFrame(row, fg_color='transparent')
        layer['class_frame'].grid(row=7, column=0, columnspan=3, sticky='ew', padx=7, pady=4)
        ctk.CTkLabel(
            row, text='Прозрачность', anchor='w',
            font=AppTheme.caption_font()).grid(
            row=2, column=0, columnspan=2, sticky='w', padx=(8, 3), pady=(0, 2))
        slider = ctk.CTkSlider(
            row, from_=0, to=1, width=105,
            command=lambda value: self.set_alpha(layer, value))
        slider.set(.8)
        layer['slider'] = slider
        slider.grid(row=3, column=0, columnspan=3, sticky='ew',
                    padx=(9, 9), pady=(0, 5))
        ctk.CTkLabel(row, text='Размер точек', anchor='w',
                     font=AppTheme.caption_font()).grid(
            row=4, column=0, columnspan=3, sticky='w', padx=(8, 3), pady=(0, 2))
        size_slider = ctk.CTkSlider(
            row, from_=0.25, to=4.0, number_of_steps=75,
            command=lambda value: self.set_marker_scale(layer, value))
        size_slider.set(1.0)
        size_slider.grid(row=5, column=0, columnspan=3, sticky='ew',
                         padx=9, pady=(0, 8))
        layer['size_slider'] = size_slider
        if not base:
            self.layers.append(layer)
        return layer

    def set_base(self, item, data):
        is_ml = bool(item and data and item['category'] == 'ML-кластеры' and data['kind'] == 'points')
        if self.base and (not is_ml or self.base['item']['path'] != item['path'] or self.base['data'] is not data):
            self.base['row'].destroy()
            self.base = None
        if is_ml:
            for layer in list(self.layers):
                if layer['item']['path'] == item['path']:
                    self.remove(layer)
            if self.base is None:
                self.base = self._append_layer(item, data, base=True)
                self.base['alpha'] = 1.
                self.base['slider'].set(1.)
            self.base['artists'] = [self.pane.scatter, self.pane.edges]
            if data.get('continuous') and data.get('class_labels') is None:
                self.pane.scatter.set_cmap('inferno')
                self.pane.scatter.set_clim(0, 1)
            self.base.pop('_viewport', None)
        self._apply_filters()

    def _class_controls(self, layer):
        if 'styles' in layer or not layer.get('data'):
            return
        layer['styles'] = class_styles(layer['data'])
        cached = self._style_cache.get(layer['item']['path'], {})
        for key, style in layer['styles'].items():
            if key in cached:
                style.update(color=cached[key]['color'], visible=cached[key]['visible'])
        self._style_cache[layer['item']['path']] = layer['styles']
        host = layer['class_frame']
        if layer['styles']:
            def toggle():
                if host.winfo_manager():
                    host.grid_remove()
                    layer['classes_toggle'].configure(text=f'▸ Классы · {len(layer["styles"])}')
                else:
                    host.grid()
                    layer['classes_toggle'].configure(text=f'▾ Классы · {len(layer["styles"])}')
            button = ctk.CTkButton(layer['row'], text=f'▾ Классы · {len(layer["styles"])}',
                anchor='w', fg_color='transparent', command=toggle)
            button.grid(row=6, column=0, columnspan=3, sticky='ew', padx=5)
            layer['classes_toggle'] = button
        host.grid_columnconfigure(0, weight=1)
        for index, (key, style) in enumerate(layer['styles'].items()):
            variable = tk.BooleanVar(value=style['visible'])
            def changed(k=key, v=variable):
                layer['styles'][k]['visible'] = v.get()
                layer.pop('_viewport', None)
                self.redraw()
            box = ctk.CTkCheckBox(host, text=f"{style['name']} · {style['count']}",
                variable=variable, width=120, font=AppTheme.caption_font(), command=changed)
            box.grid(row=index, column=0, sticky='w', pady=4)
            button = ctk.CTkButton(host, text='', width=24, height=24,
                fg_color=style['color'], hover_color=style['color'])
            button.configure(command=lambda k=key, b=button: self.choose_class_color(layer, k, b))
            button.grid(row=index, column=1, sticky='e', padx=(4, 0))
        if layer['item']['category'] == 'ML-кластеры' and not layer['styles']:
            ctk.CTkLabel(host, text='Состояния не сохранены; показана исходная оценка',
                wraplength=245, justify='left', text_color=AppTheme.TEXT_SECONDARY).grid(sticky='ew')

    def choose_class_color(self, layer, key, button):
        _, color = colorchooser.askcolor(color=layer['styles'][key]['color'],
            title=layer['styles'][key]['name'], parent=self.winfo_toplevel())
        if color:
            layer['styles'][key]['color'] = color
            button.configure(fg_color=color, hover_color=color)
            layer.pop('_viewport', None)
            self.redraw()

    def open_layer_picker(self):
        """Open one explorer; never create Tk children while root is closing."""
        from utils.tk_lifecycle import active_widget_root
        root = active_widget_root(self.pane)
        if root is None:
            return

        dialog = self._dialog
        if dialog is not None:
            try:
                if dialog.winfo_exists():
                    dialog.lift()
                    dialog.focus_set()
                    return
            except tk.TclError:
                pass
            self._dialog = None

        if not (self.pane.payload and self.pane.payload.get('spatial')):
            self.status.configure(text='Сначала выберите пространственный результат')
            return

        from utils.layer_picker import LayerPicker
        try:
            dialog = LayerPicker(
                root, items=self.pane._experiment_items,
                base_item=self.pane.selected_item, base_data=self.pane.payload,
                folder=self.pane.folder,
                added_ids={result_id(layer['item']) for layer in self.layers},
                on_add=self.add_items, on_close=self._picker_closed,
                state=self._dialog_state, executor=self.pane._executor)
        except tk.TclError:
            # CTk may receive a previously queued click after WM_DELETE_WINDOW.
            # Do not mask genuine errors while the application is alive.
            if active_widget_root(self.pane) is None:
                return
            raise

        if active_widget_root(self.pane) is None:
            dialog.close()
            return
        self._dialog = dialog
        self._picker_root = root
        self._picker_open = True
        if hasattr(root, 'register_child_window'):
            root.register_child_window(dialog)

    def _picker_closed(self, state):
        root = self._picker_root
        dialog = self._dialog
        if root is not None and dialog is not None and hasattr(root, 'unregister_child_window'):
            root.unregister_child_window(dialog)
        self._dialog_state = state
        self._dialog = None
        self._picker_root = None
        self._picker_open = False

    def toggle_layer_picker(self):
        if self._dialog is not None:
            self.hide_layer_picker()
        else:
            self.open_layer_picker()

    def hide_layer_picker(self):
        dialog = self._dialog
        if dialog is not None:
            try:
                dialog.close()
            except tk.TclError:
                # Parent Tk interpreter may already be unavailable.
                self._picker_closed(self._dialog_state)

    def add_items(self, validated):
        """Commit an already verified batch atomically, without I/O on Tk."""
        for item, data in validated:
            if item.get('path') == (self.pane.selected_item or {}).get('path'):
                return
            if any(result_id(layer['item']) == result_id(item) for layer in self.layers):
                return
        for item, data in validated:
            self._append_layer(item, data=data)
        self._apply_filters()
        self.redraw()

    def _on_layer_chosen(self, _value=None):
        # The selector is an input, not an additional permanent toolbar.
        self.add_selected()
        self.hide_layer_picker()

    def add_selected(self):
        item = self.items.get(self.selector.get())
        if not item or any(layer['item'].get('path') == item['path'] for layer in self.layers):
            return
        self._append_layer(item)
        self.status.configure(text='Загрузка слоя…')
        if self.job is None:
            self.job = self.after(20, self.poll)

    def choose_color(self, layer):
        # Native Tk colour dialog allows selecting any RGB shade.
        _rgb, color = colorchooser.askcolor(color=layer['color'],
                                           title='Цвет точек слоя', parent=self.winfo_toplevel())
        if not color:
            return
        layer['color'] = color
        layer['swatch'].configure(fg_color=color, hover_color=color)
        data = layer.get('data') or {}
        # ML cluster colours encode numerical classes/anomaly: do not override.
        if layer['item']['category'] != 'ML-кластеры' or 'colors' not in data:
            if layer['artists']:
                layer['artists'][0].set_color(color)
        if len(layer['artists']) > 1:
            layer['artists'][1].set_color(color)
        self.redraw()

    def set_marker_scale(self, layer, value):
        layer['marker_scale'] = float(value)
        layer.pop('_viewport', None)
        self.update_marker_sizes()
        self.pane.canvas.draw_idle()

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
        for index, layer in enumerate(([self.base] if self.base else []) + self.layers):
            data = layer['data']
            compatible = (data is not None and base_shape == tuple(data['shape'])
                          and data.get('source_id') == (pane.payload or {}).get('source_id'))
            visible = bool(layer['visible'].get() and compatible)
            self._class_controls(layer)
            if compatible and not layer['artists']:
                points = np.asarray(data['points'])
                clustered = layer['item']['category'] == 'ML-кластеры' and 'colors' in data
                kwargs = (dict(c=data['colors'], cmap='inferno', vmin=0.0, vmax=1.0) if clustered and data.get('continuous') else
                          dict(c=data['colors'], cmap='tab20') if clustered else dict(c=layer['color']))
                marker = 'o'
                size = 12
                dots = pane.ax.scatter(points[:, 0], points[:, 1], s=size, marker=marker,
                                       linewidths=0, zorder=5+index, **kwargs)
                lines = LineCollection(data.get('edges', []), colors=layer['color'], linewidths=.7, zorder=4+index)
                pane.ax.add_collection(lines)
                layer['artists'] = [dots, lines]
            for artist in layer['artists']:
                artist.set_visible(visible)
                artist.set_alpha(layer['alpha'])
            if visible:
                item = layer['item']
                label = self._compact_layer_name(item)
                if layer.get('styles'):
                    for style in layer['styles'].values():
                        if style['visible'] and style['count']:
                            handles.append(Line2D([], [], marker='o', linestyle='', color=style['color'],
                                label=f"{item.get('algorithm') or 'ML'} · {style['name']}" +
                                      (f" · {item.get('channel') or ''} s={item.get('scale') or '—'}" if self.layers else '')))
                elif item['category'] == 'ML-кластеры':
                    handles.append(Line2D([], [], marker='o', linestyle='', color='#edbd58', label=label + ' · исходная оценка'))
                else:
                    handles.append(Line2D([], [], marker='o', linestyle='', color=layer['color'], label=label))
            if layer.get('error'):
                messages.append(layer['error'])
            elif data is not None and not compatible:
                messages.append('слой скрыт: нужны тот же источник и размеры пространственной карты')
            elif layer.get('future') is not None:
                messages.append('Загрузка слоя…')

        self.status.configure(text='; '.join(dict.fromkeys(messages)) or f'Добавленные слои: {len(self.layers)}')
        self.update_marker_sizes()
        pane.canvas.draw_idle()

    def update_marker_sizes(self):
        """Update visible geometry only, retaining full source data for export."""
        x0, x1 = sorted(self.pane.ax.get_xlim())
        y0, y1 = sorted(self.pane.ax.get_ylim())
        for layer in ([self.base] if getattr(self, 'base', None) else []) + self.layers:
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
            indices, rgba = visible_points(layer['data'], layer.get('styles', {}), indices)
            layer['artists'][0].set_offsets(points[indices, :2])
            if rgba is not None:
                layer['artists'][0].set_array(None)
                layer['artists'][0].set_facecolors(rgba)
            elif layer['item']['category'] == 'ML-кластеры' and 'colors' in layer['data']:
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
            layer['artists'][0].set_sizes([max(.01, min(12., 5000/max(1, count))) * layer.get('marker_scale', 1.0)])

    def remove(self, layer):
        if layer.get('future') is not None:
            layer['future'].cancel()
        for artist in layer['artists']:
            artist.remove()
        layer['row'].destroy()
        self.layers.remove(layer)
        for i, remaining in enumerate(self.layers):
            remaining['row'].grid_configure(row=i+4)
        self._apply_filters()
        self.redraw()

    def clear(self):
        self.hide_layer_picker()
        for layer in list(self.layers):
            self.remove(layer)

    def destroy(self):
        self.hide_layer_picker()
        if self.job is not None:
            self.after_cancel(self.job)
            self.job = None
        for layer in self.layers:
            if layer.get('future') is not None:
                layer['future'].cancel()
        super().destroy()
