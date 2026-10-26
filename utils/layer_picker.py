"""Modal, experiment-wide result picker for overlays.

CustomTkinter tree and result cards. The result catalogue remains independent
of the filters on the 'Данные' tab, and no heavy result is loaded on navigation.
"""
from tkinter import messagebox
import tkinter as tk
import customtkinter as ctk

from history.result_data import load_result
from utils.layer_catalog import GROUPS, card_title, details, result_id, search_results, task_id
from utils.layer_compatibility import check_loaded, preflight
from utils.rendering import enable_dark_titlebar
from utils.layer_picker_widgets import (ACTIVE, BG, BORDER, BLUE, CARD, FAINT,
                                        MUTED, PANEL, SURFACE, TEXT, ExplorerTree,
                                        VirtualResultList, icon)

MAX_BATCH = 20


class LayerPicker(ctk.CTkToplevel):
    """Search, multi-select and atomically add compatible overlay results."""

    # Avoid delayed native Windows title bar recoloring after Tk shutdown.
    _deactivate_windows_window_header_manipulation = True

    def __init__(self, parent, *, items, base_item, base_data, folder,
                 added_ids, on_add, on_close, state=None, executor=None):
        super().__init__(parent)
        self.title('Выбор данных слоя — Wavelets Analysis Studio')
        self._state = dict(state or {})
        self.geometry(self._state.get('geometry', '1050x710'))
        self.minsize(820, 535)
        self.transient(parent.winfo_toplevel())
        self.configure(fg_color=BG)
        self.protocol('WM_DELETE_WINDOW', self.close)
        # Match the main app's dark Windows caption without CTk's delayed titlebar calls.
        self.bind('<Map>', self._on_map, add='+')
        self.on_add = on_add
        self.on_close = on_close
        self.base_item = base_item
        self.base_data = base_data
        self.folder = folder
        self.executor = executor
        self.items = [it for it in items if it.get('category') in GROUPS]
        self.by_id = {result_id(it): it for it in self.items}
        self.added_ids = set(added_ids)
        self.selected = set()
        self.selection_order = []
        self._busy = False
        self._waiting = None
        self._poll_job = None
        self._search_job = None
        self._todo = []
        self._loaded = []
        self._alive = True
        self._titlebar_job = None
        self._selected_scope = None
        self._nav_width = int(self._state.get('sash', 355))
        self._drag_x = 0

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # No redundant title or intro card: the app window title already names the dialog.
        toolbar = ctk.CTkFrame(self, fg_color='transparent')
        toolbar.grid(row=0, column=0, sticky='ew', padx=22, pady=(21, 15))
        toolbar.grid_columnconfigure(0, weight=1)
        search_box = ctk.CTkFrame(toolbar, fg_color=PANEL, border_width=1,
                                  border_color=BORDER, corner_radius=12, height=43)
        search_box.grid(row=0, column=0, sticky='ew', padx=(0, 15))
        search_box.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(search_box, text='', image=icon('search', '#858585', 18),
                     width=20).grid(row=0, column=0, padx=(14, 7), pady=7)
        self.search_var = tk.StringVar()
        self.search = ctk.CTkEntry(search_box, textvariable=self.search_var,
                                   placeholder_text='Поиск по названию, масштабу, каналу, алгоритму...',
                                   fg_color='transparent', border_width=0,
                                   text_color=TEXT, height=36,
                                   font=ctk.CTkFont(size=12))
        self.search.grid(row=0, column=1, sticky='ew', padx=(0, 9), pady=3)
        self.search_var.trace_add('write', lambda *_: self._schedule_search())
        self.everywhere = tk.BooleanVar(value=False)
        self.scope_switch = ctk.CTkSwitch(toolbar, text='Весь эксперимент',
                                          font=ctk.CTkFont(size=11), width=158,
                                          switch_width=36, switch_height=20,
                                          progress_color=BLUE, button_color='#dadada',
                                          fg_color='#505050', text_color=MUTED,
                                          variable=self.everywhere, command=self.refresh)
        self.scope_switch.grid(row=0, column=1, sticky='e')

        self.split = ctk.CTkFrame(self, fg_color='transparent')
        self.split.grid(row=1, column=0, sticky='nsew', padx=22)
        self.split.grid_rowconfigure(0, weight=1)
        self.split.grid_columnconfigure(0, minsize=self._nav_width, weight=0)
        self.split.grid_columnconfigure(1, minsize=10, weight=0)
        self.split.grid_columnconfigure(2, weight=1, minsize=360)

        nav = ctk.CTkFrame(self.split, fg_color=PANEL, corner_radius=16,
                           border_width=1, border_color=BORDER, width=self._nav_width)
        nav.grid(row=0, column=0, sticky='nsew')
        nav.grid_columnconfigure(0, weight=1)
        nav.grid_rowconfigure(1, weight=1)
        nav.grid_propagate(False)
        self.nav = nav
        ctk.CTkLabel(nav, text='Дерево результатов', font=ctk.CTkFont(size=12, weight='bold'),
                     text_color=MUTED, anchor='w').grid(row=0, column=0, sticky='ew',
                                                         padx=16, pady=(16, 10))
        self.tree = ExplorerTree(nav, self.items, self._on_tree_select,
                                 preferred_task=task_id(self.base_item) if self.base_item else None,
                                 saved_scope=self._state.get('scope'))
        self.tree.grid(row=1, column=0, sticky='nsew', padx=(7, 4), pady=(0, 10))
        self._selected_scope = self.tree.selected_scope

        grabber = ctk.CTkFrame(self.split, fg_color=BG, width=10, corner_radius=0)
        grabber.grid(row=0, column=1, sticky='ns')
        grabber.bind('<Button-1>', self._begin_resize)
        grabber.bind('<B1-Motion>', self._resize)

        main = ctk.CTkFrame(self.split, fg_color=PANEL, corner_radius=16,
                            border_width=1, border_color=BORDER)
        main.grid(row=0, column=2, sticky='nsew')
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(1, weight=1)
        heading = ctk.CTkFrame(main, fg_color='transparent')
        heading.grid(row=0, column=0, sticky='ew', padx=15, pady=(14, 11))
        heading.grid_columnconfigure(0, weight=1)
        self.count_label = ctk.CTkLabel(heading, text='Результаты',
                                        text_color=MUTED, anchor='w',
                                        font=ctk.CTkFont(size=12, weight='bold'))
        self.count_label.grid(row=0, column=0, sticky='w')
        self.sort = ctk.CTkOptionMenu(
            heading, values=['Масштаб ↑', 'Масштаб ↓', 'Сначала новые', 'Канал', 'Название'],
            width=153, height=31, corner_radius=9,
            fg_color=SURFACE, button_color='#2e2e2e', button_hover_color='#404040',
            dropdown_fg_color=SURFACE, dropdown_hover_color=ACTIVE,
            font=ctk.CTkFont(size=11), command=lambda _v: self.refresh()
        )
        self.sort.grid(row=0, column=1, sticky='e')
        self.result_list = VirtualResultList(main, self._toggle, self._show_details)
        self.result_list.grid(row=1, column=0, sticky='nsew', padx=(13, 9), pady=(0, 12))

        # Compact footer, no duplicate technical description block.
        footer = ctk.CTkFrame(self, fg_color='transparent')
        footer.grid(row=2, column=0, sticky='ew', padx=24, pady=(14, 20))
        footer.grid_columnconfigure(0, weight=1)
        self.status = ctk.CTkLabel(footer, text='Выбрано: 0', anchor='w',
                                   text_color=MUTED, font=ctk.CTkFont(size=11))
        self.status.grid(row=0, column=0, sticky='ew')
        self.cancel_button = ctk.CTkButton(footer, text='Отмена', width=105,
                                           height=34, corner_radius=9,
                                           fg_color=SURFACE, hover_color='#343434',
                                           text_color=TEXT, command=self.close)
        self.cancel_button.grid(row=0, column=1, padx=(10, 8))
        self.add_button = ctk.CTkButton(footer, text='Добавить', width=145,
                                        height=34, corner_radius=9, fg_color='#28699d',
                                        hover_color='#3385c4', command=self.add_selected)
        self.add_button.grid(row=0, column=2)
        self._refresh_status()
        self.refresh()
        self.after(40, self._grab_if_open)
        # DWM colors the *native* caption; CTk's old delayed manipulation stays disabled.
        self._titlebar_job = self.after(90, self._apply_native_titlebar)

    def _on_map(self, event):
        if event.widget is self and getattr(self, '_alive', False):
            self._apply_native_titlebar()

    def _apply_native_titlebar(self):
        self._titlebar_job = None
        from utils.tk_lifecycle import active_widget_root
        if self._alive and active_widget_root(self) is not None:
            enable_dark_titlebar(self, caption_color=BG, text_color=TEXT)

    def _begin_resize(self, event):
        self._drag_x = event.x_root
        self._drag_nav_width = self._nav_width

    def _resize(self, event):
        new = self._drag_nav_width + event.x_root - self._drag_x
        limit = max(285, self.split.winfo_width() - 415)
        self._nav_width = max(260, min(int(new), min(520, limit)))
        self.split.grid_columnconfigure(0, minsize=self._nav_width)
        self.nav.configure(width=self._nav_width)

    def _on_tree_select(self, scope):
        self._selected_scope = scope
        self.refresh()

    def _scope(self, item):
        scope = self._selected_scope
        if scope is None:
            return True
        task, group, algorithm = scope
        return (task_id(item) == task and
                (group is None or item.get('category') == group) and
                (algorithm is None or (item.get('algorithm') or 'Другой алгоритм') == algorithm))

    def refresh(self):
        if not hasattr(self, 'result_list'):
            return
        sorter = {'Масштаб ↑': 'scale', 'Масштаб ↓': 'scale_desc', 'Сначала новые': 'recent',
                  'Канал': 'channel', 'Название': 'name'}[self.sort.get()]
        scope = None if self.everywhere.get() else self._scope
        results = search_results(self.items, scope=scope, query=self.search_var.get(), order=sorter)
        active_scope = self._selected_scope
        label = 'Все результаты' if self.everywhere.get() else (
            active_scope[2] or ('ML' if active_scope[1] == 'ML-кластеры'
                                else active_scope[1] or 'Все результаты')
            if active_scope else 'Все результаты'
        )
        self.count_label.configure(text=f'Результаты · {label}  ({len(results)})')
        self.result_list.set_results(results, self.selected, self.added_ids)

    def _schedule_search(self):
        if self._search_job is not None:
            try:
                self.after_cancel(self._search_job)
            except tk.TclError:
                pass
        self._search_job = self.after(120, self._perform_search)

    def _perform_search(self):
        self._search_job = None
        self.refresh()

    def _show_details(self, item):
        """Diagnostic information on explicit double-click only, not in cards."""
        messagebox.showinfo('Сведения о результате', details(item), parent=self)

    def _toggle(self, item):
        if self._busy:
            return
        identifier = result_id(item)
        if identifier in self.added_ids:
            self.status.configure(text='Этот результат уже добавлен')
            return
        if identifier in self.selected:
            self.selected.remove(identifier)
            self.selection_order.remove(identifier)
        else:
            self.selected.add(identifier)
            self.selection_order.append(identifier)
        self.result_list.set_selected(self.selected)
        self._refresh_status()

    def _refresh_status(self):
        self.status.configure(text=f'Выбрано: {len(self.selected)}' +
                              (f'  ·  Уже добавлено: {len(self.added_ids)}' if self.added_ids else ''))
        self.add_button.configure(state='normal' if self.selected and not self._busy else 'disabled')

    def add_selected(self):
        if self._busy or not self.selected:
            return
        if len(self.selected) > MAX_BATCH:
            messagebox.showwarning('Слишком много слоёв',
                f'За один раз можно добавить не более {MAX_BATCH} слоёв.', parent=self)
            return
        items = [self.by_id[key] for key in self.selection_order]
        errors = []
        for item in items:
            ok, reason = preflight(item, self.base_item, self.base_data)
            if not ok:
                errors.append(f'{card_title(item)}: {reason}')
        if errors:
            messagebox.showwarning('Несовместимые результаты',
                'Ни один слой не добавлен. Исключите несовместимые результаты:\n\n' +
                '\n'.join(errors[:8]), parent=self)
            return
        self._busy = True
        self._todo = list(items)
        self._loaded = []
        self.cancel_button.configure(text='Остановить')
        self._refresh_status()
        self._start_next()

    def _start_next(self):
        if not self._alive:
            return
        if not self._todo:
            loaded = self._loaded
            self._busy = False
            self._waiting = None
            self.on_add(loaded)
            self.close()
            return
        item = self._todo.pop(0)
        self.status.configure(text=f'Проверка слоёв: {len(self._loaded) + 1} из {len(self.selection_order)}…')
        self._waiting = (item, self.executor.submit(load_result, item, self.folder))
        self._poll_job = self.after(45, self._poll)

    def _poll(self):
        self._poll_job = None
        if not self._alive or not self._waiting:
            return
        item, future = self._waiting
        if not future.done():
            self._poll_job = self.after(45, self._poll)
            return
        self._waiting = None
        try:
            data = future.result()
            ok, reason = check_loaded(data, self.base_data)
            if not ok:
                raise ValueError(reason)
        except Exception as error:
            self._busy = False
            self._todo.clear()
            self._loaded.clear()
            self.cancel_button.configure(text='Отмена')
            self._refresh_status()
            messagebox.showerror('Слой не добавлен',
                f'Не удалось проверить {card_title(item)}:\n{error}\n\nНи один слой не добавлен.', parent=self)
            return
        self._loaded.append((item, data))
        self._start_next()

    def _grab_if_open(self):
        from utils.tk_lifecycle import active_widget_root
        if self._alive and active_widget_root(self) is not None:
            try:
                self.grab_set()
                self.search.focus_set()
            except tk.TclError:
                pass

    def safe_destroy(self):
        self.close()

    def close(self):
        if not getattr(self, '_alive', False):
            return
        self._alive = False
        for name in ('_poll_job', '_search_job', '_titlebar_job'):
            job = getattr(self, name, None)
            if job is not None:
                try:
                    self.after_cancel(job)
                except tk.TclError:
                    pass
                setattr(self, name, None)
        if self._waiting:
            self._waiting[1].cancel()
        try:
            geometry = self.geometry()
        except tk.TclError:
            geometry = self._state.get('geometry', '1050x710')
        self.on_close({'scope': self._selected_scope,
                       'sash': self._nav_width, 'geometry': geometry})
        try:
            self.grab_release()
        except tk.TclError:
            pass
        try:
            self.destroy()
        except tk.TclError:
            pass
