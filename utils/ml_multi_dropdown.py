"""Compact dark, task-aware multi-select popup used by ML batch configuration cards.

Values are stable internal identifiers; only labels are localized.  The caller
owns persistence and supplies the choices derived from the active task.
"""
from __future__ import annotations

import tkinter as tk
import customtkinter as ctk


class MLMultiDropdown(ctk.CTkFrame):
    """Multi-select with a checkbox for 'Все' and one per actual task value.

    get() returns '*' when all values are selected, comma-separated identifiers
    when some are selected, or '' when none are selected (invalid at planning).
    """

    def __init__(self, master, *, options, value='*', subject='значения',
                 on_change=None, **kwargs):
        super().__init__(master, fg_color='transparent', **kwargs)
        from utils.tk_lifecycle import install_callback_cleanup
        install_callback_cleanup(self)
        self._choice_pairs = [(str(code), str(label)) for code, label in options]
        self._subject = subject
        self._on_change = on_change
        self._selection = set()
        self._popup = None
        self._popup_vars = {}
        self._all_var = None
        self._open_button = ctk.CTkButton(
            self, text='', height=35, anchor='w', corner_radius=7,
            fg_color='#34373C', hover_color='#414852',
            border_width=1, border_color='#5A5F67',
            command=self._open_popup)
        self._open_button.pack(fill='x', expand=True)
        self.set(value, notify=False)
        self.bind('<Destroy>', self._on_destroy, add='+')

    def _on_destroy(self, event):
        if event.widget is self:
            self._close_popup()

    def set(self, value, *, notify=False):
        values = {code for code, _ in self._choice_pairs}
        if isinstance(value, (tuple, list, set)):
            tokens = [str(x).strip() for x in value]
        else:
            tokens = [x.strip() for x in str(value if value is not None else '*').split(',')]
        if any(x.lower() in {'*', 'all', 'все'} for x in tokens):
            self._selection = values
        else:
            self._selection = values.intersection(tokens)
        self._update_label()
        if notify and self._on_change:
            self._on_change(self.get())

    def get(self):
        codes = [code for code, _ in self._choice_pairs if code in self._selection]
        if codes and len(codes) == len(self._choice_pairs):
            return '*'
        return ','.join(codes)

    def _update_label(self):
        chosen = [label for code, label in self._choice_pairs if code in self._selection]
        if not chosen:
            text = 'Ничего не выбрано'
        elif len(chosen) == len(self._choice_pairs):
            text = f'Все {self._subject} ({len(chosen)})'
        elif len(chosen) <= 3:
            text = ', '.join(chosen)
        else:
            text = f'{", ".join(chosen[:2])} и ещё {len(chosen)-2}'
        self._open_button.configure(text='▾  ' + text)

    def _changed(self):
        self._update_label()
        if self._on_change:
            self._on_change(self.get())

    def _all_changed(self):
        enabled = bool(self._all_var.get())
        self._selection = {code for code, _ in self._choice_pairs} if enabled else set()
        for code, var in self._popup_vars.items():
            var.set(code in self._selection)
        self._changed()

    def _item_changed(self, code):
        if self._popup_vars[code].get():
            self._selection.add(code)
        else:
            self._selection.discard(code)
        if self._all_var is not None:
            self._all_var.set(bool(self._choice_pairs) and len(self._selection) == len(self._choice_pairs))
        self._changed()

    def _open_popup(self):
        if self._popup is not None or not self._choice_pairs:
            return
        popup = tk.Toplevel(self)
        self._popup = popup
        popup.withdraw()
        popup.overrideredirect(True)
        popup.configure(background='#535965')
        popup.transient(self.winfo_toplevel())
        width = min(max(300, self.winfo_width()), self.winfo_screenwidth()-24)
        # A distinct select-all header, a small divider and individual options.
        height = min(390, 115 + 36 * len(self._choice_pairs)
                     + (40 if len(self._choice_pairs) > 10 else 0))
        height = min(height, self.winfo_screenheight()-70)
        x = max(8, min(self.winfo_rootx(), self.winfo_screenwidth()-width-8))
        below = self.winfo_rooty()+self.winfo_height()+4
        above = self.winfo_rooty()-height-4
        y = below if below+height < self.winfo_screenheight()-36 else max(8, above)
        popup.geometry(f'{width}x{height}+{x}+{y}')
        frame = ctk.CTkFrame(popup, corner_radius=9, fg_color='#27292E',
                             border_width=1, border_color='#555B64')
        frame.pack(fill='both', expand=True)
        self._all_var = tk.BooleanVar(value=len(self._selection) == len(self._choice_pairs))
        all_header = ctk.CTkFrame(frame, corner_radius=7, fg_color='#303E4D',
                                 border_width=1, border_color='#45657E')
        all_header.pack(fill='x', padx=9, pady=(9, 7))
        select_all = ctk.CTkCheckBox(
            all_header, text=f'Все {self._subject}', variable=self._all_var,
            command=self._all_changed, height=31,
            fg_color='#287AB6', hover_color='#357DAE',
            font=ctk.CTkFont(size=13, weight='bold'))
        select_all.pack(fill='x', padx=10, pady=6)
        ctk.CTkFrame(frame, height=2, fg_color='#485362',
                     corner_radius=0).pack(fill='x', padx=11, pady=(0, 4))
        ctk.CTkLabel(frame, text='ОТДЕЛЬНЫЕ ВАРИАНТЫ',
                     font=ctk.CTkFont(size=10, weight='bold'),
                     text_color='#909CAA', anchor='w', height=14).pack(
                         fill='x', padx=15, pady=(0, 2))
        search = None
        if len(self._choice_pairs) > 10:
            search = ctk.CTkEntry(frame, placeholder_text='Найти значение…', height=30)
            search.pack(fill='x', padx=11, pady=(3, 5))
        list_frame = ctk.CTkScrollableFrame(frame, fg_color='transparent',
            scrollbar_button_color='#59616B', scrollbar_button_hover_color='#73808B')
        list_frame.pack(fill='both', expand=True, padx=(4, 2), pady=(0, 4))
        self._popup_vars = {
            code: tk.BooleanVar(value=code in self._selection)
            for code, _ in self._choice_pairs
        }
        rows = []
        for code, label in self._choice_pairs:
            checkbox = ctk.CTkCheckBox(list_frame, text=label, height=32,
                variable=self._popup_vars[code],
                command=lambda c=code: self._item_changed(c))
            checkbox.pack(fill='x', padx=9, pady=2)
            rows.append((label, checkbox))
        if search is not None:
            def filter_rows(_event=None):
                pattern = search.get().casefold().strip()
                for label, checkbox in rows:
                    if pattern in label.casefold():
                        if not checkbox.winfo_manager():
                            checkbox.pack(fill='x', padx=9, pady=2)
                    else:
                        checkbox.pack_forget()
            search.bind('<KeyRelease>', filter_rows)
        close = ctk.CTkButton(frame, text='Готово', height=30,
                               command=self._close_popup)
        close.pack(fill='x', padx=11, pady=(0, 9))
        from utils.mousewheel import bind_popup_mousewheel
        bind_popup_mousewheel(popup, list_frame)
        popup.bind('<Escape>', lambda _: self._close_popup())
        def outside(event):
            if not (popup.winfo_rootx() <= event.x_root < popup.winfo_rootx()+popup.winfo_width()
                    and popup.winfo_rooty() <= event.y_root < popup.winfo_rooty()+popup.winfo_height()):
                self._close_popup()
        popup.bind('<ButtonPress-1>', outside)
        popup.deiconify()
        popup.lift()
        try:
            popup.grab_set()
        except tk.TclError:
            self._close_popup()

    def _close_popup(self):
        popup = self._popup
        self._popup = None
        self._popup_vars = {}
        self._all_var = None
        if popup is not None:
            try:
                if popup.grab_current() == popup:
                    popup.grab_release()
                popup.destroy()
            except tk.TclError:
                pass


class MLSingleDropdown(ctk.CTkFrame):
    """Single-choice companion to MLMultiDropdown with the same dark popup.

    Codes (``row``, ``col``, ``coordinates_knn``...) never appear in the UI;
    get()/set() operate exclusively on these persistent internal codes.
    """

    def __init__(self, master, *, options, value, on_change=None, **kwargs):
        super().__init__(master, fg_color='transparent', **kwargs)
        from utils.tk_lifecycle import install_callback_cleanup
        install_callback_cleanup(self)
        self._choice_pairs = [(str(code), str(label)) for code, label in options]
        self._on_change = on_change
        self._selected_code = ''
        self._popup = None
        self._open_button = ctk.CTkButton(
            self, text='', anchor='w', height=35, corner_radius=7,
            fg_color='#34373C', hover_color='#414852',
            border_width=1, border_color='#5A5F67',
            command=self._open_popup)
        self._open_button.pack(fill='x', expand=True)
        self.set(value, notify=False)
        self.bind('<Destroy>', self._on_destroy, add='+')

    def _on_destroy(self, event):
        if event.widget is self:
            self._close_popup()

    def set(self, value, *, notify=False):
        codes = {code for code, _ in self._choice_pairs}
        value = str(value)
        if value not in codes and codes:
            value = 'knn' if 'knn' in codes else self._choice_pairs[0][0]
        self._selected_code = value
        label = next((title for code, title in self._choice_pairs if code == value), 'Не выбрано')
        self._open_button.configure(text='▾  ' + label)
        if notify and self._on_change:
            self._on_change(self.get())

    def get(self):
        return self._selected_code

    def _open_popup(self):
        if self._popup is not None or not self._choice_pairs:
            return
        popup = tk.Toplevel(self)
        self._popup = popup
        popup.withdraw()
        popup.overrideredirect(True)
        popup.configure(background='#535965')
        popup.transient(self.winfo_toplevel())
        width = min(max(300, self.winfo_width()), self.winfo_screenwidth()-24)
        # Compact for three choices; scroll only when more choices are present.
        count = len(self._choice_pairs)
        height = min(400, 20 + 41 * count + (12 if self._choice_pairs[0][0] == '*' else 0))
        height = min(height, self.winfo_screenheight()-70)
        x = max(8, min(self.winfo_rootx(), self.winfo_screenwidth()-width-8))
        below = self.winfo_rooty()+self.winfo_height()+4
        above = self.winfo_rooty()-height-4
        y = below if below+height < self.winfo_screenheight()-36 else max(8, above)
        popup.geometry(f'{width}x{height}+{x}+{y}')
        frame = ctk.CTkFrame(popup, fg_color='#27292E', corner_radius=9,
                            border_width=1, border_color='#555B64')
        frame.pack(fill='both', expand=True)
        if count > 8:
            items_frame = ctk.CTkScrollableFrame(
                frame, fg_color='transparent',
                scrollbar_button_color='#59616B',
                scrollbar_button_hover_color='#73808B')
            items_frame.pack(fill='both', expand=True, padx=5, pady=6)
        else:
            items_frame = ctk.CTkFrame(frame, fg_color='transparent')
            items_frame.pack(fill='both', expand=True, padx=5, pady=5)

        for index, (code, label) in enumerate(self._choice_pairs):
            selected = code == self._selected_code
            is_all = code == '*' and index == 0
            fg_color = ('#303E4D' if selected and is_all else
                        '#29455D' if selected else '#27292E')
            button = ctk.CTkButton(
                items_frame, text=('✓  ' if selected else '    ') + label,
                anchor='w', height=34, corner_radius=6,
                fg_color=fg_color, hover_color='#3C4B5B',
                text_color='#F1F3F6',
                border_width=1 if is_all and selected else 0,
                border_color='#45657E',
                command=lambda v=code: self._choose(v))
            button.pack(fill='x', padx=5, pady=2)
            if is_all and count > 1:
                ctk.CTkFrame(items_frame, height=1, corner_radius=0,
                             fg_color='#485362').pack(fill='x', padx=7, pady=(5, 6))

        if count > 8:
            from utils.mousewheel import bind_popup_mousewheel
            bind_popup_mousewheel(popup, items_frame)
        popup.bind('<Escape>', lambda _: self._close_popup())
        def outside(event):
            if not (popup.winfo_rootx() <= event.x_root < popup.winfo_rootx()+popup.winfo_width()
                    and popup.winfo_rooty() <= event.y_root < popup.winfo_rooty()+popup.winfo_height()):
                self._close_popup()
        popup.bind('<ButtonPress-1>', outside)
        popup.deiconify()
        popup.lift()
        try:
            popup.grab_set()
        except tk.TclError:
            self._close_popup()

    def _choose(self, value):
        self.set(value, notify=True)
        self._close_popup()

    def _close_popup(self):
        popup = self._popup
        self._popup = None
        if popup is not None:
            try:
                if popup.grab_current() == popup:
                    popup.grab_release()
                popup.destroy()
            except tk.TclError:
                pass
