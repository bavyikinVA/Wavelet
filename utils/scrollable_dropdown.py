"""Dark, searchable popup selectors for CustomTkinter.

The existing CTkComboBox/CTkOptionMenu value, callback and configure API stays intact.
Only the popup is replaced; no application data is stored by this module.
"""
import tkinter as tk
import customtkinter as ctk


_BG = '#25272C'
_BORDER = '#51555E'
_ROW = '#30333A'
_HOVER = '#3C4551'
_ACTIVE = '#244B70'
_TEXT = '#F1F3F6'
_MUTED = '#AEB6C1'


def _split_result(value):
    """Return a human-readable title and subtitle without changing selection values."""
    parts = [part.strip() for part in value.split(' · ')]
    if len(parts) < 3:
        return value, ''
    if parts[0].lower().startswith('task '):
        main = f'{parts[0]}  ·  {parts[1]}'
        return main, ' · '.join(parts[2:])
    return ' · '.join(parts[:2]), ' · '.join(parts[2:])


class _ScrollableDropdown:
    def __init__(self, *args, dropdown_wide=False, dropdown_search=False, **kwargs):
        self._dropdown_wide = dropdown_wide
        self._dropdown_search = dropdown_search
        self._popup = None
        super().__init__(*args, **kwargs)

    def _open_dropdown_menu(self):
        if self.cget('state') == 'disabled':
            return
        self._close_popup()
        values = list(self.cget('values'))
        popup = tk.Toplevel(self)
        self._popup = popup
        popup.withdraw()
        popup.overrideredirect(True)
        popup.configure(background=_BORDER)
        popup.transient(self.winfo_toplevel())

        wide = self._dropdown_wide
        width = min(max(self.winfo_width(), 560 if wide else 255),
                    max(200, self.winfo_screenwidth() - 24))
        height = min((420 if wide else 320), max(85, self.winfo_screenheight() - 80))
        x = self.winfo_rootx() + self.winfo_width() - width
        x = max(8, min(x, self.winfo_screenwidth() - width - 8))
        below = self.winfo_rooty() + self.winfo_height() + 4
        above = self.winfo_rooty() - height - 4
        y = below if below + height <= self.winfo_screenheight() - 38 else max(8, above)
        popup.geometry(f'{width}x{height}+{x}+{y}')

        container = ctk.CTkFrame(popup, fg_color=_BG, corner_radius=9,
                                 border_width=1, border_color=_BORDER)
        container.pack(fill='both', expand=True)
        search_var = tk.StringVar()
        if self._dropdown_search or len(values) > 12:
            search = ctk.CTkEntry(container, textvariable=search_var,
                                  placeholder_text='Поиск по результатам…' if wide else 'Поиск…',
                                  height=35, fg_color=_ROW, border_color=_BORDER,
                                  text_color=_TEXT, corner_radius=7)
            search.pack(fill='x', padx=9, pady=(9, 5))
        else:
            search = None

        frame = ctk.CTkScrollableFrame(container, fg_color='transparent',
                                       corner_radius=0,
                                       scrollbar_button_color='#555C66',
                                       scrollbar_button_hover_color='#778392')
        frame.pack(fill='both', expand=True, padx=(5, 3), pady=(5 if search is None else 2, 6))
        self._popup_list = frame

        def choose(value):
            self._close_popup()
            if self.winfo_exists():
                self._dropdown_callback(value)

        def populate(*_):
            if self._popup is not popup:
                return
            for child in frame.winfo_children():
                child.destroy()
            term = search_var.get().casefold().strip()
            matching = [value for value in values if term in str(value).casefold()]
            if not matching:
                ctk.CTkLabel(frame, text='Ничего не найдено', text_color=_MUTED,
                             height=44).pack(fill='x')
                return
            current = self.get()
            for value in matching:
                selected = value == current
                row = ctk.CTkFrame(frame, fg_color=_ACTIVE if selected else 'transparent',
                                   corner_radius=7, height=50 if wide else 33)
                row.pack(fill='x', padx=3, pady=2)
                row.pack_propagate(False)
                inner = ctk.CTkFrame(row, fg_color='transparent')
                inner.pack(fill='both', expand=True, padx=10, pady=5 if wide else 1)
                if wide:
                    title, subtitle = _split_result(str(value))
                    label = ctk.CTkLabel(inner, text=title, text_color=_TEXT,
                                         anchor='w', height=19)
                    label.pack(fill='x')
                    detail = ctk.CTkLabel(inner, text=subtitle or ' ', text_color=_MUTED,
                                          font=ctk.CTkFont(size=11), anchor='w', height=17)
                    detail.pack(fill='x')
                    targets = (row, inner, label, detail)
                else:
                    label = ctk.CTkLabel(inner, text=str(value), text_color=_TEXT,
                                         anchor='w', height=28)
                    label.pack(fill='x')
                    targets = (row, inner, label)
                if selected:
                    ctk.CTkLabel(inner, text='✓', text_color='#83C5FF',
                                 width=19).place(relx=1.0, rely=0.5, anchor='e')
                for widget in targets:
                    widget.bind('<Button-1>', lambda event, v=value: choose(v), add='+')
                    widget.bind('<Enter>', lambda event, r=row, s=selected: r.configure(
                        fg_color=_ACTIVE if s else _HOVER), add='+')
                    widget.bind('<Leave>', lambda event, r=row, s=selected: r.configure(
                        fg_color=_ACTIVE if s else 'transparent'), add='+')

        search_var.trace_add('write', populate)
        populate()
        popup.bind('<Escape>', lambda _: self._close_popup())
        def close_if_outside(event):
            if not (popup.winfo_rootx() <= event.x_root < popup.winfo_rootx() + popup.winfo_width()
                    and popup.winfo_rooty() <= event.y_root < popup.winfo_rooty() + popup.winfo_height()):
                self._close_popup()
        popup.bind('<ButtonPress-1>', close_if_outside)
        popup.deiconify()
        popup.lift()
        popup.grab_set()
        if search is not None:
            search.focus_set()
        else:
            frame.focus_set()

    def _close_popup(self):
        popup = getattr(self, '_popup', None)
        self._popup = None
        self._popup_list = None
        if popup is not None:
            try:
                if popup.grab_current() == popup:
                    popup.grab_release()
                popup.destroy()
            except tk.TclError:
                pass

    def destroy(self):
        self._close_popup()
        super().destroy()


class ScrollableComboBox(_ScrollableDropdown, ctk.CTkComboBox):
    pass


class ScrollableOptionMenu(_ScrollableDropdown, ctk.CTkOptionMenu):
    pass
