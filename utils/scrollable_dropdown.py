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
        from utils.tk_lifecycle import install_callback_cleanup
        install_callback_cleanup(self)

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

        frame = tk.Frame(container, bg=_BG)
        frame.pack(fill='both', expand=True, padx=6, pady=6)
        canvas = tk.Canvas(frame, bg=_BG, highlightthickness=0, takefocus=True)
        scrollbar = ctk.CTkScrollbar(frame, command=canvas.yview)
        scrollbar.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)
        self._popup_list = canvas
        row_height = 54 if wide else 35
        matching = []
        active = 0

        def choose(value):
            self._close_popup()
            if self.winfo_exists():
                self._dropdown_callback(value)
            return 'break'

        def paint(*_):
            if self._popup is not popup:
                return
            canvas.delete('all')
            width = max(100, canvas.winfo_width())
            first = max(0, int(canvas.canvasy(0) // row_height))
            last = min(len(matching), first + max(1, canvas.winfo_height() // row_height) + 2)
            if not matching:
                canvas.create_text(12, 22, text='Ничего не найдено', fill=_MUTED, anchor='w')
            for index in range(first, last):
                value = matching[index]
                y = index * row_height
                canvas.create_rectangle(2, y+1, width-2, y+row_height-2,
                    fill=_ACTIVE if index == active else _ROW, outline='')
                title, subtitle = _split_result(str(value)) if wide else (str(value), '')
                canvas.create_text(12, y+16, text=title, fill=_TEXT, anchor='w', font=('Segoe UI', 10))
                if wide:
                    canvas.create_text(12, y+36, text=subtitle, fill=_MUTED, anchor='w', font=('Segoe UI', 9))

        def scrolled(first, last):
            scrollbar.set(first, last)
            paint()
        canvas.configure(yscrollcommand=scrolled)
        canvas.bind('<Configure>', paint)

        def populate(*_):
            nonlocal matching, active
            term = search_var.get().casefold().strip()
            matching = [value for value in values if term in str(value).casefold()]
            active = matching.index(self.get()) if self.get() in matching else 0
            canvas.configure(scrollregion=(0, 0, 1, max(1, len(matching))*row_height))
            canvas.yview_moveto(active / max(1, len(matching)))
            paint()

        def click(event):
            index = int(canvas.canvasy(event.y) // row_height)
            if 0 <= index < len(matching):
                return choose(matching[index])
            return 'break'

        def wheel(event):
            canvas.yview_scroll(-int(event.delta / 120) or (-1 if event.delta > 0 else 1), 'units')
            return 'break'

        def navigate(event):
            nonlocal active
            if event.keysym == 'Return':
                return choose(matching[active]) if matching else 'break'
            active = max(0, min(len(matching)-1,
                0 if event.keysym == 'Home' else len(matching)-1 if event.keysym == 'End'
                else active + (-1 if event.keysym == 'Up' else 1)))
            top = canvas.canvasy(0)
            if active*row_height < top or (active+1)*row_height > top+canvas.winfo_height():
                canvas.yview_moveto(active / max(1, len(matching)))
            paint()
            return 'break'

        canvas.configure(yscrollincrement=row_height)
        canvas.bind('<Button-1>', click)
        popup.bind('<MouseWheel>', wheel)
        for key in ('Up', 'Down', 'Home', 'End', 'Return'):
            popup.bind('<'+key+'>', navigate)
        search_var.trace_add('write', populate)
        self._popup_search = search_var
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
            canvas.focus_set()

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
