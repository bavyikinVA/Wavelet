"""Контекстные подсказки: компактный info-badge и единая hover-card."""
import tkinter as tk
import customtkinter as ctk
from utils.help_content import HELP_CONTENT


class _HelpManager:
    def __init__(self, root):
        self.root = root
        self.popup = None
        self.current = None
        self.pending = None
        self.anchor = None

    def cancel(self):
        if self.pending is not None:
            try:
                self.root.after_cancel(self.pending)
            except tk.TclError:
                pass
            self.pending = None

    def hide(self, _event=None):
        self.cancel()
        if self.popup is not None:
            try:
                self.popup.destroy()
            except tk.TclError:
                pass
            self.popup = None
        self.current = None
        self.anchor = None

    def schedule(self, icon, key):
        self.cancel()
        if self.current != key:
            self.hide()
        self.anchor = icon
        self.pending = self.root.after(350, lambda: self.show(icon, key))

    def show(self, icon, key):
        self.pending = None
        if not icon.winfo_exists() or not icon.winfo_ismapped():
            return
        item = HELP_CONTENT[key]

        popup = tk.Toplevel(self.root)
        popup.withdraw()
        popup.overrideredirect(True)
        magic = "#010203"
        popup.configure(bg=magic)
        try:
            popup.attributes('-topmost', True)
        except tk.TclError:
            pass
        try:
            popup.wm_attributes('-transparentcolor', magic)
        except tk.TclError:
            pass

        card = ctk.CTkFrame(
            popup,
            fg_color="#15181d",
            corner_radius=14,
            border_width=1,
            border_color="#313845",
        )
        card.pack(fill='both', expand=True)

        head = ctk.CTkFrame(card, fg_color='transparent')
        head.pack(fill='x', padx=16, pady=(14, 10))
        ctk.CTkLabel(
            head,
            text=item.title,
            font=ctk.CTkFont(size=13, weight='bold'),
            text_color='#f4f7fb',
            anchor='w'
        ).pack(side='left', fill='x', expand=True)

        icon_badge = ctk.CTkFrame(
            head,
            width=18,
            height=18,
            corner_radius=9,
            fg_color='transparent',
            border_width=1,
            border_color='#7e8a99',
        )
        icon_badge.pack(side='right')
        icon_badge.pack_propagate(False)
        ctk.CTkLabel(
            icon_badge,
            text='i',
            font=ctk.CTkFont(size=11, weight='bold'),
            text_color='#dce3ea',
        ).pack(expand=True)

        ctk.CTkLabel(
            card,
            text=item.description,
            font=ctk.CTkFont(size=11),
            text_color='#eef2f6',
            justify='left',
            anchor='w',
            wraplength=340,
        ).pack(fill='x', padx=16, pady=(0, 12))

        has_extra = any([item.effect, item.values, item.default])
        if has_extra:
            divider = ctk.CTkFrame(card, fg_color='#2a303a', height=1, corner_radius=0)
            divider.pack(fill='x', padx=16, pady=(0, 12))

        if item.effect:
            ctk.CTkLabel(
                card,
                text='Влияние на результат',
                font=ctk.CTkFont(size=11, weight='bold'),
                text_color='#aeb8c4',
                anchor='w'
            ).pack(fill='x', padx=16, pady=(0, 6))
            ctk.CTkLabel(
                card,
                text=item.effect,
                font=ctk.CTkFont(size=11),
                text_color='#eef2f6',
                justify='left',
                anchor='w',
                wraplength=340,
            ).pack(fill='x', padx=16, pady=(0, 12))

        meta_rows = []
        if item.values:
            meta_rows.append(('Допустимые значения', item.values))
        if item.default:
            meta_rows.append(('По умолчанию', item.default))

        if meta_rows:
            meta = ctk.CTkFrame(card, fg_color='transparent')
            meta.pack(fill='x', padx=16, pady=(0, 14))
            meta.grid_columnconfigure(0, weight=0)
            meta.grid_columnconfigure(1, weight=1)
            for row_idx, (label, value) in enumerate(meta_rows):
                ctk.CTkLabel(
                    meta,
                    text=label,
                    font=ctk.CTkFont(size=10),
                    text_color='#aeb8c4',
                    anchor='w',
                ).grid(row=row_idx, column=0, sticky='w', pady=(0, 8 if row_idx < len(meta_rows) - 1 else 0), padx=(0, 18))
                ctk.CTkLabel(
                    meta,
                    text=value,
                    font=ctk.CTkFont(size=10, weight='bold'),
                    text_color='#eef2f6',
                    anchor='e',
                    justify='right',
                    wraplength=170,
                ).grid(row=row_idx, column=1, sticky='e', pady=(0, 8 if row_idx < len(meta_rows) - 1 else 0))

        popup.update_idletasks()
        width, height = popup.winfo_reqwidth(), popup.winfo_reqheight()
        screen_w, screen_h = popup.winfo_screenwidth(), popup.winfo_screenheight()
        x = icon.winfo_rootx() + icon.winfo_width() + 10
        y = icon.winfo_rooty() - 6
        if x + width > screen_w - 12:
            x = icon.winfo_rootx() - width - 10
        if x < 8:
            x = 8
        y = max(8, min(y, screen_h - height - 40))
        popup.geometry(f'+{x}+{y}')
        popup.deiconify()
        self.popup = popup
        self.current = key


def _manager(widget):
    root = widget.winfo_toplevel()
    manager = getattr(root, '_wavelets_help_manager', None)
    if manager is None:
        manager = _HelpManager(root)
        root._wavelets_help_manager = manager
        root.bind_all('<MouseWheel>', manager.hide, add='+')
        root.bind_all('<Button-4>', manager.hide, add='+')
        root.bind_all('<Button-5>', manager.hide, add='+')
        root.bind('<Escape>', manager.hide, add='+')
        root.bind(
            '<Configure>',
            lambda event: manager.hide() if manager.popup and event.widget == root else None,
            add='+'
        )
    return manager


class HelpIcon(ctk.CTkFrame):
    """Small rounded info badge. Hover opens a contextual card."""

    def __init__(self, master, help_key, **kwargs):
        if help_key not in HELP_CONTENT:
            raise KeyError(f'Unknown help key: {help_key}')
        super().__init__(
            master,
            width=18,
            height=18,
            corner_radius=9,
            fg_color='transparent',
            border_width=1,
            border_color='#768394',
            **kwargs,
        )
        self.pack_propagate(False)
        self.help_key = help_key
        self._help_manager = _manager(self)
        self._label = ctk.CTkLabel(
            self,
            text='i',
            font=ctk.CTkFont(size=11, weight='bold'),
            text_color='#d8e0e8',
            cursor='hand2',
        )
        self._label.pack(expand=True)
        for widget in (self, self._label):
            widget.bind('<Enter>', self._enter, add='+')
            widget.bind('<Leave>', self._leave, add='+')
            widget.bind('<Destroy>', self._leave, add='+')

    def _enter(self, _event=None):
        self.configure(border_color='#66b7f0', fg_color='#1f2732')
        self._label.configure(text_color='#eaf6ff')
        self._help_manager.schedule(self, self.help_key)

    def _leave(self, _event=None):
        try:
            self.configure(border_color='#768394', fg_color='transparent')
            self._label.configure(text_color='#d8e0e8')
        except (tk.TclError, ValueError):
            pass
        if self._help_manager.anchor is self:
            self._help_manager.hide()
