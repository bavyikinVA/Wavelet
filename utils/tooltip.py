import tkinter as tk

import customtkinter as ctk

from utils.theme import AppTheme


class HoverTooltip:
    """Неблокирующая подсказка, появляющаяся при наведении на виджет."""

    def __init__(
        self,
        widget,
        text=None,
        *,
        text_provider=None,
        delay_ms=450,
        wraplength=360,
        offset_x=12,
        offset_y=10,
    ):
        self.widget = widget
        self.text = text or ""
        self.text_provider = text_provider
        self.delay_ms = int(delay_ms)
        self.wraplength = int(wraplength)
        self.offset_x = int(offset_x)
        self.offset_y = int(offset_y)
        self._after_id = None
        self._window = None

        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self.hide, add="+")
        widget.bind("<ButtonPress>", self.hide, add="+")

    def _current_text(self):
        if callable(self.text_provider):
            try:
                return str(self.text_provider() or "")
            except Exception:
                return ""
        return str(self.text or "")

    def _schedule(self, _event=None):
        self.hide()
        self._after_id = self.widget.after(self.delay_ms, self.show)

    def show(self):
        self._after_id = None
        text = self._current_text().strip()
        if not text or not self.widget.winfo_exists():
            return

        self._window = tk.Toplevel(self.widget)
        self._window.wm_overrideredirect(True)
        try:
            self._window.attributes("-topmost", True)
        except tk.TclError:
            pass

        frame = ctk.CTkFrame(
            self._window,
            corner_radius=8,
            border_width=1,
            border_color=AppTheme.BORDER,
        )
        frame.pack(fill="both", expand=True)
        ctk.CTkLabel(
            frame,
            text=text,
            font=AppTheme.caption_font(),
            text_color=AppTheme.TEXT_ON_DARK,
            justify="left",
            anchor="w",
            wraplength=self.wraplength,
        ).pack(padx=10, pady=8)

        x = self.widget.winfo_rootx() + self.offset_x
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + self.offset_y
        self._window.geometry(f"+{x}+{y}")

    def hide(self, _event=None):
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        if self._window is not None:
            try:
                self._window.destroy()
            except Exception:
                pass
            self._window = None
