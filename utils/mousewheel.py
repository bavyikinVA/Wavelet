"""Mouse-wheel routing for scrollable popup menus on Windows, Linux and macOS.

Bind at the widget level so a popup captures wheel events before any parent
page-wide ``bind_all`` handler can redirect them to the wrong scroll region.
"""
from __future__ import annotations
import tkinter as tk


def scroll_units(event):
    """Signed Tk scrolling units: negative = up, positive = down."""
    button = getattr(event, 'num', None)
    if button == 4:  # X11 / Linux wheel up
        return -1
    if button == 5:  # X11 / Linux wheel down
        return 1
    delta = getattr(event, 'delta', 0) or 0
    if delta == 0:
        return 0
    # Windows typically uses +/-120; some touchpads / macOS use smaller steps.
    magnitude = max(1, int(abs(delta) / 120))
    return -magnitude if delta > 0 else magnitude


def bind_popup_mousewheel(popup, scrollable):
    """Route the wheel over any child of an open CTkScrollableFrame to its canvas.

    Call after populating a dropdown; safe to repeat after a live search rebuild.
    CTkScrollableFrame's internal canvas is an existing CustomTkinter API detail.
    """
    canvas = getattr(scrollable, '_parent_canvas', None)
    if canvas is None:
        return

    def handle(event):
        units = scroll_units(event)
        if units:
            try:
                canvas.yview_scroll(units, 'units')
            except (tk.TclError, AttributeError):  # an already destroyed popup
                pass
        return 'break'

    pending = [popup]
    while pending:
        widget = pending.pop()
        try:
            pending.extend(widget.winfo_children())
            if getattr(widget, '_wavelets_wheel_canvas', None) is canvas:
                continue
            widget._wavelets_wheel_canvas = canvas
            widget.bind('<MouseWheel>', handle, add='+')
            widget.bind('<Button-4>', handle, add='+')
            widget.bind('<Button-5>', handle, add='+')
        except (AttributeError, TypeError, tk.TclError):
            continue
