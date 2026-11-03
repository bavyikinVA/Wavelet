"""Ограничение размера окна рабочей областью монитора (без GUI-зависимостей)."""

import sys


def fit_centered(width, height, bounds, *, anchor=None, minimum=(480, 360), margin=24):
    """Вернуть (w,h,x,y), умещающиеся в bounds=(left,top,right,bottom).

    anchor — (x,y) центра родительского окна; если его нет, центр монитора.
    """
    left, top, right, bottom = (int(v) for v in bounds)
    usable_w = max(1, right - left)
    usable_h = max(1, bottom - top)
    w = min(max(int(minimum[0]), int(width)), max(1, usable_w - margin * 2))
    h = min(max(int(minimum[1]), int(height)), max(1, usable_h - margin * 2))
    if anchor is None:
        anchor = ((left + right) / 2, (top + bottom) / 2)
    x = max(left, min(round(anchor[0] - w / 2), right - w))
    y = max(top, min(round(anchor[1] - h / 2), bottom - h))
    return w, h, x, y


def _windows_work_area(widget):
    """Рабочая область монитора окна-родителя с учётом панели задач Windows."""
    if sys.platform != 'win32':
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class MONITORINFO(ctypes.Structure):
            _fields_ = [('cbSize', wintypes.DWORD), ('rcMonitor', wintypes.RECT),
                        ('rcWork', wintypes.RECT), ('dwFlags', wintypes.DWORD)]

        api = ctypes.WinDLL('user32', use_last_error=True)
        api.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
        api.GetAncestor.restype = wintypes.HWND
        api.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
        api.MonitorFromWindow.restype = wintypes.HANDLE
        api.GetMonitorInfoW.argtypes = [wintypes.HANDLE, ctypes.POINTER(MONITORINFO)]
        api.GetMonitorInfoW.restype = wintypes.BOOL
        hwnd = api.GetAncestor(widget.winfo_id(), 2)  # GA_ROOT
        monitor = api.MonitorFromWindow(hwnd, 2)  # MONITOR_DEFAULTTONEAREST
        info = MONITORINFO()
        info.cbSize = ctypes.sizeof(MONITORINFO)
        if monitor and api.GetMonitorInfoW(monitor, ctypes.byref(info)):
            r = info.rcWork
            return int(r.left), int(r.top), int(r.right), int(r.bottom)
    except (OSError, AttributeError, ValueError):
        return None
    return None


def place_centered(dialog, master=None, *, width=1100, height=760, center_on_monitor=False):
    """Поставить окно по центру родителя/монитора; Tk-совместимый fallback.

    На Linux/X11 Tk не предоставляет portable API для панелей рабочего стола;
    используется virtual root, на Wayland положение может выбрать compositor.
    """
    dialog.update_idletasks()
    target = master if master is not None else dialog
    bounds = _windows_work_area(target)
    if bounds is None:
        try:
            vx, vy = target.winfo_vrootx(), target.winfo_vrooty()
            vw, vh = target.winfo_vrootwidth(), target.winfo_vrootheight()
            if vw <= 1 or vh <= 1:
                raise ValueError('Недоступен virtual root')
            bounds = vx, vy, vx + vw, vy + vh
        except (AttributeError, ValueError):
            bounds = (0, 0, dialog.winfo_screenwidth(), dialog.winfo_screenheight())
    anchor = None
    if master is not None and not center_on_monitor:
        try:
            if master.winfo_viewable():
                anchor = (master.winfo_rootx() + master.winfo_width() / 2,
                          master.winfo_rooty() + master.winfo_height() / 2)
        except (AttributeError, ValueError):
            pass
    w, h, x, y = fit_centered(width, height, bounds, anchor=anchor)
    # Tk's '-100' geometry syntax means 100 px from the RIGHT edge, not
    # position x=-100 on a monitor located left of the primary display.
    # '+-100' explicitly selects the absolute negative coordinate instead.
    # Call native Tk geometry to avoid CTk scaling physical pixel coordinates
    # from GetMonitorInfoW a second time on high-DPI Windows displays.
    import tkinter as tk
    tk.Toplevel.geometry(dialog, f'{w}x{h}+{x}+{y}')
