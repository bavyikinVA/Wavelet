"""Windows descendant-window buffering; a no-op on other platforms."""
import sys


def enable_buffered_paint(widget):
    if sys.platform != 'win32':
        return False
    import ctypes
    from ctypes import wintypes
    api = ctypes.WinDLL('user32', use_last_error=True)
    api.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
    api.GetAncestor.restype = wintypes.HWND
    api.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
    api.GetWindowLongW.restype = wintypes.LONG
    api.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.LONG]
    api.SetWindowLongW.restype = wintypes.LONG
    handle = api.GetAncestor(widget.winfo_id(), 2)  # GA_ROOT: Tk's native wrapper.
    if not handle:
        return False
    index, composited = -20, 0x02000000  # GWL_EXSTYLE, WS_EX_COMPOSITED
    style = api.GetWindowLongW(handle, index)
    ctypes.set_last_error(0)
    previous = api.SetWindowLongW(handle, index, style | composited)
    if not previous and ctypes.get_last_error():
        return False
    return bool(api.GetWindowLongW(handle, index) & composited)
