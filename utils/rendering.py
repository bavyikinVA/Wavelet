"""Windows descendant-window buffering; a no-op on other platforms."""
import sys


def enable_dark_titlebar(widget):
    """Apply the native dark caption without CTk's nested update() calls."""
    if sys.platform != 'win32':
        return False
    import ctypes
    from ctypes import wintypes
    user = ctypes.WinDLL('user32', use_last_error=True)
    user.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
    user.GetAncestor.restype = wintypes.HWND
    handle = user.GetAncestor(widget.winfo_id(), 2)
    dwm = ctypes.WinDLL('dwmapi')
    dwm.DwmSetWindowAttribute.argtypes = [wintypes.HWND, wintypes.DWORD,
                                         ctypes.c_void_p, wintypes.DWORD]
    dwm.DwmSetWindowAttribute.restype = ctypes.c_long
    dark = wintypes.BOOL(True)
    for attribute in (20, 19):
        if dwm.DwmSetWindowAttribute(handle, attribute, ctypes.byref(dark), ctypes.sizeof(dark)) == 0:
            return True
    return False


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
