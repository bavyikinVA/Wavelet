"""Windows descendant-window buffering; a no-op on other platforms."""
import sys


def enable_dark_titlebar(widget, *, caption_color=None, text_color=None):
    """Set a Windows-native dark caption without CTk's nested update() calls.

    Optional caption/text colors are supported on Windows 11. On Windows 10,
    the immersive dark mode flag still works when custom colors are unsupported.
    """
    if sys.platform != 'win32':
        return False
    import ctypes
    from ctypes import wintypes
    user = ctypes.WinDLL('user32', use_last_error=True)
    user.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
    user.GetAncestor.restype = wintypes.HWND
    handle = user.GetAncestor(widget.winfo_id(), 2)  # GA_ROOT, the native Tk wrapper
    if not handle:
        return False
    dwm = ctypes.WinDLL('dwmapi')
    dwm.DwmSetWindowAttribute.argtypes = [wintypes.HWND, wintypes.DWORD,
                                         ctypes.c_void_p, wintypes.DWORD]
    dwm.DwmSetWindowAttribute.restype = ctypes.c_long
    dark = wintypes.BOOL(True)
    dark_enabled = False
    for attribute in (20, 19):
        if dwm.DwmSetWindowAttribute(handle, attribute, ctypes.byref(dark), ctypes.sizeof(dark)) == 0:
            dark_enabled = True
            break

    def set_color(attribute, hex_color):
        if not hex_color:
            return False
        value = hex_color.lstrip('#')
        if len(value) != 6:
            return False
        try:
            rgb = int(value, 16)
        except ValueError:
            return False
        # COLORREF uses BGR byte order, unlike the #RRGGBB used by CTk.
        bgr = ((rgb & 0xff) << 16) | (rgb & 0xff00) | ((rgb >> 16) & 0xff)
        color = wintypes.DWORD(bgr)
        return dwm.DwmSetWindowAttribute(handle, attribute,
                                         ctypes.byref(color), ctypes.sizeof(color)) == 0

    colored = set_color(35, caption_color)  # DWMWA_CAPTION_COLOR
    colored = set_color(36, text_color) or colored  # DWMWA_TEXT_COLOR
    return dark_enabled or colored


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
