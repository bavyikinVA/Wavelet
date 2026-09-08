import sys
import unittest
from unittest.mock import patch
from utils.rendering import enable_buffered_paint


class RenderingTests(unittest.TestCase):
    def test_other_platforms_do_not_access_native_window(self):
        with patch('utils.rendering.sys.platform', 'linux'):
            self.assertFalse(enable_buffered_paint(None))

    @unittest.skipUnless(sys.platform == 'win32', 'Windows painting style')
    def test_buffering_preserves_window_style_and_is_idempotent(self):
        import ctypes
        from ctypes import wintypes
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        try:
            root.update_idletasks()
            api = ctypes.WinDLL('user32')
            api.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
            api.GetAncestor.restype = wintypes.HWND
            api.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
            api.GetWindowLongW.restype = wintypes.LONG
            handle = api.GetAncestor(root.winfo_id(), 2)
            before = api.GetWindowLongW(handle, -20)
            self.assertTrue(enable_buffered_paint(root))
            self.assertEqual(api.GetWindowLongW(handle, -20), before | 0x02000000)
            self.assertTrue(enable_buffered_paint(root))
            self.assertEqual(api.GetWindowLongW(handle, -20), before | 0x02000000)
        finally:
            root.destroy()
