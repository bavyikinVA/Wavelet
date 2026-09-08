import time
import unittest
import customtkinter as ctk
from utils.animation import animate_visibility


def finish_animations(root, *widgets):
    deadline = time.monotonic() + 3
    while any(getattr(w, '_disclosure', {}).get('job') for w in widgets):
        root.update()
        if time.monotonic() > deadline:
            raise AssertionError('Animation did not finish')
        time.sleep(.005)


class AnimationTests(unittest.TestCase):
    def test_collapse_reverse_and_restore_geometry(self):
        root = ctk.CTk()
        root.withdraw()
        frame = ctk.CTkFrame(root)
        frame.pack()
        ctk.CTkLabel(frame, text='Content', height=80).pack()
        root.update_idletasks()
        show = lambda: frame.pack()
        try:
            animate_visibility(frame, False, show, frame.pack_forget)
            self.assertEqual(frame.winfo_manager(), 'pack')
            finish_animations(root, frame)
            self.assertEqual(frame.winfo_manager(), '')
            animate_visibility(frame, True, show, frame.pack_forget)
            root.update()
            animate_visibility(frame, False, show, frame.pack_forget)
            animate_visibility(frame, True, show, frame.pack_forget)
            finish_animations(root, frame)
            self.assertEqual(frame.winfo_manager(), 'pack')
            self.assertTrue(frame.pack_propagate())
            self.assertIsNone(frame._disclosure['job'])
        finally:
            for callback in root.tk.call('after', 'info'):
                root.tk.call('after', 'cancel', callback)
            root.destroy()
