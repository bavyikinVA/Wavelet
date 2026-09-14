"""Small Tk-only startup view; no images or scientific libraries to load."""
import math
import tkinter as tk
import sys


class StartupSplash(tk.Toplevel):
    def __init__(self, master, on_close):
        super().__init__(master)
        self.withdraw()
        self.title("Wavelets Analysis Studio")
        self.overrideredirect(True)
        self.configure(background="#111923")
        width, height = 560, 300
        self._splash_width = width
        self._splash_height = height
        x, y = self._center_coordinates(width, height)
        self.geometry(f"{width}x{height}+{x}+{y}")
        self.protocol("WM_DELETE_WINDOW", on_close)
        self.bind('<Escape>', lambda _event: on_close())
        tk.Button(self, text="×", command=on_close, font=("Segoe UI", 16),
                  bg="#111923", fg="#9aabba", activebackground="#263544",
                  activeforeground="white", bd=0, cursor="hand2").place(x=520, y=8, width=28)
        canvas = tk.Canvas(self, width=480, height=80, bg="#111923", highlightthickness=0)
        canvas.place(x=40, y=42)
        for offset, color in ((0, '#55b6d9'), (9, '#3d647d')):
            points = []
            for index in range(481):
                position = (index - 240) / 75
                points.extend((index, 40 + offset + 28 * math.exp(-position * position / 2)
                               * math.cos(position * 5)))
            canvas.create_line(*points, fill=color, width=2, smooth=True)
        tk.Label(self, text="Wavelets Analysis Studio", bg="#111923", fg="#edf4fa",
                 font=("Segoe UI", 23, "bold")).place(x=38, y=137)
        tk.Label(self, text="Студия исследования изображений", bg="#111923", fg="#9aabba",
                 font=("Segoe UI", 11)).place(x=40, y=181)
        self.status = tk.Label(self, text="Подготовка рабочей области…", anchor='w',
                               bg="#111923", fg="#b4c9d9", font=("Segoe UI", 10))
        self.status.place(x=40, y=233, width=480)
        self.track = tk.Canvas(self, width=480, height=3, bg="#263544", highlightthickness=0)
        self.track.place(x=40, y=269)
        self.indicator = self.track.create_rectangle(0, 0, 90, 3, fill="#55b6d9", outline='')
        self._frame = 0
        self._animation = None
        self.deiconify()
        self.lift()
        # После фактического показа Windows может применить DPI/monitor scaling.
        # Повторно центрируем уже созданное native-окно.
        self.after(1, self._recenter)
        self.after(80, self._recenter)
        self._tick()

    def _center_coordinates(self, width, height):
        """Return coordinates centered on the user's current Windows monitor.

        Tk's winfo_screenwidth() can describe a virtual/logical desktop under
        mixed-DPI multi-monitor Windows setups. Prefer the foreground monitor
        when Win32 information is available.
        """
        if sys.platform == "win32":
            try:
                import ctypes
                from ctypes import wintypes

                user32 = ctypes.windll.user32

                # Prefer the monitor of the foreground window (IDE/terminal).
                hwnd = user32.GetForegroundWindow()
                if hwnd:
                    MONITOR_DEFAULTTONEAREST = 2
                    monitor = user32.MonitorFromWindow(
                        hwnd, MONITOR_DEFAULTTONEAREST
                    )

                    class MONITORINFO(ctypes.Structure):
                        _fields_ = [
                            ("cbSize", wintypes.DWORD),
                            ("rcMonitor", wintypes.RECT),
                            ("rcWork", wintypes.RECT),
                            ("dwFlags", wintypes.DWORD),
                        ]

                    info = MONITORINFO()
                    info.cbSize = ctypes.sizeof(MONITORINFO)
                    if user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
                        left = info.rcWork.left
                        top = info.rcWork.top
                        right = info.rcWork.right
                        bottom = info.rcWork.bottom

                        work_width = right - left
                        work_height = bottom - top

                        return (
                            left + max(0, (work_width - width) // 2),
                            top + max(0, (work_height - height) // 2),
                        )
            except Exception:
                pass

        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        return (
            max(0, (screen_width - width) // 2),
            max(0, (screen_height - height) // 2),
        )

    def _recenter(self):
        if not self.winfo_exists():
            return
        x, y = self._center_coordinates(
            self._splash_width,
            self._splash_height,
        )
        self.geometry(
            f"{self._splash_width}x{self._splash_height}+{x}+{y}"
        )

    def _tick(self):
        self._frame = (self._frame + 1) % 120
        x = (self._frame / 119) * 570 - 90
        self.track.coords(self.indicator, x, 0, x + 90, 3)
        self._animation = self.after(16, self._tick)

    def set_status(self, text):
        self.status.configure(text=text)

    def show_error(self, error):
        if self._animation is not None:
            self.after_cancel(self._animation)
            self._animation = None
        self.status.place(x=40, y=216, width=480, height=48)
        self.status.configure(text=f"Не удалось запустить студию: {error}",
                              fg="#f19999", wraplength=475)

    def destroy(self):
        if self._animation is not None:
            self.after_cancel(self._animation)
            self._animation = None
        super().destroy()
