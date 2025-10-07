import tkinter as tk
import customtkinter as ctk
from typing import Optional, Callable


class TkinterApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self._is_destroyed = False
        self._pending_callbacks = {}  # Словарь активных after callbacks
        self._child_windows = []  # Список дочерних окон
        self.protocol("WM_DELETE_WINDOW", self.safe_destroy)

    def register_child_window(self, window):
        """Регистрация дочернего окна для безопасного закрытия"""
        self._child_windows.append(window)

    def unregister_child_window(self, window):
        """Удаление дочернего окна из списка"""
        if window in self._child_windows:
            self._child_windows.remove(window)

    def safe_destroy(self):
        """Безопасное уничтожение приложения"""
        if self._is_destroyed:
            return

        self._is_destroyed = True

        # Сначала безопасно закрываем все дочерние окна
        for child_window in self._child_windows[:]:
            try:
                if hasattr(child_window, 'safe_destroy'):
                    child_window.safe_destroy()
                elif hasattr(child_window, 'destroy'):
                    child_window.destroy()
            except Exception as e:
                print(f'Exception while destroying {child_window}: {e}')
        self._child_windows.clear()

        # Отменяем все запланированные callbacks
        for callback_id in list(self._pending_callbacks.keys()):
            try:
                self.after_cancel(callback_id)
                del self._pending_callbacks[callback_id]
            except tk.TclError:
                pass

        try:
            self.quit()
            self.destroy()
        except tk.TclError:
            pass

    def after_safe(self, ms: int, func: Callable, *args) -> Optional[str]:
        """Безопасный вызов after с отслеживанием callback'ов"""
        if self._is_destroyed:
            return None

        try:
            def safe_func():
                if not self._is_destroyed:
                    try:
                        func(*args)
                    except Exception as e:
                        print(f"Callback error: {e}")
                # Удаляем callback из отслеживаемых после выполнения
                if callback_id in self._pending_callbacks:
                    del self._pending_callbacks[callback_id]

            callback_id = self.after(ms, safe_func)
            self._pending_callbacks[callback_id] = True
            return callback_id
        except tk.TclError:
            return None

    def after_cancel_safe(self, callback_id: str) -> None:
        """Безопасная отмена callback'а"""
        if callback_id in self._pending_callbacks:
            try:
                self.after_cancel(callback_id)
                del self._pending_callbacks[callback_id]
            except tk.TclError:
                pass


class ScrollableFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        # Создаем canvas и скроллбар
        self.canvas = tk.Canvas(
            self,
            bg=self._apply_appearance_mode(self.cget("fg_color")),
            highlightthickness=0
        )
        self.scrollbar = ctk.CTkScrollbar(self, orientation="vertical", command=self.canvas.yview)
        self.scrollable_frame = ctk.CTkFrame(self.canvas)

        # Настраиваем скроллинг
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        # Создаем окно в canvas для скроллируемого фрейма
        self.canvas_frame_id = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        # Обновляем ширину scrollable_frame при изменении размера canvas
        def update_frame_width(event):
            canvas_width = event.width
            self.canvas.itemconfig(self.canvas_frame_id, width=canvas_width)

        self.canvas.bind("<Configure>", update_frame_width)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Упаковываем элементы
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Привязываем события мыши для скроллинга
        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)

    def _bind_mousewheel(self, event):
        """Привязка колесика мыши"""
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self, event):
        """Отвязка колесика мыши"""
        self.canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event):
        """Обработка скроллинга колесиком мыши"""
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


class ResizablePanedWindow(ctk.CTkFrame):
    """Разделяемое окно с возможностью изменения размеров панелей"""

    def __init__(self, master, orientation="horizontal", **kwargs):
        super().__init__(master, **kwargs)
        self.orientation = orientation
        self.panes = []
        self.sash_positions = []
        self.setup_ui()

    def setup_ui(self):
        """Настройка UI разделяемого окна"""
        if self.orientation == "horizontal":
            self.grid_rowconfigure(0, weight=1)
            self.grid_columnconfigure(0, weight=1)
            self.grid_columnconfigure(2, weight=1)
        else:
            self.grid_columnconfigure(0, weight=1)
            self.grid_rowconfigure(0, weight=1)
            self.grid_rowconfigure(2, weight=1)

    def add_pane(self, widget, weight=1):
        """Добавление панели"""
        pane_index = len(self.panes)

        if pane_index == 0:
            # Первая панель
            if self.orientation == "horizontal":
                widget.grid(row=0, column=0, sticky="nsew", padx=(0, 2))
                self.grid_columnconfigure(0, weight=weight)
            else:
                widget.grid(row=0, column=0, sticky="nsew", pady=(0, 2))
                self.grid_rowconfigure(0, weight=weight)

        elif pane_index == 1:
            # Разделитель и вторая панель
            self._create_sash()
            if self.orientation == "horizontal":
                widget.grid(row=0, column=2, sticky="nsew", padx=(2, 0))
                self.grid_columnconfigure(2, weight=weight)
            else:
                widget.grid(row=2, column=0, sticky="nsew", pady=(2, 0))
                self.grid_rowconfigure(2, weight=weight)

        self.panes.append(widget)
        return widget

    def _create_sash(self):
        """Создание разделителя"""
        self.sash = ctk.CTkFrame(self, width=4, height=4, fg_color="#cccccc")
        self.sash.bind("<Button-1>", self._on_sash_press)
        self.sash.bind("<B1-Motion>", self._on_sash_drag)
        self.sash.bind("<Enter>", self._on_sash_enter)
        self.sash.bind("<Leave>", self._on_sash_leave)

        if self.orientation == "horizontal":
            self.sash.grid(row=0, column=1, sticky="ns")
        else:
            self.sash.grid(row=1, column=0, sticky="ew")

    def _on_sash_enter(self, event):
        """Изменение курсора при наведении на разделитель"""
        if self.orientation == "horizontal":
            self.sash.configure(cursor="sb_h_double_arrow")
        else:
            self.sash.configure(cursor="sb_v_double_arrow")

    def _on_sash_leave(self, event):
        """Возврат курсора при уходе с разделителя"""
        self.sash.configure(cursor="")

    def _on_sash_press(self, event):
        """Начало перемещения разделителя"""
        self.sash_start_pos = (event.x_root, event.y_root)
        if self.orientation == "horizontal":
            self.pane1_width = self.panes[0].winfo_width()
        else:
            self.pane1_height = self.panes[0].winfo_height()

    def _on_sash_drag(self, event):
        """Перемещение разделителя"""
        if self.orientation == "horizontal":
            delta = event.x_root - self.sash_start_pos[0]
            new_width = max(100, self.pane1_width + delta)
            total_width = self.winfo_width()
            if new_width < total_width - 100:  # Минимальная ширина второй панели
                self.grid_columnconfigure(0, weight=new_width)
                self.grid_columnconfigure(2, weight=total_width - new_width)
        else:
            delta = event.y_root - self.sash_start_pos[1]
            new_height = max(100, self.pane1_height + delta)
            total_height = self.winfo_height()
            if new_height < total_height - 100:  # Минимальная высота второй панели
                self.grid_rowconfigure(0, weight=new_height)
                self.grid_rowconfigure(2, weight=total_height - new_height)


class CollapsibleFrame(ctk.CTkFrame):
    def __init__(self, master, title="", **kwargs):
        super().__init__(master, **kwargs)
        self.content = None
        self.toggle_btn = None
        self.header = None
        self.title = title
        self.is_expanded = True
        self.setup_ui()

    def setup_ui(self):
        """Настройка UI сворачиваемого фрейма"""
        # Заголовок
        self.header = ctk.CTkFrame(self, fg_color="transparent")
        self.header.pack(fill="x", padx=5, pady=(5, 0))

        self.toggle_btn = ctk.CTkButton(
            self.header,
            text=f"▼ {self.title}",
            command=self.toggle,
            height=30,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="transparent",
            hover_color="#3a3a3a",
            anchor="w"
        )
        self.toggle_btn.pack(fill="x")

        # Контент
        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.pack(fill="x", padx=10, pady=(5, 10))

    def toggle(self):
        """Свернуть/развернуть фрейм"""
        self.is_expanded = not self.is_expanded

        if self.is_expanded:
            self.toggle_btn.configure(text=f"▼ {self.title}")
            self.content.pack(fill="x", padx=10, pady=(5, 10))
        else:
            self.toggle_btn.configure(text=f"▶ {self.title}")
            self.content.pack_forget()

    def add_widget(self, widget, **pack_args):
        """Добавить виджет в контентную область"""
        default_pack_args = {"fill": "x", "padx": 5, "pady": 2}
        default_pack_args.update(pack_args)
        widget.pack(in_=self.content, **default_pack_args)