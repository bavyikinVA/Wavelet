import logging
import queue

import customtkinter as ctk

from utils.gui import TkinterApp
from utils.theme import AppTheme


class LoggingHandler(logging.Handler):
    """Обработчик логов для вывода в GUI"""

    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        try:
            msg = self.format(record)
            self.log_queue.put(msg)
        except Exception as e:
            self.log_queue.put(str(e))


class ProgressManager:
    def __init__(self, parent: TkinterApp):
        self._update_logs_callback_id = None
        self.logger = None
        self.clear_logs_btn = None
        self.percent_label = None
        self.log_text = None
        self.progress_bar = None
        self.progress_label = None
        self.frame = None
        self.logs_expanded = False
        self.logs_container = None
        self.toggle_logs_btn = None
        self.parent = parent
        self.log_queue = queue.Queue()
        self._is_active = True
        self.setup_ui()
        self.setup_logging()

        # Добавляем обработчик закрытия окна
        self.parent.bind("<Destroy>", self._on_parent_destroy)

    def _on_parent_destroy(self, event):
        """Обработчик уничтожения родительского окна"""
        if event.widget == self.parent:
            self.safe_destroy()

    def setup_ui(self):
        """Настройка UI элементов прогресса"""
        # Основной фрейм
        self.frame = ctk.CTkFrame(self.parent)
        self.frame.pack(fill="x", padx=20, pady=10)

        # Верхняя строка: текущий статус и управление журналом.
        status_frame = ctk.CTkFrame(self.frame, fg_color="transparent")
        status_frame.pack(fill="x", padx=10, pady=(8, 2))

        self.progress_label = ctk.CTkLabel(
            status_frame,
            text="Готов к работе",
            font=AppTheme.body_font()
        )
        self.progress_label.pack(side="left", fill="x", expand=True, anchor="w")

        self.toggle_logs_btn = ctk.CTkButton(
            status_frame,
            text="Показать журнал",
            command=self.toggle_logs,
            width=125,
            height=AppTheme.COMPACT_CONTROL_HEIGHT,
            fg_color="transparent",
            hover_color=AppTheme.BORDER,
            font=AppTheme.caption_font()
        )
        self.toggle_logs_btn.pack(side="right")

        # Фрейм для прогресс-бара и процентов
        progress_bar_frame = ctk.CTkFrame(self.frame, fg_color="transparent")
        progress_bar_frame.pack(fill="x", padx=10, pady=5)

        # Прогресс-бар
        self.progress_bar = ctk.CTkProgressBar(progress_bar_frame)
        self.progress_bar.pack(side="left", fill="x", expand=True)
        self.progress_bar.set(0)

        # Метка процентов
        self.percent_label = ctk.CTkLabel(
            progress_bar_frame,
            text="0%",
            font=AppTheme.caption_font(),
            width=40
        )
        self.percent_label.pack(side="right", padx=(10, 0))

        # Журнал создаётся сразу, но по умолчанию остаётся свёрнутым.
        self.logs_container = ctk.CTkFrame(self.frame, fg_color="transparent")
        self.log_text = ctk.CTkTextbox(
            self.logs_container,
            height=AppTheme.LOG_EXPANDED_HEIGHT,
            wrap="word",
            font=AppTheme.monospace_font()
        )
        self.log_text.pack(fill="both", expand=True, pady=(6, 6))
        self.log_text.configure(state="disabled")

        # Кнопка очистки логов
        self.clear_logs_btn = ctk.CTkButton(
            self.logs_container,
            text="Очистить логи",
            command=self.clear_logs,
            width=100,
            height=AppTheme.COMPACT_CONTROL_HEIGHT
        )
        self.clear_logs_btn.pack(anchor="e", pady=(0, 6))

    def toggle_logs(self):
        """Развернуть или свернуть подробный журнал вычислений."""
        self.logs_expanded = not self.logs_expanded
        if self.logs_expanded:
            self.logs_container.pack(fill="both", expand=True, padx=10, pady=(0, 4))
            self.toggle_logs_btn.configure(text="Скрыть журнал")
        else:
            self.logs_container.pack_forget()
            self.toggle_logs_btn.configure(text="Показать журнал")

    def setup_logging(self):
        """Настройка системы логирования"""
        self.logger = logging.getLogger('WaveletApp')
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )

        # GUI handler
        gui_handler = LoggingHandler(self.log_queue)
        gui_handler.setFormatter(formatter)
        gui_handler.setLevel(logging.INFO)
        self.logger.addHandler(gui_handler)

        # File handler
        file_handler = logging.FileHandler('wavelet_analysis.log', encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)

        # Запуск обновления логов
        self._update_logs_callback_id = self.parent.after_safe(100, self._update_logs)

    def _update_logs(self):
        """Обновление логов в GUI"""
        if not self._is_active:
            return

        try:
            while True:
                msg = self.log_queue.get_nowait()
                if self._is_active:
                    self.log_text.configure(state="normal")
                    self.log_text.insert("end", msg + "\n")
                    self.log_text.see("end")
                    self.log_text.configure(state="disabled")
        except queue.Empty:
            pass
        finally:
            if self._is_active:
                self._update_logs_callback_id = self.parent.after_safe(100, self._update_logs)

    def update_progress(self, value: float, message: str = ""):
        """Обновление прогресса"""
        if not self._is_active:
            return

        value = max(0.0, min(1.0, value))  # Ограничение 0-1
        self.progress_bar.set(value)
        self.percent_label.configure(text=f"{int(value * 100)}%")

        if message and self._is_active:
            self.progress_label.configure(text=message)
            self.logger.info(message)

    def log_info(self, message: str):
        """Логирование информационного сообщения"""
        if self._is_active:
            self.logger.info(message)

    def log_error(self, message: str):
        """Логирование ошибки"""
        if self._is_active:
            self.logger.error(message)
            self.progress_label.configure(text=f"Ошибка: {message}")

    def log_debug(self, message: str):
        """Логирование отладочной информации"""
        if self._is_active:
            self.logger.debug(message)

    def clear_logs(self):
        """Очистка логов в GUI"""
        if self._is_active:
            self.log_text.configure(state="normal")
            self.log_text.delete("1.0", "end")
            self.log_text.configure(state="disabled")

    def reset(self):
        """Сброс прогресса"""
        if self._is_active:
            self.update_progress(0.0, "Готов к работе")

    def safe_destroy(self):
        """Безопасное уничтожение менеджера прогресса"""
        self._is_active = False

        # Отменяем все pending callbacks
        if hasattr(self, '_update_logs_callback_id') and self._update_logs_callback_id:
            self.parent.after_cancel_safe(self._update_logs_callback_id)
