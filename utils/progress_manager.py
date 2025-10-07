import logging
import queue

import customtkinter as ctk

from utils.gui import TkinterApp


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


class SafeProgressManager:
    def __init__(self, parent: TkinterApp):
        self._update_logs_callback_id = None
        self.logger = None
        self.clear_logs_btn = None
        self.percent_label = None
        self.log_text = None
        self.progress_bar = None
        self.progress_label = None
        self.progress_frame = None
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
        self.progress_frame = ctk.CTkFrame(self.parent)
        self.progress_frame.pack(fill="x", padx=20, pady=10)

        # Метка статуса
        self.progress_label = ctk.CTkLabel(
            self.progress_frame,
            text="Готов к работе",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.progress_label.pack(anchor="w", padx=10, pady=(10, 5))

        # Фрейм для прогресс-бара и процентов
        progress_bar_frame = ctk.CTkFrame(self.progress_frame, fg_color="transparent")
        progress_bar_frame.pack(fill="x", padx=10, pady=5)

        # Прогресс-бар
        self.progress_bar = ctk.CTkProgressBar(progress_bar_frame)
        self.progress_bar.pack(side="left", fill="x", expand=True)
        self.progress_bar.set(0)

        # Метка процентов
        self.percent_label = ctk.CTkLabel(
            progress_bar_frame,
            text="0%",
            font=ctk.CTkFont(size=10),
            width=40
        )
        self.percent_label.pack(side="right", padx=(10, 0))

        # Текстовое поле для логов
        self.log_text = ctk.CTkTextbox(
            self.progress_frame,
            height=120,
            wrap="word",
            font=ctk.CTkFont(family="Consolas", size=10)
        )
        self.log_text.pack(fill="both", expand=True, padx=10, pady=10)
        self.log_text.configure(state="disabled")

        # Кнопка очистки логов
        self.clear_logs_btn = ctk.CTkButton(
            self.progress_frame,
            text="Очистить логи",
            command=self.clear_logs,
            width=100,
            height=25
        )
        self.clear_logs_btn.pack(anchor="e", padx=10, pady=(0, 10))

    def setup_logging(self):
        """Настройка системы логирования"""
        self.logger = logging.getLogger('WaveletApp')
        self.logger.setLevel(logging.INFO)

        # Форматтер
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


# Обновим ProgressManager для совместимости
class ProgressManager(SafeProgressManager):
    """Алиас для совместимости"""
    pass