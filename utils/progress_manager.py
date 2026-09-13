import logging
import os
import queue
import threading
import time
from datetime import datetime
from tkinter import filedialog

import customtkinter as ctk

from utils.gui import TkinterApp
from utils.theme import AppTheme


class LoggingHandler(logging.Handler):
    """Thread-safe bridge between Python logging and the GUI journal."""

    def __init__(self, log_queue: queue.Queue, record_callback=None):
        super().__init__()
        self.log_queue = log_queue
        self.record_callback = record_callback

    def emit(self, record):
        try:
            msg = self.format(record)
            self.log_queue.put(msg)
            if self.record_callback is not None:
                self.record_callback(msg)
        except Exception as error:
            self.log_queue.put(str(error))


class ProgressManager:
    """Persistent execution panel with separate run and stage progress."""

    def __init__(self, parent: TkinterApp):
        self.parent = parent
        self.log_queue = queue.Queue()
        self._is_active = True
        self._update_logs_callback_id = None
        self._timer_callback_id = None
        self._started_at = None
        self._run_title = ""
        self._stages = []
        self._stage_index = 0
        self._stage_progress = 0.0
        self._run_log_lines = []
        self._log_lock = threading.Lock()
        self.logs_expanded = False
        self._result_callback = None
        self._folder_callback = None
        self._history_callback = None
        from compute.run_control import RunControl
        self.run_control = RunControl()
        self.setup_ui()
        self.setup_logging()
        self.parent.bind("<Destroy>", self._on_parent_destroy)

    def _on_parent_destroy(self, event):
        if event.widget == self.parent:
            self.safe_destroy()

    def setup_ui(self):
        self.frame = ctk.CTkFrame(
            self.parent, border_width=1, border_color=AppTheme.BORDER
        )
        self.frame.pack(fill="x", padx=20, pady=(4, 10))

        status_frame = ctk.CTkFrame(self.frame, fg_color="transparent")
        status_frame.pack(fill="x", padx=12, pady=(8, 2))
        self.state_badge = ctk.CTkLabel(
            status_frame, text="● ГОТОВО", text_color=AppTheme.TEXT_SECONDARY,
            font=AppTheme.caption_font(), width=84, anchor="w"
        )
        self.state_badge.pack(side="left")
        self.progress_label = ctk.CTkLabel(
            status_frame, text="Готов к работе",
            font=AppTheme.section_title_font(), anchor="w"
        )
        self.progress_label.pack(side="left", fill="x", expand=True, padx=(6, 8))
        self.timer_label = ctk.CTkLabel(
            status_frame, text="00:00", font=AppTheme.caption_font(), width=68
        )
        self.timer_label.pack(side="right", padx=(8, 0))
        self.toggle_logs_btn = ctk.CTkButton(
            status_frame, text="Показать журнал", command=self.toggle_logs,
            width=125, height=AppTheme.COMPACT_CONTROL_HEIGHT,
            fg_color="transparent", hover_color=AppTheme.BORDER,
            font=AppTheme.caption_font()
        )
        self.toggle_logs_btn.pack(side="right")

        self.stage_label = ctk.CTkLabel(
            self.frame, text="Этапы появятся после запуска",
            font=AppTheme.caption_font(), text_color=AppTheme.TEXT_SECONDARY,
            anchor="w"
        )
        self.stage_label.pack(fill="x", padx=12, pady=(1, 1))

        self._create_progress_row("Общий прогресс", "overall")
        self._create_progress_row("Текущий этап", "stage")

        self.completion_actions = ctk.CTkFrame(self.frame, fg_color="transparent")
        self.result_btn = ctk.CTkButton(
            self.completion_actions, text="Открыть результаты", width=150,
            height=AppTheme.COMPACT_CONTROL_HEIGHT,
            command=lambda: self._call_action(self._result_callback)
        )
        self.result_btn.pack(side="left")
        self.folder_btn = ctk.CTkButton(
            self.completion_actions, text="Открыть папку", width=125,
            height=AppTheme.COMPACT_CONTROL_HEIGHT,
            command=lambda: self._call_action(self._folder_callback)
        )
        self.folder_btn.pack(side="left", padx=(8, 0))
        self.history_btn = ctk.CTkButton(
            self.completion_actions, text="К предыдущим запускам", width=180,
            height=AppTheme.COMPACT_CONTROL_HEIGHT,
            command=lambda: self._call_action(self._history_callback)
        )
        self.history_btn.pack(side="left", padx=(8, 0))

        self.logs_container = ctk.CTkFrame(self.frame, fg_color="transparent")
        self.log_text = ctk.CTkTextbox(
            self.logs_container, height=AppTheme.LOG_EXPANDED_HEIGHT,
            wrap="word", font=AppTheme.monospace_font()
        )
        self.log_text.pack(fill="both", expand=True, pady=(6, 6))
        self.log_text.configure(state="disabled")
        log_actions = ctk.CTkFrame(self.logs_container, fg_color="transparent")
        log_actions.pack(fill="x", pady=(0, 6))
        self.save_logs_btn = ctk.CTkButton(
            log_actions, text="Сохранить логи", command=self.export_logs,
            width=120, height=AppTheme.COMPACT_CONTROL_HEIGHT
        )
        self.save_logs_btn.pack(side="right")
        self.clear_logs_btn = ctk.CTkButton(
            log_actions, text="Очистить логи", command=self.clear_logs,
            width=110, height=AppTheme.COMPACT_CONTROL_HEIGHT,
            fg_color="transparent", hover_color=AppTheme.BORDER
        )
        self.clear_logs_btn.pack(side="right", padx=(0, 8))

    def _create_progress_row(self, title, prefix):
        row = ctk.CTkFrame(self.frame, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=2)
        ctk.CTkLabel(
            row, text=title, width=108, anchor="w", font=AppTheme.caption_font()
        ).pack(side="left")
        bar = ctk.CTkProgressBar(row)
        bar.pack(side="left", fill="x", expand=True)
        bar.set(0)
        label = ctk.CTkLabel(
            row, text="0%", width=42, font=AppTheme.caption_font()
        )
        label.pack(side="right", padx=(8, 0))
        setattr(self, f"{prefix}_progress_bar", bar)
        setattr(self, f"{prefix}_percent_label", label)
        # Compatibility for older code and tests.
        if prefix == "overall":
            self.progress_bar = bar
            self.percent_label = label

    def setup_logging(self):
        self.logger = logging.getLogger("WaveletApp")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
        )
        gui_handler = LoggingHandler(self.log_queue, self._remember_log_line)
        gui_handler.setFormatter(formatter)
        gui_handler.setLevel(logging.INFO)
        self.logger.addHandler(gui_handler)
        file_handler = logging.FileHandler("wavelet_analysis.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self._update_logs_callback_id = self.parent.after_safe(100, self._update_logs)

    def _remember_log_line(self, line):
        with self._log_lock:
            self._run_log_lines.append(line)

    def _ui(self, callback):
        if self._is_active:
            self.parent.after_safe(0, callback)

    def begin_run(self, title, stages):
        self._run_title = title
        self._stages = list(stages) or ["Выполнение"]
        self._stage_index = 0
        self._stage_progress = 0.0
        self._started_at = time.monotonic()
        with self._log_lock:
            self._run_log_lines = []
        self._ui(lambda: self._apply_begin_run(title))
        self._schedule_timer()

    def _apply_begin_run(self, title):
        if self.run_control.event.is_set():
            return
        self.completion_actions.pack_forget()
        self.state_badge.configure(text="● ВЫПОЛНЯЕТСЯ", text_color=AppTheme.INFO)
        self.progress_label.configure(text=title, text_color=AppTheme.TEXT_ON_DARK)
        self._set_bars(0.0, 0.0)
        self._apply_stage_label()

    def begin_stage(self, name, index=None):
        self.run_control.check()
        if name not in self._stages:
            self._stages.append(name)
        self._stage_index = self._stages.index(name) if index is None else int(index)
        self._stage_index = max(0, min(self._stage_index, len(self._stages) - 1))
        self._stage_progress = 0.0
        self._ui(lambda: self._apply_begin_stage(name))
        self.log_info(f"Этап {self._stage_index + 1}/{len(self._stages)}: {name}")

    def _apply_begin_stage(self, name):
        if self.run_control.event.is_set():
            return
        self.progress_label.configure(text=name, text_color=AppTheme.TEXT_ON_DARK)
        self._apply_stage_label()
        self._set_bars(self._overall_value(), 0.0)

    def complete_stage(self):
        self.update_progress(1.0)

    def update_progress(self, value: float, message: str = ""):
        """Update progress inside the active stage, preserving global progress."""
        self.run_control.check()
        value = max(0.0, min(1.0, float(value)))
        self._stage_progress = value
        self._ui(lambda: self._apply_progress(value, message))
        if message:
            self.logger.info(message)

    def _apply_progress(self, value, message):
        if self.run_control.event.is_set():
            return
        self._set_bars(self._overall_value(), value)
        if message:
            self.progress_label.configure(text=message, text_color=AppTheme.TEXT_ON_DARK)
        self._apply_stage_label()

    def _overall_value(self):
        if not self._stages:
            return self._stage_progress
        return min(1.0, (self._stage_index + self._stage_progress) / len(self._stages))

    def _set_bars(self, overall, stage):
        self.overall_progress_bar.set(overall)
        self.overall_percent_label.configure(text=f"{int(overall * 100)}%")
        self.stage_progress_bar.set(stage)
        self.stage_percent_label.configure(text=f"{int(stage * 100)}%")

    def _apply_stage_label(self):
        if self._stages:
            self.stage_label.configure(
                text=(f"Этап {self._stage_index + 1} из {len(self._stages)} · "
                      f"{self._stages[self._stage_index]}")
            )

    def finish_run(self, message, *, result_callback=None, folder_callback=None,
                   history_callback=None):
        self._result_callback = result_callback
        self._folder_callback = folder_callback
        self._history_callback = history_callback
        self._stage_index = max(0, len(self._stages) - 1)
        self._stage_progress = 1.0
        self._ui(lambda: self._apply_finished(message))

    def _apply_finished(self, message):
        self._cancel_timer()
        if self._started_at is not None:
            elapsed = int(time.monotonic() - self._started_at)
            self.timer_label.configure(text=f"{elapsed // 60:02d}:{elapsed % 60:02d}")
        self._started_at = None
        self.state_badge.configure(text="● ГОТОВО", text_color=AppTheme.SUCCESS)
        self.progress_label.configure(text=message, text_color=AppTheme.TEXT_ON_DARK)
        count = len(self._stages)
        self.stage_label.configure(text=f"Все этапы завершены · {count} из {count}")
        self._set_bars(1.0, 1.0)
        self.completion_actions.pack(fill="x", padx=12, pady=(4, 8))

    def fail_run(self, message):
        self._ui(lambda: self._apply_failed(message))

    def cancel_run(self):
        def apply():
            self._cancel_timer()
            self._started_at = None
            self.state_badge.configure(text='● ОТМЕНЕНО', text_color=AppTheme.TEXT_SECONDARY)
            self.progress_label.configure(text='Расчёт отменён. Частичные файлы сохранены.')
            self.completion_actions.pack_forget()
        self._ui(apply)

    def _apply_failed(self, message):
        self._cancel_timer()
        self.state_badge.configure(text="● ОШИБКА", text_color=AppTheme.DANGER)
        self.progress_label.configure(text=message, text_color=AppTheme.DANGER)
        self.completion_actions.pack_forget()

    def _schedule_timer(self):
        self._cancel_timer()
        self._timer_callback_id = self.parent.after_safe(250, self._tick_timer)

    def _tick_timer(self):
        if not self._is_active or self._started_at is None:
            return
        elapsed = int(time.monotonic() - self._started_at)
        self.timer_label.configure(text=f"{elapsed // 60:02d}:{elapsed % 60:02d}")
        self._timer_callback_id = self.parent.after_safe(250, self._tick_timer)

    def _cancel_timer(self):
        if self._timer_callback_id is not None:
            self.parent.after_cancel_safe(self._timer_callback_id)
            self._timer_callback_id = None

    def _call_action(self, callback):
        if callback is not None:
            callback()

    def show_completion_actions(self, **callbacks):
        self._result_callback = callbacks.get("result_callback")
        self._folder_callback = callbacks.get("folder_callback")
        self._history_callback = callbacks.get("history_callback")

    def toggle_logs(self):
        from utils.animation import animate_visibility
        self.logs_expanded = not self.logs_expanded
        if self.logs_expanded:
            self.toggle_logs_btn.configure(text="Скрыть журнал")
        else:
            self.toggle_logs_btn.configure(text="Показать журнал")
        animate_visibility(self.logs_container, self.logs_expanded,
                           lambda: self.logs_container.pack(fill='both', expand=True, padx=10, pady=(0, 4)),
                           self.logs_container.pack_forget)

    def _update_logs(self):
        if not self._is_active:
            return
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_text.configure(state="normal")
                self.log_text.insert("end", msg + "\n")
                self.log_text.see("end")
                self.log_text.configure(state="disabled")
        except queue.Empty:
            pass
        finally:
            if self._is_active:
                self._update_logs_callback_id = self.parent.after_safe(100, self._update_logs)

    def set_status(self, message: str, color=None):
        self._ui(lambda: self.progress_label.configure(
            text=message,
            **({"text_color": color} if color is not None else {})
        ))

    def log_info(self, message: str):
        if self._is_active:
            self.logger.info(message)

    def log_error(self, message: str):
        if self._is_active:
            self.logger.error(message)

    def log_debug(self, message: str):
        if self._is_active:
            self.logger.debug(message)

    def clear_logs(self):
        if self._is_active:
            self.log_text.configure(state="normal")
            self.log_text.delete("1.0", "end")
            self.log_text.configure(state="disabled")

    def _log_document(self, header=None):
        with self._log_lock:
            lines = list(self._run_log_lines)
        heading = [
            "WAVELET ANALYSIS — ЖУРНАЛ ИССЛЕДОВАНИЯ",
            f"Экспорт: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        ]
        if self._run_title:
            heading.append(f"Запуск: {self._run_title}")
        if header:
            heading.extend(str(header).splitlines())
        return "\n".join(heading + ["", *lines, ""])

    def export_logs(self):
        filename = filedialog.asksaveasfilename(
            parent=self.parent, title="Сохранить журнал",
            defaultextension=".txt", filetypes=[("Текстовый файл", "*.txt")],
            initialfile=datetime.now().strftime("wavelet_log_%Y-%m-%d_%H-%M.txt")
        )
        if filename:
            with open(filename, "w", encoding="utf-8") as file:
                file.write(self._log_document())
            self.log_info(f"Журнал сохранён: {filename}")

    def save_run_log(self, output_dir, header=None):
        if not output_dir:
            return ""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "Журнал_запуска.txt")
        with open(path, "w", encoding="utf-8") as file:
            file.write(self._log_document(header))
        return path

    def reset(self):
        self._started_at = None
        self._stages = []
        self._stage_index = 0
        self._stage_progress = 0.0
        self._ui(self._apply_reset)

    def _apply_reset(self):
        self._cancel_timer()
        self.state_badge.configure(text="● ГОТОВО", text_color=AppTheme.TEXT_SECONDARY)
        self.progress_label.configure(text="Готов к работе", text_color=AppTheme.TEXT_ON_DARK)
        self.stage_label.configure(text="Этапы появятся после запуска")
        self.timer_label.configure(text="00:00")
        self._set_bars(0.0, 0.0)
        self.completion_actions.pack_forget()

    def safe_destroy(self):
        self._is_active = False
        self._cancel_timer()
        if self._update_logs_callback_id:
            self.parent.after_cancel_safe(self._update_logs_callback_id)
