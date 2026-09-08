"""Единые визуальные настройки приложения Wavelet Analysis."""

import customtkinter as ctk


class AppTheme:
    """Цвета, размеры, интервалы и типографика интерфейса."""

    # Цвета действий и состояний
    PRIMARY = "#2b5b84"
    PRIMARY_HOVER = "#1e4160"
    SUCCESS = "#28a745"
    SUCCESS_HOVER = "#218838"
    DANGER = "#dc3545"
    DANGER_HOVER = "#c82333"
    WARNING = "#d6a84b"
    INFO = "#17a2b8"

    # Нейтральные цвета
    DISABLED = "#6c757d"
    DISABLED_HOVER = "#5a6268"
    DISABLED_DARK = "#50545a"
    BORDER = "#3a3a3a"
    ACTIVE_BORDER = "#007bff"
    TEXT_SECONDARY = "#a9a9a9"
    TEXT_ON_DARK = "white"
    MUTED = "gray"
    GPU_AVAILABLE = "#69b47c"
    MODE_2D = "#6a5330"
    # Навигация намеренно ахроматическая: глобальная синяя тема CTk не должна
    # окрашивать заголовки рабочих вкладок.
    NAV_BACKGROUND = "#242424"
    NAV_SELECTED_BACKGROUND = "#2b2b2b"
    NAV_ACTIVE = "#9b9b9b"
    NAV_HOVER = "#303030"
    NAV_TEXT_ACTIVE = "#f2f2f2"
    NAV_TEXT_INACTIVE = "#a9a9a9"

    # Размеры и интервалы
    WINDOW_PADDING = 8
    PANEL_GAP = 6
    SECTION_GAP = 6
    CONTENT_PADDING = 12
    BUTTON_HEIGHT = 36
    SMALL_BUTTON_HEIGHT = 28
    COMPACT_CONTROL_HEIGHT = 26
    BADGE_HEIGHT = 20
    PRIMARY_BUTTON_HEIGHT = 46
    NAV_HEIGHT = 50
    ACTION_BUTTON_WIDTH = 230
    SIDEBAR_WIDTH = 420
    DROPDOWN_WIDTH = 320
    LOG_EXPANDED_HEIGHT = 120

    @staticmethod
    def page_title_font():
        return ctk.CTkFont(size=18, weight="bold")

    @staticmethod
    def panel_title_font():
        return ctk.CTkFont(size=16, weight="bold")

    @staticmethod
    def section_title_font():
        return ctk.CTkFont(size=13, weight="bold")

    @staticmethod
    def body_font():
        return ctk.CTkFont(size=12)

    @staticmethod
    def caption_font():
        return ctk.CTkFont(size=10)

    @staticmethod
    def monospace_font():
        return ctk.CTkFont(family="Consolas", size=10)
