"""Встроенная прокручиваемая справка редактора изображений.

Не использует общий hover-tooltip: для прокрутки пользователь должен иметь
возможность переместить указатель внутрь карточки, не закрывая её.
"""

import customtkinter as ctk


# Формулировки хранятся отдельно от верстки и пригодны для проверки тестами.
HELP_SECTIONS = (
    (
        'Прямоугольная обрезка',
        (
            '1. Выберите «Прямоугольная».',
            '2. Зажмите левую кнопку мыши и выделите область.',
            '3. Отпустите кнопку — обрезка применится автоматически.',
        ),
    ),
    (
        'Полигональная обрезка',
        (
            '1. Выберите «Полигональная».',
            '2. Нажимайте ЛКМ по изображению, чтобы добавить вершины контура.',
            '3. Для удаления последней точки нажмите ПКМ или Ctrl + Z.',
            '4. После задания минимум трёх точек нажмите «Применить контур».',
            '«Очистить контур» удалит все точки; «Отмена» вернёт прямоугольный режим.',
        ),
    ),
    (
        'Отмена изменений',
        (
            'Ctrl + Z — шаг назад: обрезка, вершина полигона или изменение масштаба.',
            '«Сбросить» — вернуться к исходному изображению. Сброс тоже можно отменить.',
        ),
    ),
    (
        'Просмотр изображения',
        (
            'Ctrl + колесо мыши — изменить масштаб предпросмотра.',
            'Ctrl + 0 — вписать изображение в рабочую область.',
            'Колесо мыши — вертикальная прокрутка; Shift + колесо — горизонтальная.',
        ),
    ),
)


class CropperHelpOverlay(ctk.CTkFrame):
    """Контекстная карточка внутри окна кроппера, а не отдельный Toplevel.

    Поэтому она доступна для прокрутки, даже когда основное окно модальное.
    """

    def __init__(self, master, *, on_close):
        # Единая прямоугольная форма: без круглых углов, через которые
        # проступал прямоугольный фон вложенных элементов на Windows.
        super().__init__(
            master, fg_color='#191D23', corner_radius=0,
            border_width=1, border_color='#46515D',
        )
        # Содержимое с прокруткой не должно растягивать карточку за пределы
        # размеров, выбранных в toggle_help().
        self.pack_propagate(False)
        header = ctk.CTkFrame(self, fg_color='transparent')
        header.pack(fill='x', padx=(16, 10), pady=(14, 4))
        ctk.CTkLabel(
            header, text='Как пользоваться обрезкой',
            font=ctk.CTkFont(size=16, weight='bold'),
            text_color='#F3F6FA', anchor='w',
        ).pack(side='left', fill='x', expand=True)
        ctk.CTkButton(
            header, text='×', width=28, height=28, command=on_close,
            corner_radius=7, fg_color='#2A3038', hover_color='#3B4551',
            font=ctk.CTkFont(size=19), text_color='#E2E8F0',
        ).pack(side='right')

        self.content = ctk.CTkScrollableFrame(
            self, fg_color='transparent', corner_radius=0,
            scrollbar_button_color='#536273',
            scrollbar_button_hover_color='#73869B',
        )
        self.content.pack(fill='both', expand=True, padx=(9, 5), pady=(0, 12))

        for index, (title, steps) in enumerate(HELP_SECTIONS):
            section = ctk.CTkFrame(self.content, fg_color='transparent')
            section.pack(fill='x', padx=(6, 9), pady=(4 if index else 0, 12))
            ctk.CTkLabel(
                section, text=title,
                font=ctk.CTkFont(size=14, weight='bold'),
                text_color='#86C5F4', anchor='w',
            ).pack(fill='x', pady=(0, 6))
            for step in steps:
                ctk.CTkLabel(
                    section, text=step,
                    font=ctk.CTkFont(size=13),
                    text_color='#E5EAF0', justify='left', anchor='w',
                    wraplength=390,
                ).pack(fill='x', pady=(0, 5))
