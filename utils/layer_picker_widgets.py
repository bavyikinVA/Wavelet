"""CustomTkinter-only presentation widgets for the layer explorer.

Tk Canvas is used exclusively as an O(visible rows) scrolling viewport. Cards,
labels, controls and the folder tree are genuine themed CustomTkinter widgets.
"""
from functools import lru_cache
import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageDraw

from utils.layer_catalog import (GROUPS, GROUP_LABELS, algorithm_line,
                                 card_footer, category_color, category_tree,
                                 parameter_line, point_scope, process_line,
                                 result_id, task_id, title)

BG = '#171717'
PANEL = '#1c1c1c'
SURFACE = '#222222'
CARD = '#171717'
ACTIVE = '#1d2d3c'
BLUE = '#438ad0'
BLUE_LIGHT = '#a9d5ff'
TEXT = '#ededed'
MUTED = '#a4a4a4'
FAINT = '#858585'
BORDER = '#323232'
CARD_HEIGHT = 154
ROW_HEIGHT = 166
MAX_POOL = 18


@lru_cache(maxsize=128)
def icon(kind, color='#a5a5a5', size=18):
    """Antialiased monochrome icons, rendered at 4x resolution."""
    ratio = 4
    S = size * ratio
    im = Image.new('RGBA', (S, S), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)
    def xy(*coords):
        return tuple(int(v * ratio) for v in coords)
    stroke = max(2, round(1.7 * ratio))
    if kind == 'folder':
        draw.rounded_rectangle(xy(2, 6, size-2, size-3), radius=3*ratio,
                               outline=color, width=stroke)
        draw.line(xy(3, 6, 7, 3, size-8, 3, size-5, 6), fill=color, width=stroke, joint='curve')
    elif kind == 'file':
        draw.rounded_rectangle(xy(4, 2, size-4, size-2), radius=2*ratio,
                               outline=color, width=stroke)
        draw.line(xy(7, size*.54, size-7, size*.54), fill=color, width=stroke)
    elif kind == 'search':
        draw.ellipse(xy(3, 3, size-7, size-7), outline=color, width=stroke)
        draw.line(xy(size-7, size-7, size-2, size-2), fill=color, width=stroke)
    elif kind == 'right':
        draw.line(xy(size*.39, size*.25, size*.64, size*.5, size*.39, size*.75),
                  fill=color, width=stroke, joint='curve')
    elif kind == 'down':
        draw.line(xy(size*.24, size*.4, size*.5, size*.65, size*.76, size*.4),
                  fill=color, width=stroke, joint='curve')
    elif kind in ('check', 'empty', 'added'):
        draw.rounded_rectangle(xy(2, 2, size-2, size-2), radius=4*ratio,
                               fill=color if kind in ('check', 'added') else None,
                               outline=color, width=stroke)
        if kind in ('check', 'added'):
            draw.line(xy(size*.27, size*.5, size*.44, size*.66, size*.75, size*.34),
                      fill='#ffffff', width=stroke, joint='curve')
    elif kind == 'info':
        draw.ellipse(xy(2, 2, size-2, size-2), outline=color, width=stroke)
        draw.ellipse(xy(size*.47, size*.25, size*.55, size*.33), fill=color)
        draw.line(xy(size*.51, size*.45, size*.51, size*.73), fill=color, width=stroke)
    return ctk.CTkImage(light_image=im, dark_image=im, size=(size, size))


class ExplorerTree(ctk.CTkScrollableFrame):
    """Theme-consistent folder tree with no ttk widgets or native selection boxes."""

    def __init__(self, master, items, on_choose, preferred_task=None, saved_scope=None):
        super().__init__(master, fg_color='transparent', corner_radius=0,
                         scrollbar_button_color='#555555', scrollbar_button_hover_color='#707070')
        self.on_choose = on_choose
        self.data = category_tree(items)
        self.expanded = set()
        if self.data:
            first = preferred_task if preferred_task in self.data else next(iter(self.data))
            self.expanded.add((first, None, None))
            if 'ML-кластеры' in self.data[first]['groups']:
                self.expanded.add((first, 'ML-кластеры', None))
            fallback = ((first, 'ML-кластеры', None)
                        if 'ML-кластеры' in self.data[first]['groups'] else (first, None, None))
            candidate = (tuple(saved_scope) if isinstance(saved_scope, (tuple, list))
                         and len(saved_scope) == 3 else None)
            if candidate is not None:
                t, group, algo = candidate
                groups = self.data.get(t, {}).get('groups', {})
                if group is not None and group not in groups:
                    candidate = None
                elif algo is not None and algo not in groups.get(group, {}).get('algorithms', {}):
                    candidate = None
                elif t not in self.data:
                    candidate = None
            self.selected_scope = candidate or fallback
        else:
            self.selected_scope = None
        if self.selected_scope:
            t, group, _ = self.selected_scope
            self.expanded.add((t, None, None))
            if group:
                self.expanded.add((t, group, None))
        self._entries = []
        self._render()

    def _build_row(self, scope, label, depth, count=None, branch=False, bold=False):
        selected = scope == self.selected_scope
        bg = ACTIVE if selected else PANEL
        line = ctk.CTkFrame(self, fg_color=bg, corner_radius=11, height=37)
        line.pack(fill='x', padx=(8 + 14 * depth, 7), pady=2)
        line.pack_propagate(False)
        is_open = scope in self.expanded
        arrow_kind = 'down' if is_open else 'right'
        if branch:
            arrow = ctk.CTkButton(line, image=icon(arrow_kind, '#8e9bab', 14), text='',
                                  width=23, height=30, fg_color='transparent',
                                  hover_color='#2a3949', corner_radius=7,
                                  command=lambda s=scope: self._toggle(s))
            arrow.pack(side='left', padx=(3, 1))
        else:
            ctk.CTkLabel(line, text='', width=23).pack(side='left', padx=(3, 1))
        icon_kind = 'file' if depth == 2 else 'folder'
        icon_color = BLUE if selected else '#949da7'
        button = ctk.CTkButton(
            line, text=label, image=icon(icon_kind, icon_color, 17), compound='left',
            anchor='w', height=34, fg_color='transparent', hover_color='#2c343e',
            font=ctk.CTkFont(size=12, weight='bold' if bold else 'normal'),
            text_color=TEXT if selected or bold else '#cecece', corner_radius=9,
            command=lambda s=scope: self.choose(s),
        )
        button.pack(side='left', fill='x', expand=True, padx=(0, 2))
        if count is not None:
            count_label = ctk.CTkLabel(line, text=str(count), width=26,
                                       text_color='#858d98', font=ctk.CTkFont(size=10))
            count_label.pack(side='right', padx=(0, 4))
            count_label.bind('<Button-1>', lambda _e, s=scope: self.choose(s))
        self._entries.append(line)

    def _render(self):
        for row in self._entries:
            row.destroy()
        self._entries.clear()
        for task, entry in self.data.items():
            root = (task, None, None)
            total = sum(group['count'] for group in entry['groups'].values())
            self._build_row(root, entry['name'], 0, total, branch=True, bold=True)
            if root not in self.expanded:
                continue
            for group in GROUPS:
                group_entry = entry['groups'].get(group)
                if not group_entry:
                    continue
                scope = (task, group, None)
                self._build_row(scope, GROUP_LABELS.get(group, group), 1,
                                group_entry['count'], branch=group == 'ML-кластеры')
                if group == 'ML-кластеры' and scope in self.expanded:
                    for algorithm, count in sorted(group_entry['algorithms'].items()):
                        self._build_row((task, group, algorithm), algorithm, 2, count)

    def _toggle(self, scope):
        if scope in self.expanded:
            self.expanded.remove(scope)
        else:
            self.expanded.add(scope)
        self._render()

    def choose(self, scope):
        self.selected_scope = scope
        task, group, algorithm = scope
        if algorithm is None and (group is None or group == 'ML-кластеры'):
            self.expanded.add(scope)
        self._render()
        self.on_choose(scope)


class ResultCard(ctk.CTkFrame):
    """Readable, selectable result card with its own consistent CTk widgets."""

    def __init__(self, master, on_select, on_details):
        super().__init__(master, fg_color=CARD, corner_radius=13,
                         border_width=1, border_color='#2a2a2a', height=CARD_HEIGHT)
        self.on_select = on_select
        self.on_details = on_details
        self.item = None
        self._last_rendered = None
        self.grid_columnconfigure(1, weight=1)
        self.grid_propagate(False)

        self.check_icon = ctk.CTkLabel(self, text='', width=28, height=28,
                                      image=icon('empty', '#858d95', 19))
        self.check_icon.grid(row=0, column=0, rowspan=4, padx=(12, 9), pady=(10, 0), sticky='n')
        self.primary = ctk.CTkLabel(self, text='', anchor='w', text_color=TEXT,
                                     font=ctk.CTkFont(size=13, weight='bold'), height=23)
        self.primary.grid(row=0, column=1, sticky='ew', pady=(8, 0))
        self.tag = ctk.CTkLabel(self, text='', width=78, corner_radius=8,
                               fg_color='#263343', text_color='#a9d5ff',
                               font=ctk.CTkFont(size=10), height=22)
        self.tag.grid(row=0, column=2, padx=(8, 12), pady=(10, 0), sticky='e')
        self.algorithm = ctk.CTkLabel(self, text='', anchor='w',
                                     font=ctk.CTkFont(size=12, weight='bold'),
                                     text_color='#c7d8e8', height=20)
        self.algorithm.grid(row=1, column=1, columnspan=2, sticky='ew', padx=(0, 12))
        self.process = ctk.CTkLabel(self, text='', anchor='w',
                                   font=ctk.CTkFont(size=11), text_color='#b7b7b7', height=18)
        self.process.grid(row=2, column=1, columnspan=2, sticky='ew', padx=(0, 12))
        self.params = ctk.CTkLabel(self, text='', anchor='w',
                                  font=ctk.CTkFont(size=10), text_color='#929aa6', height=19)
        self.params.grid(row=3, column=1, columnspan=2, sticky='ew', padx=(0, 12))
        self.footer = ctk.CTkLabel(self, text='', anchor='w',
                                  font=ctk.CTkFont(size=10), text_color='#777f89', height=17)
        self.footer.grid(row=4, column=1, columnspan=2, sticky='ew', padx=(0, 12), pady=(2, 12))
        for widget in (self, self.check_icon, self.primary, self.tag,
                       self.algorithm, self.process, self.params, self.footer):
            widget.bind('<Button-1>', self._click, add='+')
            widget.bind('<Double-Button-1>', self._details, add='+')

    def _click(self, _event):
        if self.item is not None:
            self.on_select(self.item)

    def _details(self, _event):
        if self.item is not None:
            self.on_details(self.item)

    def set_item(self, item, selected=False, added=False):
        self.item = item
        state = (result_id(item), selected, added)
        if state == self._last_rendered:
            return
        self._last_rendered = state
        tint = ACTIVE if selected else '#1f2527' if added else CARD
        self.configure(fg_color=tint, border_width=2 if selected else 1,
                       border_color=BLUE if selected else '#3a434a' if added else '#303030')
        checkbox_kind = 'check' if selected else 'added' if added else 'empty'
        checkbox_color = BLUE if selected else '#527b63' if added else '#777e88'
        self.check_icon.configure(image=icon(checkbox_kind, checkbox_color, 19))
        self.primary.configure(text=title(item))
        alg = item.get('algorithm') if item.get('category') == 'ML-кластеры' else GROUP_LABELS.get(item.get('category'), item.get('category', ''))
        self.tag.configure(text='Добавлен' if added else alg or 'Результат',
                           fg_color='#2b4739' if added else '#243443' if selected else '#282b31',
                           text_color='#b2e0c5' if added else category_color(item))
        self.algorithm.configure(text=algorithm_line(item))
        self.process.configure(text=process_line(item))
        self.params.configure(text=parameter_line(item))
        self.footer.configure(text=card_footer(item))


class VirtualResultList(ctk.CTkFrame):
    """Virtual list of real CTk cards; only the visible pool is ever created."""

    def __init__(self, master, on_select, on_details):
        super().__init__(master, fg_color='transparent', corner_radius=0)
        self.on_select = on_select
        self.on_details = on_details
        self.results = []
        self.selected = set()
        self.added = set()
        self._slots = []
        self._windows = []
        self.canvas = tk.Canvas(self, bg=PANEL, highlightthickness=0, bd=0,
                                height=360, takefocus=1, yscrollincrement=14)
        self.canvas.pack(side='left', fill='both', expand=True, padx=(2, 0))
        self.scroll = ctk.CTkScrollbar(self, orientation='vertical', command=self._scroll_to,
                                      button_color='#515151', button_hover_color='#747474')
        self.scroll.pack(side='right', fill='y', padx=(3, 3))
        self.canvas.configure(yscrollcommand=self.scroll.set)
        self.empty = ctk.CTkLabel(self, text='В этой категории ничего не найдено\nПопробуйте изменить запрос',
                                  text_color=FAINT, font=ctk.CTkFont(size=12),
                                  justify='center')
        self.canvas.bind('<Configure>', lambda _e: self.redraw())
        self.canvas.bind('<MouseWheel>', self._wheel)
        self.canvas.bind('<Button-4>', lambda _e: self._wheel_steps(-3))
        self.canvas.bind('<Button-5>', lambda _e: self._wheel_steps(3))
        self.canvas.bind('<Key-Up>', lambda _e: self._wheel_steps(-3))
        self.canvas.bind('<Key-Down>', lambda _e: self._wheel_steps(3))
        self.canvas.bind('<Prior>', lambda _e: self._wheel_steps(-20))
        self.canvas.bind('<Next>', lambda _e: self._wheel_steps(20))
        for _ in range(MAX_POOL):
            card = ResultCard(self.canvas, self.on_select, self.on_details)
            win = self.canvas.create_window(7, 0, anchor='nw', window=card,
                                            height=self._scaled_card_height(card), width=360, state='hidden')
            self._slots.append(card)
            self._windows.append(win)
            self._bind_card_wheel(card)

    def _bind_card_wheel(self, card):
        for child in [card, *card.winfo_children()]:
            child.bind('<MouseWheel>', self._wheel, add='+')
            child.bind('<Button-4>', lambda _e: self._wheel_steps(-3), add='+')
            child.bind('<Button-5>', lambda _e: self._wheel_steps(3), add='+')

    @staticmethod
    def _scaled_card_height(card):
        # Canvas is plain Tk and does NOT scale its create_window dimensions.
        # Match CustomTkinter's DPI-scaled card geometry to prevent clipping.
        scaling = float(getattr(card, '_widget_scaling', 1.0))
        return max(1, round(CARD_HEIGHT * scaling))

    def _row_height(self):
        card = self._slots[0] if self._slots else None
        scaling = float(getattr(card, '_widget_scaling', 1.0)) if card else 1.0
        return max(1, round(ROW_HEIGHT * scaling))

    def set_results(self, results, selected, added=()):
        self.results = list(results)
        self.selected = set(selected)
        self.added = set(added)
        if self.results:
            self.empty.place_forget()
        else:
            self.empty.place(relx=0.49, rely=0.5, anchor='center')
        self.canvas.yview_moveto(0)
        self.redraw()

    def set_selected(self, selected):
        self.selected = set(selected)
        self.redraw()

    def _scroll_to(self, *args):
        self.canvas.yview(*args)
        self.redraw()

    def _wheel_steps(self, steps):
        self.canvas.yview_scroll(int(steps), 'units')
        self.redraw()
        return 'break'

    def _wheel(self, event):
        amount = max(1, abs(event.delta) // 120)
        return self._wheel_steps(-amount * 3 if event.delta > 0 else amount * 3)

    def redraw(self):
        width = max(280, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        row_height = self._row_height()
        self.canvas.configure(scrollregion=(0, 0, width, max(height, len(self.results) * row_height)))
        first = int(self.canvas.canvasy(0) // row_height)
        visible = min(MAX_POOL, max(1, (height + row_height * 2) // row_height))
        for slot, (card, win) in enumerate(zip(self._slots, self._windows)):
            idx = first + slot
            if idx >= len(self.results) or slot >= visible:
                self.canvas.itemconfigure(win, state='hidden')
                card.item = None
                continue
            item = self.results[idx]
            identity = result_id(item)
            card.set_item(item, identity in self.selected, identity in self.added)
            self.canvas.coords(win, 7, idx * row_height + 3)
            self.canvas.itemconfigure(win, state='normal', width=max(260, width - 22),
                                      height=self._scaled_card_height(card))
