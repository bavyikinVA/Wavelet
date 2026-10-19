"""Dark, bounded-size ML report presented above its owner on completion."""
import customtkinter as ctk

from ml.run_report import STATUS, summary
from utils.theme import AppTheme
from utils.tk_lifecycle import install_callback_cleanup


class MLRunReport(ctk.CTkToplevel):
    # CTk's Windows titlebar implementation withdraws and restores the window
    # asynchronously; use the same native dark-titlebar helper as the main app.
    _deactivate_windows_window_header_manipulation = True
    PAGE_SIZE = 40
    BACKGROUND = '#242424'
    SURFACE = '#2b2b2b'
    TEXT = '#f2f2f2'

    def __init__(self, master, rows, on_results, on_continue):
        super().__init__(master, fg_color=self.BACKGROUND)
        self.withdraw()
        install_callback_cleanup(self)
        self.rows = rows
        self.page = 0
        self._row_widgets = {}
        self.title('Результаты ML-плана')
        scale = self._get_window_scaling()
        width = min(1120, int((self.winfo_screenwidth()-80) / scale))
        height = min(740, int((self.winfo_screenheight()-100) / scale))
        self.geometry(f'{width}x{height}')
        self.minsize(min(760, width), min(520, height))
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)
        ctk.CTkLabel(self, text='Результаты ML-плана', text_color=self.TEXT,
            font=ctk.CTkFont(size=23, weight='bold')).grid(row=0, column=0, sticky='w', padx=24, pady=(22, 5))
        self.summary_label = ctk.CTkLabel(self, text=summary(rows), text_color=AppTheme.TEXT_SECONDARY,
            font=ctk.CTkFont(size=13), anchor='w', justify='left', wraplength=width-48)
        self.summary_label.grid(row=1, column=0, sticky='ew', padx=24, pady=(0, 18))
        header = ctk.CTkFrame(self, fg_color=self.SURFACE, corner_radius=8)
        header.grid(row=2, column=0, sticky='ew', padx=(20, 40))
        self._columns(header)
        for column, text in enumerate(('Алгоритм / задача', 'Набор данных / параметры', 'Статус')):
            ctk.CTkLabel(header, text=text, text_color=AppTheme.TEXT_SECONDARY,
                font=ctk.CTkFont(size=12, weight='bold'), anchor='w').grid(
                    row=0, column=column, sticky='ew', padx=14, pady=10)
        self.table = ctk.CTkScrollableFrame(self, fg_color=self.BACKGROUND, corner_radius=0,
            scrollbar_button_color='#505050', scrollbar_button_hover_color='#707070')
        self.table.grid(row=3, column=0, sticky='nsew', padx=(20, 24), pady=(4, 0))
        self.table.grid_columnconfigure(0, weight=1)
        pager = ctk.CTkFrame(self, fg_color='transparent')
        pager.grid(row=4, column=0, sticky='ew', padx=24, pady=(8, 10))
        self.page_label = ctk.CTkLabel(pager, text='', text_color=AppTheme.TEXT_SECONDARY,
                                      font=ctk.CTkFont(size=12))
        self.page_label.pack(side='left')
        self.next_button = ctk.CTkButton(pager, text='Далее →', width=95, height=30,
            fg_color=AppTheme.NAV_HOVER, hover_color='#3b3b3b', command=lambda: self.show_page(self.page+1))
        self.next_button.pack(side='right')
        self.prev_button = ctk.CTkButton(pager, text='← Назад', width=95, height=30,
            fg_color=AppTheme.NAV_HOVER, hover_color='#3b3b3b', command=lambda: self.show_page(self.page-1))
        self.prev_button.pack(side='right', padx=8)
        actions = ctk.CTkFrame(self, fg_color='transparent')
        actions.grid(row=5, column=0, sticky='ew', padx=24, pady=(0, 22))
        def finish(callback):
            self.destroy()
            callback()
        self.results_button = ctk.CTkButton(actions, text='Результаты', height=38,
            font=ctk.CTkFont(size=14), fg_color=AppTheme.PRIMARY, hover_color=AppTheme.PRIMARY_HOVER,
            command=lambda: finish(on_results))
        self.results_button.pack(side='right', padx=(12, 0))
        self.continue_button = ctk.CTkButton(actions, text='Продолжить', height=38,
            font=ctk.CTkFont(size=14), fg_color='#353535', hover_color='#454545',
            border_width=1, border_color=AppTheme.BORDER, command=lambda: finish(on_continue))
        self.continue_button.pack(side='right')
        self.protocol('WM_DELETE_WINDOW', lambda: finish(on_continue))
        self.show_page(0)
        self.bind('<Configure>', self._resize, add='+')
        self.after_idle(self._present)

    def _columns(self, frame):
        scale = frame._get_widget_scaling()
        frame.grid_columnconfigure(0, minsize=int(170 * scale), weight=0)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(2, minsize=int(170 * scale), weight=0)

    def _present(self):
        owner = self.master.winfo_toplevel()
        if owner.state() == 'iconic':
            owner.deiconify()
        if owner.state() != 'withdrawn':
            self.transient(owner)
        self.update_idletasks()
        x = max(0, min(owner.winfo_rootx() + (owner.winfo_width()-self.winfo_width())//2,
                       self.winfo_screenwidth()-self.winfo_width()))
        y = max(0, min(owner.winfo_rooty() + (owner.winfo_height()-self.winfo_height())//2,
                       self.winfo_screenheight()-self.winfo_height()-45))
        self.geometry(f'+{x}+{y}')
        self.deiconify()
        from utils.rendering import enable_dark_titlebar
        enable_dark_titlebar(self)
        self.lift()
        self.attributes('-topmost', True)
        self.focus_force()
        self.grab_set()
        # Only the initial presentation is topmost; keep normal owner stacking afterwards.
        self.after(250, lambda: self.attributes('-topmost', False))

    def _resize(self, event):
        if event.widget is not self:
            return
        width = int(event.width / self._get_window_scaling())
        self.summary_label.configure(wraplength=max(300, width-48))
        for _, labels in self._row_widgets.values():
            for label in labels:
                label.configure(wraplength=max(180, width-420))

    def show_page(self, page):
        last = max(0, (len(self.rows)-1)//self.PAGE_SIZE)
        self.page = max(0, min(page, last))
        for widget in self.table.winfo_children():
            widget.destroy()
        self._row_widgets.clear()
        start = self.page*self.PAGE_SIZE
        stop = min(start+self.PAGE_SIZE, len(self.rows))
        for index in range(start, stop):
            row = self.rows[index]
            frame = ctk.CTkFrame(self.table, fg_color=self.SURFACE, corner_radius=8,
                                border_width=1, border_color=self.SURFACE)
            frame.grid(row=index-start, column=0, sticky='ew', pady=(0, 5))
            self._columns(frame)
            left = ctk.CTkFrame(frame, fg_color='transparent')
            left.grid(row=0, column=0, sticky='nw', padx=14, pady=12)
            ctk.CTkLabel(left, text=row['algorithm'], text_color=self.TEXT,
                font=ctk.CTkFont(size=14, weight='bold'), anchor='w').pack(anchor='w')
            ctk.CTkLabel(left, text=row['task'], text_color=AppTheme.TEXT_SECONDARY,
                font=ctk.CTkFont(size=12), anchor='w').pack(anchor='w')
            middle = ctk.CTkFrame(frame, fg_color='transparent')
            middle.grid(row=0, column=1, sticky='ew', padx=14, pady=12)
            dataset = ctk.CTkLabel(middle, text=row['dataset'], text_color=self.TEXT,
                font=ctk.CTkFont(size=14), anchor='w', justify='left', wraplength=550)
            dataset.pack(fill='x')
            config = row['configuration']
            short = (f"k = {config.get('n_clusters', 5)} · seed = {config.get('random_state', 42)}"
                     if config['algorithm'] == 'kmeans' else
                     f"ε = {config.get('eps', .8)} · min_samples = {config.get('min_samples', 5)}")
            parameters = ctk.CTkLabel(middle, text=short, text_color=AppTheme.TEXT_SECONDARY,
                font=ctk.CTkFont(size=12), anchor='w', justify='left', wraplength=550)
            parameters.pack(fill='x')
            status = row['status']
            bg, fg = (('#1e3c2d', '#a0e2b5') if status in ('completed', 'reused') else
                      ('#482b2e', '#ffb3b8') if status == 'failed' else ('#3d3729', '#e6cd8b'))
            badge = ctk.CTkLabel(frame, text=STATUS[status], width=142, height=30,
                fg_color=bg, text_color=fg, corner_radius=6, wraplength=140,
                font=ctk.CTkFont(size=12, weight='bold'))
            badge.grid(row=0, column=2, padx=14, pady=12, sticky='e')
            self._row_widgets[index] = (frame, (dataset, parameters))
        self.page_label.configure(text=f'{start+1 if self.rows else 0}–{stop} из {len(self.rows)} операций')
        self.prev_button.configure(state='normal' if self.page else 'disabled')
        self.next_button.configure(state='normal' if self.page < last else 'disabled')
        if last == 0:
            self.prev_button.pack_forget()
            self.next_button.pack_forget()
        self.table._parent_canvas.yview_moveto(0)

    def destroy(self):
        if self.grab_current() is self:
            self.grab_release()
        super().destroy()
