"""Bounded dropdowns for CustomTkinter selectors, with native list scrolling."""
import tkinter as tk
import customtkinter as ctk


class _ScrollableDropdown:
    def _open_dropdown_menu(self):
        if self.cget('state') == 'disabled':
            return
        self._close_popup()
        popup = self._popup = tk.Toplevel(self)
        popup.withdraw()
        popup.overrideredirect(True)
        popup.transient(self.winfo_toplevel())
        values = list(self.cget('values'))
        rows = max(1, min(10, len(values)))
        box = self._popup_list = tk.Listbox(
            popup, height=rows, exportselection=False, activestyle='dotbox',
            background='#242424', foreground='#f2f2f2',
            selectbackground='#404040', selectforeground='white',
            font=self._apply_font_scaling(self.cget('dropdown_font')),
            borderwidth=0, highlightthickness=1)
        scrollbar = tk.Scrollbar(popup, orient='vertical', command=box.yview)
        box.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        box.pack(fill='both', expand=True)
        for value in values:
            box.insert('end', value)
        if self.get() in values:
            index = values.index(self.get())
            box.selection_set(index)
            box.activate(index)
            box.see(index)
        def choose(event=None):
            selected = box.curselection()
            value = box.get(selected[0]) if selected else None
            self._close_popup()
            if value is not None:
                self._dropdown_callback(value)
            return 'break'
        box.bind('<ButtonRelease-1>', choose)
        box.bind('<Return>', choose)
        popup.bind('<Escape>', lambda event: self._close_popup())
        def check_focus():
            if getattr(self, '_popup', None) is popup:
                focused = popup.focus_get()
                if focused is None or not str(focused).startswith(str(popup) + '.'):
                    self._close_popup()
        popup.bind('<FocusOut>', lambda event: popup.after_idle(check_focus))
        def outside(event):
            if not (popup.winfo_rootx() <= event.x_root < popup.winfo_rootx()+popup.winfo_width()
                    and popup.winfo_rooty() <= event.y_root < popup.winfo_rooty()+popup.winfo_height()):
                self._close_popup()
                return 'break'
        popup.bind('<ButtonPress-1>', outside)
        popup.update_idletasks()
        width = min(max(self.winfo_width(), 240), self.winfo_screenwidth())
        height = min(popup.winfo_reqheight(), self.winfo_screenheight() // 2)
        x = max(0, min(self.winfo_rootx(), self.winfo_screenwidth()-width))
        y = self.winfo_rooty()+self.winfo_height()
        if y+height > self.winfo_screenheight():
            y = max(0, self.winfo_rooty()-height)
        popup.geometry(f'{width}x{height}+{x}+{y}')
        popup.deiconify()
        popup.grab_set()
        box.focus_set()

    def _close_popup(self):
        popup = getattr(self, '_popup', None)
        if popup is not None:
            popup.grab_release()
            popup.destroy()
            self._popup = None

    def destroy(self):
        self._close_popup()
        super().destroy()


class ScrollableComboBox(_ScrollableDropdown, ctk.CTkComboBox):
    pass


class ScrollableOptionMenu(_ScrollableDropdown, ctk.CTkOptionMenu):
    pass
