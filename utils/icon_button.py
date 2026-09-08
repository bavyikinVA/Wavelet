"""Small monochrome layout controls with hover hints."""
import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageDraw
from utils.theme import AppTheme


class IconButton(ctk.CTkButton):
    def __init__(self, parent, icon, hint, command):
        image = Image.new('RGBA', (48, 48))
        draw = ImageDraw.Draw(image)
        color = '#c5c5c5'
        if icon in ('sidebar', 'bottom', 'focus'):
            draw.rounded_rectangle((8, 9, 40, 39), radius=4, outline=color, width=3)
            if icon == 'sidebar':
                draw.line((19, 10, 19, 38), fill=color, width=3)
            elif icon == 'bottom':
                draw.line((9, 29, 39, 29), fill=color, width=3)
            else:
                draw.rectangle((17, 17, 31, 31), outline=color, width=2)
        elif icon == 'refresh':
            draw.arc((10, 10, 38, 38), 35, 320, fill=color, width=3)
            draw.line((30, 9, 39, 14, 32, 21), fill=color, width=3)
        else:
            for y, x in ((14, 18), (25, 31), (36, 21)):
                draw.line((8, y, 40, y), fill=color, width=2)
                draw.ellipse((x-4, y-4, x+4, y+4), fill='#292929', outline=color, width=2)
        self._icon_image = ctk.CTkImage(image, image, size=(22, 22))
        super().__init__(parent, text='', image=self._icon_image, width=32, height=30,
                         corner_radius=7, fg_color='transparent', hover_color=AppTheme.NAV_HOVER,
                         command=command)
        self.hint = hint
        self._tip = None
        self._tip_id = None
        self.bind('<Enter>', self._schedule_tip, add='+')
        self.bind('<Leave>', self._hide_tip, add='+')
        self.bind('<Button-1>', self._hide_tip, add='+')

    def _schedule_tip(self, event=None):
        self._hide_tip()
        self._tip_id = self.after(500, self._show_tip)

    def _show_tip(self):
        self._tip_id = None
        self._tip = tk.Toplevel(self)
        self._tip.overrideredirect(True)
        self._tip.geometry(f'+{self.winfo_rootx()}+{self.winfo_rooty()+self.winfo_height()+5}')
        tk.Label(self._tip, text=self.hint, background='#303030', foreground='white', padx=8, pady=5).pack()

    def _hide_tip(self, event=None):
        if self._tip_id:
            self.after_cancel(self._tip_id)
            self._tip_id = None
        if self._tip:
            self._tip.destroy()
            self._tip = None

    def destroy(self):
        self._hide_tip()
        super().destroy()
