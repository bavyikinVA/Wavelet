import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk


def pipette(image_path):
    def get_pixel_rgb(event):
        nonlocal click_count, colors

        x, y = event.x, event.y
        r, g, b = original_img.convert('RGB').getpixel((x, y))

        click_count += 1
        colors.append((r, g, b))

        if click_count <= 2:
            show_color_square(r, g, b)

        if click_count == 2:
            root.unbind("<Button-1>")
            save_colors()
            show_message()
            root.destroy()

    def save_colors():
        with open("color_tone.txt", 'w') as f:
            for i, color in enumerate(colors, start=1):
                f.write(f'{color[0]} {color[1]} {color[2]}\n')
            

    def show_message():
        message = f'Выбранные цвета:\n'
        for i, color in enumerate(colors, start=1):
            message += f'{i} цвет: {color[0]} {color[1]} {color[2]}\n'
        message += f'Цвета сохранены в файл color_tone.txt.'
        messagebox.showinfo("Цвета", message, icon="info")

    def show_color_square(r, g, b):
        color_square_root = tk.Toplevel(root)
        color_square_root.title("Цвет пикселя")
        color_square_root.geometry("100x100")

        color_square = tk.Canvas(color_square_root, width=100, height=100)
        color_square.create_rectangle(0, 0, 100, 100, fill=f'#{r:02x}{g:02x}{b:02x}')
        color_square.pack()

    root = tk.Tk()
    root.title("Pipette Tool")

    original_img = Image.open(image_path)
    canvas = tk.Canvas(root, width=original_img.width, height=original_img.height)
    canvas.pack()
    img = ImageTk.PhotoImage(file=image_path, master=root)
    canvas.create_image(0, 0, anchor="nw", image=img)
    color_label = tk.Label(root, text="Нажмите на экран, чтобы получить цвета пикселей."
                                      "\nМаксимум можно выбрать 2 цвета!")
    color_label.pack()

    click_count = 0
    colors = []

    root.bind("<Button-1>", get_pixel_rgb)

    root.mainloop(1)
