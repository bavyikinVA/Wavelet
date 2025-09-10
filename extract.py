import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog


class ImageCropper:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Cropper")

        # Переменные состояния
        self.drawing = False
        self.points = []
        self.img = None
        self.img_copy = None
        self.output_path = "cropped_result.png"

        # Создание интерфейса
        self.create_widgets()

    def create_widgets(self):
        # Фрейм для кнопок
        self.button_frame = Frame(self.root)
        self.button_frame.pack(fill=X, pady=5)

        # Кнопки
        self.btn_open = Button(self.button_frame, text="Open Image", command=self.open_image)
        self.btn_open.pack(side=LEFT, padx=5)

        self.btn_crop = Button(self.button_frame, text="Crop Image", command=self.crop_image)
        self.btn_crop.pack(side=LEFT, padx=5)
        self.btn_crop.config(state=DISABLED)

        self.btn_reset = Button(self.button_frame, text="Reset", command=self.reset_contour)
        self.btn_reset.pack(side=LEFT, padx=5)
        self.btn_reset.config(state=DISABLED)

        self.btn_exit = Button(self.button_frame, text="Exit", command=self.exit_app)
        self.btn_exit.pack(side=RIGHT, padx=5)

        # Метка для инструкций
        self.lbl_instructions = Label(self.root,
                                      text="1. Откройте изображение\n2. Нарисуйте контур мышью\n3. Нажмите 'Crop Image'")
        self.lbl_instructions.pack()

    def open_image(self):
        file_types = [("Image files", "*.png *.jpg *.jpeg *.bmp")]
        file_path = filedialog.askopenfilename(filetypes=file_types)

        if not file_path:
            return

        try:
            self.img = cv2.imread(file_path)
            if self.img is None:
                raise Exception("Не удалось загрузить изображение")

            self.img_copy = self.img.copy()
            cv2.namedWindow("Image Cropper")
            cv2.setMouseCallback("Image Cropper", self.mouse_callback)

            # Активируем кнопки
            self.btn_crop.config(state=NORMAL)
            self.btn_reset.config(state=NORMAL)

            # Показываем изображение
            cv2.imshow("Image Cropper", self.img_copy)

        except Exception as e:
            print(f"Error: {e}")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.points = [(x, y)]
            self.img_copy = self.img.copy()
            cv2.circle(self.img_copy, (x, y), 3, (0, 255, 0), -1)
            cv2.imshow("Image Cropper", self.img_copy)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.points.append((x, y))
                if len(self.points) > 1:
                    self.img_copy = self.img.copy()
                    cv2.polylines(self.img_copy, [np.array(self.points)], False, (0, 255, 0), 2)
                    cv2.imshow("Image Cropper", self.img_copy)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if len(self.points) > 2:
                self.img_copy = self.img.copy()
                cv2.polylines(self.img_copy, [np.array(self.points)], True, (0, 255, 0), 2)
                cv2.imshow("Image Cropper", self.img_copy)

    def crop_image(self):
        if len(self.points) > 2:
            # Создаем маску
            mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
            pts = np.array([self.points], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)

            # Создаем белый фон
            white_bg = np.ones_like(self.img) * 255

            # Копируем только выбранную область
            result = white_bg.copy()
            result[mask == 255] = self.img[mask == 255]

            # Сохраняем результат
            cv2.imwrite(self.output_path, result)

            # Показываем результат
            cv2.imshow("Cropped Result", result)
            print(f"Изображение сохранено как '{self.output_path}'")

    def reset_contour(self):
        self.points.clear()
        if self.img is not None:
            self.img_copy = self.img.copy()
            cv2.imshow("Image Cropper", self.img_copy)
        print("Контур сброшен")

    def exit_app(self):
        cv2.destroyAllWindows()
        self.root.quit()


if __name__ == "__main__":
    root = Tk()
    app = ImageCropper(root)
    root.mainloop()