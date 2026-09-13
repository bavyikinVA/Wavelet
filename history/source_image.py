"""Keep the effective input (including crop) inside each run directory."""
import os

import numpy as np
from PIL import Image

from Gram_Shmidt import change_channels


def save_run_image(task):
    if task.original_image is None:
        raise ValueError("Нет изображения для сохранения в каталог запуска")
    if not task.task_folder_path:
        raise ValueError("Каталог запуска не создан")
    path = os.path.join(task.task_folder_path, "image.png")
    # original_image is the RGB input returned by the cropper, before channel
    # transformation. Saving transformed floats as RGB would lose information.
    Image.fromarray(task.original_image).save(path, format="PNG")
    with Image.open(path) as saved:
        image = np.array(saved.convert("RGB"))
    channels = [image[:, :, index].copy() for index in range(3)]
    data = (change_channels(task.color1, task.color2, channels)
            if task.gram_schmidt_applied else channels)
    # Bind the source only after both saving and preparation succeed.
    task.original_image = image
    task.data_copy = [channel.copy() for channel in channels]
    task.data = data
    task.image_path = path
    return path
