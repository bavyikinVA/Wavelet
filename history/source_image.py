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
    from history.atomic_io import atomic_binary_writer
    with atomic_binary_writer(path) as stream:
        Image.fromarray(task.original_image).save(stream, format="PNG")
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


def release_task_source(task):
    """Drop large runtime buffers after their durable input snapshot exists.

    The input path and scientific settings remain available for lazy restoration.
    This is intentionally only called by the disk-backed batch lifecycle.
    """
    if task.original_image is not None:
        from history.pipeline_resume import cwt_signature_payload
        cwt_signature_payload(task)  # capture decoded-pixel fingerprint before unloading
    task.original_image = None
    task.data = []
    task.data_copy = []
    from compute.channel_analysis import clear_prepared_channels
    clear_prepared_channels(task)
    task.result = {}
    task.knn_results = {}
    task.statistics_results = {}
    task.synchronization_results = {}


def ensure_task_source(task, row=None):
    """Load one validated effective RGB image, never the mutable source path.

    Hash verification is on the compressed snapshot bytes; the decoded RGB is
    recreated with exactly the same data-copy contract as the eager resume.
    Do not fall back to ``task.image_path`` if a snapshot is missing/corrupt.
    """
    if task.original_image is not None:
        return
    from pathlib import Path
    import hashlib
    from PIL import Image
    from Gram_Shmidt import change_channels

    path = (row or {}).get('input_path') or getattr(task, '_input_snapshot_path', None)
    expected_hash = (row or {}).get('input_sha256') or getattr(task, '_input_snapshot_sha256', None)
    if not path or not expected_hash:
        raise ValueError('Для задачи отсутствует сохранённый снимок входа')
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Снимок входного изображения отсутствует: {path}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != expected_hash:
        raise ValueError(f'Повреждён снимок входного изображения: {path}')
    with Image.open(path) as saved:
        image = np.array(saved.convert('RGB'))
    channels = [image[:, :, index].copy() for index in range(3)]
    data = (change_channels(task.color1, task.color2, channels)
            if task.gram_schmidt_applied else channels)
    # Publish only after validation and decoding: partial loads cannot escape.
    task.original_image = image
    task.data_copy = [channel.copy() for channel in channels]
    task.data = data
    task._input_snapshot_path = str(path)
    task._input_snapshot_sha256 = expected_hash
    # Do not replace task.image_path: it can point to the saved attempt image.
    if getattr(task, '_restore_cwt_signature', False):
        from history.pipeline_resume import cwt_signature
        task.last_completed_cwt_signature = cwt_signature(task)
        task._restore_cwt_signature = False


def stage_task_input(task, experiment_root):
    """Durably stage an effective cropped RGB input before it joins a big queue.

    Unlike batch snapshots, this is an editable per-task source; every batch
    makes its own immutable snapshot inside .runs. Only one staged input needs
    to be decoded when an inactive task is reactivated.
    """
    from pathlib import Path
    from uuid import uuid4
    from history.atomic_io import atomic_binary_writer
    from history.attempts import digest

    if task.original_image is None:
        raise ValueError('Отсутствует изображение для сохранения')
    if not experiment_root:
        raise ValueError('Не выбран каталог исследования')
    if not getattr(task, 'persistent_id', None):
        task.persistent_id = uuid4().hex
    destination = (Path(experiment_root) / '.inputs' /
                   (str(task.persistent_id) + '.png')).resolve()
    from compute.disk_budget import check_snapshot_capacity
    check_snapshot_capacity(task, destination.parent)
    with atomic_binary_writer(destination) as stream:
        Image.fromarray(task.original_image).save(stream, format='PNG')
    checksum = digest(destination)
    # Publish the pointers only after the PNG is complete and checksummed.
    task._input_snapshot_path = str(destination)
    task._input_snapshot_sha256 = checksum
    # Store the decoded-pixel fingerprint in settings alongside the PNG hash.
    # The two hashes are not interchangeable: PNG hashes compressed bytes,
    # while CWT provenance hashes the exact RGB array used for computation.
    from history.pipeline_resume import cwt_signature_payload
    cwt_signature_payload(task)
    return destination


def attach_workspace_input(task, experiment_root, *, defer_verify=False):
    """Attach a prior task without eagerly retaining its image or KNN arrays.

    Modern tasks reuse the immutable checksum recorded in the SQLite settings.
    For legacy studies without staging, decode the original once and stage it.
    """
    from pathlib import Path
    from history.attempts import digest

    root = Path(experiment_root)
    if not getattr(task, 'persistent_id', None):
        raise ValueError('У задачи отсутствует постоянный идентификатор')
    staged = (root / '.inputs' / (str(task.persistent_id) + '.png')).resolve()
    if staged.is_file():
        saved_hash = getattr(task, '_input_snapshot_sha256', None)
        # Fast research opening: do not hash every compressed PNG in the
        # workspace. ensure_task_source() verifies the saved hash before use.
        if defer_verify and saved_hash:
            task._input_snapshot_path = str(staged)
            return staged
        actual_hash = digest(staged)
        if saved_hash and saved_hash != actual_hash:
            raise ValueError(f'Повреждён сохранённый источник задачи: {staged}')
        task._input_snapshot_path = str(staged)
        task._input_snapshot_sha256 = actual_hash
        return staged
    original = Path(getattr(task, 'image_path', '') or '')
    if not original.is_file():
        raise FileNotFoundError(f'Не найден исходный файл задачи: {original}')
    with Image.open(original) as source:
        task.original_image = np.asarray(source.convert('RGB'))
    try:
        return stage_task_input(task, experiment_root)
    finally:
        release_task_source(task)
