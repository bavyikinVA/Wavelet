import os
import tempfile
import unittest

import numpy as np
from PIL import Image

from compute.processing_task import ProcessingTask
from history import RunHistoryStore
from history.source_image import save_run_image
from Gram_Shmidt import change_channels


class RunImageTests(unittest.TestCase):
    def test_effective_crop_is_saved_and_referenced_in_history(self):
        with tempfile.TemporaryDirectory() as directory:
            task = ProcessingTask()
            task.task_folder_path = directory
            task.image_path = os.path.join(directory, 'missing-original.jpg')
            crop = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
            task.original_image = crop.copy()
            path = save_run_image(task)
            with Image.open(path) as image:
                self.assertEqual(image.format, 'PNG')
                np.testing.assert_array_equal(np.asarray(image), crop)
            np.testing.assert_array_equal(np.stack(task.data, axis=-1), crop)
            store = RunHistoryStore(os.path.join(directory, 'history.sqlite3'))
            run = store.add_run(task=task, status='completed', duration_seconds=1,
                                settings=task.settings_snapshot())
            self.assertEqual(store.get_run(run)['image_path'], path)
            self.assertEqual(store.get_settings(run)['image_path'], path)

    def test_new_run_gets_own_copy_without_reapplying_transform_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            task = ProcessingTask()
            task.original_image = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
            original = task.original_image.copy()
            task.color1 = np.array([1., 1., 0.])
            task.color2 = np.array([0., 1., 0.])
            task.gram_schmidt_applied = True
            expected = change_channels(task.color1, task.color2,
                                       [original[:, :, i] for i in range(3)])
            paths = []
            for name in ('run1', 'run2'):
                task.task_folder_path = os.path.join(directory, name)
                os.mkdir(task.task_folder_path)
                paths.append(save_run_image(task))
                np.testing.assert_allclose(task.data, expected)
                np.testing.assert_array_equal(task.original_image, original)
            self.assertNotEqual(*paths)
            self.assertTrue(all(os.path.isfile(path) for path in paths))


if __name__ == '__main__':
    unittest.main()
