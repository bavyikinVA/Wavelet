import ast
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock
from uuid import uuid4
import time

import numpy as np
from compute.processing_task import ProcessingTask
from compute.validation import validate_scales
from history.source_image import save_run_image


def method(class_name, name, namespace):
    tree = ast.parse((Path(__file__).parents[1] / 'main.py').read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'main.py', 'exec'), namespace)
    return namespace[name]


class IsolationTests(unittest.TestCase):
    def test_loading_new_source_detaches_results_and_keeps_old_png(self):
        with tempfile.TemporaryDirectory() as directory:
            task = ProcessingTask()
            task.original_image = np.zeros((3, 4, 3), dtype=np.uint8)
            task.task_folder_path = directory
            old = save_run_image(task)
            original_bytes = Path(old).read_bytes()
            task.knn_results = {'old': 1}
            task.result = {0: 1}
            task.ml_result = {'old': 1}
            task.color1 = np.array([1, 2, 3])
            import cv2
            new_path = str(Path(directory) / 'new.png')
            from PIL import Image
            Image.fromarray(np.full((3, 4, 3), 200, dtype=np.uint8)).save(new_path)
            load = method('ImageProcessor', 'load_image_for_task',
                          {'np': np, 'cv2': cv2, 'run_cropper': lambda master: new_path})
            self.assertTrue(load(SimpleNamespace(progress=Mock()), task))
            self.assertEqual(task.knn_results, {})
            self.assertEqual(task.result, {})
            self.assertIsNone(task.ml_result)
            self.assertIsNone(task.color1)
            self.assertEqual(task.task_folder_path, '')
            self.assertEqual(Path(old).read_bytes(), original_bytes)

    def test_repeated_ml_uses_new_folder_and_preserves_features(self):
        with tempfile.TemporaryDirectory() as directory:
            task = ProcessingTask()
            task.task_name = 'Experiment'
            task.task_folder_path = directory
            task.knn_results = {'features': [1, 2]}
            create = method('ImageProcessor', 'create_task_folder',
                            {'os': os, 'time': time, 'uuid4': uuid4})
            processor = SimpleNamespace(root_folder_path=directory, progress=Mock())
            processor.create_task_folder = lambda task: create(processor, task)
            start = method('App', '_start_ml_clustering',
                           {'mb': Mock(), 'threading': Mock(), 'AppTheme': SimpleNamespace(TEXT_SECONDARY='gray')})
            app = SimpleNamespace(current_task=task, _ml_thread=None, image_processor=processor,
                                  _store_ml_settings=Mock(), ml_run_button=Mock(), ml_result_label=Mock(),
                                  progress_manager=Mock(), _ml_clustering_worker=Mock())
            start(app)
            first = task.task_folder_path
            app._ml_thread = None
            start(app)
            self.assertNotEqual(first, task.task_folder_path)
            self.assertNotEqual(directory, first)
            self.assertEqual(task.knn_results, {'features': [1, 2]})

    def test_bad_scale_file_preserves_previous_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scales.txt'
            path.write_text('2 0 nan')
            task = ProcessingTask()
            task.scales = np.array([2., 4.])
            task.num_scale = 2
            load = method('ImageProcessor', 'load_scales_from_file_for_task',
                          {'np': np, 'validate_scales': validate_scales})
            with self.assertRaises(ValueError):
                load(SimpleNamespace(progress=Mock()), task, path)
            np.testing.assert_array_equal(task.scales, [2., 4.])
            self.assertEqual(task.num_scale, 2)
