import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location('gram_test', ROOT / 'Gram_Shmidt.py')
GRAM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GRAM)


def load_method(class_name, method_name, namespace):
    tree = ast.parse((ROOT / 'main.py').read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    exec(compile(ast.Module(body=[method], type_ignores=[]), 'main.py', 'exec'), namespace)
    return namespace[method_name]


class GramSchmidtTests(unittest.TestCase):
    def setUp(self):
        self.data = np.arange(18., dtype=float).reshape(3, 2, 3)

    def test_rejects_zero_collinear_nearly_collinear_and_invalid_colors(self):
        cases = [([0, 0, 0], [1, 0, 0]), ([1, 0, 0], [0, 0, 0]),
                 ([100, 50, 25], [200, 100, 50]),
                 ([1, 0, 0], [1, 1e-8, 0]),
                 ([np.nan, 0, 0], [0, 1, 0]),
                 ([1, 0, 0], [0, np.inf, 0]),
                 ([1, 2], [0, 1, 0])]
        for first, second in cases:
            with self.subTest(first=first, second=second):
                before = self.data.copy()
                with np.errstate(all='raise'), self.assertRaises(ValueError):
                    GRAM.change_channels(first, second, self.data)
                np.testing.assert_array_equal(self.data, before)

    def test_valid_colors_preserve_existing_transform_direction_and_energy(self):
        # q1=(1,1,0)/sqrt(2), q2=(-1,1,0)/sqrt(2), q3=(0,0,1).
        q = np.array([[1, -1, 0], [1, 1, 0], [0, 0, np.sqrt(2)]]) / np.sqrt(2)
        actual = GRAM.change_channels([1, 1, 0], [0, 1, 0], self.data)
        expected = np.einsum('ij,jhw->ihw', q, self.data)
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        np.testing.assert_allclose(np.sum(actual ** 2, axis=0), np.sum(self.data ** 2, axis=0))

    def test_color_brightness_does_not_change_basis(self):
        expected = GRAM.change_channels([1, 1, 0], [0, 1, 0], self.data)
        actual = GRAM.change_channels([1e-200, 1e-200, 0], [0, 1e200, 0], self.data)
        np.testing.assert_allclose(actual, expected)

    def test_rejected_transform_preserves_task_and_gui_state(self):
        transform = load_method('ImageProcessor', 'gram_shmidt_transform_for_task',
                                {'change_channels': GRAM.change_channels})
        task = SimpleNamespace(color1=np.array([1., 0, 0]), color2=np.array([2., 0, 0]),
                               data=self.data, gram_schmidt_applied=False)
        processor = SimpleNamespace(progress=Mock())
        processor.gram_shmidt_transform_for_task = lambda task: transform(processor, task)
        messages = Mock()
        callback = load_method('App', 'gramm_shmidt_transform', {'np': np, 'mb': messages})
        app = SimpleNamespace(current_task=task, image_processor=processor,
                              progress_manager=Mock(), gram_shmidt_button=Mock(),
                              _update_tasks_display=Mock())
        callback(app)
        self.assertIs(task.data, self.data)
        self.assertFalse(task.gram_schmidt_applied)
        messages.showwarning.assert_called_once()
        app.gram_shmidt_button.configure.assert_not_called()
        app._update_tasks_display.assert_not_called()


if __name__ == '__main__':
    unittest.main()
