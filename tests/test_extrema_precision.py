"""Exercise real point-stage methods without importing the GUI/CUDA startup."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np


def load_point_methods():
    path = Path(__file__).parents[1] / "main.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    processor = next(node for node in tree.body
                     if isinstance(node, ast.ClassDef) and node.name == "ImageProcessor")
    methods = [node for node in processor.body
               if isinstance(node, ast.FunctionDef)
               and node.name in {"find_extremes", "compute_points"}]
    for method in methods:
        method.decorator_list = []
    namespace = {"np": np}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


METHODS = load_point_methods()


class DetectionReached(Exception):
    pass


class ExtremaPrecisionTests(unittest.TestCase):
    def test_small_maxima_and_minima_in_both_directions(self):
        signal = np.array([[0.0001, 0.0004, 0.0002, -0.0004, 0.0001]])
        for matrix, maxima_index, minima_index, expected_max, expected_min in (
            (signal, 1, 3, [[1, 0]], [[3, 0]]),
            (signal.T, 2, 4, [[0, 1]], [[0, 3]]),
        ):
            with self.subTest(shape=matrix.shape):
                original = matrix.copy()
                result = METHODS["find_extremes"](matrix, True, True, True, True)
                self.assertEqual(result[maxima_index], expected_max)
                self.assertEqual(result[minima_index], expected_min)
                np.testing.assert_array_equal(matrix, original)

    def test_positive_rescaling_preserves_extrema(self):
        matrix = np.array([[1.0, 4.0, 2.0, -4.0, 1.0]])
        expected = METHODS["find_extremes"](matrix, True, True, True, True)[1:]
        for factor in (1e-6, 1e-3, 1e3):
            actual = METHODS["find_extremes"](matrix * factor, True, True, True, True)[1:]
            self.assertEqual(actual, expected)

    def test_flat_plateau_remains_excluded_by_strict_rule(self):
        result = METHODS["find_extremes"](
            np.array([[0.0, 1.0, 1.0, 0.0]]), True, True, True, True
        )
        self.assertEqual(result[1:], ([], [], [], []))

    def test_pipeline_passes_unrounded_coefficients_to_cpu_and_gpu(self):
        source = np.array([[1.0001, 1.0004, 1.0002]])
        for use_gpu in (False, True):
            with self.subTest(use_gpu=use_gpu):
                captured = []

                def detect(coefs=None, **kwargs):
                    captured.append(np.asarray(coefs).copy())
                    # Stop at the stage boundary, before interpolation/export.
                    raise DetectionReached

                namespace = dict(METHODS)
                namespace["ExtremesFinder"] = SimpleNamespace(find_extremes_gpu=detect)
                # Rebind only globals; execute the unchanged production method.
                from types import FunctionType
                compute = FunctionType(METHODS["compute_points"].__code__, namespace)
                processor = SimpleNamespace(
                    progress=Mock(), backend=SimpleNamespace(use_gpu=use_gpu),
                    knn_processor=Mock(), find_extremes=detect,
                )
                task = SimpleNamespace(num_scale=1, scales=[2.0], result={0: [[source]]})
                plan = SimpleNamespace(maxima_required=False)
                with self.assertRaises(DetectionReached):
                    compute(processor, task, True, True, True, True,
                            False, False, False, False, False, "normal", plan)
                np.testing.assert_array_equal(captured[0], source)
                peaks = METHODS["find_extremes"](captured[0], True, False, True, False)
                self.assertEqual(peaks[1], [[1, 0]])


if __name__ == "__main__":
    unittest.main()
