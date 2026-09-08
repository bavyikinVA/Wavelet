import importlib.util
from pathlib import Path
import unittest

import numpy as np


def load_module(name, filename):
    path = Path(__file__).parents[1] / 'compute' / 'extremes' / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CPU = load_module('envelope_cpu_test', 'interpol.py')
SIGNAL = np.array([0., 2., 0., 5., 0., 2., 0., 1., 0.])
SUPPORTS = [[], [3], [1, 3], [1, 3, 5], [1, 3, 5, 7]]
EXPECTED = [None, [5.] * 9,
            [2., 2., 3.5, 5., 5., 5., 5., 5., 5.],
            [2., 2., 3.5, 5., 3.5, 2., 2., 2., 2.],
            [2., 2., 3.5, 5., 3.5, 2., 1.5, 1., 1.]]


class EnvelopeSupportTests(unittest.TestCase):
    def test_zero_to_four_supports_and_constant_tails(self):
        for support, expected in zip(SUPPORTS, EXPECTED):
            with self.subTest(support=support):
                actual = CPU.interpolate_envelope(SIGNAL, support)
                if expected is None:
                    self.assertIsNone(actual)
                else:
                    np.testing.assert_allclose(actual, expected)

    def test_points_in_rows_and_columns_for_both_envelopes(self):
        for support in SUPPORTS:
            image = np.stack([SIGNAL, -SIGNAL])
            maxima = [(x, 0) for x in support]
            minima = [(x, 1) for x in support]
            expected = ([(3, 0)], [(3, 1)]) if len(support) >= 3 else ([], [])
            with self.subTest(support=support):
                self.assertEqual(CPU.get_row_envelopes(image, maxima, minima), expected)
                self.assertEqual(CPU.get_column_envelopes(
                    image.T, [(y, x) for x, y in maxima], [(y, x) for x, y in minima]),
                    tuple([(y, x) for x, y in points] for points in expected))


class GPUEnvelopeSupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() == 0:
                raise unittest.SkipTest('No CUDA device')
            cp.zeros(1)
        except Exception as error:
            raise unittest.SkipTest(f'CUDA unavailable: {error}') from error
        cls.cp = cp
        cls.gpu = load_module('envelope_gpu_test', 'gpu_envelopes.py').GPUEnvelopeProcessor()

    def test_gpu_interpolants_match_expected_in_both_directions(self):
        cp = self.cp
        batch = cp.asarray(np.tile(SIGNAL, (len(SUPPORTS), 1)), dtype=cp.float32)
        mask = cp.zeros(batch.shape, dtype=cp.bool_)
        for i, support in enumerate(SUPPORTS):
            if support:
                mask[i, support] = True
        for method in (self.gpu._interpolate_batch_gpu, self.gpu._interpolate_cols_batch_gpu):
            actual = cp.asnumpy(method(batch, mask, len(SIGNAL)))
            self.assertTrue(np.isnan(actual[0]).all())
            for i in range(1, len(SUPPORTS)):
                np.testing.assert_allclose(actual[i], EXPECTED[i], rtol=1e-6, atol=1e-6)

    def test_gpu_peaks_match_cpu_for_small_support_sets(self):
        image = np.stack([SIGNAL, -SIGNAL])
        for support in SUPPORTS:
            maxima = [(x, 0) for x in support]
            minima = [(x, 1) for x in support]
            with self.subTest(support=support):
                self.assertEqual(self.gpu.get_row_envelopes_gpu(image, maxima, minima),
                                 CPU.get_row_envelopes(image, maxima, minima))
                transposed_max = [(y, x) for x, y in maxima]
                transposed_min = [(y, x) for x, y in minima]
                self.assertEqual(self.gpu.get_col_envelopes_gpu(image.T, transposed_max, transposed_min),
                                 CPU.get_column_envelopes(image.T, transposed_max, transposed_min))


if __name__ == '__main__':
    unittest.main()
