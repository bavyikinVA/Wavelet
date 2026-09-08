"""Boundary regressions against independently padded NumPy signals."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CPU = load_module("cpu_boundary_test", "compute/wavelets/cpu_wavelet.py")


def reference_cwt(signal, scales):
    result = []
    for scale in scales:
        padding = int(7 * scale) // 2 + 1
        extended = np.pad(signal, padding, mode="symmetric")
        positions = np.arange(-padding, len(signal) + padding)
        offsets = (positions[None, :] - np.arange(len(signal))[:, None]) / scale
        kernel = 0.75 * np.exp(-offsets ** 2 / 2) * np.cos(2 * np.pi * offsets)
        result.append(kernel @ extended / np.sqrt(scale))
    return np.asarray(result)


class Morlet1DBoundaryTests(unittest.TestCase):
    def test_short_signals_match_explicit_symmetric_padding(self):
        for signal in ([2.0], [1.0, -2.0], [0.0, 3.0, -1.0, 2.0]):
            for scales in ([0.5, 1.0], [2.0, 5.0, 12.0]):
                with self.subTest(signal=signal, scales=scales):
                    data = np.asarray(signal)
                    actual = CPU.morlet_wavelet_with_padding(data, np.asarray(scales))
                    np.testing.assert_allclose(actual, reference_cwt(data, scales),
                                               rtol=1e-12, atol=1e-12)

    def test_scale_is_independent_of_other_scales_and_order(self):
        signal = np.array([0.0, 2.0, -1.0, 4.0])
        alone = CPU.morlet_wavelet_with_padding(signal, np.array([2.0]))[0]
        for scales, index in (([2.0, 12.0], 0), ([12.0, 2.0, 0.5], 1)):
            actual = CPU.morlet_wavelet_with_padding(signal, np.asarray(scales))[index]
            np.testing.assert_array_equal(actual, alone)


class Morlet1DGPUBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() == 0:
                raise unittest.SkipTest("No CUDA device")
            cp.zeros(1)
        except (ImportError, OSError) as error:
            raise unittest.SkipTest(str(error)) from error
        except Exception as error:
            # Device/runtime availability only; kernel failures below must fail.
            raise unittest.SkipTest(f"CUDA unavailable: {error}") from error
        gpu = load_module("gpu_boundary_test", "compute/wavelets/cupy_wavelet.py")
        cls.processor = gpu.CupyWaveletGPU()

    def test_gpu_regular_and_chunked_match_reference_and_cpu(self):
        scales = np.array([0.5, 2.0, 5.0, 12.0], dtype=np.float32)
        for signal in ([2.0], [1.0, -2.0], [0.0, 3.0, -1.0, 2.0]):
            batch = np.array([signal], dtype=np.float32)
            expected = reference_cwt(batch[0].astype(float), scales)
            cpu = CPU.morlet_wavelet_with_padding(batch[0].astype(float), scales)
            for compute in (self.processor.compute_batch_signals,
                            self.processor._compute_batch_chunked):
                with self.subTest(signal=signal, method=compute.__name__):
                    actual = compute(batch, scales)[0]
                    # Float32 sums and CUDA transcendental functions differ from CPU.
                    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=5e-6)
                    np.testing.assert_allclose(actual, cpu, rtol=5e-5, atol=5e-6)

    def test_gpu_scale_is_independent_of_other_scales(self):
        batch = np.array([[0.0, 2.0, -1.0, 4.0]], dtype=np.float32)
        alone = self.processor.compute_batch_signals(batch, np.array([2.0]))[:, 0]
        combined = self.processor.compute_batch_signals(batch, np.array([12.0, 2.0]))[:, 1]
        np.testing.assert_array_equal(alone, combined)


if __name__ == "__main__":
    unittest.main()
