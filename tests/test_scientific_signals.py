import unittest
import numpy as np

from compute.validation import validate_scales
from compute.wavelets.cpu_wavelet import morlet_wavelet_with_padding as cwt
from compute.wavelets.morlet_2d import cwt2d_morlet_cpu


class ScientificSignalTests(unittest.TestCase):
    def test_invalid_scales_rejected_by_validator_and_cpu(self):
        for scales in ([], [0.], [-1.], [np.nan], [np.inf]):
            with self.subTest(scales=scales):
                with self.assertRaises(ValueError):
                    validate_scales(scales)
                with self.assertRaises(ValueError):
                    cwt(np.ones(8), np.array(scales))

    def test_centered_constant_is_zero_in_1d_and_2d(self):
        signal = np.full(64, 123.)
        np.testing.assert_array_equal(cwt(signal - signal.mean(), np.array([2., 8.])), 0)
        np.testing.assert_allclose(cwt2d_morlet_cpu(np.full((32, 32), 123.), [2., 4.], [0., 90.]), 0, atol=1e-6)

    def test_sinusoid_has_expected_frequency_selectivity_and_linearity(self):
        x = np.arange(256)
        signal = np.cos(2 * np.pi * x / 8)
        scales = np.array([4., 8., 16.])
        result = cwt(signal, scales)
        energy = np.mean(result[:, 64:-64] ** 2, axis=1)
        self.assertGreater(energy[1], 10 * max(energy[0], energy[2]))
        np.testing.assert_allclose(cwt(3 * signal, scales), 3 * result, rtol=1e-11, atol=1e-11)

    def test_crop_differs_at_edge_but_matches_interior(self):
        x = np.arange(512)
        signal = np.cos(2 * np.pi * x / 13 + .3)
        full = cwt(signal, np.array([5.]))[0, 128:384]
        crop = cwt(signal[128:384], np.array([5.]))[0]
        error = np.abs(full - crop)
        self.assertLess(error[40:-40].max(), 1e-3)
        self.assertGreater(error[:10].max(), 1e-2)

    def test_2d_orientation_tracks_texture_normal(self):
        y, x = np.mgrid[:96, :96]
        for angle in (0., 45., 90.):
            radians = np.deg2rad(angle)
            image = np.cos(2 * np.pi * (x * np.cos(radians) + y * np.sin(radians)) / 12)
            result = cwt2d_morlet_cpu(image, [12 * 6 / (2 * np.pi)], [angle, angle + 90])
            energy = np.mean(np.abs(result[0, :, 30:-30, 30:-30]) ** 2, axis=(1, 2))
            self.assertGreater(energy[0], 20 * energy[1])


class ScientificGPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() == 0:
                raise RuntimeError('No CUDA device')
            cp.zeros(1)
        except Exception as error:
            raise unittest.SkipTest(str(error)) from error
        from compute.wavelets.cupy_wavelet import CupyWaveletGPU
        cls.gpu = CupyWaveletGPU()

    def test_cpu_gpu_agree_on_sinusoid_and_centered_constant(self):
        x = np.arange(128)
        signals = np.stack([np.cos(2 * np.pi * x / 8), np.zeros(128)])
        scales = np.array([4., 8., 16.])
        expected = np.stack([cwt(signal, scales) for signal in signals])
        np.testing.assert_allclose(self.gpu.compute_batch_signals(signals, scales), expected, rtol=1e-4, atol=1e-5)
        for scale in (0., -1., np.nan, np.inf):
            with self.assertRaises(ValueError):
                self.gpu.compute_batch_signals(signals, np.array([scale]))

    def test_cpu_gpu_agree_on_2d_texture(self):
        from compute.wavelets.morlet_2d import cwt2d_morlet_gpu
        y, x = np.mgrid[:48, :48]
        image = np.cos(2 * np.pi * (x + y) / 12)
        args = (image, [3., 6.], [0., 45., 90.])
        np.testing.assert_allclose(cwt2d_morlet_gpu(*args), cwt2d_morlet_cpu(*args), rtol=1e-4, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
