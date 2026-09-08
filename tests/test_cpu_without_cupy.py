"""Exercise the real import graph in a fresh process without GPU packages."""
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


class CPUOnlyTests(unittest.TestCase):
    def test_app_import_and_cpu_stages_without_gpu_packages(self):
        script = textwrap.dedent('''
            import importlib.abc
            import sys
            class NoGPU(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split('.')[0] in {'cupy', 'cupyx', 'pycuda', 'pyopencl'}:
                        raise ModuleNotFoundError('GPU package blocked: ' + fullname, name=fullname)
            sys.meta_path.insert(0, NoGPU())
            import numpy as np
            import main
            import compute
            from compute.extremes.interpolator import Interpolator
            from compute.knn.knn_cpu import KNNProcessor
            from compute.wavelets.morlet_2d import cwt2d_morlet_cpu
            backend = compute.ComputeBackend(use_gpu=True)
            assert not backend.use_gpu
            assert backend.get_backend_info()['available']
            assert not backend.set_use_gpu(True)
            signal = np.array([0., 2., 0., 5., 0., 2., 0.])
            coefs = compute.morlet_wavelet_with_padding(signal, np.array([2.]))
            assert coefs.shape == (1, 7) and np.isfinite(coefs).all()
            points = main.ImageProcessor.find_extremes(signal[None], True, False, True, False)
            assert points[1] == [[1, 0], [3, 0], [5, 0]]
            upper, lower = Interpolator(backend).get_envelopes(signal[None], points[1], [])
            assert upper == [(3, 0)] and lower == []
            neighbors = KNNProcessor(use_gpu=True).find_k_nearest_neighbors(
                np.array([[0., 0.], [1., 0.], [3., 0.]]), 1)
            assert len(neighbors) == 3
            result = cwt2d_morlet_cpu(np.outer(signal, signal), [2.], [0.])
            assert np.isfinite(result).all()
            assert not any(name.split('.')[0] in {'cupy', 'cupyx', 'pycuda', 'pyopencl'}
                           for name in sys.modules)
        ''')
        result = subprocess.run(
            [sys.executable, '-c', script], cwd=Path(__file__).parents[1],
            capture_output=True, text=True, errors='replace', timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
