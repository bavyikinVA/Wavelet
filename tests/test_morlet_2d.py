import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "compute" / "wavelets" / "morlet_2d.py"
SPEC = importlib.util.spec_from_file_location("morlet_2d_direct", MODULE_PATH)
MORLET_2D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MORLET_2D)


class Morlet2DTests(unittest.TestCase):
    def test_cpu_transform_returns_complex_scale_angle_tensor(self):
        image = np.arange(64, dtype=np.float32).reshape(8, 8)

        result = MORLET_2D.cwt2d_morlet_cpu(
            image,
            scales=[2.0],
            orientations=[0.0, 90.0],
        )

        self.assertEqual(result.shape, (1, 2, 8, 8))
        self.assertTrue(np.iscomplexobj(result))
        self.assertTrue(np.all(np.isfinite(result)))


if __name__ == "__main__":
    unittest.main()
