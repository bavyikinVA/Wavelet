import tempfile
from pathlib import Path
import unittest
import numpy as np
from PIL import Image
from history.result_data import load_result


class ResultDataTests(unittest.TestCase):
    def test_native_companions_share_source_coordinates(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            Image.new('RGB', (60, 40)).save(root / 'Изображение.png')
            png = root / 'вейвлет.png'
            Image.new('RGB', (800, 600)).save(png)
            np.savetxt(png.with_suffix('.txt'), np.arange(2400).reshape(40, 60), delimiter=',')
            data = load_result(dict(path=str(png), category='Вейвлеты'), folder)
            self.assertEqual(data['kind'], 'map')
            self.assertEqual(data['shape'], (40, 60))
            self.assertEqual(data['array'][10, 20], 620)
            points = root / 'экстремумы.npz'
            np.savez_compressed(points, points=[[20, 10]], shape=[40, 60])
            data = load_result(dict(path=str(points), category='Экстремумы'), folder)
            np.testing.assert_array_equal(data['points'], [[20, 10]])
            self.assertEqual(data['shape'], (40, 60))
            self.assertTrue(data['spatial'])

    def test_baked_plot_without_numbers_is_not_spatial(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'экстремумы.png'
            Image.new('RGB', (80, 60)).save(path)
            data = load_result(dict(path=str(path), category='Экстремумы'), folder)
            self.assertFalse(data['spatial'])
            self.assertTrue(data['legacy'])
