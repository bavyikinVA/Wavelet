import tempfile
import time
from test_animation import finish_animations
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import customtkinter as ctk
import numpy as np
from PIL import Image

from utils.result_viewer import ResultsPanel
from history.result_catalog import discover_results, save_coefficient_preview


def settle(root, panel):
    deadline = time.monotonic() + 10
    while panel.left._future is not None or panel.right._future is not None:
        root.update()
        if time.monotonic() > deadline:
            raise AssertionError('Result loading timed out')
        time.sleep(.02)
    root.update_idletasks()


class ResultViewerTests(unittest.TestCase):
    def test_filter_both_panes_and_reuse_canvas(self):
        with tempfile.TemporaryDirectory() as folder:
            Image.fromarray(np.zeros((20, 30, 3), dtype=np.uint8)).save(Path(folder) / 'Изображение.png')
            np.savetxt(Path(folder) / 'вейвлет_2.txt', np.ones((20, 30)), delimiter=',')
            np.savetxt(Path(folder) / 'вейвлет_3.txt', np.zeros((20, 30)), delimiter=',')
            root = ctk.CTk()
            root.withdraw()
            panel = ResultsPanel(root)
            try:
                panel.load_folder(folder)
                settle(root, panel)
                axes = panel.right.ax
                artist = panel.right.image_artist
                with patch.object(panel.right, 'render_payload', wraps=panel.right.render_payload) as render:
                    panel.load_folder(folder)
                    panel.filter()
                    settle(root, panel)
                    render.assert_not_called()
                panel.filter('Вейвлеты')
                settle(root, panel)
                for pane in (panel.left, panel.right):
                    self.assertEqual(len(pane.items), 2)
                    self.assertTrue(all(item['category'] == 'Вейвлеты' for item in pane.items.values()))
                panel.search.insert(0, '_3')
                panel.filter()
                settle(root, panel)
                self.assertEqual(list(panel.left.items), ['вейвлет_3.txt'])
                self.assertEqual(list(panel.right.items), ['вейвлет_3.txt'])
                self.assertIs(panel.right.ax, axes)
                self.assertIs(panel.right.image_artist, artist)
                panel.search.insert('end', 'missing')
                panel.filter()
                self.assertIsNone(panel.left.array)
                self.assertIsNone(panel.right.array)
            finally:
                for callback in root.tk.call('after', 'info'):
                    root.tk.call('after', 'cancel', callback)
                panel.destroy()
                root.destroy()

    def test_linked_view_maps_relative_region_and_survives_scale_change(self):
        with tempfile.TemporaryDirectory() as folder:
            for name, shape in [('a', (40, 60)), ('b', (80, 120)), ('c', (20, 30))]:
                np.save(Path(folder) / (name + '.npy'), np.ones(shape))
            root = ctk.CTk()
            root.withdraw()
            panel = ResultsPanel(root)
            try:
                panel.load_folder(folder)
                settle(root, panel)
                panel.right.selector.set('b.npy')
                panel.right.show_selected()
                settle(root, panel)
                panel.linked.set(True)
                panel.left.ax.set_xlim(9.5, 29.5)
                panel.left.ax.set_ylim(24.5, 4.5)
                np.testing.assert_allclose(panel.right.ax.get_xlim(), [19.5, 59.5])
                np.testing.assert_allclose(panel.right.ax.get_ylim(), [49.5, 9.5])
                view = panel.left.normalized_view()
                panel.right.selector.set('c.npy')
                panel.right.show_selected()
                settle(root, panel)
                np.testing.assert_allclose(panel.right.normalized_view(), view)
                panel.left.scroll_zoom(SimpleNamespace(inaxes=panel.left.ax, xdata=15., ydata=12., step=1))
                np.testing.assert_allclose(panel.right.normalized_view(), panel.left.normalized_view())
                self.assertLess(np.diff(panel.left.ax.get_xlim())[0], 20.)
                panel.linked.set(False)
                before = panel.right.normalized_view()
                panel.left.ax.set_xlim(0, 5)
                np.testing.assert_allclose(panel.right.normalized_view(), before)

                self.assertEqual(str(panel.panes.cget('orient')), 'horizontal')
                panel.left.toggle_details()
                finish_animations(root, panel.left.controls)
                self.assertEqual(panel.left.controls.winfo_manager(), 'grid')
                panel.left.toggle_details()
                finish_animations(root, panel.left.controls)
                self.assertEqual(panel.left.controls.winfo_manager(), '')
                root.update_idletasks()
            finally:
                for callback in root.tk.call('after', 'info'):
                    root.tk.call('after', 'cancel', callback)
                panel.destroy()
                root.destroy()

    def test_legacy_wavelet_text_is_displayed_as_map(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'Расчет_вейвлетов_масштаб_2.txt'
            np.savetxt(path, np.arange(12.).reshape(3, 4), delimiter=',')
            root = ctk.CTk()
            root.withdraw()
            panel = ResultsPanel(root)
            try:
                panel.load_folder(folder)
                settle(root, panel)
                np.testing.assert_array_equal(panel.right.array, np.arange(12.).reshape(3, 4))
                root.update_idletasks()
            finally:
                for callback in root.tk.call('after', 'info'):
                    root.tk.call('after', 'cancel', callback)
                panel.destroy()
                root.destroy()

    def test_results_survive_cleared_memory_and_support_multiple_formats(self):
        with tempfile.TemporaryDirectory() as folder:
            Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8)).save(Path(folder) / 'Изображение.png')
            coefficients = np.arange(20.).reshape(4, 5)
            path = save_coefficient_preview(folder, '1D_масштаб_2', coefficients)
            del coefficients
            (Path(folder) / 'Статистика.csv').write_text('scale,count\n2,3', encoding='utf-8')
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(Path(folder) / 'Кластеры.png')
            root = ctk.CTk()
            root.withdraw()
            panel = ResultsPanel(root)
            try:
                panel.load_folder(folder)
                settle(root, panel)
                self.assertEqual(len(panel.artifacts), 4)
                self.assertIn('Источник', {a['category'] for a in panel.artifacts})
                label = next(a['label'] for a in panel.artifacts if a['path'] == str(path))
                panel.right.selector.set(label)
                panel.right.show_selected()
                settle(root, panel)
                np.testing.assert_array_equal(panel.right.array, np.arange(20.).reshape(4, 5))
                panel.right.selector.set('Статистика.csv')
                panel.right.show_selected()
                settle(root, panel)
                self.assertEqual(panel.right.payload['kind'], 'series')
                panel.category.set('ML-кластеры')
                panel.filter()
                self.assertEqual(list(panel.right.items), ['Кластеры.png'])
                root.update_idletasks()
            finally:
                for callback in root.tk.call('after', 'info'):
                    root.tk.call('after', 'cancel', callback)
                panel.destroy()
                root.destroy()

    def test_complex_preview_and_missing_folder(self):
        with tempfile.TemporaryDirectory() as folder:
            path = save_coefficient_preview(folder, '2D_Morlet', np.ones((512, 512)) * 1j)
            array = np.load(path)
            self.assertEqual(array.shape, (256, 256))
            self.assertTrue(np.iscomplexobj(array))
            root = ctk.CTk()
            root.withdraw()
            panel = ResultsPanel(root)
            try:
                panel.load_folder(folder)
                settle(root, panel)
                panel.right.component.set('Фаза')
                panel.right.render_payload()
                panel.right.show_selected()
                settle(root, panel)
                np.testing.assert_allclose(panel.right.array, np.pi / 2)
                panel.load_folder(str(Path(folder) / 'missing'))
                self.assertEqual(panel.artifacts, [])
                self.assertIsNone(panel.right.array)
                root.update_idletasks()
            finally:
                for callback in root.tk.call('after', 'info'):
                    root.tk.call('after', 'cancel', callback)
                panel.destroy()
                root.destroy()


