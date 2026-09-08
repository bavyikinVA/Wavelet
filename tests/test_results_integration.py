"""Regression: successful calculation clears tensors but results still open."""
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
from unittest.mock import patch

import numpy as np
from PIL import Image
import main
from test_result_viewer import settle
from test_animation import finish_animations
from history import RunHistoryStore
from compute.processing_task import ProcessingTask
from history.result_data import load_result


class ResultsIntegrationTests(unittest.TestCase):
    def test_graph_exports_keep_native_data_without_csv_option(self):
        with tempfile.TemporaryDirectory() as folder:
            main.ImageProcessor.save_upper_envelope_maxima_row_histogram(
                folder, 'Статистика', [2, 4], [3, 7], 5, save_csv=False)
            data = load_result(dict(path=str(Path(folder) / 'Статистика.png'), category='Статистика'), folder)
            np.testing.assert_array_equal(data['array'], [[2, 3], [4, 7]])
            self.assertFalse(data['spatial'])
            main.ImageProcessor.save_extremes_graphic(folder, 'Экстремумы', [[2, 3]], original_img_shape=(8, 10))
            data = load_result(dict(path=str(Path(folder) / 'Экстремумы.png'), category='Экстремумы'), folder)
            self.assertEqual(data['shape'], (8, 10))
            np.testing.assert_array_equal(data['points'], [[2, 3]])

    def test_main_window_completion_opens_disk_backed_results_tab(self):
        # CTk keeps process-global appearance/DPI state. Exercise the complete
        # application in its own interpreter, as it runs outside the test suite.
        project = Path(__file__).resolve().parents[1]
        script = (
            "import sys; sys.path.insert(0, 'tests'); "
            "from test_results_integration import ResultsIntegrationTests; "
            "ResultsIntegrationTests()._check_main_window_completion()"
        )
        result = subprocess.run([sys.executable, '-c', script], cwd=project,
                                capture_output=True, text=True, errors='replace', timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def _check_main_window_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            store = RunHistoryStore(str(Path(folder) / 'history.sqlite3'))
            image_path = Path(folder) / 'Изображение.png'
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_path)
            np.save(Path(folder) / '2D_Morlet.npy', np.ones((8, 8), dtype=np.complex64))
            # Keep automated checks withdrawn: the delayed startup maximizer
            # must not re-open a test window on the user's desktop.
            with patch.object(main, 'RunHistoryStore', return_value=store), \
                    patch.object(main.App, '_maximize_properly', return_value=None):
                app = main.App()
            app.withdraw()
            try:
                task = ProcessingTask()
                task.task_name = 'Test'
                task.task_folder_path = folder
                task.original_image = np.zeros((8, 8, 3), dtype=np.uint8)
                task.result = {}  # Real post-calculation state.
                app.current_task = task
                app.image_processor.tasks = [task]
                app._update_tasks_display()
                card = app.task_widgets[0]
                task.task_name = 'Updated'
                app._update_tasks_display()
                self.assertIs(app.task_widgets[0], card)
                self.assertEqual(app._task_cards[id(task)]['title'].cget('text'), 'Updated')
                app.show_success_message(1., 1)
                app.update()
                settle(app, app.results_panel)
                self.assertEqual(app.results_panel.folder, folder)
                self.assertGreaterEqual(len(app.results_panel.artifacts), 2)
                self.assertIsNotNone(app.results_panel.right.array)
                self.assertIn('ГОТОВО', app.progress_manager.state_badge.cget('text'))
                app._toggle_focus_layout()
                finish_animations(app, app.tasks_panel, app.progress_manager.frame)
                self.assertEqual(app.tasks_panel.winfo_manager(), '')
                self.assertEqual(app.progress_manager.frame.winfo_manager(), '')
                app._toggle_focus_layout()
                finish_animations(app, app.tasks_panel, app.progress_manager.frame)
                self.assertEqual(app.tasks_panel.winfo_manager(), 'grid')
                self.assertEqual(app.progress_manager.frame.winfo_manager(), 'pack')
            finally:
                for callback in app.tk.call('after', 'info'):
                    app.tk.call('after', 'cancel', callback)
                app.safe_destroy()


