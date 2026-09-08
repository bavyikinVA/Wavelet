import os
import sqlite3
import tempfile
import unittest
import importlib.util
from pathlib import Path

import numpy as np

from history import (
    RunHistoryStore, restore_feature_checkpoint, save_feature_checkpoint,
)


MODULE_PATH = Path(__file__).parents[1] / "compute" / "processing_task.py"
SPEC = importlib.util.spec_from_file_location("processing_task_history", MODULE_PATH)
PROCESSING_TASK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROCESSING_TASK)
ProcessingTask = PROCESSING_TASK.ProcessingTask


class RunHistoryTests(unittest.TestCase):
    def test_transaction_commits_and_connection_closes(self):
        with tempfile.TemporaryDirectory() as directory:
            store = RunHistoryStore(os.path.join(directory, "history.sqlite3"))
            task = ProcessingTask()
            run_id = store.add_run(task=task, status="completed", duration_seconds=1,
                                   settings=task.settings_snapshot())
            with store._connect() as connection:
                connection.execute(
                    "UPDATE research_runs SET researcher_note = ? WHERE id = ?",
                    ("committed", run_id),
                )
            self.assertEqual(store.get_run(run_id)["researcher_note"], "committed")
            # Keep a reference: closure must not depend on garbage collection.
            with self.assertRaises(sqlite3.ProgrammingError):
                connection.execute("SELECT 1")

    def test_failed_transaction_rolls_back_and_connection_closes(self):
        with tempfile.TemporaryDirectory() as directory:
            store = RunHistoryStore(os.path.join(directory, "history.sqlite3"))
            task = ProcessingTask()
            run_id = store.add_run(task=task, status="completed", duration_seconds=1,
                                   settings=task.settings_snapshot())
            with self.assertRaisesRegex(RuntimeError, "abort"):
                with store._connect() as connection:
                    connection.execute(
                        "UPDATE research_runs SET researcher_note = ? WHERE id = ?",
                        ("must roll back", run_id),
                    )
                    raise RuntimeError("abort")
            self.assertEqual(store.get_run(run_id)["researcher_note"], "")
            with self.assertRaises(sqlite3.ProgrammingError):
                connection.execute("SELECT 1")
            # The store remains usable after a failed transaction.
            store.update_note(run_id, "recovered")
            self.assertEqual(store.get_run(run_id)["researcher_note"], "recovered")

    def test_public_operations_release_database_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "history.sqlite3")
            store = RunHistoryStore(path)
            task = ProcessingTask()
            run_id = store.add_run(task=task, status="completed", duration_seconds=1,
                                   settings=task.settings_snapshot())
            store.list_runs()
            store.get_settings(run_id)
            store.get_run(run_id)
            store.update_note(run_id, "note")
            self.assertTrue(store.delete_run(run_id))
            self.assertFalse(store.delete_run(run_id))
            with self.assertRaises(KeyError):
                store.update_note(run_id, "missing")
            os.remove(path)
            self.assertFalse(os.path.exists(path))

    def test_completed_run_preserves_reusable_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            store = RunHistoryStore(os.path.join(directory, "history.sqlite3"))
            task = ProcessingTask()
            task.task_name = "Задача 1"
            task.image_path = os.path.join(directory, "source.png")
            task.scales = np.array([2, 4, 6])
            task.num_scale = 3
            task.apply_pipeline_preset("Подготовка данных для ML")
            task.ml_algorithm = "dbscan"
            task.ml_eps = 1.25

            run_id = store.add_run(
                task=task, status="completed", duration_seconds=3.5,
                settings=task.settings_snapshot(), ml_summary="Кластеров: 2"
            )

            rows = store.list_runs()
            restored = store.get_settings(run_id)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["ml_algorithm"], "dbscan")
            self.assertEqual(restored["scales"], [2, 4, 6])
            self.assertEqual(restored["ml_eps"], 1.25)

    def test_snapshot_can_be_applied_to_a_fresh_task(self):
        original = ProcessingTask()
        original.scales = np.array([1, 3, 5], dtype=float)
        original.num_scale = 3
        original.ml_algorithm = "dbscan"
        original.ml_min_samples = 8

        restored = ProcessingTask()
        restored.apply_settings_snapshot(original.settings_snapshot())

        np.testing.assert_array_equal(restored.scales, original.scales)
        self.assertEqual(restored.ml_algorithm, "dbscan")
        self.assertEqual(restored.ml_min_samples, 8)
        self.assertEqual(restored.task_folder_path, "")

    def test_history_note_can_be_saved_without_changing_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            store = RunHistoryStore(os.path.join(directory, "history.sqlite3"))
            task = ProcessingTask()
            task.task_name = "Опыт"
            run_id = store.add_run(
                task=task, status="completed", duration_seconds=1,
                settings=task.settings_snapshot()
            )
            store.update_note(run_id, "Проверить блоки по 5 масштабов")
            row = store.list_runs()[0]
            self.assertEqual(
                row["researcher_note"], "Проверить блоки по 5 масштабов"
            )
            self.assertEqual(store.get_settings(run_id)["analysis_mode"], "1d")

    def test_execution_plan_and_card_summaries_are_reproducible(self):
        task = ProcessingTask()
        task.scales = np.array([1, 2, 3, 4], dtype=float)
        task.num_scale = 4
        task.apply_pipeline_preset("Подготовка данных для ML")
        task.ml_algorithm = "dbscan"

        self.assertIn("Вейвлет-преобразование", task.execution_stage_names())
        self.assertIn("ML-кластеризация", task.execution_stage_names())
        self.assertEqual(task.scales_summary(), "1–4, шаг 1 · 4 масштабов")
        self.assertIn("вейвлеты → экстремумы", task.executed_stage_summary())
        self.assertIn("ML: DBSCAN", task.nuances_summary())

    def test_feature_checkpoint_restores_knn_for_repeated_ml(self):
        with tempfile.TemporaryDirectory() as directory:
            task = ProcessingTask()
            task.task_folder_path = directory
            task.knn_results = {
                ("red", 2.0): {"max_by_row": [[1, 2], [3, 4]]}
            }
            path = save_feature_checkpoint(task)
            self.assertTrue(os.path.isfile(path))

            restored = ProcessingTask()
            stages = restore_feature_checkpoint(restored, directory)
            self.assertIn("KNN-признаки", stages)
            self.assertEqual(restored.knn_results, task.knn_results)


if __name__ == "__main__":
    unittest.main()
