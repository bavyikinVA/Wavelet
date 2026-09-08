import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).parents[1] / "compute" / "processing_task.py"
SPEC = importlib.util.spec_from_file_location("processing_task_direct", MODULE_PATH)
PROCESSING_TASK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROCESSING_TASK)
ProcessingTask = PROCESSING_TASK.ProcessingTask


class PipelinePlanTests(unittest.TestCase):
    def test_ml_preset_includes_clustering_after_knn(self):
        task = ProcessingTask()
        task.apply_pipeline_preset("Подготовка данных для ML")

        plan = task.resolve_pipeline()

        self.assertTrue(plan.knn)
        self.assertTrue(plan.ml_requested)
        self.assertTrue(plan.ml)

    def test_ml_stage_is_not_enabled_for_regular_knn(self):
        task = ProcessingTask()
        task.calculate_knn = True
        task.pipeline_preset = "Пользовательский"

        plan = task.resolve_pipeline()

        self.assertTrue(plan.knn)
        self.assertFalse(plan.ml)

    def test_wavelet_only_has_no_point_pipeline(self):
        task = ProcessingTask()
        task.apply_pipeline_preset("Только вейвлет")

        plan = task.resolve_pipeline()

        self.assertFalse(plan.point_pipeline_required)
        self.assertFalse(plan.extrema)
        self.assertFalse(plan.envelopes)
        self.assertFalse(plan.knn)
        self.assertFalse(plan.statistics)
        self.assertFalse(plan.synchronization)

    def test_knn_automatically_requires_envelopes_and_extrema(self):
        task = ProcessingTask()
        task.apply_pipeline_preset("Только вейвлет")
        task.calculate_knn = True
        task.find_maxima = False
        task.find_minima = False

        plan = task.resolve_pipeline()

        self.assertTrue(plan.knn)
        self.assertTrue(plan.envelopes)
        self.assertTrue(plan.extrema)
        self.assertTrue(plan.maxima_required)
        self.assertIn("envelopes", plan.automatic_reasons())

    def test_statistics_do_not_enable_knn_or_synchronization(self):
        task = ProcessingTask()
        task.apply_pipeline_preset("Только вейвлет")
        task.calculate_statistics = True

        plan = task.resolve_pipeline()

        self.assertTrue(plan.statistics)
        self.assertTrue(plan.envelopes)
        self.assertTrue(plan.extrema)
        self.assertFalse(plan.knn)
        self.assertFalse(plan.synchronization)

    def test_calculation_does_not_require_any_output_format(self):
        task = ProcessingTask()
        task.apply_pipeline_preset("Огибающие и статистики")
        task.output_extremes_text = False
        task.output_extremes_image = False
        task.output_envelopes_text = False
        task.output_envelopes_image = False
        task.statistics_output_csv = False
        task.statistics_output_image = False

        plan = task.resolve_pipeline()

        self.assertTrue(plan.extrema)
        self.assertTrue(plan.envelopes)
        self.assertTrue(plan.statistics)

    def test_statistics_require_maxima_even_if_user_unchecked_them(self):
        task = ProcessingTask()
        task.apply_pipeline_preset("Огибающие и статистики")
        task.find_maxima = False
        task.find_minima = True

        plan = task.resolve_pipeline()

        self.assertTrue(plan.maxima_required)

    def test_output_formats_do_not_start_calculation(self):
        task = ProcessingTask()
        task.apply_pipeline_preset("Только вейвлет")
        task.output_extremes_text = True
        task.output_extremes_image = True
        task.output_knn_text = True
        task.statistics_output_csv = True

        plan = task.resolve_pipeline()

        self.assertFalse(plan.extrema)
        self.assertFalse(plan.knn)
        self.assertFalse(plan.statistics)

    def test_row_dependent_stages_are_unavailable_without_rows(self):
        task = ProcessingTask()
        task.apply_pipeline_preset("Полный 1D-анализ")
        task.process_rows = False
        task.process_columns = True

        plan = task.resolve_pipeline()

        self.assertFalse(plan.point_pipeline_required)
        self.assertFalse(plan.knn)
        self.assertFalse(plan.statistics)
        self.assertFalse(plan.synchronization)

    def test_2d_has_separate_base_pipeline(self):
        task = ProcessingTask()
        task.apply_pipeline_preset("Полный 1D-анализ")
        task.set_analysis_mode("2d")

        plan = task.resolve_pipeline()

        self.assertEqual(plan.mode, "2d")
        self.assertFalse(plan.point_pipeline_required)
        self.assertFalse(plan.extrema)
        self.assertFalse(plan.envelopes)
        self.assertFalse(plan.knn)

    def test_pipeline_settings_are_independent_between_tasks(self):
        first = ProcessingTask()
        second = ProcessingTask()
        first.apply_pipeline_preset("Полный 1D-анализ")

        self.assertTrue(first.resolve_pipeline().knn)
        self.assertFalse(second.resolve_pipeline().knn)
        self.assertEqual(second.pipeline_preset, "Только вейвлет")

    def test_preset_does_not_overwrite_output_formats(self):
        task = ProcessingTask()
        task.output_wavelet_numpy = True
        task.output_wavelet_image = False

        task.apply_pipeline_preset("Полный 1D-анализ")

        self.assertTrue(task.output_wavelet_numpy)
        self.assertFalse(task.output_wavelet_image)

    def test_analysis_mode_does_not_replace_loaded_source(self):
        task = ProcessingTask()
        original_image = object()
        channel_data = [object(), object(), object()]
        channel_copy = [object(), object(), object()]
        task.image_path = "sample.png"
        task.original_image = original_image
        task.data = channel_data
        task.data_copy = channel_copy

        task.set_analysis_mode("2d")

        self.assertEqual(task.image_path, "sample.png")
        self.assertIs(task.original_image, original_image)
        self.assertIs(task.data, channel_data)
        self.assertIs(task.data_copy, channel_copy)

    def test_tasks_keep_independent_sources_and_modes(self):
        first = ProcessingTask()
        second = ProcessingTask()
        first.image_path = "first.png"
        second.image_path = "second.png"

        second.set_analysis_mode("2d")

        self.assertEqual(first.image_path, "first.png")
        self.assertEqual(first.analysis_mode, "1d")
        self.assertEqual(second.image_path, "second.png")
        self.assertEqual(second.analysis_mode, "2d")


if __name__ == "__main__":
    unittest.main()
