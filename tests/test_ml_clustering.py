import os
import tempfile
import unittest

import numpy as np

from ml.clustering import ClusteringError, extract_knn_features, run_clustering


def make_knn_results():
    points = np.asarray([
        [5, 5], [6, 5], [5, 6], [6, 6],
        [50, 50], [51, 50], [50, 51], [51, 51],
    ], dtype=float)
    neighbors = {}
    for index in range(len(points)):
        group_start = 0 if index < 4 else 4
        indices = [value for value in range(group_start, group_start + 4) if value != index]
        deltas = points[indices] - points[index]
        neighbors[index] = {
            "indices": indices,
            "distances": np.linalg.norm(deltas, axis=1).tolist(),
            "angles": [float((index * 11 + offset * 7) % 360) for offset in range(3)],
        }
    return {
        ("red", 10.0): {
            "max_by_row": {"points": points, "neighbors": neighbors}
        }
    }


class MLClusteringTests(unittest.TestCase):
    def test_extracts_features_without_intermediate_text_files(self):
        table = extract_knn_features(
            make_knn_results(), point_filter="max_by_row",
            feature_set="coordinates_knn"
        )
        self.assertEqual(table.values.shape[0], 8)
        self.assertIn("distance_mean", table.feature_names)
        self.assertIn("x", table.feature_names)

    def test_kmeans_saves_table_and_two_visualizations(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_clustering(
                make_knn_results(), algorithm="kmeans", output_root=directory,
                point_filter="max_by_row", feature_set="coordinates_knn",
                n_clusters=2, image=np.zeros((60, 60, 3), dtype=np.uint8),
            )
            self.assertEqual(result["metrics"]["cluster_count"], 2)
            self.assertTrue(os.path.isfile(result["table_path"]))
            self.assertTrue(os.path.isfile(result["feature_plot_path"]))
            self.assertTrue(os.path.isfile(result["image_plot_path"]))
            with open(result["table_path"], encoding="utf-8-sig") as stream:
                self.assertIn("Кластер", stream.readline())

    def test_dbscan_detects_dense_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_clustering(
                make_knn_results(), algorithm="dbscan", output_root=directory,
                point_filter="max_by_row", feature_set="coordinates_knn",
                standardize=False, eps=3.0, min_samples=2,
            )
            self.assertEqual(result["metrics"]["cluster_count"], 2)
            self.assertEqual(result["metrics"]["noise_count"], 0)

    def test_reports_missing_selected_point_type(self):
        with self.assertRaises(ClusteringError):
            extract_knn_features(
                make_knn_results(), point_filter="min_by_row",
                feature_set="knn"
            )


if __name__ == "__main__":
    unittest.main()
