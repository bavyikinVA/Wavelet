"""Read-only validation of persisted feature stages; no GUI or ML imports."""
import gzip
import pickle
import zipfile
import zlib

import numpy as np

from history.checkpoints import CHECKPOINT_VERSION, existing_checkpoint_path
from history.pipeline_resume import load_point_cache
from compute.knn.result_format import materialize_knn_payload, neighbor_arrays


def feature_stage_available(task, stage):
    try:
        with gzip.open(existing_checkpoint_path(task.task_folder_path), 'rb') as stream:
            payload = pickle.load(stream)
        if not isinstance(payload, dict) or payload.get('version') != CHECKPOINT_VERSION:
            return False
        data = payload.get(stage + '_results')
        if not isinstance(data, dict):
            return False
        axes = [axis for axis, enabled in (('row', task.process_rows), ('col', task.process_columns)) if enabled]
        channels = list(task.analysis_channel_codes) or list(task.analysis_channel_codes_for_settings())
        if not axes or not channels or not len(task.scales):
            return False
        if stage == 'statistics':
            for axis in axes:
                for channel in channels:
                    for feature in ('row', 'col'):
                        item = data.get((axis, feature, channel), {})
                        scales = np.asarray(item.get('scales', []))
                        counts = np.asarray(item.get('upper_envelope_maxima_counts', []))
                        if (not np.array_equal(scales, task.scales) or counts.shape != scales.shape
                                or not np.isfinite(counts).all() or (counts < 0).any()):
                            return False
                        blocks = item.get('scale_blocks', {})
                        for size in task.scale_block_sizes:
                            if int(size) > 1 and int(size) not in blocks:
                                return False
            return True
        if stage != 'knn':
            return False
        expected = set()
        for axis in axes:
            for channel in channels:
                for scale in task.scales:
                    for feature in ('row', 'col'):
                        for enabled, kind, label in ((task.find_maxima, 'upper', 'max'), (task.find_minima, 'lower', 'min')):
                            if not enabled:
                                continue
                            points = load_point_cache(task.task_folder_path, 'env', axis, feature, channel, scale, kind, as_array=True)
                            if points is None:
                                return False
                            if len(points) < 2:
                                continue  # KNN intentionally omits these groups.
                            key = (axis, channel, float(scale), label + '_by_' + ('row' if feature == 'row' else 'column'))
                            expected.add(key)
                            group = materialize_knn_payload(data[key])
                            if not np.array_equal(group['points'], points):
                                return False
                            indices, distances, angles = neighbor_arrays(group['neighbors'], point_count=len(points))
                            if (indices.ndim != 2 or indices.shape != distances.shape or indices.shape[0] != len(points)
                                    or indices.shape[1] != min(task.k_neighbors, len(points) - 1)):
                                return False
                            valid = indices >= 0
                            if (indices[valid] >= len(points)).any() or not np.isfinite(distances[valid]).all() or (distances[valid] < 0).any():
                                return False
                            if angles is None or angles.shape != indices.shape or not np.isfinite(angles[valid]).all():
                                return False
        return set(data) == expected
    except (OSError, EOFError, ValueError, TypeError, KeyError, AttributeError, pickle.UnpicklingError, zipfile.BadZipFile, zlib.error):
        return False
