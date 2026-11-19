"""Availability checks for the optional, independent ROI point pipeline.

Normal full-image checks stay in pipeline_resume/artifact_validation. ROI results
are stored per region and must never be mistaken for legacy point caches.
"""
from __future__ import annotations

import gzip
import pickle
from pathlib import Path

from history.checkpoints import CHECKPOINT_VERSION, existing_checkpoint_path
from result_naming import point_stem, scale_folder_name
from compute.knn.result_format import is_disk_knn_payload


def roi_stage_available(task, stage: str) -> bool:
    if not task.has_roi or not task.task_folder_path:
        return False
    root = Path(task.task_folder_path)
    if not (root / 'roi' / 'regions.json').is_file():
        return False
    metadata = task.roi_descriptors()
    if not metadata or task.roi_fingerprint() is None:
        return False

    if stage in ('extrema', 'envelopes'):
        axes = [axis for axis, enabled in (('row', task.process_rows),
                 ('col', task.process_columns)) if enabled]
        codes = list(task.analysis_channel_codes) or list(task.analysis_channel_codes_for_settings())
        if not axes or not codes or not len(task.scales):
            return False
        artifact = 'ext' if stage == 'extrema' else 'env'
        folder = 'extremes' if stage == 'extrema' else 'envelopes'
        kinds = (('max', bool(task.find_maxima)), ('min', bool(task.find_minima))) if stage == 'extrema' else (
            ('upper', bool(task.find_maxima)), ('lower', bool(task.find_minima)))
        for region in metadata:
            for axis in axes:
                for feature_axis in ('row', 'col'):
                    for channel in codes:
                        for scale in task.scales:
                            region_root = (root / 'roi' / f"region_{int(region['id']):04d}" /
                                           'scales' / scale_folder_name(scale) / folder)
                            for kind, enabled in kinds:
                                if not enabled:
                                    continue
                                result = region_root / (point_stem(artifact, axis, feature_axis,
                                                           channel, scale, kind) + '.npz')
                                if not result.is_file() or result.stat().st_size == 0:
                                    return False
        return True

    if stage in ('knn', 'statistics'):
        checkpoint = Path(existing_checkpoint_path(task.task_folder_path))
        if not checkpoint.is_file():
            return False
        try:
            with gzip.open(checkpoint, 'rb') as stream:
                payload = pickle.load(stream)
        except (OSError, EOFError, ValueError, TypeError, pickle.UnpicklingError):
            return False
        if (not isinstance(payload, dict) or payload.get('version') != CHECKPOINT_VERSION
                or payload.get('roi_sha256') != task.roi_fingerprint()):
            return False
        if stage == 'statistics':
            return (root / 'roi' / 'statistics.csv').is_file()
        values = payload.get('knn_results')
        if not isinstance(values, dict):
            return False
        valid_ids = {int(r['id']) for r in metadata}
        for key, item in values.items():
            if not (isinstance(key, tuple) and len(key) == 4
                    and '@roi' in str(key[1])):
                return False
            region_text = str(key[1]).rsplit('@roi', 1)[-1]
            if not region_text.isdecimal() or int(region_text) not in valid_ids:
                return False
            if is_disk_knn_payload(item) and not Path(item['path']).is_file():
                return False
        return True
    return False
