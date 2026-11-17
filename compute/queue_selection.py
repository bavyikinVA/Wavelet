"""Select only work that is still required in an existing research workspace.

The previous batch journal is a durable record of outcomes; it is *not* proof
that current result files are intact or match the current scientific settings.
We therefore skip an earlier completed task only when both its input/settings
and its requested scientific artifacts can be verified. Inspection hydrates one
staged image at a time, not the entire queue.
"""
from __future__ import annotations

import json
from pathlib import Path

from compute.batch_runner import global_failure
from history.pipeline_state import minimal_execution_plan
from history.source_image import ensure_task_source, release_task_source


TERMINAL_COMPLETED = 'completed'
# Paths change when save_run_image() moves image_path to a new attempt folder.
# Source identity is checked separately by content hash and the CWT signature.
NON_SCIENTIFIC_SETTINGS = {'image_path', 'input_snapshot_sha256'}


def previous_attempts(root):
    """Last durable batch row for each persistent task id; malformed logs ignored."""
    latest = {}
    folder = Path(root or '.') / '.runs'
    if not folder.is_dir():
        return latest
    journals = []
    for path in folder.glob('*/batch.json'):
        try:
            document = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, UnicodeError, ValueError):
            continue
        if document.get('schema_version') != 1 or not isinstance(document.get('tasks'), list):
            continue
        journals.append((str(document.get('created_at') or ''), str(path), document))
    for _, _, document in sorted(journals):
        for row in document['tasks']:
            if isinstance(row, dict) and row.get('task_id'):
                latest[row['task_id']] = row
    return latest


def _settings_match(task, previous):
    saved = previous.get('settings')
    if not isinstance(saved, dict):
        return False
    current = task.settings_snapshot()
    return {
        key: value for key, value in current.items()
        if key not in NON_SCIENTIFIC_SETTINGS
    } == {
        key: value for key, value in saved.items()
        if key not in NON_SCIENTIFIC_SETTINGS
    }


def _ml_outputs_complete(task, previous, control):
    """Proof of a completed integrated ML plan without repeating clustering.

    An ML plan may expand into multiple operations. Its prior batch summary is
    used as the expected number, and *all* referenced result files are checked
    against hashes from the operation journal. Unknown/legacy ML state is not
    assumed complete.
    """
    if not getattr(task, 'ml_plan', None):
        return True
    summary = previous.get('ml_summary') or {}
    try:
        expected = int(summary.get('completed', 0))
        failed = int(summary.get('failed', 0))
    except (ValueError, TypeError):
        return False
    if expected <= 0 or failed:
        return False
    path = Path(task.task_folder_path) / '.ml_operations.json'
    try:
        document = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, ValueError):
        return False
    if not isinstance(document, dict):
        return False
    completed = 0
    from ml.operation_state import file_digest
    for entry in document.values():
        if control is not None:
            control.check()
        if not isinstance(entry, dict) or entry.get('status') != 'completed':
            continue
        artifacts = entry.get('artifacts')
        result = entry.get('result')
        if not isinstance(artifacts, dict) or not isinstance(result, dict):
            continue
        try:
            if all(file_digest(result[key], control.check if control else lambda: None) == artifacts[key]
                   for key in ('table_path', 'summary_path', 'feature_plot_path')):
                completed += 1
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return completed >= expected


def _complete_2d_coefficients(task, control):
    """Validate every complex Morlet output for a previously completed 2D task.

    2D is not yet part of the 1D stage-state DAG; do not rely on it for reuse.
    Verify exactly the expected channel/scale/orientation files instead.
    """
    import numpy as np
    from result_naming import cwt2d_stem, scale_folder_name
    image = getattr(task, 'original_image', None)
    if image is None:
        return False
    expected_shape = tuple(image.shape[:2])
    channels = task.analysis_channel_codes_for_settings()
    scales = getattr(task, 'scales', [])
    angles = getattr(task, 'orientations', [])
    if not channels or not len(scales) or not angles:
        return False
    folder = Path(task.task_folder_path) / 'scales'
    for channel in channels:
        for scale in scales:
            for angle in angles:
                if control is not None:
                    control.check()
                path = (folder / scale_folder_name(scale) / 'wavelets'
                        / (cwt2d_stem(channel, scale, angle) + '.npy'))
                try:
                    values = np.load(path, mmap_mode='r', allow_pickle=False)
                    if values.shape != expected_shape or values.dtype.kind != 'c':
                        return False
                    for start in range(0, len(values), 256):
                        if control is not None:
                            control.check()
                        if not np.isfinite(values[start:start + 256]).all():
                            return False
                except (OSError, ValueError, EOFError):
                    return False
                finally:
                    if 'values' in locals() and isinstance(values, np.memmap):
                        values._mmap.close()
                    values = None
    return True


def select_pending_tasks(tasks, root, control=None, progress=None):
    """Return (to_execute, previously_ready) without loading all images at once.

    Conservative rules: new/failed/cancelled/interrupted/modified tasks run.
    No automatic science rerun is attempted merely because a previous batch
    exists. A previously complete task can be skipped only if its stored
    provenance and all requested stage artifacts still validate.
    """
    history = previous_attempts(root)
    if not history:
        return tuple(tasks), ()

    pending, ready = [], []
    total = len(tasks)
    for index, task in enumerate(tasks, 1):
        if control is not None:
            control.check()
        if progress is not None:
            progress(index, total, task)
        row = history.get(getattr(task, 'persistent_id', None))
        if (row is None or row.get('status') != TERMINAL_COMPLETED
                or not getattr(task, 'reuse_existing_results', True)
                or not getattr(task, 'task_folder_path', None)
                or not Path(task.task_folder_path).is_dir()
                or row.get('history_pending')
                or not _settings_match(task, row)):
            pending.append(task)
            continue
        expected = row.get('input_sha256')
        current_hash = getattr(task, '_input_snapshot_sha256', None)
        if not expected or not current_hash or expected != current_hash:
            pending.append(task)
            continue
        try:
            # ensure_task_source returns early when the GUI currently holds
            # decoded pixels. Validate its backing snapshot nonetheless.
            if getattr(task, 'original_image', None) is not None:
                from history.attempts import digest
                path = Path(getattr(task, '_input_snapshot_path', '') or '')
                if not path.is_file() or digest(
                        path, control.check if control else lambda: None) != current_hash:
                    pending.append(task)
                    continue
            ensure_task_source(task)
            science_incomplete = (
                not _complete_2d_coefficients(task, control)
                if getattr(task, 'analysis_mode', '1d') == '2d'
                else bool(minimal_execution_plan(task)))
            if science_incomplete or not _ml_outputs_complete(task, row, control):
                pending.append(task)
            else:
                ready.append(task)
        except Exception as error:
            if global_failure(error):
                raise
            # A corrupt source/result must be reported by the normal batch
            # journal instead of stopping selection of another 100 tasks.
            pending.append(task)
        finally:
            release_task_source(task)
    return tuple(pending), tuple(ready)
