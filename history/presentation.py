"""History presentation helpers shared by SQLite/PostgreSQL/offline workspaces.

History in the UI is ordered by the date of the *research*, not by the time
its SQLite index happened to be edited or synchronized.
"""
from datetime import datetime, timezone


def _instant(value):
    """Comparable UTC instant; accepts legacy timestamps with no timezone."""
    if not value:
        return float('-inf')
    try:
        moment = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    except (ValueError, TypeError):
        return float('-inf')
    if moment.tzinfo is None:
        # Old history may have local naive values; preserving textual order
        # is more useful than interpreting them in the user's current TZ.
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).timestamp()


def sort_researches(records, order='Сначала новые'):
    """Stable chronological order, deterministic even for equal dates."""
    column = 'updated_at' if order == 'Недавно изменённые' else 'created_at'
    reverse = order != 'Сначала старые'
    return sorted(records, key=lambda record: (
        _instant(record.get(column)), str(record.get('id', ''))), reverse=reverse)


def sort_runs(records):
    return sorted(records, key=lambda record: (
        _instant(record.get('created_at')), int(record.get('id') or 0)), reverse=True)


def format_date(value):
    if not value:
        return 'дата неизвестна'
    try:
        moment = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    except (TypeError, ValueError):
        return str(value).replace('T', ' ')
    return moment.strftime('%d.%m.%Y %H:%M')


def display_status(status):
    return {
        'completed': 'Готово', 'failed': 'Ошибка', 'partial_failed': 'Частично',
        'cancelled': 'Отменено', 'interrupted': 'Прервано',
        'running': 'Выполняется', 'pending': 'Ожидает',
    }.get(status, str(status or 'Неизвестно'))


def group_runs_by_task(runs):
    result = {}
    for run in sort_runs(runs):
        result.setdefault(run.get('task_id'), []).append(run)
    return result


def history_roi_summary(snapshot):
    """Display the ROI count recorded by a task/run, not the current mask files.

    Snapshots predate ROI support; those should not acquire a misleading label.
    The scientific pipeline performs CWT on the full image and filters point
    processing to separately selected ROI masks.
    """
    if not isinstance(snapshot, dict):
        return None
    regions = snapshot.get('roi_regions')
    if isinstance(regions, list):
        count = len(regions)
    else:
        count = 0
    if not (count and snapshot.get('roi_enabled', True)):
        return None
    suffix = ('область' if count % 10 == 1 and count % 100 != 11 else
              'области' if count % 10 in (2, 3, 4) and count % 100 not in (12, 13, 14)
              else 'областей')
    return (f'{count} {suffix} · CWT по всему изображению; '
            'экстремумы, огибающие и KNN — отдельно по ROI')


def history_snapshot_changes(task_meta, run_meta, *, task_image=None, run_image=None):
    """Only historical settings differing from the task's current snapshot.

    When task settings are unavailable, show the full run snapshot instead.
    Returned fields are display labels and values, independent of any ORM.
    """
    if not isinstance(run_meta, dict):
        return []
    reference = task_meta if isinstance(task_meta, dict) else None
    fields = (
        ('Преобразование', 'wavelet'),
        ('Масштабы', 'scales'),
        ('План этапов', 'stages'),
        ('Каналы и параметры', 'nuances'),
    )
    changes = [
        (label, run_meta[key])
        for label, key in fields
        if run_meta.get(key) not in (None, '')
        and (reference is None or run_meta.get(key) != reference.get(key))
    ]
    # A run can have different ROI annotations than the task currently saved.
    # Show even a change back to full-image processing when it is relevant.
    run_roi = run_meta.get('roi')
    task_roi = reference.get('roi') if reference is not None else None
    if run_roi != task_roi and (run_roi or task_roi):
        changes.append(('Области ROI', run_roi or 'Не использовались'))
    if run_image and (reference is None or run_image != task_image):
        from pathlib import Path
        # Support Windows paths even when testing on another platform.
        from ntpath import basename as win_basename
        changes.append(('Изображение', win_basename(str(run_image)) or Path(run_image).name))
    return changes
