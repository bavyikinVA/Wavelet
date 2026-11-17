"""Readable, bounded-memory presentation of durable batch outcomes.

No traceback, hashes or raw JSON are shown by default; the original batch.json
and offline forensic report remain available to researchers for auditing.
"""
from __future__ import annotations

STATUS = {
    'completed': 'Успешно',
    'partial_failed': 'Частично',
    'failed': 'Ошибка',
    'cancelled': 'Отменено',
    'interrupted': 'Прервано',
    'pending': 'Не выполнено',
    'running': 'Выполняется',
    'saving': 'Сохранение',
}


def _stages(settings):
    if settings.get('analysis_mode') == '2d':
        return '2D-вейвлет Морле'
    stages = ['Вейвлеты']
    knn = bool(settings.get('calculate_knn'))
    stats = bool(settings.get('calculate_statistics'))
    sync = bool(settings.get('calculate_synchronization'))
    env = bool(settings.get('calculate_envelopes')) or knn or stats or sync
    ext = bool(settings.get('calculate_extrema')) or env
    if ext:
        stages.append('экстремумы')
    if env:
        stages.append('огибающие')
    if knn:
        stages.append('KNN')
    if stats:
        stages.append('статистики')
    if sync:
        stages.append('синхронизация')
    if settings.get('ml_plan'):
        stages.append('ML')
    return ', '.join(stages)


def _parameters(settings):
    elements = []
    scales = settings.get('scales') or []
    if scales:
        shown = ', '.join(f'{float(x):g}' for x in scales[:5])
        if len(scales) > 5:
            shown += f' и ещё {len(scales)-5}'
        elements.append('Масштабы: ' + shown)
    if settings.get('calculate_knn'):
        elements.append('KNN: ' + str(settings.get('k_neighbors', 5)) + ' соседей')
    channels = settings.get('channel_representation', 'rgb')
    if channels == 'rgb':
        if settings.get('rgb_channel_mode') == 'single':
            elements.append('Канал: ' + str(settings.get('rgb_single_channel', 'R')))
        else:
            elements.append('Каналы: RGB ×3')
    elif channels == 'grayscale':
        elements.append('Каналы: оттенки серого')
    else:
        elements.append('Каналы: Грамм–Шмидт')
    return ' · '.join(elements)


def _reason(row):
    error = (row.get('error') or row.get('preparation_error') or
             row.get('provenance_error') or '').strip()
    if error:
        return error
    if row.get('status') == 'partial_failed':
        failures = (row.get('ml_summary') or {}).get('failed', 0)
        if failures:
            return f'Ошибок ML-операций: {failures}'
    return ''


def readable_rows(journal):
    return [dict(
        name=row.get('task_name') or f"Задача {row.get('task_number', '?')}",
        stages=_stages(row.get('settings') or {}),
        parameters=_parameters(row.get('settings') or {}),
        status=row.get('status', 'pending'),
        status_text=STATUS.get(row.get('status'), 'Не выполнено'),
        error=_reason(row),
        history_pending=bool(row.get('history_pending')),
    ) for row in journal.document.get('tasks', [])]


def readable_summary(rows, skipped=0):
    success = sum(row['status'] == 'completed' for row in rows)
    partial = sum(row['status'] == 'partial_failed' for row in rows)
    failed = sum(row['status'] == 'failed' for row in rows)
    unfinished = len(rows) - success - partial - failed
    parts = [f'Всего: {len(rows)}', f'Успешно: {success}']
    if partial:
        parts.append(f'Частично: {partial}')
    if failed:
        parts.append(f'С ошибкой: {failed}')
    if unfinished:
        parts.append(f'Не завершено: {unfinished}')
    if skipped:
        parts.append(f'Готовых пропущено: {skipped}')
    if any(row['history_pending'] for row in rows):
        parts.append('История ожидает сохранения')
    return ' · '.join(parts)
