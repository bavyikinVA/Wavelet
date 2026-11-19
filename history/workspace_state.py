"""Self-contained, atomically written workspace metadata independent of any DB.

Each experiment has a single task index and one JSON document per task. Scientific
artifacts, immutable attempts and staged inputs keep their existing formats.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from history.atomic_io import atomic_write_text

SCHEMA_VERSION = 1


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def _read(path):
    document = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(document, dict) or document.get('schema_version') != SCHEMA_VERSION:
        raise ValueError(f'Повреждён или неподдерживаемый файл состояния: {path}')
    return document


def _write(path, document):
    atomic_write_text(path, json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False))


def _portable_path(root, value):
    if not value:
        return ''
    path = Path(value)
    if path.is_absolute():
        try:
            return str(path.relative_to(root))
        except ValueError:
            return str(path)
    return str(path)


def _absolute_path(root, value):
    if not value:
        return ''
    path = Path(value)
    return str(path if path.is_absolute() else root / path)


class WorkspaceState:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.base = self.root / '.workspace'
        self.index_path = self.base / 'experiment.json'
        self.tasks_path = self.base / 'tasks'

    @property
    def exists(self):
        return self.index_path.is_file()

    def ensure(self, experiment_id, name, *, bootstrap_complete=None, created_at=None, note=None):
        if not experiment_id or not self.root.is_dir():
            raise ValueError('Не найден каталог исследования или ID')
        if self.exists:
            index = _read(self.index_path)
            if index.get('id') != experiment_id:
                raise ValueError('Каталог принадлежит другому исследованию')
        else:
            index = dict(schema_version=SCHEMA_VERSION, id=experiment_id,
                name=name, note=note or '', created_at=created_at or timestamp(), task_ids=[], deleted_task_ids=[],
                bootstrap_complete=(bootstrap_complete is not False))
        new_name = name or index['name']
        new_note = index.get('note', '') if note is None else note
        new_complete = index.get('bootstrap_complete', True) if bootstrap_complete is None else bool(bootstrap_complete)
        update_creation = bool(created_at and index.get('created_at') != created_at)
        if (index['name'] != new_name or index.get('note', '') != new_note
                or index.get('bootstrap_complete', True) != new_complete
                or update_creation or not self.index_path.exists()):
            if update_creation:
                index['created_at'] = created_at
            index['name'] = new_name
            index['note'] = new_note
            index['bootstrap_complete'] = new_complete
            index['updated_at'] = timestamp()
            _write(self.index_path, index)
        return index

    def update_metadata(self, experiment_id, name, note):
        from history.metadata import normalize_metadata
        name, note = normalize_metadata(name, note)
        current = _read(self.index_path)
        if current['id'] != experiment_id:
            raise ValueError('Каталог принадлежит другому исследованию')
        if current['name'] != name or current.get('note', '') != note:
            current['name'], current['note'] = name, note
            current['updated_at'] = timestamp()
            _write(self.index_path, current)
        return current

    def update_task_metadata(self, experiment_id, task_id, name, note):
        from history.metadata import normalize_metadata
        name, note = normalize_metadata(name, note)
        index = _read(self.index_path)
        if index['id'] != experiment_id or task_id not in index['task_ids']:
            raise KeyError('Задача не принадлежит исследованию')
        path = self.tasks_path / (task_id + '.json')
        row = _read(path)
        if row['id'] != task_id:
            raise ValueError('Несовпадение идентификатора задачи')
        if row['name'] != name or row.get('note', '') != note:
            row['name'], row['note'] = name, note
            row['updated_at'] = timestamp()
            _write(path, row)
        return row

    def align_created_at(self, experiment_id, recorded_date):
        """Preserve the original research creation date while bootstrapping ORM history."""
        if not recorded_date or not self.exists:
            return
        index = _read(self.index_path)
        if index.get('id') != experiment_id:
            raise ValueError('Каталог принадлежит другому исследованию')
        if index.get('created_at') != recorded_date:
            index['created_at'] = recorded_date
            _write(self.index_path, index)

    @property
    def bootstrap_complete(self):
        return self.exists and _read(self.index_path).get('bootstrap_complete', True)

    def mark_bootstrap_complete(self):
        index = _read(self.index_path)
        if index.get('bootstrap_complete') is False:
            index['bootstrap_complete'] = True
            index['updated_at'] = timestamp()
            _write(self.index_path, index)

    def save_task_row(self, experiment_id, name, row):
        """Save a database-compatible row (including serialized settings JSON)."""
        if not row.get('id'):
            raise ValueError('У задачи отсутствует постоянный ID')
        index = self.ensure(experiment_id, name)
        task_id = str(row['id'])
        if task_id in index.get('deleted_task_ids', []):
            raise ValueError('Нельзя восстановить удалённый идентификатор задачи')
        record = dict(schema_version=SCHEMA_VERSION, id=task_id,
            experiment_id=experiment_id, task_number=int(row['task_number']),
            name=row['name'], note=row.get('note', '') or '',
            folder_path=_portable_path(self.root, row.get('folder_path', '')),
            settings_json=row['settings_json'], updated_at=timestamp())
        # Save first, then update the root index. No write if nothing changed.
        target = self.tasks_path / (task_id + '.json')
        previous = _read(target) if target.exists() else None
        fields = ('experiment_id', 'task_number', 'name', 'note', 'folder_path', 'settings_json')
        if previous is None or any(previous[k] != record[k] for k in fields):
            _write(target, record)
        if task_id not in index['task_ids']:
            index['task_ids'].append(task_id)
            index['updated_at'] = timestamp()
            _write(self.index_path, index)
        return task_id

    def _task_number(self, task_id):
        return int(_read(self.tasks_path / (task_id + '.json'))['task_number'])

    def save_task(self, experiment_id, name, task):
        if not getattr(task, 'persistent_id', None):
            task.persistent_id = uuid4().hex
        return self.save_task_row(experiment_id, name, dict(
            id=task.persistent_id, task_number=int(task.task_id),
            name=task.task_name, note=getattr(task, 'note', '') or '',
            folder_path=task.task_folder_path or '',
            settings_json=json.dumps(task.settings_snapshot(), ensure_ascii=False, allow_nan=False)))

    def remove_task(self, experiment_id, task_id, remaining_tasks):
        index = _read(self.index_path)
        if index['id'] != experiment_id:
            raise ValueError('Несовпадение ID исследования')
        # A tombstone prevents a stale orphan file being recovered after a crash.
        index['deleted_task_ids'] = list(dict.fromkeys([*index.get('deleted_task_ids', []), task_id]))
        index['task_ids'] = [getattr(task,'persistent_id') for task in remaining_tasks]
        for task in remaining_tasks:
            if not task.persistent_id:
                raise ValueError('У оставшейся задачи отсутствует ID')
            row = _read(self.tasks_path / (task.persistent_id + '.json'))
            row['task_number'] = int(task.task_id)
            row['name'] = task.task_name
            row['note'] = getattr(task, 'note', '') or ''
            row['updated_at'] = timestamp()
            _write(self.tasks_path / (task.persistent_id + '.json'), row)
        index['updated_at'] = timestamp()
        _write(self.index_path, index)

    def open(self):
        index = _read(self.index_path)
        if index.get('bootstrap_complete') is False:
            raise ValueError('Импорт исследования из старой SQLite был прерван; '
                'для завершения требуется исходная база истории')
        if not isinstance(index.get('task_ids'), list):
            raise ValueError('Повреждён список задач исследования')
        ids = index['task_ids']
        if len(ids) != len(set(ids)):
            raise ValueError('Повторяющиеся ID в индексе исследования')
        excluded = set(index.get('deleted_task_ids', []))
        # Recover orphan task saved before an interrupted index update, but never
        # resurrect intentionally deleted tasks.
        recovered = []
        for path in self.tasks_path.glob('*.json') if self.tasks_path.exists() else ():
            if path.stem in ids or path.stem in excluded:
                continue
            row = _read(path)
            if row.get('experiment_id') == index['id'] and row.get('id') == path.stem:
                recovered.append(path.stem)
        if recovered:
            ids = [*ids, *recovered]
            ids.sort(key=self._task_number)
            index['task_ids'] = ids
            index['updated_at'] = timestamp()
            _write(self.index_path, index)
        rows = []
        for task_id in ids:
            row = _read(self.tasks_path / (task_id + '.json'))
            if row.get('id') != task_id or row.get('experiment_id') != index['id']:
                raise ValueError('Неконсистентный JSON задачи')
            row['folder_path'] = _absolute_path(self.root, row.get('folder_path', ''))
            data = {key:row[key] for key in ('id','experiment_id','task_number','name',
                'folder_path','settings_json','updated_at')}
            data['note'] = row.get('note', '') or ''
            rows.append(data)
        rows.sort(key=lambda r:r['task_number'])
        record = dict(id=index['id'], name=index['name'], root_path=str(self.root),
            created_at=index['created_at'], updated_at=index['updated_at'], note=index.get('note', ''),
            task_count=len(rows), run_count=0, ml_count=0,
            total_duration_seconds=None)
        return record, rows
