"""Resilient research-history facade: files are authoritative, ORM is an index.

No change to scientific output formats. Database failures are visible and do not
prevent saving a workspace; durable attempt delivery still reports errors so
batch.json / MLHistoryOutbox do not falsely acknowledge registrations.
"""
import json
from pathlib import Path
from threading import RLock
from types import SimpleNamespace

from sqlalchemy.exc import SQLAlchemyError

from history.atomic_io import atomic_write_text
from history.store import RunHistoryStore, _default_database_path
from history.workspace_state import WorkspaceState, _read
from history.presentation import sort_researches


class ResilientHistoryService:
    def __init__(self, database_path=None, *, database_url=None,
                 db_store_factory=RunHistoryStore, registry_path=None):
        self._database_path, self._database_url = database_path, database_url
        self._factory = db_store_factory
        self._db = None
        self._error = None
        self._roots = {}
        self._ensured = {}
        self._lock = RLock()
        self._registry_error = None
        self._sync_errors = {}
        # The locator stores only workspace paths, never task data, secrets or
        # scientific results. Stub stores in tests do not write to the user's home.
        if registry_path is None and (db_store_factory is RunHistoryStore or database_path):
            registry_path = Path(database_path or _default_database_path()).parent / 'workspaces.json'
        self._registry_path = Path(registry_path) if registry_path is not None else None
        self._load_registry()
        self.reconnect(replay=False)

    def _load_registry(self):
        path = self._registry_path
        if path is None or not path.exists():
            return
        try:
            document = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(document, dict) or document.get('schema_version') != 1 or not isinstance(document.get('roots'), list):
                raise ValueError('Некорректный реестр папок исследований')
        except (OSError, TypeError, ValueError) as error:
            # Preserve the corrupt locator; explicit folder opening still works.
            self._registry_error = error
            return
        for root in document['roots']:
            try:
                state = WorkspaceState(root)
                if not state.exists:
                    continue
                index = json.loads(state.index_path.read_text(encoding='utf-8'))
                if index.get('schema_version') != 1 or not index.get('id'):
                    raise ValueError('Некорректный индекс исследования')
                self._roots[index['id']] = str(state.root)
            except (OSError, TypeError, ValueError) as error:
                self._sync_errors[str(root)] = str(error)

    def _save_registry(self):
        if self._registry_path is None or self._registry_error is not None:
            return
        payload = {'schema_version': 1, 'roots': sorted(set(self._roots.values()))}
        try:
            atomic_write_text(self._registry_path,
                json.dumps(payload, ensure_ascii=False, indent=2))
        except OSError as error:
            # A broken user-level locator must not mask a successful scientific
            # workspace write. Explicit 'Open from folder' remains possible.
            self._registry_error = error

    def _remember_root(self, experiment_id, root):
        root = str(Path(root).resolve())
        if self._roots.get(experiment_id) != root:
            self._roots[experiment_id] = root
            self._save_registry()

    @property
    def available(self):
        return self._db is not None and self._error is None

    @property
    def last_error(self):
        return str(self._error) if self._error else None

    def _mark_unavailable(self, error):
        # The underlying store may still hold a connection pool after a failed
        # query; the next explicit reconnect builds a fresh engine.
        self._error = error
        self._db = None

    def _try_db(self, method, *args, **kwargs):
        if not self.available:
            return None
        try:
            return getattr(self._db, method)(*args, **kwargs)
        except (SQLAlchemyError, OSError, ConnectionError, TimeoutError) as error:
            self._mark_unavailable(error)
            return None

    def reconnect(self, *, replay=True):
        """Explicit recovery; returning True means DB is usable, not merely configured."""
        with self._lock:
            try:
                if self._database_url is not None:
                    db = self._factory(database_url=self._database_url)
                elif self._database_path is not None:
                    db = self._factory(self._database_path)
                else:
                    db = self._factory()
            except Exception as error:
                self._mark_unavailable(error)
                return False
            old = self._db
            self._db, self._error = db, None
            self._ensured.clear()
            if old is not None and old is not db and hasattr(old, 'close'):
                old.close()
            if replay:
                self.sync_known_workspaces()
            return self.available

    def bootstrap_legacy_workspaces(self):
        """Create portable task metadata for studies saved before P0-2.

        Runs in a worker, after GUI startup; never overwrites an existing
        workspace index, never decodes images or touches scientific outputs.
        """
        if not self.available:
            return 0
        created = 0
        registry_changed = False
        for record in self._db.list_experiments(limit=10**9):
            root = Path(record['root_path'])
            if not root.is_dir():
                continue
            resolved = str(root.resolve())
            if self._roots.get(record['id']) != resolved:
                self._roots[record['id']] = resolved
                registry_changed = True
            state = WorkspaceState(root)
            if state.exists and state.bootstrap_complete:
                try:
                    # Carry forward legacy experiment notes only when the file
                    # pre-dates the new metadata field (do not override edits).
                    if 'note' not in _read(state.index_path) and record.get('note'):
                        state.update_metadata(record['id'], record['name'], record['note'])
                    state.align_created_at(record['id'], record['created_at'])
                except (OSError, ValueError, TypeError) as error:
                    self._sync_errors[str(root)] = str(error)
                continue
            try:
                self.prepare_restore(record['id'])
            except (OSError, ValueError, TypeError) as error:
                self._sync_errors[str(root)] = str(error)
            else:
                created += 1
        if registry_changed:
            self._save_registry()
        return created

    def sync_known_workspaces(self):
        """Rebuild a usable DB index without hiding corrupt on-disk studies."""
        self._sync_errors.clear()
        for root in list(self._roots.values()):
            if not WorkspaceState(root).exists:
                continue
            try:
                self.sync_workspace(root)
            except (OSError, ValueError, TypeError, RuntimeError) as error:
                self._sync_errors[root] = str(error)
            except SQLAlchemyError as error:
                self._mark_unavailable(error)
                break
        return self.available

    @property
    def sync_errors(self):
        return dict(self._sync_errors)

    def open_workspace_folder(self, root):
        record, rows = WorkspaceState(root).open()
        self._remember_root(record['id'], record['root_path'])
        return record, rows

    def ensure_experiment(self, experiment_id, root_path, name=None):
        with self._lock:
            state = WorkspaceState(root_path)
            if not name and state.exists:
                name = _read(state.index_path).get('name')
            name = name or f'Исследование {experiment_id[:8]}'
            # Mandatory filesystem write; no database exception can undo it.
            state.ensure(experiment_id, name)
            self._remember_root(experiment_id, root_path)
            signature = (str(Path(root_path).resolve()), name)
            if self._ensured.get(experiment_id) != signature and self.available:
                self._try_db('ensure_experiment', experiment_id, root_path, name)
                if self.available:
                    self._ensured[experiment_id] = signature

    def update_experiment_metadata(self, experiment_id, name, note):
        """File first, ORM second; successful edits survive database outages."""
        from history.metadata import normalize_metadata
        name, note = normalize_metadata(name, note)
        with self._lock:
            root = self._roots.get(experiment_id)
            if not root:
                raise KeyError('Каталог исследования неизвестен')
            WorkspaceState(root).update_metadata(experiment_id, name, note)
            self._ensured.pop(experiment_id, None)
            self._try_db('update_experiment_metadata', experiment_id, name, note)
        return name, note

    def update_task_metadata(self, experiment_id, task_id, name, note):
        """Rename/annotate a task without reading its image or scientific arrays."""
        from history.metadata import normalize_metadata
        name, note = normalize_metadata(name, note)
        with self._lock:
            root = self._roots.get(experiment_id)
            if not root:
                raise KeyError('Каталог исследования неизвестен')
            WorkspaceState(root).update_task_metadata(experiment_id, task_id, name, note)
            self._try_db('update_task_metadata', experiment_id, task_id, name, note)
        return name, note

    def save_task(self, experiment_id, task):
        with self._lock:
            root = self._roots.get(experiment_id)
            if not root:
                raise ValueError('Сначала создайте файловое исследование')
            state = WorkspaceState(root)
            index = _read(state.index_path)
            if index.get('bootstrap_complete') is False:
                raise ValueError('Незавершённый импорт файлов исследования')
            name = index['name']
            task_id = WorkspaceState(root).save_task(experiment_id, name, task)
            self._try_db('save_task', experiment_id, task)
            return task_id

    def remove_task(self, experiment_id, task_id, remaining_tasks):
        with self._lock:
            if not experiment_id or not task_id:
                return
            WorkspaceState(self._roots[experiment_id]).remove_task(
                experiment_id, task_id, remaining_tasks)
            self._try_db('remove_task', experiment_id, task_id, remaining_tasks)

    def close(self):
        """Release pooled database connections during application shutdown."""
        with self._lock:
            if self._db is not None and hasattr(self._db, 'close'):
                self._db.close()
            self._db = None

    def _required_db(self):
        if not self.available:
            raise RuntimeError(f'База истории недоступна: {self.last_error or "ожидает подключения"}')
        return self._db

    def add_run(self, **kwargs):
        try:
            return self._required_db().add_run(**kwargs)
        except (SQLAlchemyError, OSError, ConnectionError, TimeoutError) as error:
            self._mark_unavailable(error)
            raise

    def find_ml_operation(self, *args, **kwargs):
        try:
            return self._required_db().find_ml_operation(*args, **kwargs)
        except (SQLAlchemyError, OSError, ConnectionError, TimeoutError) as error:
            self._mark_unavailable(error)
            raise

    def list_runs(self, limit=100):
        return self._try_db('list_runs', limit=limit) or []

    def list_experiments(self, limit=100):
        rows = self._try_db('list_experiments', limit=limit)
        if rows is not None:
            changed = False
            for row in rows:
                resolved = str(Path(row['root_path']).resolve())
                if self._roots.get(row['id']) != resolved:
                    self._roots[row['id']] = resolved
                    changed = True
            if changed:
                self._save_registry()
            return sort_researches(rows)
        restored = []
        for root in list(self._roots.values()):
            try:
                restored.append(WorkspaceState(root).open()[0])
            except (OSError, ValueError, TypeError) as error:
                self._sync_errors[str(root)] = str(error)
        return sort_researches(restored)[:limit]

    def get_experiment(self, experiment_id):
        root = self._roots.get(experiment_id)
        if root and WorkspaceState(root).exists:
            return WorkspaceState(root).open()[0]
        row = self._try_db('get_experiment', experiment_id)
        if row is not None:
            self._remember_root(experiment_id, row['root_path'])
            return row
        raise KeyError(f'Исследование {experiment_id} недоступно; откройте папку исследования')

    def list_tasks(self, experiment_id):
        root = self._roots.get(experiment_id)
        if root and WorkspaceState(root).exists:
            return WorkspaceState(root).open()[1]
        rows = self._try_db('list_tasks', experiment_id)
        if rows is None:
            raise KeyError('Для исследования нет доступного файлового описания')
        return rows

    def prepare_restore(self, experiment_id):
        """Convert legacy DB-only tasks to file metadata on explicit opening.

        Do not bulk-write thousands of workspace JSONs while listing history.
        """
        root = self._roots.get(experiment_id)
        if not root or not Path(root).is_dir():
            return
        state = WorkspaceState(root)
        if state.exists and state.bootstrap_complete:
            return
        record = self._required_db().get_experiment(experiment_id)
        rows = self._required_db().list_tasks(experiment_id)
        state.ensure(experiment_id, record['name'], bootstrap_complete=False,
                     created_at=record['created_at'], note=record.get('note', ''))
        for row in rows:
            state.save_task_row(experiment_id, record['name'], row)
        state.mark_bootstrap_complete()

    def get_run(self, run_id):
        result = self._try_db('get_run', run_id)
        if result is None:
            raise KeyError(f'Запуск {run_id} не найден или история недоступна')
        return result

    def get_settings(self, run_id):
        return json.loads(self.get_run(run_id)['settings_json'])

    def update_note(self, run_id, note):
        return self._required_db().update_note(run_id, note)

    def delete_run(self, run_id):
        return self._required_db().delete_run(run_id)

    def sync_workspace(self, root):
        """Replay workspace, old batch journals and ML outboxes without calculations."""
        record, rows = self.open_workspace_folder(root)
        db = self._required_db()
        db.ensure_experiment(record['id'], record['root_path'], record['name'])
        db.update_experiment_metadata(record['id'], record['name'], record.get('note', ''))
        db.replace_tasks(record['id'], rows)
        from compute.batch_runner import BatchJournal, summarize
        stats = dict(experiment_id=record['id'], tasks=len(rows), pipeline=0, ml=0)
        task_index = {row['id']: row for row in rows}
        for journal_path in sorted((Path(root)/'.runs').glob('*/batch.json')):
            journal = BatchJournal.load(journal_path)
            dirty = False
            for current in journal.document['tasks']:
                for attempt in [*current.get('previous_attempts', []), current]:
                    outcome = attempt.get('outcome') or attempt.get('status')
                    status = attempt.get('status')
                    if status == 'saving':
                        # Only a sealed attempt may be delivered after a crash.
                        if not attempt.get('attempt_manifest'):
                            continue
                    elif status not in ('completed','failed','partial_failed','cancelled','interrupted'):
                        continue
                    if not attempt.get('history_pending') or outcome not in (
                            'completed','failed','partial_failed','cancelled','interrupted'):
                        continue
                    source = task_index.get(attempt.get('task_id'))
                    if source is None:
                        continue
                    task = SimpleNamespace(persistent_id=source['id'],
                        task_folder_path=attempt.get('output_path') or source['folder_path'])
                    db.add_run(task=task, attempt_id=attempt['attempt_id'],
                        status=outcome,
                        duration_seconds=attempt.get('duration_seconds',0),
                        settings={**attempt.get('settings', {}),
                            'pipeline_attempt_id':attempt['attempt_id']},
                        error_message=attempt.get('error',''),
                        ml_summary=str(attempt.get('ml_summary', {})),
                        output_path=task.task_folder_path)
                    attempt['history_pending'] = False
                    if status == 'saving':
                        attempt['status'] = outcome
                    attempt.pop('history_error', None)
                    dirty = True
                    stats['pipeline'] += 1
            if dirty:
                journal.document['summary'] = summarize(journal.document['tasks'])
                journal.document['status'] = journal.document['summary']['status']
                journal.flush()
        for outbox_path in sorted((Path(root)/'.ml-history').glob('*.json')):
            rows_outbox = json.loads(outbox_path.read_text(encoding='utf-8'))
            if not isinstance(rows_outbox, list):
                raise ValueError(f'Повреждён ML outbox: {outbox_path}')
            dirty = False
            for entry in rows_outbox:
                if entry.get('delivered'):
                    continue
                # Identify task by persisted path, do not guess a task number.
                output = entry.get('payload', {}).get('output_path', '')
                matched = [r for r in rows if r['folder_path'] and
                    (output == r['folder_path'] or output.startswith(r['folder_path'] + '/'))]
                if len(matched) != 1:
                    # Windows paths may use \\; normalize both separators.
                    norm = lambda p: str(p).replace('\\', '/').rstrip('/')
                    matched = [r for r in rows if r['folder_path'] and
                        (norm(output) == norm(r['folder_path']) or
                         norm(output).startswith(norm(r['folder_path']) + '/'))]
                if len(matched) != 1:
                    continue
                task = SimpleNamespace(persistent_id=matched[0]['id'],
                    task_folder_path=matched[0]['folder_path'])
                db.add_run(task=task, attempt_id=entry['attempt_id'], **entry['payload'])
                entry['delivered'] = True
                entry.pop('error', None)
                dirty = True
                stats['ml'] += 1
            if dirty:
                atomic_write_text(outbox_path, json.dumps(rows_outbox, ensure_ascii=False, indent=2, allow_nan=False))
        return stats
