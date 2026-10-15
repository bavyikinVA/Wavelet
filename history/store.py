"""Persistent experiment -> task -> run model (schema v2; test data reset)."""
import json
import os
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime
from uuid import uuid4


def _default_database_path():
    if sys.platform == 'win32':
        base = os.environ.get('LOCALAPPDATA', os.path.expanduser('~'))
    elif sys.platform == 'darwin':
        base = os.path.join(os.path.expanduser('~'), 'Library', 'Application Support')
    else:
        base = os.environ.get('XDG_DATA_HOME', os.path.join(os.path.expanduser('~'), '.local', 'share'))
    return os.path.join(base, 'WaveletAnalysis', 'research_history.sqlite3')


def _now():
    return datetime.now().astimezone().isoformat(timespec='seconds')


class RunHistoryStore:
    """Experiments, saved tasks and immutable execution records."""

    def __init__(self, database_path=None):
        self.database_path = database_path or _default_database_path()
        os.makedirs(os.path.dirname(os.path.abspath(self.database_path)), exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.database_path, timeout=30)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute('PRAGMA foreign_keys = ON')
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self):
        with self._connect() as db:
            version = db.execute('PRAGMA user_version').fetchone()[0]
            if version < 2:
                # The development-stage database is intentionally reset, not migrated.
                db.execute('DROP TABLE IF EXISTS research_runs')
                db.execute('DROP TABLE IF EXISTS runs')
                db.execute('DROP TABLE IF EXISTS tasks')
                db.execute('DROP TABLE IF EXISTS experiments')
            db.executescript('''
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    root_path TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    note TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
                    task_number INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    folder_path TEXT NOT NULL DEFAULT '',
                    settings_json TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL,
                    UNIQUE(experiment_id, task_number)
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
                    run_type TEXT NOT NULL CHECK(run_type IN ('pipeline', 'ml')),
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    duration_seconds REAL NOT NULL DEFAULT 0,
                    output_path TEXT NOT NULL DEFAULT '',
                    ml_algorithm TEXT NOT NULL DEFAULT '',
                    ml_summary TEXT NOT NULL DEFAULT '',
                    error_message TEXT NOT NULL DEFAULT '',
                    researcher_note TEXT NOT NULL DEFAULT '',
                    settings_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_tasks_experiment ON tasks(experiment_id, task_number);
                CREATE INDEX IF NOT EXISTS idx_runs_task ON runs(task_id, id);
                PRAGMA user_version = 2;
            ''')

    def ensure_experiment(self, experiment_id, root_path, name=None):
        if not experiment_id or not root_path:
            raise ValueError('Идентификатор и каталог исследования обязательны')
        with self._connect() as db:
            db.execute('''INSERT INTO experiments(id,name,root_path,created_at,updated_at)
                VALUES(?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
                name=excluded.name, updated_at=excluded.updated_at''',
                (experiment_id, name or f'Исследование {experiment_id[:8]}', root_path, _now(), _now()))

    def save_task(self, experiment_id, task):
        task_uuid = getattr(task, 'persistent_id', None)
        if not task_uuid:
            task_uuid = uuid4().hex
            task.persistent_id = task_uuid
        snapshot = json.dumps(task.settings_snapshot(), ensure_ascii=False)
        with self._connect() as db:
            db.execute('''INSERT INTO tasks(id,experiment_id,task_number,name,folder_path,settings_json,updated_at)
                VALUES(?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
                name=excluded.name, folder_path=excluded.folder_path,
                settings_json=excluded.settings_json,updated_at=excluded.updated_at''',
                (task_uuid, experiment_id, int(task.task_id), task.task_name,
                 task.task_folder_path or '', snapshot, _now()))
            db.execute('UPDATE experiments SET updated_at=? WHERE id=?', (_now(), experiment_id))
        return task_uuid

    def add_run(self, *, task, status, duration_seconds, settings,
                ml_summary='', error_message='', ml_algorithm=None, run_type=None,
                output_path=None):
        task_id = getattr(task, 'persistent_id', None)
        if not task_id:
            raise ValueError('Сначала зарегистрируйте задачу в SQLite')
        algorithm = ml_algorithm if ml_algorithm is not None else ''
        kind = run_type or ('ml' if algorithm else 'pipeline')
        if kind not in ('pipeline', 'ml'):
            raise ValueError('Неизвестный тип запуска')
        with self._connect() as db:
            cursor = db.execute('''INSERT INTO runs(task_id,run_type,created_at,status,duration_seconds,
                output_path,ml_algorithm,ml_summary,error_message,settings_json)
                VALUES(?,?,?,?,?,?,?,?,?,?)''',
                (task_id, kind, _now(), status, float(duration_seconds),
                 output_path if output_path is not None else (task.task_folder_path or ''),
                 algorithm, ml_summary, error_message,
                 json.dumps(settings, ensure_ascii=False)))
            db.execute('''UPDATE tasks SET folder_path=?,settings_json=?,updated_at=? WHERE id=?''',
                       (task.task_folder_path or '', json.dumps(settings, ensure_ascii=False), _now(), task_id))
            db.execute('''UPDATE experiments SET updated_at=? WHERE id=(SELECT experiment_id FROM tasks WHERE id=?)''',
                       (_now(), task_id))
            return cursor.lastrowid

    _RUN_SELECT = '''SELECT r.*,t.name AS task_name,t.task_number,t.experiment_id,
        t.folder_path AS task_folder_path,e.root_path AS experiment_root,
        e.name AS experiment_name,json_extract(r.settings_json,'$.image_path') AS image_path,
        json_extract(r.settings_json,'$.analysis_mode') AS analysis_mode,
        json_extract(r.settings_json,'$.pipeline_preset') AS pipeline_preset
        FROM runs r JOIN tasks t ON r.task_id=t.id
        JOIN experiments e ON t.experiment_id=e.id'''

    def list_runs(self, limit=100):
        with self._connect() as db:
            rows = db.execute(self._RUN_SELECT + ' ORDER BY r.id DESC LIMIT ?', (int(limit),)).fetchall()
        return [dict(row) for row in rows]

    def get_run(self, run_id):
        with self._connect() as db:
            row = db.execute(self._RUN_SELECT + ' WHERE r.id=?', (int(run_id),)).fetchone()
        if row is None:
            raise KeyError(f'Запуск {run_id} не найден')
        return dict(row)

    def get_settings(self, run_id):
        return json.loads(self.get_run(run_id)['settings_json'])

    def list_experiments(self, limit=100):
        with self._connect() as db:
            rows = db.execute('''SELECT e.*,
                (SELECT COUNT(*) FROM tasks t WHERE t.experiment_id=e.id) AS task_count,
                (SELECT COUNT(*) FROM runs r JOIN tasks t ON t.id=r.task_id
                   WHERE t.experiment_id=e.id AND r.run_type='ml') AS ml_count,
                (SELECT COUNT(*) FROM runs r JOIN tasks t ON t.id=r.task_id
                   WHERE t.experiment_id=e.id) AS run_count
                FROM experiments e ORDER BY updated_at DESC LIMIT ?''', (int(limit),)).fetchall()
        return [dict(row) for row in rows]

    def get_experiment(self, experiment_id):
        with self._connect() as db:
            row = db.execute('SELECT * FROM experiments WHERE id=?', (experiment_id,)).fetchone()
        if row is None:
            raise KeyError(f'Исследование {experiment_id} не найдено')
        return dict(row)

    def list_tasks(self, experiment_id):
        with self._connect() as db:
            rows = db.execute('SELECT * FROM tasks WHERE experiment_id=? ORDER BY task_number', (experiment_id,)).fetchall()
        return [dict(row) for row in rows]

    def remove_task(self, experiment_id, task_id, remaining_tasks):
        """Remove a workspace task and its run records, keeping files untouched."""
        if not experiment_id or not task_id:
            return
        with self._connect() as db:
            db.execute('DELETE FROM tasks WHERE id=? AND experiment_id=?',
                       (task_id, experiment_id))
            # Two-phase numbering avoids conflicts with UNIQUE(experiment_id,task_number).
            for index, task in enumerate(remaining_tasks, 1):
                if getattr(task, 'persistent_id', None):
                    db.execute('UPDATE tasks SET task_number=? WHERE id=? AND experiment_id=?',
                               (1000000 + index, task.persistent_id, experiment_id))
            for index, task in enumerate(remaining_tasks, 1):
                if getattr(task, 'persistent_id', None):
                    db.execute('UPDATE tasks SET task_number=?,name=?,updated_at=? WHERE id=?',
                               (index, task.task_name, _now(), task.persistent_id))
            db.execute('UPDATE experiments SET updated_at=? WHERE id=?', (_now(), experiment_id))

    def update_note(self, run_id, note):
        with self._connect() as db:
            cursor = db.execute('UPDATE runs SET researcher_note=? WHERE id=?', (str(note).strip(), int(run_id)))
        if not cursor.rowcount:
            raise KeyError(f'Запуск {run_id} не найден')

    def delete_run(self, run_id):
        with self._connect() as db:
            cursor = db.execute('DELETE FROM runs WHERE id=?', (int(run_id),))
        return cursor.rowcount > 0
