"""Database-independent run-history repository: SQLAlchemy ORM, SQLite/PostgreSQL.

Scientific data belongs to the workspace, not to this secondary database index.
Existing SQLite databases are opened in-place with no destructive migration.
"""
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from sqlalchemy import create_engine, event, inspect, select, func, text, case
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool, StaticPool

from history.models import Base, Experiment, Task, Run


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


def _database_url(database_path=None, database_url=None):
    """WAVELETS_DATABASE_URL selects SQLAlchemy backend (e.g. postgresql+psycopg)."""
    if database_url:
        return database_url
    if database_path and '://' in str(database_path):
        return str(database_path)
    if database_path:
        return 'sqlite:///' + Path(database_path).absolute().as_posix()
    configured = os.environ.get('WAVELETS_DATABASE_URL')
    return configured or 'sqlite:///' + Path(_default_database_path()).absolute().as_posix()


class RunHistoryStore:
    """ORM-only CRUD with the same public API and JSON row shapes as the old store."""

    def __init__(self, database_path=None, *, database_url=None):
        self.database_url = _database_url(database_path, database_url)
        url = make_url(self.database_url)
        self.database_path = url.database if url.get_backend_name() == 'sqlite' else None
        options = {'pool_pre_ping': True}
        if url.get_backend_name() == 'postgresql':
            # A dead server must not keep the Tk startup splash blocked.
            # A URL query parameter may override the default when necessary.
            if 'connect_timeout' not in url.query:
                options['connect_args'] = {'connect_timeout': 3}
        if url.get_backend_name() == 'sqlite':
            filename = url.database
            if filename and filename != ':memory:':
                os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
            # In-memory SQLite must retain the same DB-API connection.
            if not filename or filename == ':memory:':
                options.update(poolclass=StaticPool, connect_args={'check_same_thread': False})
            else:
                # Short timeout allows the GUI's DB-offline mode to start quickly.
                options.update(poolclass=NullPool, connect_args={'timeout': 1.5})
        self.engine = create_engine(self.database_url, **options)
        if url.get_backend_name() == 'sqlite':
            @event.listens_for(self.engine, 'connect')
            def sqlite_foreign_keys(dbapi_connection, _connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute('PRAGMA foreign_keys=ON')
                cursor.close()
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        self._initialize()

    @contextmanager
    def _connect(self):
        """Legacy diagnostic DB-API access used by existing SQLite-only tests.

        Production operations exclusively use ORM sessions; never use this for CRUD.
        """
        raw = self.engine.raw_connection()
        dbapi = raw.dbapi_connection
        try:
            yield dbapi
            raw.commit()
        except BaseException:
            raw.rollback()
            raise
        finally:
            raw.close()

    @contextmanager
    def _session(self):
        with self.Session.begin() as session:
            yield session

    def _initialize(self):
        # Only additive migration required by old schema: runs.attempt_id.
        # SQLAlchemy metadata create_all is idempotent and never drops existing data.
        with self.engine.begin() as conn:
            inspector = inspect(conn)
            if 'runs' in inspector.get_table_names():
                columns = {column['name'] for column in inspector.get_columns('runs')}
                if 'attempt_id' not in columns:
                    conn.execute(text('ALTER TABLE runs ADD COLUMN attempt_id TEXT'))
        with self.engine.begin() as conn:
            inspector = inspect(conn)
            if 'tasks' in inspector.get_table_names():
                fields = {column['name'] for column in inspector.get_columns('tasks')}
                if 'note' not in fields:
                    # Additive, portable migration: valid for SQLite and PostgreSQL.
                    conn.execute(text("ALTER TABLE tasks ADD COLUMN note TEXT NOT NULL DEFAULT ''"))
        Base.metadata.create_all(self.engine)

    def ensure_experiment(self, experiment_id, root_path, name=None):
        if not experiment_id or not root_path:
            raise ValueError('Идентификатор и каталог исследования обязательны')
        with self._session() as db:
            row = db.get(Experiment, experiment_id)
            timestamp = _now()
            if row is None:
                db.add(Experiment(id=experiment_id,
                    name=name or f'Исследование {experiment_id[:8]}',
                    root_path=str(root_path), created_at=timestamp,
                    updated_at=timestamp, note=''))
            else:
                # Never replace a user title with a default on routine saves.
                if name is not None:
                    row.name = name
                row.root_path = str(root_path)
                row.updated_at = timestamp

    def save_task(self, experiment_id, task):
        task_id = getattr(task, 'persistent_id', None)
        if not task_id:
            task_id = uuid4().hex
            task.persistent_id = task_id
        snapshot = json.dumps(task.settings_snapshot(), ensure_ascii=False)
        with self._session() as db:
            row = db.get(Task, task_id)
            if row is None:
                row = Task(id=task_id, experiment_id=experiment_id,
                    task_number=int(task.task_id), name=task.task_name,
                    note=getattr(task, 'note', '') or '', folder_path=task.task_folder_path or '',
                    settings_json=snapshot, updated_at=_now())
                db.add(row)
            else:
                if row.experiment_id != experiment_id:
                    raise ValueError('Задача принадлежит другому исследованию')
                row.task_number = int(task.task_id)
                row.name = task.task_name
                row.note = getattr(task, 'note', '') or ''
                row.folder_path = task.task_folder_path or ''
                row.settings_json = snapshot
                row.updated_at = _now()
            experiment = db.get(Experiment, experiment_id)
            if experiment is not None:
                experiment.updated_at = _now()
        return task_id

    def add_run(self, *, task, status, duration_seconds, settings,
                ml_summary='', error_message='', ml_algorithm=None, run_type=None,
                output_path=None, attempt_id=None):
        task_id = getattr(task, 'persistent_id', None)
        if not task_id:
            raise ValueError('У задачи отсутствует постоянный идентификатор')
        algorithm = ml_algorithm if ml_algorithm is not None else ''
        kind = run_type or ('ml' if algorithm else 'pipeline')
        if kind not in ('pipeline', 'ml'):
            raise ValueError('Неизвестный тип запуска')
        with self._session() as db:
            if attempt_id:
                previous = db.scalar(select(Run).where(Run.attempt_id == attempt_id))
                if previous is not None:
                    if previous.task_id != task_id:
                        raise ValueError('Идентификатор попытки принадлежит другой задаче')
                    return previous.id
            row = Run(task_id=task_id, run_type=kind, created_at=_now(),
                status=status, duration_seconds=float(duration_seconds),
                output_path=output_path if output_path is not None else (task.task_folder_path or ''),
                ml_algorithm=algorithm, ml_summary=ml_summary,
                error_message=error_message, researcher_note='',
                settings_json=json.dumps(settings, ensure_ascii=False), attempt_id=attempt_id)
            db.add(row)
            saved_task = db.get(Task, task_id)
            if saved_task is None:
                raise ValueError('Сначала зарегистрируйте задачу в истории')
            saved_task.folder_path = task.task_folder_path or ''
            saved_task.settings_json = json.dumps(settings, ensure_ascii=False)
            saved_task.updated_at = _now()
            experiment = db.get(Experiment, saved_task.experiment_id)
            if experiment is not None:
                experiment.updated_at = _now()
            db.flush()
            return row.id

    @staticmethod
    def _record(row):
        return {c.name:getattr(row,c.name) for c in row.__table__.columns}

    def _run_record(self, run, task, experiment):
        record = self._record(run)
        record.update(task_name=task.name, task_number=task.task_number,
            experiment_id=task.experiment_id, task_folder_path=task.folder_path,
            experiment_root=experiment.root_path, experiment_name=experiment.name)
        try:
            settings = json.loads(run.settings_json)
        except (ValueError, TypeError):
            settings = {}
        for key in ('image_path', 'analysis_mode', 'pipeline_preset'):
            record[key] = settings.get(key)
        return record

    def find_ml_operation(self, task, operation_id, output_path):
        with self._session() as db:
            records = db.scalars(select(Run).where(Run.task_id == task.persistent_id,
                Run.run_type == 'ml', Run.status == 'completed',
                Run.output_path == output_path).order_by(Run.id.desc())).all()
            for row in records:
                try:
                    if json.loads(row.settings_json).get('ml_operation_id') == operation_id:
                        return row.id
                except (ValueError, TypeError):
                    continue
        return None

    def list_runs(self, limit=100):
        with self._session() as db:
            rows = db.execute(select(Run, Task, Experiment).join(Task, Run.task_id == Task.id)
                .join(Experiment, Task.experiment_id == Experiment.id)
                .order_by(Run.created_at.desc(), Run.id.desc()).limit(int(limit))).all()
            return [self._run_record(*row) for row in rows]

    def get_run(self, run_id):
        with self._session() as db:
            row = db.execute(select(Run, Task, Experiment).join(Task, Run.task_id == Task.id)
                .join(Experiment, Task.experiment_id == Experiment.id)
                .where(Run.id == int(run_id))).first()
            if row is None:
                raise KeyError(f'Запуск {run_id} не найден')
            return self._run_record(*row)

    def get_settings(self, run_id):
        return json.loads(self.get_run(run_id)['settings_json'])

    def list_experiments(self, limit=100):
        """One portable aggregate SELECT, not three queries per research card."""
        task_counts = (select(Task.experiment_id.label('experiment_id'),
                func.count(Task.id).label('task_count'))
            .group_by(Task.experiment_id).subquery())
        run_counts = (select(Task.experiment_id.label('experiment_id'),
                func.count(Run.id).label('run_count'),
                func.sum(case((Run.run_type == 'ml', 1), else_=0)).label('ml_count'),
                func.sum(Run.duration_seconds).label('total_duration_seconds'))
            .join(Run, Run.task_id == Task.id).group_by(Task.experiment_id).subquery())
        with self._session() as db:
            query = (select(Experiment,
                    func.coalesce(task_counts.c.task_count, 0),
                    func.coalesce(run_counts.c.run_count, 0),
                    func.coalesce(run_counts.c.ml_count, 0),
                    func.coalesce(run_counts.c.total_duration_seconds, 0.0))
                .outerjoin(task_counts, task_counts.c.experiment_id == Experiment.id)
                .outerjoin(run_counts, run_counts.c.experiment_id == Experiment.id)
                .order_by(Experiment.created_at.desc(), Experiment.id.desc()).limit(int(limit)))
            return [{**self._record(exp), 'task_count':int(tasks),
                'run_count':int(runs), 'ml_count':int(ml),
                'total_duration_seconds':float(duration)}
                for exp, tasks, runs, ml, duration in db.execute(query)]

    def get_experiment(self, experiment_id):
        with self._session() as db:
            row = db.get(Experiment, experiment_id)
            if row is None:
                raise KeyError(f'Исследование {experiment_id} не найдено')
            return self._record(row)

    def list_tasks(self, experiment_id):
        with self._session() as db:
            return [self._record(row) for row in db.scalars(
                select(Task).where(Task.experiment_id == experiment_id).order_by(Task.task_number))]

    def remove_task(self, experiment_id, task_id, remaining_tasks):
        if not experiment_id or not task_id:
            return
        with self._session() as db:
            doomed = db.get(Task, task_id)
            if doomed is not None and doomed.experiment_id == experiment_id:
                db.delete(doomed)
                db.flush()
            # Avoid UNIQUE(experiment_id,task_number) while swapping numbers.
            for index, task in enumerate(remaining_tasks, 1):
                row = db.get(Task, getattr(task, 'persistent_id', None))
                if row is not None and row.experiment_id == experiment_id:
                    row.task_number = 1000000 + index
            db.flush()
            for index, task in enumerate(remaining_tasks, 1):
                row = db.get(Task, getattr(task, 'persistent_id', None))
                if row is not None and row.experiment_id == experiment_id:
                    row.task_number, row.name, row.updated_at = index, task.task_name, _now()
                    row.note = getattr(task, 'note', '') or '' 
            experiment = db.get(Experiment, experiment_id)
            if experiment is not None:
                experiment.updated_at = _now()

    def iter_runs(self, *, page_size=500):
        """Iterate all history records in stable keyset pages (backend neutral)."""
        cursor = 0
        while True:
            with self._session() as db:
                rows = db.execute(select(Run, Task, Experiment)
                    .join(Task, Run.task_id == Task.id)
                    .join(Experiment, Task.experiment_id == Experiment.id)
                    .where(Run.id > cursor).order_by(Run.id.asc()).limit(page_size)).all()
                page = [self._run_record(*row) for row in rows]
            if not page:
                break
            for record in page:
                yield record
            cursor = page[-1]['id']

    def import_run(self, record, *, source_namespace):
        """Losslessly import historic run metadata, preserving timestamps and notes.

        Old NULL-attempt rows receive a deterministic import identity, making
        repeated migrations idempotent without relying on SQLite-specific SQL.
        """
        attempt_id = record.get('attempt_id') or f"import:{source_namespace}:{record['id']}"
        with self._session() as db:
            previous = db.scalar(select(Run).where(Run.attempt_id == attempt_id))
            if previous is not None:
                if previous.task_id != record['task_id']:
                    raise ValueError('Коллизия идентификатора попытки при импорте')
                return previous.id, False
            if db.get(Task, record['task_id']) is None:
                raise ValueError('Сначала импортируйте задачу')
            row = Run(task_id=record['task_id'], run_type=record['run_type'],
                created_at=record['created_at'], status=record['status'],
                duration_seconds=record['duration_seconds'],
                output_path=record.get('output_path') or '',
                ml_algorithm=record.get('ml_algorithm') or '',
                ml_summary=record.get('ml_summary') or '',
                error_message=record.get('error_message') or '',
                researcher_note=record.get('researcher_note') or '',
                settings_json=record['settings_json'], attempt_id=attempt_id)
            db.add(row)
            db.flush()
            return row.id, True

    def replace_tasks(self, experiment_id, rows):
        """Make an ORM index reflect the authoritative workspace after offline edits.

        Single transaction, two-phase numbering and deletes prevent uniqueness
        conflicts, including swaps and removed tasks. Unchanged run records
        remain associated with the same immutable task IDs.
        """
        desired = {row['id']: row for row in rows}
        with self._session() as db:
            existing = list(db.scalars(select(Task).where(Task.experiment_id == experiment_id)))
            for row in existing:
                if row.id not in desired:
                    db.delete(row)
                else:
                    row.task_number = 1000000 + int(desired[row.id]['task_number'])
            db.flush()
            for data in rows:
                row = db.get(Task, data['id'])
                if row is None:
                    row = Task(id=data['id'], experiment_id=experiment_id)
                    db.add(row)
                elif row.experiment_id != experiment_id:
                    raise ValueError('ID задачи уже занят другим исследованием')
                row.task_number = int(data['task_number'])
                row.name = data['name']
                row.note = data.get('note', '') or ''
                row.folder_path = data.get('folder_path') or ''
                row.settings_json = data['settings_json']
                row.updated_at = data.get('updated_at') or _now()
            experiment = db.get(Experiment, experiment_id)
            if experiment is not None:
                experiment.updated_at = _now()

    def update_experiment_metadata(self, experiment_id, name, note):
        from history.metadata import normalize_metadata
        name, note = normalize_metadata(name, note)
        with self._session() as db:
            record = db.get(Experiment, experiment_id)
            if record is None:
                raise KeyError(f'Исследование {experiment_id} не найдено')
            record.name = name
            record.note = note
            record.updated_at = _now()

    def update_task_metadata(self, experiment_id, task_id, name, note):
        from history.metadata import normalize_metadata
        name, note = normalize_metadata(name, note)
        with self._session() as db:
            record = db.get(Task, task_id)
            if record is None or record.experiment_id != experiment_id:
                raise KeyError(f'Задача {task_id} не найдена в исследовании')
            record.name = name
            record.note = note
            record.updated_at = _now()

    def update_note(self, run_id, note):
        with self._session() as db:
            row = db.get(Run, int(run_id))
            if row is None:
                raise KeyError(f'Запуск {run_id} не найден')
            row.researcher_note = str(note).strip()

    def delete_run(self, run_id):
        with self._session() as db:
            row = db.get(Run, int(run_id))
            if row is None:
                return False
            db.delete(row)
            return True

    def close(self):
        self.engine.dispose()
