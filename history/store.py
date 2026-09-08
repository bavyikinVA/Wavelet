import json
import os
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime


def _default_database_path():
    """Return a stable, OS-appropriate location outside result folders."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
    elif sys.platform == "darwin":
        base = os.path.join(os.path.expanduser("~"), "Library", "Application Support")
    else:
        base = os.environ.get(
            "XDG_DATA_HOME", os.path.join(os.path.expanduser("~"), ".local", "share")
        )
    return os.path.join(base, "WaveletAnalysis", "research_history.sqlite3")


class RunHistoryStore:
    """SQLite storage for reproducible runs; JSON is only an internal payload."""

    def __init__(self, database_path=None):
        self.database_path = database_path or _default_database_path()
        os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self):
        """Commit/rollback the transaction, then always release the database."""
        connection = sqlite3.connect(self.database_path)
        try:
            connection.row_factory = sqlite3.Row
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self):
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS research_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    task_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    duration_seconds REAL NOT NULL DEFAULT 0,
                    image_path TEXT NOT NULL DEFAULT '',
                    output_path TEXT NOT NULL DEFAULT '',
                    analysis_mode TEXT NOT NULL,
                    pipeline_preset TEXT NOT NULL,
                    ml_algorithm TEXT NOT NULL DEFAULT '',
                    ml_summary TEXT NOT NULL DEFAULT '',
                    error_message TEXT NOT NULL DEFAULT '',
                    researcher_note TEXT NOT NULL DEFAULT '',
                    settings_json TEXT NOT NULL
                )
                """
            )
            columns = {
                row["name"] for row in connection.execute(
                    "PRAGMA table_info(research_runs)"
                ).fetchall()
            }
            if "researcher_note" not in columns:
                connection.execute(
                    "ALTER TABLE research_runs ADD COLUMN researcher_note "
                    "TEXT NOT NULL DEFAULT ''"
                )

    def add_run(self, *, task, status, duration_seconds, settings,
                ml_summary="", error_message="", ml_algorithm=None):
        created_at = datetime.now().astimezone().isoformat(timespec="seconds")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO research_runs (
                    created_at, task_name, status, duration_seconds, image_path,
                    output_path, analysis_mode, pipeline_preset, ml_algorithm,
                    ml_summary, error_message, settings_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    task.task_name,
                    status,
                    float(duration_seconds),
                    task.image_path or "",
                    task.task_folder_path or "",
                    task.analysis_mode,
                    task.pipeline_preset,
                    (ml_algorithm if ml_algorithm is not None else
                     (task.ml_algorithm
                      if task.pipeline_preset == "Подготовка данных для ML" else "")),
                    ml_summary,
                    error_message,
                    json.dumps(settings, ensure_ascii=False),
                ),
            )
            return cursor.lastrowid

    def list_runs(self, limit=100):
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM research_runs ORDER BY id DESC LIMIT ?", (int(limit),)
            ).fetchall()
        return [dict(row) for row in rows]

    def get_settings(self, run_id):
        with self._connect() as connection:
            row = connection.execute(
                "SELECT settings_json FROM research_runs WHERE id = ?", (int(run_id),)
            ).fetchone()
        if row is None:
            raise KeyError(f"Запуск {run_id} не найден")
        return json.loads(row["settings_json"])

    def get_run(self, run_id):
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM research_runs WHERE id = ?", (int(run_id),)
            ).fetchone()
        if row is None:
            raise KeyError(f"Запуск {run_id} не найден")
        return dict(row)

    def update_note(self, run_id, note):
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE research_runs SET researcher_note = ? WHERE id = ?",
                (str(note).strip(), int(run_id)),
            )
        if cursor.rowcount == 0:
            raise KeyError(f"Запуск {run_id} не найден")

    def delete_run(self, run_id):
        """Delete only the history record; result files remain untouched."""
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM research_runs WHERE id = ?", (int(run_id),)
            )
        return cursor.rowcount > 0
