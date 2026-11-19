"""Portable SQLAlchemy 2 ORM mapping of the existing research history schema.

Names, storage types and constraints intentionally match legacy SQLite files.
"""
from sqlalchemy import CheckConstraint, ForeignKey, Index, Integer, Float, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Experiment(Base):
    __tablename__ = 'experiments'
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    root_path: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    created_at: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[str] = mapped_column(Text, nullable=False)
    note: Mapped[str] = mapped_column(Text, nullable=False, default='')
    tasks: Mapped[list['Task']] = relationship(back_populates='experiment', cascade='all, delete-orphan', passive_deletes=True)


class Task(Base):
    __tablename__ = 'tasks'
    __table_args__ = (UniqueConstraint('experiment_id', 'task_number'),
                      Index('idx_tasks_experiment', 'experiment_id', 'task_number'))
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    experiment_id: Mapped[str] = mapped_column(Text, ForeignKey('experiments.id', ondelete='CASCADE'), nullable=False)
    task_number: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    folder_path: Mapped[str] = mapped_column(Text, nullable=False, default='')
    settings_json: Mapped[str] = mapped_column(Text, nullable=False, default='{}')
    note: Mapped[str] = mapped_column(Text, nullable=False, default='')
    updated_at: Mapped[str] = mapped_column(Text, nullable=False)
    experiment: Mapped[Experiment] = relationship(back_populates='tasks')
    runs: Mapped[list['Run']] = relationship(back_populates='task', cascade='all, delete-orphan', passive_deletes=True)


class Run(Base):
    __tablename__ = 'runs'
    __table_args__ = (CheckConstraint("run_type IN ('pipeline', 'ml')"),
                      Index('idx_runs_task', 'task_id', 'id'),
                      Index('idx_runs_attempt', 'attempt_id', unique=True))
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(Text, ForeignKey('tasks.id', ondelete='CASCADE'), nullable=False)
    run_type: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    output_path: Mapped[str] = mapped_column(Text, nullable=False, default='')
    ml_algorithm: Mapped[str] = mapped_column(Text, nullable=False, default='')
    ml_summary: Mapped[str] = mapped_column(Text, nullable=False, default='')
    error_message: Mapped[str] = mapped_column(Text, nullable=False, default='')
    researcher_note: Mapped[str] = mapped_column(Text, nullable=False, default='')
    settings_json: Mapped[str] = mapped_column(Text, nullable=False)
    attempt_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    task: Mapped[Task] = relationship(back_populates='runs')
