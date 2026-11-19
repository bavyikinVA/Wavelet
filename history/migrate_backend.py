"""CLI: move all historical index records from a SQLite file to any SQLAlchemy DB.

Example: python -m history.migrate_backend --source-sqlite C:/.../research_history.sqlite3 \\
    --target-url postgresql+psycopg://user:password@host:5432/wavelets
Scientific result files are NOT copied, modified or re-computed.
"""
import argparse
import hashlib
import os
from pathlib import Path

from history.store import RunHistoryStore


def migrate(source_path, target_url, *, dry_run=False):
    source_path = Path(source_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f'Исходная SQLite не найдена: {source_path}')
    if target_url.startswith('sqlite:///'):
        target = Path(target_url.removeprefix('sqlite:///')).expanduser().resolve()
        if source_path == target:
            raise ValueError('Нельзя импортировать базу саму в себя')
    source = RunHistoryStore(str(source_path))
    experiments = source.list_experiments(limit=10**9)
    tasks = {exp['id']:source.list_tasks(exp['id']) for exp in experiments}
    if dry_run:
        runs = sum(1 for _ in source.iter_runs())
        return {'experiments':len(experiments), 'tasks':sum(map(len,tasks.values())),
            'runs':runs, 'imported':0, 'skipped':0, 'dry_run':True}
    target = RunHistoryStore(database_url=target_url)
    namespace = hashlib.sha256(str(source_path).encode('utf-8')).hexdigest()[:16]
    # Prevent an accidental rerun from deleting target-only tasks.
    for experiment in experiments:
        try:
            target.get_experiment(experiment['id'])
        except KeyError:
            existing = []
        else:
            existing = target.list_tasks(experiment['id'])
        source_ids = {row['id'] for row in tasks[experiment['id']]}
        extra = {row['id'] for row in existing} - source_ids
        if extra:
            raise ValueError(f'В целевой БД есть {len(extra)} дополнительных задач '
                             f'исследования {experiment["id"]}; импорт отменён')
    for experiment in experiments:
        target.ensure_experiment(experiment['id'], experiment['root_path'], experiment['name'])
        target.update_experiment_metadata(experiment['id'], experiment['name'],
                                          experiment.get('note', ''))
        target.replace_tasks(experiment['id'], tasks[experiment['id']])
    inserted, skipped = 0, 0
    for record in source.iter_runs():
        _, created = target.import_run(record, source_namespace=namespace)
        inserted += int(created)
        skipped += int(not created)
    source.close()
    target.close()
    return {'experiments':len(experiments), 'tasks':sum(map(len,tasks.values())),
        'runs':inserted+skipped, 'imported':inserted, 'skipped':skipped, 'dry_run':False}


def main():
    parser = argparse.ArgumentParser(description='Перенос индекса истории SQLite → SQLAlchemy ORM backend')
    parser.add_argument('--source-sqlite', required=True, help='Путь к существующей базе SQLite')
    parser.add_argument('--target-url', default=os.environ.get('WAVELETS_DATABASE_URL'),
                        help='SQLAlchemy URL цели, или WAVELETS_DATABASE_URL из окружения')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not args.target_url:
        parser.error('Укажите --target-url или переменную WAVELETS_DATABASE_URL')
    results = migrate(args.source_sqlite, args.target_url, dry_run=args.dry_run)
    for key, value in results.items():
        print(f'{key}: {value}')


if __name__ == '__main__':
    main()
