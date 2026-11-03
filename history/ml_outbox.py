"""Durable, idempotent ML history delivery, independent of computation."""
import json
from pathlib import Path
from uuid import uuid4

from history.atomic_io import atomic_write_text


def outbox_path(folder):
    folder = Path(folder).resolve()
    # Delivery acknowledgements must not mutate a sealed scientific directory.
    return folder.parent / '.ml-history' / (folder.name + '.json')


class MLHistoryOutbox:
    def __init__(self, folder, store, task, persist, warn):
        self.path = outbox_path(folder)
        self.store, self.task, self.persist, self.warn = store, task, persist, warn
        self.available = True
        # A corrupt outbox must never be silently replaced.
        self.records = json.loads(self.path.read_text(encoding='utf8')) if self.path.exists() else []
        if not isinstance(self.records, list) or any(not isinstance(r, dict) for r in self.records):
            raise ValueError('Invalid ML history outbox')

    def save(self):
        atomic_write_text(self.path, json.dumps(self.records, ensure_ascii=False, indent=2, allow_nan=False))

    @property
    def pending(self):
        return sum(not row.get('delivered') for row in self.records)

    def enqueue(self, **payload):
        self.records.append(dict(attempt_id='ml-' + uuid4().hex, payload=payload, delivered=False))
        self.save()
        self.flush()

    def contains(self, identity, output):
        return any(r['payload']['settings'].get('ml_operation_id') == identity
                   and r['payload'].get('output_path') == output for r in self.records)

    def flush(self):
        for row in self.records:
            if row.get('delivered') or not self.available:
                continue
            try:
                self.persist(self.task)
                self.store.add_run(task=self.task, attempt_id=row['attempt_id'], **row['payload'])
            except Exception as error:
                self.available = False
                row['error'] = f'{type(error).__name__}: {error}'
                self.warn(f'История ML ожидает записи; результаты сохранены: {error}')
                self.save()
            else:
                row['delivered'] = True
                row.pop('error', None)
                # If this acknowledgement fails, replay uses the same SQLite unique ID.
                self.save()
        return self.pending
