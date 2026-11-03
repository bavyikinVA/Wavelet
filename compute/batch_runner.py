"""Sequential, backend-neutral execution with a durable journal before work.

One writer owns a batch. GPU selection/fallback belongs to ComputeBackend.
History is a secondary index: its failure must not replace a scientific error.
"""
import errno
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from compute.run_control import RunCancelled
from history.atomic_io import atomic_write_text


def now():
    return datetime.now(timezone.utc).isoformat()


class JournalWriteError(RuntimeError):
    pass


class BatchJournal:
    def __init__(self, path, document):
        self.path = Path(path)
        self.document = document

    @classmethod
    def create(cls, root, tasks, requested_backend='gpu'):
        if not root:
            raise ValueError('Каталог исследования не задан')
        batch_id = uuid4().hex
        rows = []
        for task in tasks:
            if not getattr(task, 'persistent_id', None):
                task.persistent_id = uuid4().hex
            rows.append(dict(attempt_id=uuid4().hex, task_id=task.persistent_id,
                task_number=task.task_id, task_name=task.task_name,
                settings=task.settings_snapshot(), status='pending',
                output_path=task.task_folder_path or '', history_pending=False))
        journal = cls(Path(root)/'.runs'/batch_id/'batch.json',
            dict(schema_version=1, batch_id=batch_id, created_at=now(),
                 requested_backend=requested_backend, status='pending', tasks=rows))
        import psutil
        journal.document['owner'] = {'pid':os.getpid(),'created':psutil.Process().create_time()}
        from history.batch_report import environment
        journal.document['environment'] = environment()
        journal.flush()
        return journal

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        if path.name != 'batch.json' or path.parent.parent.name != '.runs':
            raise ValueError('Выберите .runs/<идентификатор>/batch.json')
        document = json.loads(Path(path).read_text(encoding='utf8'))
        if document.get('schema_version') != 1 or not isinstance(document.get('tasks'), list):
            raise ValueError('Неподдерживаемый журнал очереди')
        return cls(path, document)

    def flush(self):
        self.document['updated_at'] = now()
        try:
            atomic_write_text(self.path, json.dumps(self.document,ensure_ascii=False,indent=2))
        except Exception as error:
            raise JournalWriteError(f'Не удалось сохранить очередь {self.path}: {error}') from error

    def update(self, row, **values):
        row.update(values)
        self.flush()

    def interrupt(self):
        """Call only after the previous owner stopped, never on a live batch."""
        for row in self.document['tasks']:
            if row['status'] == 'running':
                row['status'] = 'interrupted'
            # saving means a terminal result is durable; only the history index
            # or final journal acknowledgement remains to be reconciled.
            elif row['status'] == 'saving':
                row['status'] = row['outcome']
                row['history_pending'] = True
        self.document['status'] = 'interrupted'
        self.flush()

    def prepare_resume(self):
        owner = self.document.get('owner',{})
        if owner.get('pid') and owner['pid'] != os.getpid():
            import psutil
            try:
                process = psutil.Process(owner['pid'])
                if process.create_time() == owner.get('created'):
                    raise ValueError('Очередь принадлежит работающему процессу; закройте его перед восстановлением')
            except psutil.NoSuchProcess:
                pass
        self.interrupt()
        for row in self.document['tasks']:
            if row['status'] == 'interrupted':
                from types import SimpleNamespace
                from history.attempts import seal_attempt
                row['outcome'] = 'interrupted'
                row['attempt_manifest'] = seal_attempt(self,row,SimpleNamespace(
                    analysis_mode=row['settings'].get('analysis_mode','1d'),execution_protocol=None))
            if row['status'] in {'failed','cancelled','interrupted','partial_failed'}:
                previous = {k:v for k,v in row.items() if k != 'previous_attempts'}
                row.setdefault('previous_attempts',[]).append(previous)
                for key in ('error','traceback','history_error','outcome','started_at',
                            'finished_at','duration_seconds','ml_summary','preparation_error'):
                    row.pop(key,None)
                row.update(status='pending',attempt_id=uuid4().hex,history_pending=False)
                for key in ('initial_stages','inherited_stages','attempt_manifest','backend','provenance_error'):
                    row.pop(key,None)
        self.flush()


def global_failure(error):
    return (isinstance(error, (JournalWriteError, MemoryError)) or
            isinstance(error, OSError) and error.errno in {errno.ENOSPC, errno.EROFS})


def summarize(rows):
    counts = {state:sum(row['status']==state for row in rows) for state in
              ('completed','partial_failed','failed','cancelled','pending','interrupted')}
    counts['history_pending'] = sum(bool(attempt.get('history_pending')) for row in rows
        for attempt in [row,*row.get('previous_attempts',[])])
    if counts['cancelled']:
        status = 'cancelled'
    elif counts['pending'] or counts['interrupted']:
        status = 'interrupted'
    elif counts['failed'] or counts['partial_failed']:
        status = 'partial_failed' if counts['completed'] or counts['partial_failed'] else 'failed'
    elif counts['history_pending']:
        status = 'history_pending'
    else:
        status = 'completed'
    return dict(status=status, **counts)


def commit_history(journal, row, task, record_history, log):
    try:
        record_history(task, row)
    except Exception as error:
        row['history_pending'] = True
        row['history_error'] = f'{type(error).__name__}: {error}'
        log(f"История {task.task_name} ожидает повторной записи: {error}")
    else:
        row['history_pending'] = False
        row.pop('history_error',None)
    journal.flush()
    return not row['history_pending']


def run_batch(journal, tasks, execute, record_history, control, log):
    """execute returns an ML summary; scientific failures stay local to a task."""
    rows = journal.document['tasks']
    if len(tasks) != len(rows):
        raise ValueError('Состав очереди не соответствует журналу')
    journal.document['status'] = 'running'
    import psutil
    journal.document['owner'] = {'pid':os.getpid(),'created':psutil.Process().create_time()}
    journal.flush()
    history_available = True
    for task, row in zip(tasks, rows):
        for previous in row.get('previous_attempts',[]):
            if previous.get('history_pending') and history_available:
                history_available = commit_history(journal,previous,task,record_history,log)
        if row['status'] in {'completed','partial_failed','failed','cancelled'}:
            if row.get('history_pending') and history_available:
                history_available = commit_history(journal,row,task,record_history,log)
            continue
        try:
            control.check()
        except RunCancelled:
            journal.document['status'] = 'cancelled'
            journal.flush()
            raise
        started = time.monotonic()
        journal.update(row,status='running',started_at=now())
        cancelled = False
        stop = False
        try:
            ml = execute(task,row) or {}
            control.check()
            outcome = 'partial_failed' if ml.get('failed',0) else 'completed'
            row['ml_summary'] = ml
        except RunCancelled as error:
            outcome, cancelled = 'cancelled', True
            row['error'] = str(error)
        except JournalWriteError:
            raise  # Never perform more scientific work without a durable log.
        except Exception as error:
            outcome = 'failed'
            row['error'] = f'{type(error).__name__}: {error}'
            row['traceback'] = traceback.format_exc()
            stop = global_failure(error)
            log(f"Задача {task.task_name}: {row['error']}")
        journal.update(row,status='saving',outcome=outcome,
            output_path=task.task_folder_path or '',
            duration_seconds=time.monotonic()-started,finished_at=now(),
            history_pending=True)
        try:
            from history.attempts import seal_attempt
            row['attempt_manifest'] = seal_attempt(journal,row,task)
        except Exception as error:
            row['provenance_error'] = f'{type(error).__name__}: {error}'
            log(f"Протокол {task.task_name} не сохранён: {error}")
            if outcome in {'completed','partial_failed'}:
                outcome = 'failed'
                row['outcome'] = outcome
                row['error'] = row.get('error') or row['provenance_error']
            stop = True
        if history_available:
            history_available = commit_history(journal,row,task,record_history,log)
        else:
            row['history_error'] = 'Запись отложена после ошибки БД в этом пакете'
        journal.update(row,status=outcome)
        if cancelled:
            journal.document['status'] = 'cancelled'
            journal.flush()
            raise RunCancelled(row.get('error','Отменено'))
        if stop:
            break
    result = summarize(rows)
    journal.document.update(status=result['status'],summary=result,finished_at=now())
    journal.flush()
    return result


def validate_task(task):
    from compute.validation import validate_scales
    from compute.extremes.detection import normalize_peak_parameters
    validate_scales(task.scales)
    if not task.image_path or task.original_image is None:
        raise ValueError('Не загружено изображение')
    if task.channel_representation == 'gram_schmidt' and not task.has_colors_selected():
        raise ValueError('Не задан цветовой базис Gram–Schmidt')
    if task.analysis_mode == '1d':
        if not (task.process_rows or task.process_columns):
            raise ValueError('Не выбран источник 1D CWT')
        normalize_peak_parameters(task.extrema_distance,task.extrema_prominence)
    elif not task.orientations:
        raise ValueError('Не заданы ориентации 2D Morlet')


def snapshot_inputs(journal, tasks, control):
    """Keep the effective cropped input for pending tasks before computations."""
    from PIL import Image
    import hashlib
    from history.atomic_io import atomic_binary_writer
    for task, row in zip(tasks, journal.document['tasks']):
        control.check()
        try:
            validate_task(task)
            path = journal.path.parent/'inputs'/(row['attempt_id']+'.png')
            with atomic_binary_writer(path, cancel_check=control.check) as stream:
                Image.fromarray(task.original_image).save(stream,format='PNG')
            journal.update(row,input_path=str(path.resolve()),
                           input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        except JournalWriteError:
            raise
        except Exception as error:
            if global_failure(error):
                raise
            journal.update(row,preparation_error=f'{type(error).__name__}: {error}')


def restore_tasks(journal):
    """Recreate pending tasks from saved effective input, never the original crop source."""
    import hashlib
    import numpy as np
    from PIL import Image
    from compute.processing_task import ProcessingTask
    from history.pipeline_resume import cwt_signature
    tasks = []
    for row in journal.document['tasks']:
        task = ProcessingTask()
        task.apply_settings_snapshot(row['settings'])
        task.persistent_id = row['task_id']
        task.task_id, task.task_name = row['task_number'], row['task_name']
        task.task_folder_path = row.get('output_path','')
        if row.get('input_path') or row['status'] in {'pending','running','interrupted'}:
            path = Path(row.get('input_path',''))
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row.get('input_sha256'):
                row['preparation_error'] = 'Снимок входного изображения отсутствует или повреждён'
            else:
                with Image.open(path) as image:
                    task.original_image = np.array(image.convert('RGB'))
                task.image_path = str(path)
                task.data = [task.original_image[:,:,c].copy() for c in range(3)]
                task.data_copy = [channel.copy() for channel in task.data]
                # F04 requires every stage signature + artifacts before reuse.
                task.last_completed_cwt_signature = cwt_signature(task)
        tasks.append(task)
    return tasks
