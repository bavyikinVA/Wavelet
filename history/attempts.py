"""Separate attempt outputs and immutable scientific provenance."""
import hashlib
import json
import shutil
from pathlib import Path

from history.atomic_io import atomic_binary_writer, atomic_write_text


def digest(path, check=lambda:None):
    value=hashlib.sha256()
    with open(path,'rb') as stream:
        while True:
            check()
            block=stream.read(1024*1024)
            if not block: break
            value.update(block)
    return value.hexdigest()


def read_json(path):
    try:
        result=json.loads(Path(path).read_text(encoding='utf8'))
        return result if isinstance(result,dict) else {}
    except (OSError, ValueError):
        return {}


def _rebase(value, old, new):
    if isinstance(value,dict): return {k:_rebase(v,old,new) for k,v in value.items()}
    if isinstance(value,list): return [_rebase(v,old,new) for v in value]
    if isinstance(value,tuple): return tuple(_rebase(v,old,new) for v in value)
    if isinstance(value,str):
        try:
            path=Path(value)
            if path.is_absolute(): return str(new/path.relative_to(old))
        except (ValueError,OSError): pass
    return value


def prepare_output(task, processor, journal, row, reuse, check):
    """Copy, never link, reused data. Every attempt writes only its own folder."""
    from history.pipeline_state import stage_statuses
    old=Path(task.task_folder_path).resolve() if reuse else None
    initial=stage_statuses(task) if reuse else {}
    source=read_json(old/'attempt.json') if old else {}
    task.task_folder_path=''
    processor.create_task_folder(task)
    new=Path(task.task_folder_path).resolve()
    if old == new:
        raise ValueError('Новая попытка должна иметь отдельный каталог')
    journal.update(row,output_path=str(new),reused_from=str(old) if old else None,
        producer_attempt_id=source.get('attempt_id'),initial_stages=initial,
        inherited_stages=source.get('stages',{}))
    if old:
        files=[p for p in old.rglob('*') if p.is_file()]
        excluded={'attempt.json','manifest.json','execution_manifest.json','run_log.txt'}
        files=[p for p in files if p.name not in excluded and not p.name.endswith('.tmp')]
        size=sum(p.stat().st_size for p in files)
        from compute.disk_budget import SAFETY_RESERVE
        if shutil.disk_usage(new).free < size + SAFETY_RESERVE:
            import errno
            raise OSError(errno.ENOSPC,'Недостаточно места для независимой копии результатов')
        for path in files:
            check()
            if path.is_symlink(): raise ValueError('Символические ссылки в результатах не поддерживаются')
            target=new/path.relative_to(old)
            with atomic_binary_writer(target,cancel_check=check) as stream, path.open('rb') as source_file:
                while True:
                    check()
                    block=source_file.read(1024*1024)
                    if not block: break
                    stream.write(block)
        # Disk-backed checkpoints and ML journals contain absolute paths.
        # Rebase the copied versions so subsequent ML cannot write into old runs.
        from history.checkpoints import feature_checkpoint_available, restore_feature_checkpoint, save_feature_checkpoint
        if feature_checkpoint_available(str(new)):
            restore_feature_checkpoint(task,str(new))
            for field in ('knn_results','statistics_results','synchronization_results'):
                setattr(task,field,_rebase(getattr(task,field),old,new))
            save_feature_checkpoint(task,cancel_check=check)
        ml=new/'.ml_operations.json'
        if ml.is_file():
            data=read_json(ml)
            atomic_write_text(ml,json.dumps(_rebase(data,old,new),ensure_ascii=False,indent=2))
    return new


def seal_attempt(journal,row,task):
    """Scientific snapshot is immutable; history acknowledgements stay in batch.json."""
    destination=journal.path.parent/'attempts'/row['attempt_id']/'attempt.json'
    if destination.exists():
        existing=read_json(destination)
        if existing.get('attempt_id')!=row['attempt_id']:
            raise ValueError('Неверная идентичность сохранённой попытки')
        return str(destination)
    folder=Path(row['output_path']) if row.get('output_path') and 'initial_stages' in row else None
    manifest=read_json(folder/'manifest.json') if folder else {}
    if folder and hasattr(task,'resolve_pipeline'):
        from history.pipeline_state import stage_statuses, stage_signatures
        statuses,signatures=stage_statuses(task),stage_signatures(task)
        manifest['stages']={name:{'status':state,'signature':signatures[name]} for name,state in statuses.items()}
    stages={}
    for name, state in manifest.get('stages',{}).items():
        reused=(row.get('initial_stages',{}).get(name)=='ready'
                and state.get('status')=='ready' and task.analysis_mode!='2d')
        inherited=row.get('inherited_stages',{}).get(name,{})
        stages[name]={**state,'mode':'reused' if reused else 'computed' if state.get('status')=='ready' else 'not_completed',
            'producer_attempt_id':(inherited.get('producer_attempt_id') or row.get('producer_attempt_id'))
                                  if reused else row['attempt_id'],
            'producer_path':inherited.get('producer_path',row.get('reused_from')) if reused else row.get('output_path')}
    artifacts=[]
    complete=row.get('outcome') in {'completed','partial_failed'}
    if folder and folder.is_dir():
        for path in sorted(folder.rglob('*')):
            if not path.is_file() or path.name=='attempt.json' or path.name.endswith('.tmp'): continue
            artifacts.append({'path':path.relative_to(folder).as_posix(),
                'bytes':path.stat().st_size,'sha256':digest(path) if complete else None})
    protocol=getattr(task,'execution_protocol',None)
    payload={k:row.get(k) for k in ('attempt_id','task_id','task_name','started_at','finished_at',
        'duration_seconds','outcome','settings','error','traceback','ml_summary',
        'input_sha256','output_path','reused_from','producer_attempt_id')}
    payload.update(schema_version=1,batch_id=journal.document['batch_id'],stages=stages,
        output_path=str(folder) if folder else None,
        requested_backend=journal.document.get('requested_backend'),
        execution=protocol.to_dict() if protocol else read_json(folder/'execution_manifest.json') if folder else {},
        backend=row.get('backend'),artifacts=artifacts)
    text=json.dumps(payload,ensure_ascii=False,indent=2)
    # Only touch a folder assigned to this attempt. A pre-validation failure
    # may still point to a previous attempt and must not change it.
    if folder and 'initial_stages' in row:
        atomic_write_text(folder/'attempt.json',text)
    atomic_write_text(destination,text)
    return str(destination)
