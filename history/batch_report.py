"""Offline JSON + HTML reports; every export is an immutable revision."""
import html
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from uuid import uuid4

from history.atomic_io import atomic_write_text
from history.attempts import read_json


def environment():
    packages={}
    for name in ('numpy','scipy','numba','cupy-cuda12x','scikit-learn','customtkinter'):
        try: packages[name]=version(name)
        except PackageNotFoundError: packages[name]=None
    from history.attempts import digest
    root=Path(__file__).resolve().parents[1]
    files=[root/'main.py']
    for directory in ('compute','history','ml'):
        files.extend((root/directory).rglob('*.py'))
    import hashlib
    code=hashlib.sha256()
    for path in sorted(files):
        code.update(path.relative_to(root).as_posix().encode())
        code.update(digest(path).encode())
    return {'python':sys.version.split()[0],'platform':platform.platform(),
            'packages':packages,'implementation_sha256':code.hexdigest()}


LABELS={'completed':'Готово','partial_failed':'Частично выполнено','failed':'Ошибка',
    'cancelled':'Отменено','pending':'Ожидает','running':'Выполняется',
    'saving':'Фиксация результата','interrupted':'Прервано','history_pending':'Ожидает записи истории'}


def build_report(journal):
    from compute.batch_runner import summarize
    data=json.loads(json.dumps(journal.document))
    summary=summarize(data['tasks'])
    # An explicit stop between tasks has no cancelled row; keep the batch state.
    status=data.get('status',summary['status'])
    rows=[]
    for task in data['tasks']:
        attempts=[]
        for row in [*task.get('previous_attempts',[]),task]:
            path=journal.path.parent/'attempts'/row['attempt_id']/'attempt.json'
            snapshot=read_json(path)
            attempts.append({**{k:v for k,v in row.items() if k!='previous_attempts'},
                'scientific_snapshot':snapshot or None})
        rows.append({'task_id':task['task_id'],'task_name':task['task_name'],
                     'task_number':task['task_number'],'status':task['status'],'attempts':attempts})
    return {'schema_version':1,'batch_id':data['batch_id'],
        'experiment_id':data.get('experiment_id'),'status':status,'summary':summary,
        'batch_error':data.get('batch_error'),
        'created_at':data.get('created_at'),'finished_at':data.get('finished_at'),
        'exported_at':datetime.now(timezone.utc).isoformat(),
        'requested_backend':data.get('requested_backend'),'environment':data.get('environment'),
        'tasks':rows,'notes':['Неизвестные данные обозначены null. Наличие файла не означает успешное вычисление.',
        'Хеши артефактов зафиксированы в момент завершения попытки; экспорт отчёта не проверяет их заново.',
        'HTML содержит сведения отчёта; сами численные массивы хранятся в каталогах результатов.']}


def render_html(report):
    esc=lambda value:html.escape(str(value if value is not None else '—'),quote=True)
    sections=[]
    for task in report['tasks']:
        details=[]
        for row in task['attempts']:
            snapshot=row.get('scientific_snapshot') or {}
            stages=snapshot.get('stages',{})
            stage_rows=''.join('<tr><td>'+esc(stage)+'</td><td>'+esc(value.get('status'))+
                '</td><td>'+esc(value.get('mode'))+'</td><td>'+esc(value.get('producer_attempt_id'))+'</td></tr>'
                for stage,value in stages.items())
            protocol=snapshot.get('execution',{})
            actual=sorted({v.get('actual_backend','unknown') for v in protocol.get('stages',{}).values()})
            details.append(f'''<details><summary>{esc(row['attempt_id'])} · {esc(LABELS.get(row['status'],row['status']))}</summary>
<p>Время: {esc(row.get('started_at'))} → {esc(row.get('finished_at'))}; длительность: {esc(row.get('duration_seconds'))} с</p>
<p>Фактические backend: {esc(', '.join(actual) or None)}. История ожидает записи: {esc(bool(row.get('history_pending')))}</p>
<p>Источник SHA256: {esc(row.get('input_sha256'))}</p><p>Результаты: {esc(row.get('output_path'))}</p>
<p class="error">{esc(row.get('error',''))} {esc(row.get('history_error',''))} {esc(row.get('provenance_error',''))}</p>
<table><thead><tr><th>Этап</th><th>Состояние</th><th>Вычислен / reused</th><th>Попытка-производитель</th></tr></thead><tbody>{stage_rows}</tbody></table>
<details><summary>Параметры, протокол и артефакты</summary><pre>{esc(json.dumps(row,ensure_ascii=False,indent=2))}</pre></details></details>''')
        sections.append(f'<section><h2>{esc(task["task_number"])}. {esc(task["task_name"])}</h2><p>{esc(LABELS.get(task["status"],task["status"]))}</p>'+''.join(details)+'</section>')
    counts=report['summary']
    return f'''<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Отчёт исследования Wavelets</title><style>
body{{font:16px/1.5 system-ui,sans-serif;margin:32px auto;max-width:1120px;padding:0 24px;color:#192c3a;background:#f6f8fa}}
h1{{font-size:30px}}h2{{font-size:20px}}section,header{{background:white;border:1px solid #d9e1e8;border-radius:12px;padding:20px;margin:18px 0}}
summary{{cursor:pointer;font-weight:600;padding:8px 0}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border-bottom:1px solid #ddd;padding:8px;text-align:left;overflow-wrap:anywhere}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;background:#f3f6f8;padding:16px}}p{{overflow-wrap:anywhere}}.error{{color:#9b2634}}
@media print{{body{{background:white;margin:0}}section{{break-inside:avoid}}details> *{{display:block}}}}
</style></head><body><header><h1>Отчёт исследования</h1>
<p>Wavelets Analysis Studio · {esc(report['batch_id'])}</p><h2>{esc(LABELS.get(report['status'],report['status']))}</h2>
<p class="error">{esc(report.get('batch_error') or '')}</p>
<p>GPU/CPU запрошен: {esc(report['requested_backend'])}. Задач: {len(report['tasks'])}; готово: {counts['completed']}; частично: {counts['partial_failed']}; ошибок: {counts['failed']}; история ожидает записи: {counts['history_pending']}.</p>
<p>Начало: {esc(report['created_at'])}. Завершение: {esc(report['finished_at'])}. Экспорт: {esc(report['exported_at'])}.</p></header>
{''.join(sections)}<section><h2>Среда и ограничения</h2><pre>{esc(json.dumps(report['environment'],ensure_ascii=False,indent=2))}</pre>
{''.join('<p>'+esc(note)+'</p>' for note in report['notes'])}</section></body></html>'''


def export_report(journal):
    report=build_report(journal)
    folder=journal.path.parent/'reports'/uuid4().hex
    folder.mkdir(parents=True,exist_ok=False)
    atomic_write_text(folder/'report.json',json.dumps(report,ensure_ascii=False,indent=2))
    atomic_write_text(folder/'report.html',render_html(report))
    return folder/'report.html'
