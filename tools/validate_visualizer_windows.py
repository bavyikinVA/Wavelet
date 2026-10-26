"""Run the *actual* Windows viewer acceptance; never pretend this ran in Linux.

Usage from the project venv:
    python tools/validate_visualizer_windows.py --experiment "C:\\...\\wavelet_studio_..."

Produces one Markdown run log in the current directory. No DB updates or
scientific file modifications. Requires pytest, CustomTkinter and Windows GUI.
"""
import argparse
import importlib.metadata
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import median

# When launched as `python tools/validate_visualizer_windows.py`, Python
# adds tools/ (not the project root) to sys.path. Add the root explicitly so
# the in-process catalogue benchmark can import history and utils.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def timed_catalog(experiment):
    from history.result_catalog import discover_results
    from utils.layer_catalog import search_results, prepare_search_index
    from utils.result_filters import cascade_filter
    start = time.perf_counter()
    items = discover_results(str(experiment))
    prepare_search_index(items)
    duration = time.perf_counter() - start
    repeats = []
    for _ in range(30):
        now = time.perf_counter()
        search_results(items, query='масштаб')
        repeats.append((time.perf_counter()-now)*1000)
    option_times = []
    for _ in range(30):
        now = time.perf_counter()
        cascade_filter(items, {}, keys=('scale','channel','orientation'))
        option_times.append((time.perf_counter()-now)*1000)
    repeats.sort(); option_times.sort()
    return [f'Результатов: {len(items)}', f'Обнаружение файлов: {duration:.3f} c',
            f'Поиск: median={median(repeats):.3f} мс, P95={repeats[int(.95*(len(repeats)-1))]:.3f} мс',
            f'Каскад: median={median(option_times):.3f} мс, P95={option_times[int(.95*(len(option_times)-1))]:.3f} мс']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, help='Реальный эксперимент для замера каталога')
    parser.add_argument('--smoke', action='store_true', help='Только тесты визуализатора, без полного набора')
    parser.add_argument('--output', type=Path, help='Итоговый Markdown файл')
    args = parser.parse_args()
    project = PROJECT_ROOT
    path = args.output or (project / f'viewer_validation_{datetime.now():%Y%m%d_%H%M%S}.md')
    if platform.system() != 'Windows':
        raise SystemExit('Эта приёмка предназначена для Windows; текущий запуск не засчитывается.')
    try:
        import customtkinter  # noqa: F401
    except ImportError as exc:
        raise SystemExit('В активном venv нет customtkinter. Установи requirements.txt.') from exc
    tests = ([str(project/'tests'/p) for p in (
        'test_result_viewer.py','test_ml_classes.py','test_ml_display_policy.py',
        'test_result_filter_model.py','test_result_descriptors.py',
        'test_layer_picker_gui.py','test_results_integration.py',
        'test_viewer_ml_gui_regression.py')]
        if args.smoke else [str(project/'tests')])
    command = [sys.executable,'-m','pytest','-q',*tests]
    started = time.perf_counter()
    result = subprocess.run(command, cwd=project, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, errors='replace')
    elapsed = time.perf_counter()-started
    # Retain the complete pytest output. The old [-100:] truncation hid the
    # primary Tcl/Matplotlib exception behind the final assertion summary.
    full_log = path.with_name(path.stem + '_pytest.txt')
    full_log.write_text(result.stdout, encoding='utf-8')
    try:
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                              '--format=csv,noheader'], text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, timeout=8)
        gpu_text = gpu.stdout.strip() if gpu.returncode == 0 else 'nvidia-smi недоступен'
    except (OSError, subprocess.TimeoutExpired):
        gpu_text = 'nvidia-smi недоступен'
    rows = ['# Приёмка визуализатора — Windows', '',
            f'- Время: {datetime.now().isoformat(timespec="seconds")}',
            f'- Система: {platform.platform()}', f'- Python: {sys.version.split()[0]}',
            f'- CustomTkinter: {importlib.metadata.version("customtkinter")}',
            f'- Команда: `{sys.executable} -m pytest -q ...`',
            f'- Время тестов: {elapsed:.2f} с',
            f'- Код завершения: {result.returncode}',
            f'- Полный журнал pytest: `{full_log.name}`',
            '', '## Результаты pytest (последние 100 строк)', '',
            '```text', *result.stdout.splitlines()[-100:], '```', '']
    if args.experiment:
        rows += ['## Замеры каталога реального эксперимента', '']
        if args.experiment.is_dir():
            try:
                rows += ['- '+row for row in timed_catalog(args.experiment)]
            except Exception as exc:
                rows += [f'- ОШИБКА замера: {type(exc).__name__}: {exc}']
        else:
            rows += ['- Указанный каталог отсутствует.']
        rows.append('')
    rows += ['## Требуется проверить вручную', '',
             '- [ ] Масштаб Windows 100%, 125%, 150%; без обрезанных карточек',
             '- [ ] A/B: независимые цвета/скрытые классы и переключение B',
             '- [ ] Экспорт PNG с >12 классами: видимые цвета и полная легенда',
             '- [ ] DBSCAN: три сохранённые категории, отключение одного класса',
             '- [ ] Два ML-запуска: легенда отличает запуски',
             '- [ ] Закрытие окна во время загрузки без TclError',
             '- [ ] Мониторинг VRAM/RAM и сохранение исходных NPY/CSV', '']
    path.write_text('\n'.join(rows),encoding='utf-8')
    print(f'Отчёт: {path}\nКод тестов: {result.returncode}')
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
