"""In-memory ML run report; no disk writes or scientific imports."""
from collections import Counter
STATUS = {'completed': 'Успешно', 'reused': 'Готово ранее', 'failed': 'Ошибка',
          'cancelled': 'Отменено', 'pending': 'Не выполнено',
          'saving': 'Результат сохранён; история не записана'}


def new_row(task, config):
    direction, channel, scale, point_type = config['dataset_key']
    channel = {'r': 'Красный', 'g': 'Зелёный', 'b': 'Синий',
               'gray': 'Серый'}.get(str(channel).lower(), str(channel))
    axis = {'row': 'R', 'rows': 'R', 'col': 'C', 'columns': 'C'}.get(direction, direction)
    point = {'max_by_row': 'max_R', 'min_by_row': 'min_R',
             'max_by_column': 'max_C', 'min_by_column': 'min_C'}.get(point_type, point_type)
    algorithm = config['algorithm']
    parameters = (f"k={config.get('n_clusters', 5)}, seed={config.get('random_state', 42)}"
                  if algorithm == 'kmeans' else
                  f"ε={config.get('eps', .8)}, min_samples={config.get('min_samples', 5)}")
    if algorithm == 'dbscan':
        parameters += f" · размер вычислительного пакета: {config.get('dbscan_batch_size', 512)}"
    features = 'исходные измерения KNN Info'
    parameters += f" · признаки: {features}"
    parameters += ' · стандартизация: ' + ('да' if config.get('standardize', True) else 'нет')
    return dict(task=task.task_name, algorithm='K-Means' if algorithm == 'kmeans' else 'DBSCAN',
                dataset=f'{channel}_{float(scale):g}_CWT_{axis}_{point}',
                parameters=parameters, configuration=dict(config), status='pending',
                error='', output_dir='', files={}, operation_id='')


def set_result(row, status, result):
    row.update(status=status, output_dir=result['output_dir'], files={
        name: result[name] for name in ('table_path', 'summary_path', 'feature_plot_path')
        if name in result})


def summary(rows):
    counts = Counter(row['status'] for row in rows)
    parts = [f'Всего: {len(rows)}']
    parts += [f'{STATUS[key]}: {counts[key]}' for key in STATUS if counts[key]]
    return ' · '.join(parts)

