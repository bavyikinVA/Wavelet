"""Lightweight ML parameter contract shared by planning and execution."""
import math
from .errors import ClusteringError


def _number(value, name, minimum, *, integer=False, positive=False):
    try:
        number = float(value)
        if isinstance(value, bool) or not math.isfinite(number):
            raise ValueError()
        if integer and not number.is_integer():
            raise ValueError()
        if number < minimum or (positive and number == minimum):
            raise ValueError()
    except (ValueError, TypeError, OverflowError) as error:
        bound = '>' if positive else '>='
        kind = 'целое' if integer else ''
        raise ClusteringError(f'{name}: требуется {kind} конечное число {bound} {minimum}') from error
    return int(number) if integer else number


def validate_ml_parameters(config, point_count=None):
    """Return normalized active parameters; None means count is not known yet."""
    result = dict(config)
    algorithm = str(config.get('algorithm', 'kmeans')).lower()
    result['algorithm'] = algorithm
    if algorithm not in {'kmeans', 'dbscan'}:
        raise ClusteringError(f'Неизвестный алгоритм ML: {algorithm}')
    if point_count is not None and point_count < 3:
        raise ClusteringError('Недостаточно KNN-точек: требуется не меньше 3')
    if algorithm == 'kmeans':
        k = _number(config.get('n_clusters', 5), 'k', 2, integer=True)
        if point_count is not None and k >= point_count:
            raise ClusteringError(f'k: должно быть меньше числа KNN-точек ({point_count})')
        result['n_clusters'] = k
        seed = _number(config.get('random_state', 42), 'random_state', 0, integer=True)
        if seed > 2**32 - 1:
            raise ClusteringError('random_state: должно быть не больше 4294967295')
        result['random_state'] = seed
    else:
        result['eps'] = _number(config.get('eps', .8), 'eps', 0, positive=True)
        result['min_samples'] = _number(config.get('min_samples', 5), 'min_samples', 1, integer=True)
        result['dbscan_batch_size'] = _number(config.get('dbscan_batch_size', 512),
            'dbscan_batch_size', 1, integer=True)

    return result
