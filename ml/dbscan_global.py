"""Global exact Euclidean DBSCAN with bounded-memory CPU / adaptive CUDA.

Spatial *image* tiles are never used. GPU pairwise connectivity is employed
only when an O(N^2) candidate bound is small enough; on dense datasets the
exact cKDTree flood-fill is considerably faster and uses O(N) radius-query
memory. The chosen backend is reported, never misrepresented as CUDA.

The core graph and the noise mask are equivalent to DBSCAN's; border points
that touch multiple components are assigned to their nearest core, with a
stable source-index tie break. As in sklearn, border assignment can differ
without changing the core cluster partition.
"""
from __future__ import annotations

from collections import deque

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from .errors import ClusteringError

# A one-dimensional bound on possible (not necessarily real) candidate edges
# keeps CUDA from spending minutes comparing very dense feature-space blocks.
# Exact CPU flood-fill becomes cheaper long before such a graph is enumerated.
_GPU_MAX_CANDIDATE_PAIRS = 4_000_000
_GPU_MAX_CORE_POINTS = 20_000


class _UnionFind:
    def __init__(self, n):
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, a):
        parent = self.parent
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return a

    def merge(self, a, b):
        a, b = self.find(int(a)), self.find(int(b))
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def _notify(cb, fraction, message):
    if cb is not None:
        cb('clustering', max(0., min(1., fraction)), message)


def _find_cores(values, eps, min_samples):
    # The point itself is included in sklearn's min_samples convention.
    # A k-NN query requires only O(n * k) temporary memory rather than the
    # potentially O(n**2) full radius-neighborhood graph.
    kth_distances, _ = cKDTree(values).query(
        values, k=min_samples, distance_upper_bound=eps, workers=-1)
    kth = kth_distances if min_samples == 1 else kth_distances[:, -1]
    core_mask = np.asarray(kth <= eps, dtype=bool)
    return core_mask, np.flatnonzero(core_mask)


def _cpu_core_components(values, eps, min_samples, batch_size,
                         progress_callback, cancel_callback, *, precomputed=None):
    """Exact flood fill on *unseen* cores, with adaptive tree rebuilding.

    Every core is discovered precisely once and points are linked to their
    BFS seed. A radius query returns at most one O(n) list, even if a single
    component contains all n observations. No explicit edge graph is built.
    """
    if precomputed is None:
        _notify(progress_callback, .04, 'DBSCAN CPU: определение ядровых точек…')
        cores, core_ids = _find_cores(values, eps, min_samples)
    else:
        cores, core_ids = precomputed
    count = len(core_ids)
    _notify(progress_callback, .15, f'DBSCAN CPU: ядровых точек {count:,}, построение связности…')
    uf = _UnionFind(count)
    if count == 0:
        return cores, core_ids, uf

    discovered = np.zeros(count, dtype=bool)
    remaining = count
    tree_ids = np.arange(count, dtype=np.int64)
    tree = cKDTree(values[core_ids])
    next_rebuild = count // 2
    processed = 0
    # Bounded bulk-query mode uses SciPy's native parallel cKDTree search.
    # It can be considerably faster than Python-driven scalar searches in
    # high-dimensional datasets, without building a global neighbor graph.
    typical_hits = 32.
    max_query_results = 80_000  # approximate retained neighbor IDs per batch
    for seed in range(count):
        if discovered[seed]:
            continue
        if cancel_callback is not None:
            cancel_callback()
        discovered[seed] = True
        remaining -= 1
        queue = deque((seed,))
        while queue and remaining:
            if cancel_callback is not None:
                cancel_callback()
            if remaining <= next_rebuild:
                tree_ids = np.flatnonzero(~discovered)
                if not len(tree_ids):
                    break
                tree = cKDTree(values[core_ids[tree_ids]])
                next_rebuild = remaining // 2
            # In dense groups, even 100 simultaneous queries can create
            # millions of duplicate candidate IDs. Scale the number of
            # probes down using recent observed neighborhood size.
            batch_count = min(len(queue), max(1, int(batch_size)), 256,
                              max(1, int(max_query_results / max(1., typical_hits))))
            batch = [queue.popleft() for _ in range(batch_count)]
            query = values[core_ids[np.asarray(batch, dtype=np.int64)]]
            if batch_count == 1:
                neighbor_lists = [tree.query_ball_point(query[0], r=eps)]
            else:
                neighbor_lists = tree.query_ball_point(query, r=eps, workers=-1)
            hits = 0
            for neighbors in neighbor_lists:
                hits += len(neighbors)
                if not neighbors:
                    continue
                candidates = tree_ids[np.asarray(neighbors, dtype=np.int64)]
                fresh = candidates[~discovered[candidates]]
                if not len(fresh):
                    continue
                discovered[fresh] = True
                remaining -= len(fresh)
                # Any newly discovered vertex belongs to this BFS component.
                uf.parent[fresh] = seed
                queue.extend(fresh.tolist())
            typical_hits = .65 * typical_hits + .35 * (hits / batch_count)
            processed += batch_count
            if processed % max(1, batch_size) < batch_count:
                _notify(progress_callback, .15 + .70 * (count - remaining) / count,
                        f'DBSCAN CPU: найдено {count - remaining:,}/{count:,} ядровых точек')
        # BFS exhaustion means no unseen vertex is adjacent to this component.
    _notify(progress_callback, .85, f'DBSCAN CPU: компоненты ядра построены ({count:,} точек)')
    return cores, core_ids, uf


def _potential_gpu_edges(values, core_ids, eps):
    """Cheap exact upper bound using a single feature-coordinate projection."""
    ordered = np.sort(values[core_ids, 0], kind='stable')
    right = np.searchsorted(ordered, ordered + eps, side='right')
    return int(np.sum(right.astype(np.int64) - np.arange(len(ordered)) - 1))


def _gpu_core_components(values, eps, min_samples, batch_size,
                         progress_callback, cancel_callback):
    """Use CUDA for a sparse candidate graph; switch to exact CPU for dense.

    GPU pair computations are bounded and never implicitly change the radius.
    Rows of GPU FP32 candidate pairs are rechecked exactly on CPU as float64.
    Dense runs avoid quadratic candidate materialization entirely.
    Returns (core mask, IDs, DSU, actual backend, decision reason).
    """
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError('CUDA-устройство отсутствует')
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(f'CuPy/CUDA недоступны: {exc}') from exc

    _notify(progress_callback, .04, 'DBSCAN GPU: определение ядровых точек (cKDTree CPU)…')
    cores, core_ids = _find_cores(values, eps, min_samples)
    m = len(core_ids)
    _notify(progress_callback, .12, f'DBSCAN: ядровых точек {m:,}; выбор безопасного алгоритма…')
    if cancel_callback is not None:
        cancel_callback()
    if m <= 1:
        return cores, core_ids, _UnionFind(m), 'gpu-cupy', None

    if m > _GPU_MAX_CORE_POINTS:
        bound = None
    else:
        bound = _potential_gpu_edges(values, core_ids, eps)
    if bound is None or bound > _GPU_MAX_CANDIDATE_PAIRS:
        explanation = (f'плотный граф ядра ({m:,} точек, '
                       f'потенциальных пар: {bound if bound is not None else "более лимита"}); '
                       'выбран точный CPU cKDTree вместо квадратичного GPU-перебора')
        _notify(progress_callback, .15, f'DBSCAN: {explanation}')
        result = _cpu_core_components(
            values, eps, min_samples, batch_size,
            progress_callback, cancel_callback, precomputed=(cores, core_ids))
        return (*result, 'cpu-kdtree', explanation)

    core_ids = core_ids[np.argsort(values[core_ids, 0], kind='stable')]
    uf = _UnionFind(m)
    rows_per_block = max(32, min(256, int(batch_size) // 2))
    cols_per_block = min(m, max(256, min(4096, int(batch_size) * 8)))
    core_values = np.ascontiguousarray(values[core_ids], dtype=np.float64)
    center = core_values.mean(axis=0)
    centered = core_values - center
    max_abs = float(np.max(np.abs(centered)))
    scale = float(2. ** np.floor(np.log2(max_abs))) if max_abs > 0 else 1.0
    gpu_cores = cp.asarray(centered / scale, dtype=cp.float32)
    core_norms = cp.sum(gpu_cores * gpu_cores, axis=1)
    eps2 = float(eps / scale) ** 2
    try:
        for start in range(0, m, rows_per_block):
            if cancel_callback is not None:
                cancel_callback()
            stop = min(start + rows_per_block, m)
            for begin in range(start, m, cols_per_block):
                if cancel_callback is not None:
                    cancel_callback()
                end = min(begin + cols_per_block, m)
                if core_values[begin, 0] - core_values[stop - 1, 0] > eps:
                    break  # Safe: all LATER blocks are farther in first feature.
                src_roots = {uf.find(i) for i in range(start, stop)}
                if len(src_roots) == 1:
                    target_roots = {uf.find(i) for i in range(begin, end)}
                    if len(target_roots) == 1 and src_roots == target_roots:
                        continue
                dot = gpu_cores[start:stop] @ gpu_cores[begin:end].T
                d2 = (core_norms[start:stop, None]
                      + core_norms[None, begin:end] - 2 * dot)
                margin = (128 * np.finfo(np.float32).eps *
                          (core_norms[start:stop, None] + core_norms[None, begin:end])
                          + 1e-8)
                hit_r, hit_c = np.nonzero(cp.asnumpy(d2 <= eps2 + margin))
                if not len(hit_r):
                    continue
                sources = hit_r.astype(np.int64) + start
                targets = hit_c.astype(np.int64) + begin
                upper = targets > sources
                sources, targets = sources[upper], targets[upper]
                if not len(sources):
                    continue
                # Vectorized double-precision validation for every candidate
                # in a bounded block, instead of Python loops over all rows.
                diff = core_values[targets] - core_values[sources]
                exact = np.einsum('ij,ij->i', diff, diff) <= eps * eps
                sources, targets = sources[exact], targets[exact]
                for i, j in zip(sources, targets):
                    uf.merge(i, j)
                # Do not break out of the block loop here: more distant
                # blocks may contain cores of OTHER components that connect
                # to the source; the former early-break corrupted clusters.
            _notify(progress_callback, .15 + .70 * stop / m,
                    f'DBSCAN GPU: связность ядра {stop:,}/{m:,}')
        return cores, core_ids, uf, 'gpu-cupy', None
    finally:
        del gpu_cores, core_norms
        cp.get_default_memory_pool().free_all_blocks()


def _labels_from_cores(values, core_mask, core_ids, uf, eps, callback=None):
    n = len(values)
    labels = np.full(n, -1, dtype=np.int32)
    if not len(core_ids):
        return labels
    # Each connected component has exactly one representative. Use the
    # earliest original index as a stable, contiguous label-ordering rule.
    roots = np.fromiter((uf.find(i) for i in range(len(core_ids))),
                        dtype=np.int64, count=len(core_ids))
    unique_roots, inverse = np.unique(roots, return_inverse=True)
    first_indices = np.full(len(unique_roots), len(values), dtype=np.int64)
    np.minimum.at(first_indices, inverse, core_ids)
    order = np.argsort(first_indices, kind='stable')
    cluster_for_group = np.empty(len(order), dtype=np.int32)
    cluster_for_group[order] = np.arange(len(order), dtype=np.int32)
    core_labels = cluster_for_group[inverse]
    labels[core_ids] = core_labels
    borders = np.flatnonzero(~core_mask)
    if len(borders):
        tree = cKDTree(values[core_ids])
        distance, nearest = tree.query(values[borders], k=1,
                                       distance_upper_bound=eps, workers=-1)
        ok = (nearest < len(core_ids)) & (distance <= eps)
        labels[borders[ok]] = core_labels[nearest[ok]]
    return labels


def global_dbscan(values, eps, min_samples, *, device='cpu',
                  small_threshold=2500, batch_size=512,
                  progress_callback=None, cancel_callback=None):
    """Return (labels, actual_backend, diagnostic_reason)."""
    values = np.ascontiguousarray(values, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ClusteringError('DBSCAN: ожидается непустая конечная матрица признаков')
    if eps <= 0 or min_samples < 1 or batch_size < 1:
        raise ClusteringError('DBSCAN: eps > 0, min_samples ≥ 1, размер пакета ≥ 1')
    if cancel_callback is not None:
        cancel_callback()
    reason = None
    if str(device).lower() == 'gpu' and len(values) > small_threshold:
        try:
            cores, core_ids, uf, backend, reason = _gpu_core_components(
                values, eps, min_samples, batch_size,
                progress_callback, cancel_callback)
        except Exception as exc:
            if cancel_callback is not None:
                cancel_callback()
            reason = f'{type(exc).__name__}: {exc}'
            _notify(progress_callback, 0, f'DBSCAN: GPU недоступен ({reason}); переход на CPU')
        else:
            labels = _labels_from_cores(values, cores, core_ids, uf, eps)
            _notify(progress_callback, 1., f'DBSCAN: глобальная кластеризация {backend} завершена')
            return labels, backend, reason
    if len(values) <= small_threshold:
        labels = DBSCAN(eps=float(eps), min_samples=int(min_samples)).fit_predict(values)
        _notify(progress_callback, 1., 'DBSCAN: глобальная кластеризация sklearn завершена')
        return labels.astype(np.int32, copy=False), 'cpu-sklearn', reason
    cores, core_ids, uf = _cpu_core_components(values, eps, min_samples, batch_size,
                                               progress_callback, cancel_callback)
    labels = _labels_from_cores(values, cores, core_ids, uf, eps)
    _notify(progress_callback, 1., 'DBSCAN: глобальная кластеризация CPU завершена')
    return labels, 'cpu-kdtree', reason
