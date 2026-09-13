"""ML services for clustering wavelet-derived point features."""

from .errors import ClusteringError

__all__ = ["ClusteringError", "extract_knn_features", "run_clustering"]


def __getattr__(name):
    if name in {"extract_knn_features", "run_clustering"}:
        from importlib import import_module
        value = getattr(import_module('.clustering', __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
