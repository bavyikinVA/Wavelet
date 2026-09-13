"""Exact viewport queries; original coordinates remain untouched."""
import numpy as np


class PointIndex:
    def __init__(self, points):
        self.points = np.asarray(points)
        self.order = np.argsort(self.points[:, 0], kind='stable')
        self.x = self.points[self.order, 0]

    def query(self, bounds):
        x0, x1, y0, y1 = bounds
        start = np.searchsorted(self.x, x0, side='left')
        end = np.searchsorted(self.x, x1, side='right')
        indices = self.order[start:end]
        y = self.points[indices, 1]
        # Preserve drawing order, including coincident points with different colors.
        return np.sort(indices[(y >= y0) & (y <= y1)])


def clip_segments(segments, bounds):
    """Liang–Barsky clipping, including lines whose endpoints are both outside."""
    edges = np.asarray(segments, dtype=float).reshape(-1, 2, 2)
    if not len(edges):
        return edges
    x0, x1, y0, y1 = bounds
    a, b = edges[:, 0], edges[:, 1]
    delta = b-a
    lower, upper = np.zeros(len(edges)), np.ones(len(edges))
    valid = np.isfinite(edges).all(axis=(1, 2))
    for p, q in ((-delta[:, 0], a[:, 0]-x0), (delta[:, 0], x1-a[:, 0]),
                 (-delta[:, 1], a[:, 1]-y0), (delta[:, 1], y1-a[:, 1])):
        parallel = p == 0
        valid &= ~(parallel & (q < 0))
        ratio = np.divide(q, p, out=np.zeros_like(q), where=~parallel)
        lower = np.where(p < 0, np.maximum(lower, ratio), lower)
        upper = np.where(p > 0, np.minimum(upper, ratio), upper)
    valid &= lower <= upper
    return np.stack((a[valid]+lower[valid, None]*delta[valid],
                     a[valid]+upper[valid, None]*delta[valid]), axis=1)
