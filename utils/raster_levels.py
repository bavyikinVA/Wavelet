"""Cached display levels. Source arrays and their coordinates stay unchanged."""
import math
import numpy as np


class RasterLevels:
    def __init__(self, array, shape, reduce=True):
        self.levels = [np.asarray(array)]
        self.shape = shape
        self.reduce = reduce

    def _next(self):
        source = self.levels[-1]
        h, w = source.shape[:2]
        out_shape = ((h+1)//2, (w+1)//2)
        if source.ndim == 3:
            total = np.zeros((*out_shape, source.shape[2]), dtype=np.float32)
            count = np.zeros(out_shape, dtype=np.float32)
            for y in range(2):
                for x in range(2):
                    part = source[y::2, x::2]
                    ph, pw = part.shape[:2]
                    total[:ph, :pw] += part
                    count[:ph, :pw] += 1
            result = (total/count[..., None]).astype(source.dtype)
        else:
            # Keep the strongest finite coefficient of each block, including
            # isolated positive/negative peaks. Never modify numerical exports.
            dtype = source.dtype if np.issubdtype(source.dtype, np.floating) else np.float64
            result = np.full(out_shape, np.nan, dtype=dtype)
            for y in range(2):
                for x in range(2):
                    part = source[y::2, x::2]
                    ph, pw = part.shape
                    target = result[:ph, :pw]
                    candidate = part.astype(dtype, copy=False)
                    choose = np.isfinite(candidate) & (~np.isfinite(target) | (np.abs(candidate) > np.abs(target)))
                    np.copyto(target, part, where=choose)
        self.levels.append(result)

    def view(self, xlim, ylim, pixels):
        source = self.levels[0]
        h, w = source.shape[:2]
        physical_h, physical_w = self.shape
        # Empty or malformed raster artifacts must not bring down the Tk draw
        # loop.  Point caches are handled by result_data, but keep this guard
        # for legacy/corrupt user files as well.
        if h <= 0 or w <= 0 or physical_h <= 0 or physical_w <= 0:
            return None
        sx, sy = physical_w/w, physical_h/h
        x0, x1 = sorted(xlim)
        y0, y1 = sorted(ylim)
        ratio = min((x1-x0)/sx/max(1, pixels[0]), (y1-y0)/sy/max(1, pixels[1]))
        level = max(0, int(math.floor(math.log2(max(1, ratio))))) if self.reduce else 0
        level = min(level, int(math.ceil(math.log2(max(h, w)))))
        while len(self.levels) <= level:
            self._next()
        step = 2**level
        array = self.levels[level]
        left = max(0, math.floor((x0+.5)/sx/step)-1)
        right = min(array.shape[1], math.ceil((x1+.5)/sx/step)+1)
        top = max(0, math.floor((y0+.5)/sy/step)-1)
        bottom = min(array.shape[0], math.ceil((y1+.5)/sy/step)+1)
        if right <= left or bottom <= top:
            return None
        extent = (left*step*sx-.5, right*step*sx-.5, bottom*step*sy-.5, top*step*sy-.5)
        return array[top:bottom, left:right], extent, (level, left, right, top, bottom)
