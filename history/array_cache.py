"""Shared, byte-bounded cache for immutable numerical payloads."""
from collections import OrderedDict
from threading import RLock
from functools import wraps
import numpy as np


def payload_size(value):
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, dict):
        return sum(payload_size(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(payload_size(v) + 64 for v in value)
    return len(value.encode('utf-8')) if isinstance(value, str) else 32


def byte_cache(budget=256 * 1024**2):
    def decorate(function):
        cache, lock = OrderedDict(), RLock()
        used = 0
        @wraps(function)
        def read(*key):
            nonlocal used
            with lock:
                if key in cache:
                    cache.move_to_end(key)
                    return cache[key][0]
            value = function(*key)
            size = payload_size(value)
            with lock:
                if key in cache:
                    return cache[key][0]
                if size <= budget:
                    while cache and used + size > budget:
                        _, (_, removed) = cache.popitem(last=False)
                        used -= removed
                    cache[key] = (value, size)
                    used += size
            return value
        def clear():
            nonlocal used
            with lock:
                cache.clear()
                used = 0
        read.cache_clear = clear
        return read
    return decorate
