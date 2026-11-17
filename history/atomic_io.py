"""Replace a complete file on the same filesystem; never truncate its predecessor."""
from contextlib import contextmanager
import io
import os
from pathlib import Path
import tempfile


@contextmanager
def atomic_binary_writer(path, *, cancel_check=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        if cancel_check is not None:
            cancel_check()
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_text(path, text):
    with atomic_binary_writer(path) as stream:
        stream.write(text.encode('utf-8'))


def atomic_savez(path, **arrays):
    import numpy as np
    with atomic_binary_writer(path) as stream:
        np.savez(stream, **arrays)


def atomic_savez_compressed(path, **arrays):
    import numpy as np
    with atomic_binary_writer(path) as stream:
        np.savez_compressed(stream, **arrays)


def atomic_savetxt(path, values, **options):
    import numpy as np
    with atomic_binary_writer(path) as stream:
        np.savetxt(stream, values, **options)


@contextmanager
def atomic_text_writer(path, *, encoding='utf-8', newline=None, cancel_check=None):
    """Stream CSV/TXT in bounded memory; publish only after a complete write.

    Use a text wrapper without closing the underlying atomic binary stream.
    This preserves the writer's fsync and same-directory os.replace semantics.
    """
    with atomic_binary_writer(path, cancel_check=cancel_check) as stream:
        wrapper = io.TextIOWrapper(stream, encoding=encoding, newline=newline)
        try:
            yield wrapper
            wrapper.flush()
        finally:
            wrapper.detach()


def atomic_savefig(path, *, figure=None, **options):
    """Save a matplotlib image without ever exposing a half-written PNG."""
    from matplotlib import pyplot as plt
    format_name = Path(path).suffix.lstrip('.').lower()
    if not format_name:
        raise ValueError('Формат изображения должен быть указан в имени файла')
    with atomic_binary_writer(path) as stream:
        (figure if figure is not None else plt).savefig(
            stream, format=format_name, **options)
