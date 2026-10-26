"""Inspect the Tcl/Tk installation used by the current Python interpreter.

Run from the Windows project's activated .venv:
    python tools/diagnose_tk.py

Read-only diagnostic; never repairs or modifies a Python installation.
"""
from pathlib import Path
import os
import sys


def main():
    print('Executable:', sys.executable)
    print('Python base:', sys.base_prefix)
    print('TCL_LIBRARY environment:', os.environ.get('TCL_LIBRARY', '(unset)'))
    print('TK_LIBRARY environment:', os.environ.get('TK_LIBRARY', '(unset)'))
    try:
        import tkinter as tk
    except Exception as exc:
        print('Tkinter import error:', repr(exc))
        return 2
    for attempt in range(1, 6):
        root = None
        try:
            root = tk.Tk()
            root.withdraw()
            tcl_dir = Path(root.tk.eval('info library'))
            tk_dir = Path(root.tk.eval('set tk_library'))
            print(f'Root #{attempt}: Tcl {root.tk.eval("info patchlevel")}; Tk {root.tk.eval("package require Tk")}')
            for file in (tcl_dir / 'init.tcl', tk_dir / 'tk.tcl', tk_dir / 'icons.tcl'):
                print('  ', 'OK' if file.is_file() else 'MISSING', file)
                if not file.is_file():
                    return 2
        except Exception as exc:
            print(f'Root #{attempt} failed: {type(exc).__name__}: {exc}')
            return 2
        finally:
            if root is not None:
                root.destroy()
    print('Tcl/Tk check: PASS (5 roots created; libraries found).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
