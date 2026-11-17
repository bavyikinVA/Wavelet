"""Turn background Tcl errors (normally printed outside pytest) into failures."""
import pytest


@pytest.fixture(autouse=True)
def reject_background_tcl_errors(monkeypatch):
    import tkinter as tk
    roots = []
    errors = []
    original = tk.Tk.__init__

    def initialize(root, *args, **kwargs):
        original(root, *args, **kwargs)
        roots.append(root)
        # Register on the interpreter, so it survives deletion of root commands.
        root.tk.createcommand('bgerror', lambda message: errors.append(str(message)))

    monkeypatch.setattr(tk.Tk, '__init__', initialize)
    yield
    for root in roots:
        try:
            root.tk.eval('update')
            for identifier in root.tk.splitlist(root.tk.call('after', 'info')):
                script, kind = root.tk.call('after', 'info', identifier)
                words = root.tk.splitlist(script)
                if words and not root.tk.call('info', 'commands', words[0]):
                    errors.append(f'Pending {kind} refers to deleted command: {words[0]}')
        except tk.TclError as error:
            errors.append(str(error))
    assert not errors, 'Background Tcl errors: ' + '; '.join(errors)
