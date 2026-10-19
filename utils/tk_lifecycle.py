"""Cancel Tcl timers before their owning widget's Python commands are deleted.

Installed once per Tcl interpreter, without replacing tkinter or library methods.
Only commands registered on the destroyed widget are matched; other timers stay.
"""
import tkinter as tk


def _cancel_owned(widget):
    commands = set(getattr(widget, '_tclCommands', None) or ())
    cancelled = []
    for identifier in widget.tk.splitlist(widget.tk.call('after', 'info')):
        script, kind = widget.tk.call('after', 'info', identifier)
        words = widget.tk.splitlist(script)
        if words and words[0] in commands:
            # Do not use after_cancel: Misc.destroy owns deletion of commands.
            widget.tk.call('after', 'cancel', identifier)
            cancelled.append(words[0])
    return cancelled


def _transfer_ctk_trackers(widget, commands):
    """CTk's process-wide trackers must survive replacement of their timer host."""
    import customtkinter as ctk
    for tracker, method, hosts in (
        (ctk.AppearanceModeTracker, ctk.AppearanceModeTracker.update,
         ctk.AppearanceModeTracker.app_list),
        (ctk.ScalingTracker, ctk.ScalingTracker.check_dpi_scaling,
         ctk.ScalingTracker.window_widgets_dict),
    ):
        if not any(command.endswith(method.__name__) and
                   command[:-len(method.__name__)].isdigit() for command in commands):
            continue
        tracker.update_loop_running = False
        for host in list(hosts):
            try:
                # No callback may migrate to a child of the closing Tk root.
                if host is widget or (isinstance(widget, tk.Tk) and host._root() is widget):
                    continue
                if host.winfo_exists():
                    host.after(tracker.update_loop_interval, method)
                    tracker.update_loop_running = True
                    break
            except tk.TclError:
                continue


def install_callback_cleanup(widget):
    root = widget._root()
    if getattr(root, '_owned_after_cleanup', False):
        return
    root._owned_after_cleanup = True

    def destroyed(event):
        owner = event.widget
        if not isinstance(owner, tk.Misc):
            return
        cancelled = _cancel_owned(owner)
        if cancelled:
            _transfer_ctk_trackers(owner, cancelled)

    # 'all' also sees Toplevel descendants; a root binding alone does not.
    root.bind_all('<Destroy>', destroyed, add='+')
