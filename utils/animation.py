"""Interruptible, time-based disclosure animation on Tk's main loop."""
import time


def animate_visibility(widget, visible, show, hide, axis='height', duration_ms=180):
    state = getattr(widget, '_disclosure', None)
    scale = widget._get_widget_scaling()
    if state is None:
        state = widget._disclosure = dict(job=None, visible=bool(widget.winfo_manager()),
                                          pack=widget.pack_propagate(), grid=widget.grid_propagate())
        state['layout'] = 'pack' if any(child.winfo_manager() == 'pack' for child in widget.winfo_children()) else 'grid'
        def cleanup(event):
            if event.widget is widget and state['job'] is not None:
                widget.after_cancel(state['job'])
                state['job'] = None
        widget.bind('<Destroy>', cleanup, add='+')
    if state['job'] is None and state['visible'] == visible:
        return
    if state['job'] is not None:
        widget.after_cancel(state['job'])
        start = float(widget.cget(axis))
    else:
        current = widget.winfo_width() if axis == 'width' else widget.winfo_height()
        requested = widget.winfo_reqwidth() if axis == 'width' else widget.winfo_reqheight()
        if state['visible']:
            state['expanded'] = max(current, requested) / scale
            start = state['expanded']
        else:
            start = 0
            if axis == 'height' or 'expanded' not in state:
                state['expanded'] = requested / scale
    state['visible'] = visible
    end = state['expanded'] if visible else 0
    propagate = widget.pack_propagate if state['layout'] == 'pack' else widget.grid_propagate
    propagate(False)
    widget.configure(**{axis: max(1, start)})
    show()
    started = time.monotonic()

    def tick():
        state['job'] = None
        t = min(1., (time.monotonic() - started) * 1000 / duration_ms)
        eased = t * t * (3 - 2 * t)
        widget.configure(**{axis: max(1, start + (end - start) * eased)})
        if t < 1:
            state['job'] = widget.after(16, tick)
        else:
            if not visible:
                hide()
            widget.configure(**{axis: state['expanded']})
            propagate(bool(state[state['layout']]))

    state['job'] = widget.after(0, tick)
