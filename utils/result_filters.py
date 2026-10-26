"""Pure dependent result filters; no Tkinter state or callbacks.

The order of the field keys is the cascade order.  A parent selection limits
available values downstream. Invalid stale values reset to the caller's
fallback rather than raising KeyError.
"""
from dataclasses import dataclass


TRANSLATIONS = {'row': 'По строкам', 'col': 'По столбцам',
                'max': 'Максимумы', 'min': 'Минимумы',
                'upper': 'Верхняя огибающая', 'lower': 'Нижняя огибающая'}


@dataclass
class FilterStep:
    key: str
    options: list
    selected: str


def cascade_filter(items, selections, *, keys=('scale', 'channel', 'orientation',
                                                 'feature_axis', 'point_kind', 'algorithm')):
    """Return ordered UI choices and surviving artifacts, preserving 0°."""
    remaining = list(items)
    steps = []
    for key in keys:
        values = {str(item[key]) for item in remaining if item.get(key) is not None}
        def sort_key(value):
            if key in {'scale', 'orientation'}:
                try:
                    return (0, float(value))
                except (TypeError, ValueError):
                    pass
            return (1, value)
        mapping = {TRANSLATIONS.get(raw, raw): raw for raw in sorted(values, key=sort_key)}
        chosen = selections.get(key, 'Все')
        if chosen not in mapping:
            chosen = 'Все'
        steps.append(FilterStep(key, ['Все', *mapping], chosen))
        if chosen != 'Все':
            raw = mapping[chosen]
            remaining = [item for item in remaining if str(item.get(key)) == raw]
    return steps, remaining
