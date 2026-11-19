"""Validation of editable research and task metadata (independent of DB/GUI)."""

MAX_NAME_LENGTH = 120
MAX_NOTE_LENGTH = 4000


def normalize_metadata(name, note, *, allow_empty_name=False):
    if not isinstance(name, str):
        raise ValueError('Укажите название')
    clean_name = ' '.join(name.split())
    if not clean_name and not allow_empty_name:
        raise ValueError('Укажите название')
    if len(clean_name) > MAX_NAME_LENGTH:
        raise ValueError(f'Название не должно превышать {MAX_NAME_LENGTH} символов')
    if not isinstance(note, str):
        raise ValueError('Описание должно быть строкой')
    clean_note = note.strip().replace('\r\n', '\n')
    if len(clean_note) > MAX_NOTE_LENGTH:
        raise ValueError(f'Описание не должно превышать {MAX_NOTE_LENGTH} символов')
    return clean_name, clean_note
