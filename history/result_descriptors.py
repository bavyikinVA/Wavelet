"""Stable logical metadata for viewer artifacts (no disk I/O or DB migration).

The descriptor contains *known* provenance only. A pixel coordinate frame is
not silently assumed when geometric transforms/crops are undocumented.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResultDescriptor:
    artifact_id: str
    task_id: str
    category: str
    operation_id: str | None
    source_image: str | None
    coordinate_frame: str | None
    representation: str
    scale: str | None
    channel: str | None
    cwt_direction: str | None
    feature_axis: str | None
    point_kind: str | None
    algorithm: str | None


def describe_result(item, experiment_folder):
    """Portable ID relative to the experiment, plus explicit unknown fields."""
    folder = Path(experiment_folder).resolve()
    path = Path(item['path']).resolve()
    try:
        relative = path.relative_to(folder).as_posix()
    except ValueError:
        relative = str(path)
    task_folder = Path(item.get('task_folder') or folder)
    source = next((task_folder / name for name in ('image.png', 'Изображение.png')
                   if (task_folder / name).is_file()), None)
    return ResultDescriptor(
        artifact_id=f'artifact:{relative}',
        task_id=str(item.get('task_id') or ''),
        category=str(item.get('category') or ''),
        operation_id=item.get('operation_id'),
        source_image=str(source) if source is not None else None,
        coordinate_frame=item.get('coordinate_frame'),
        representation=str(item.get('file_format') or path.suffix.lstrip('.')).upper(),
        scale=str(item['scale']) if item.get('scale') is not None else None,
        channel=item.get('channel'),
        cwt_direction=item.get('cwt_direction') or item.get('direction'),
        feature_axis=item.get('feature_axis'),
        point_kind=item.get('point_kind'),
        algorithm=item.get('algorithm'),
    )
