"""Independent overlapping regions of interest stored in pixel coordinates.

The model is intentionally independent of Tk and of the CWT backends.  Each
ROI has its *own* dense boolean mask, so overlapping regions remain separate.
An empty collection means the legacy full-image pipeline, unchanged.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROI_VERSION = 1
ROI_FILENAME = "regions.npz"
ROI_META_FILENAME = "regions.json"
DEFAULT_COLORS = ("#54A8DC", "#E76770", "#72B879", "#D1A24F", "#9D84D9", "#52BDB5")


@dataclass
class ROIRegion:
    id: int
    mask: np.ndarray = field(repr=False)
    name: str = ""
    color: str = DEFAULT_COLORS[0]
    opacity: float = 0.85
    visible: bool = True

    def __post_init__(self):
        self.id = int(self.id)
        self.mask = np.asarray(self.mask, dtype=np.bool_)
        if self.mask.ndim != 2:
            raise ValueError("ROI-маска должна иметь размерность (H, W)")
        self.name = self.name or f"Область {self.id}"
        self.opacity = min(1.0, max(0.0, float(self.opacity)))

    def metadata(self):
        return {"id": self.id, "name": self.name, "color": self.color,
                "opacity": self.opacity, "visible": self.visible,
                "pixels": int(np.count_nonzero(self.mask))}


class ROICollection:
    def __init__(self, shape, regions=None):
        self.shape = tuple(int(s) for s in shape[:2])
        if len(self.shape) != 2 or min(self.shape) <= 0:
            raise ValueError("Размер ROI должен соответствовать изображению")
        self.regions: list[ROIRegion] = []
        for region in regions or []:
            self.add(region)

    def __len__(self):
        return len(self.regions)

    def add(self, region):
        if region.mask.shape != self.shape:
            raise ValueError("Размер ROI не соответствует размеру изображения")
        if any(r.id == region.id for r in self.regions):
            raise ValueError(f"Повторяющийся ROI ID: {region.id}")
        self.regions.append(region)
        return region

    def new_region(self, mask, *, name=None):
        next_id = max((r.id for r in self.regions), default=0) + 1
        return self.add(ROIRegion(next_id, np.asarray(mask, dtype=bool).copy(),
                                  name or f"Область {next_id}",
                                  DEFAULT_COLORS[(next_id - 1) % len(DEFAULT_COLORS)]))

    def get(self, region_id):
        return next((region for region in self.regions if region.id == region_id), None)

    def remove(self, region_id):
        self.regions = [r for r in self.regions if r.id != int(region_id)]

    def copy(self):
        return ROICollection(self.shape, [ROIRegion(r.id, r.mask.copy(), r.name,
                                                    r.color, r.opacity, r.visible)
                                          for r in self.regions])

    def fingerprint(self):
        """Scientific fingerprint: presentation-only properties do not invalidate results."""
        digest = hashlib.sha256()
        digest.update(b"roi-independent-overlap-v1")
        digest.update(np.asarray(self.shape, dtype="<i4").tobytes())
        for region in self.regions:
            digest.update(int(region.id).to_bytes(4, "little", signed=False))
            digest.update(np.packbits(region.mask, axis=None).tobytes())
        return digest.hexdigest()

    def metadata(self):
        return {"version": ROI_VERSION, "shape": list(self.shape),
                "regions": [r.metadata() for r in self.regions],
                "mask_sha256": self.fingerprint()}

    def save(self, task_folder):
        folder = Path(task_folder) / "roi"
        folder.mkdir(parents=True, exist_ok=True)
        packed = {f"mask_{r.id}": np.packbits(r.mask, axis=None) for r in self.regions}
        from history.atomic_io import atomic_binary_writer, atomic_write_text
        with atomic_binary_writer(folder / ROI_FILENAME) as stream:
            np.savez_compressed(stream, **packed)
        atomic_write_text(
            folder / ROI_META_FILENAME,
            json.dumps(self.metadata(), ensure_ascii=False, indent=2),
        )
        return folder

    @classmethod
    def load(cls, task_folder, expected_shape=None):
        folder = Path(task_folder) / "roi"
        meta = json.loads((folder / ROI_META_FILENAME).read_text(encoding="utf-8"))
        if meta.get("version") != ROI_VERSION:
            raise ValueError("Версия ROI не поддерживается")
        shape = tuple(int(s) for s in meta["shape"])
        if expected_shape is not None and tuple(expected_shape[:2]) != shape:
            raise ValueError("Размер сохранённых ROI не совпадает с изображением")
        result = cls(shape)
        with np.load(folder / ROI_FILENAME, allow_pickle=False) as npz:
            for item in meta["regions"]:
                packed = npz[f"mask_{int(item['id'])}"]
                mask = np.unpackbits(packed, count=int(np.prod(shape))).reshape(shape).astype(bool)
                result.add(ROIRegion(item["id"], mask, item.get("name", ""),
                                     item.get("color", DEFAULT_COLORS[0]),
                                     item.get("opacity", 0.85), item.get("visible", True)))
        if result.fingerprint() != meta.get("mask_sha256"):
            raise ValueError("Контрольная сумма ROI не совпадает с сохранёнными данными")
        return result


def polygon_mask(shape, points):
    """Rasterize a freehand polygon supplied as (x, y) *image pixel* coordinates."""
    height, width = shape[:2]
    mask = Image.new("1", (width, height), 0)
    if len(points) < 3:
        return np.zeros((height, width), dtype=bool)
    ImageDraw.Draw(mask).polygon([(int(round(x)), int(round(y))) for x, y in points], fill=1)
    return np.asarray(mask, dtype=bool)


def draw_brush(mask, points, diameter, *, erase=False):
    """Paint connected strokes directly in original-pixel coordinates."""
    mask = np.asarray(mask, dtype=bool)
    if not points:
        return mask.copy()
    image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    draw = ImageDraw.Draw(image)
    points = [(int(round(x)), int(round(y))) for x, y in points]
    radius = max(1, int(round(diameter / 2)))
    value = 0 if erase else 255
    if len(points) > 1:
        draw.line(points, fill=value, width=max(1, int(round(diameter))), joint="curve")
    for x, y in points:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=value)
    return np.asarray(image, dtype=np.uint8) > 0


def regions_touched_by_stroke(collection, points, diameter, *, visible_only=True):
    """Return IDs whose *boundary* intersects an eraser stroke.

    Masks remain independent; touching two overlapping ROI boundaries returns
    both IDs.  Only a small window around the stroke is inspected, so moving
    the eraser does not erode full-resolution masks on every mouse event.
    """
    if not points or not collection.regions:
        return []
    height, width = collection.shape
    radius = max(1, int(round(diameter / 2)))
    x0 = max(0, int(np.floor(min(p[0] for p in points))) - radius - 2)
    y0 = max(0, int(np.floor(min(p[1] for p in points))) - radius - 2)
    x1 = min(width, int(np.ceil(max(p[0] for p in points))) + radius + 3)
    y1 = min(height, int(np.ceil(max(p[1] for p in points))) + radius + 3)
    if x0 >= x1 or y0 >= y1:
        return []

    stroke_points = [(x - x0, y - y0) for x, y in points]
    stroke = draw_brush(np.zeros((y1 - y0, x1 - x0), dtype=bool),
                        stroke_points, diameter)
    if not stroke.any():
        return []

    # One-pixel halo prevents an artificial contour at the crop boundary.
    hx0, hy0 = max(0, x0 - 1), max(0, y0 - 1)
    hx1, hy1 = min(width, x1 + 1), min(height, y1 + 1)
    offset_x, offset_y = x0 - hx0, y0 - hy0
    touched = []
    for region in collection.regions:
        if visible_only and not region.visible:
            continue
        if not region.mask[y0:y1, x0:x1].any():
            continue
        halo = np.pad(region.mask[hy0:hy1, hx0:hx1], 1, mode="constant")
        center = halo[1:-1, 1:-1]
        boundary = center & ~(
            halo[:-2, 1:-1] & halo[2:, 1:-1]
            & halo[1:-1, :-2] & halo[1:-1, 2:]
        )
        boundary = boundary[offset_y:offset_y + y1 - y0,
                            offset_x:offset_x + x1 - x0]
        if np.any(boundary & stroke):
            touched.append(region.id)
    return touched
