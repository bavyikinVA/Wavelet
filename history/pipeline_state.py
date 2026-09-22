"""Stage-level validity model for incremental scientific calculations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict

from history.pipeline_resume import cwt_signature, load_complete_cwt, point_bundle_available

STATE_FILENAME = ".pipeline_state.json"
STATE_SCHEMA_VERSION = 1
STAGE_ORDER = ("wavelet", "extrema", "envelopes", "knn", "statistics")
STAGE_LABELS = {
    "wavelet": "Вейвлеты",
    "extrema": "Экстремумы",
    "envelopes": "Огибающие",
    "knn": "KNN",
    "statistics": "Статистики",
}


def _hash(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def stage_signatures(task) -> Dict[str, str]:
    wavelet = cwt_signature(task)
    extrema = _hash({
        "upstream": wavelet,
        "find_maxima": bool(getattr(task, "find_maxima", True)),
        "find_minima": bool(getattr(task, "find_minima", True)),
        "algorithm": "local-extrema-v1",
    })
    envelopes = _hash({
        "upstream": extrema,
        "algorithm": "envelope-interpolation-v1",
    })
    knn = _hash({
        "upstream": envelopes,
        "k_neighbors": int(getattr(task, "k_neighbors", 5)),
        "algorithm": "knn-angles-v1",
    })
    statistics = _hash({
        "upstream": envelopes,
        "scale_block_sizes": [int(x) for x in getattr(task, "scale_block_sizes", [])],
        "algorithm": "point-statistics-v1",
    })
    return {
        "wavelet": wavelet,
        "extrema": extrema,
        "envelopes": envelopes,
        "knn": knn,
        "statistics": statistics,
    }


def _state_path(task) -> Path | None:
    folder = getattr(task, "task_folder_path", "")
    if not folder:
        return None
    return Path(folder) / STATE_FILENAME


def load_stage_state(task) -> dict:
    path = _state_path(task)
    if path is None or not path.is_file():
        return {"schema_version": STATE_SCHEMA_VERSION, "stages": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {"schema_version": STATE_SCHEMA_VERSION, "stages": {}}
    if not isinstance(payload, dict):
        return {"schema_version": STATE_SCHEMA_VERSION, "stages": {}}
    payload.setdefault("stages", {})
    return payload


def save_stage_state(task, state: dict) -> None:
    path = _state_path(task)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    state["schema_version"] = STATE_SCHEMA_VERSION
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def mark_stage_ready(task, stage: str) -> None:
    if stage not in STAGE_ORDER:
        return
    state = load_stage_state(task)
    signatures = stage_signatures(task)
    state.setdefault("stages", {})[stage] = {"signature": signatures[stage], "status": "ready"}
    save_stage_state(task, state)


def _legacy_ready(task, stage: str) -> bool:
    """Best-effort readiness for runs created before stage-state metadata."""
    if stage == "wavelet":
        try:
            return load_complete_cwt(task) is not None
        except Exception:
            return False
    if stage == "extrema":
        try:
            return point_bundle_available(task, envelope=False)
        except Exception:
            return False
    if stage == "envelopes":
        try:
            return point_bundle_available(task, envelope=True)
        except Exception:
            return False
    return False


def stage_statuses(task) -> Dict[str, str]:
    """Return ready/missing/stale for every computational stage."""
    signatures = stage_signatures(task)
    state = load_stage_state(task)
    stored = state.get("stages", {}) if isinstance(state, dict) else {}
    statuses: Dict[str, str] = {}

    for stage in STAGE_ORDER:
        entry = stored.get(stage)
        if isinstance(entry, dict) and entry.get("signature"):
            statuses[stage] = "ready" if entry.get("signature") == signatures[stage] else "stale"
        elif _legacy_ready(task, stage):
            # Legacy artifacts are safe only when the task's remembered CWT
            # signature still matches the current scientific inputs.
            last_cwt = getattr(task, "last_completed_cwt_signature", None)
            if last_cwt and last_cwt != signatures["wavelet"]:
                statuses[stage] = "stale"
            else:
                statuses[stage] = "ready"
        else:
            statuses[stage] = "missing"

    # Dependency invalidation: a downstream stage cannot remain ready when an
    # upstream stage is stale/missing, even if its own old signature exists.
    for upstream, downstream in zip(STAGE_ORDER, STAGE_ORDER[1:]):
        if statuses[upstream] != "ready" and statuses[downstream] == "ready":
            statuses[downstream] = "stale"
    return statuses


def requested_stage_names(task):
    plan = task.resolve_pipeline()
    requested = ["wavelet"]
    if plan.extrema:
        requested.append("extrema")
    if plan.envelopes:
        requested.append("envelopes")
    if plan.knn:
        requested.append("knn")
    if plan.statistics:
        requested.append("statistics")
    return requested


def minimal_execution_plan(task):
    """Return only requested stages that are not currently valid."""
    statuses = stage_statuses(task)
    requested = requested_stage_names(task)
    return [stage for stage in requested if statuses.get(stage) != "ready"]


def format_stage_status(task) -> str:
    statuses = stage_statuses(task)
    icon = {"ready": "✓", "missing": "○", "stale": "↻"}
    words = {"ready": "готово", "missing": "нет", "stale": "устарело"}
    return "   ".join(
        f"{icon[statuses[s]]} {STAGE_LABELS[s]}: {words[statuses[s]]}"
        for s in requested_stage_names(task)
    )


def format_execution_plan(task) -> str:
    plan = minimal_execution_plan(task)
    if not plan:
        return "Все выбранные этапы уже актуальны — пересчёт не требуется"
    labels = [STAGE_LABELS[s] for s in plan]
    return "Будет рассчитано: " + " → ".join(labels)
