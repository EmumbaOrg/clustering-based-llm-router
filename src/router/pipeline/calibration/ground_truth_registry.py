"""Persistent, cross-run record of ground-truth verification results. Keyed by `task_id` alone,
never by which pinned selection a task belongs to — validity is a property of the task itself, so
a verdict stays useful across a future k/category-mix change. Grows accretively via
`calibrate.verify_ground_truth`, the only intended writer.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from ...common.artifacts import ARTIFACTS_DIR
from .grading.base import Task

logger = logging.getLogger(__name__)

REGISTRY_PATH = ARTIFACTS_DIR / "ground-truth-registry.json"
SCHEMA_VERSION = 1


def compute_task_digest(task: Task) -> str:
    """SHA-256 over the task's own prompt text. An entry whose digest no longer matches the
    task's current prompt is treated as stale (see `lookup`), not trusted forever."""
    return f"sha256:{hashlib.sha256(task.prompt.encode('utf-8')).hexdigest()}"


def load_registry(path: Path = REGISTRY_PATH) -> dict:
    if not path.exists():
        return {"schema_version": SCHEMA_VERSION, "entries": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def write_registry(registry: dict, path: Path = REGISTRY_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")
    logger.info(f"wrote ground-truth registry to {path} ({len(registry['entries'])} entries)")
    return path


def lookup(registry: dict, task_id: str, prompt_digest: str) -> dict | None:
    """Returns the cached entry for `task_id` if present and still fresh — `None` (never a stale
    verdict) if the task has never been verified, or if `prompt_digest` no longer matches what was
    verified, in which case the task is treated as unknown and re-verified rather than trusted on
    stale content."""
    entry = registry["entries"].get(task_id)
    if entry is None or entry["prompt_digest"] != prompt_digest:
        return None
    return entry


def upsert(
    registry: dict,
    task_id: str,
    source: str,
    prompt_digest: str,
    reference_outcome: str,
    reference_detail: str,
    null_outcome: str,
    null_detail: str,
    verdict: str,
    calibration_run_id: str | None = None,
) -> None:
    registry["entries"][task_id] = {
        "source": source,
        "prompt_digest": prompt_digest,
        "reference_outcome": reference_outcome,
        "reference_detail": reference_detail,
        "null_outcome": null_outcome,
        "null_detail": null_detail,
        "verdict": verdict,
        "verified_at": datetime.now(UTC).isoformat(),
        "calibration_run_id": calibration_run_id,
    }
