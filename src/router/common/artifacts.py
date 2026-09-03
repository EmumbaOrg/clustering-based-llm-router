"""Shared JSON Schema validation + write helpers for pipeline output artifacts (cluster-map.json,
model-profiles.json).

Both artifacts follow the same two-layer validation shape (see ../../README.md's "Config and
artifact schema"): standard JSON Schema structural checks, plus cross-field invariants the schema
can't express on its own. Both layers live here rather than inside the pipeline writers — each schema's own
`description` field says these invariants "must be re-checked by any consumer before trusting the
file", and the runtime is a second consumer.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from jsonschema import Draft202012Validator

from .config import REPO_ROOT

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = REPO_ROOT / "artifacts"
SCHEMA_DIR = REPO_ROOT / "artifacts-schema"

CLUSTER_MAP_SCHEMA_PATH = SCHEMA_DIR / "cluster-map.schema.json"
MODEL_PROFILES_SCHEMA_PATH = SCHEMA_DIR / "model-profiles.schema.json"


def validate_against_schema(artifact: dict, schema_path: Path, artifact_name: str) -> None:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(artifact), key=lambda e: [str(p) for p in e.path])
    if errors:
        messages = "\n".join(f"  - {'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in errors)
        logger.error(f"{artifact_name} failed schema validation ({len(errors)} errors)")
        raise ValueError(f"{artifact_name} failed schema validation:\n{messages}")


def write_json_artifact(artifact: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return path


def check_cluster_map_invariants(artifact: dict) -> None:
    """The three invariants cluster-map.schema.json's own description says it cannot express:
    clusters.length == kmeans.k, cluster ids contiguous 0..k-1 ascending, and every centroid's
    length == embedding.dimensions. Also rejects a non-finite centroid value — see
    docs/engineering-notes.md, "NaN centroid guard".
    """
    dimensions = artifact["embedding"]["dimensions"]
    k = artifact["kmeans"]["k"]
    clusters = artifact["clusters"]
    if len(clusters) != k:
        raise ValueError(f"kmeans.k={k} but clusters has {len(clusters)} entries")
    for expected_id, cluster in enumerate(clusters):
        if cluster["id"] != expected_id:
            raise ValueError(
                f"clusters must be contiguous 0..k-1 in ascending order; "
                f"expected id {expected_id} at position {expected_id}, got {cluster['id']}"
            )
        centroid = cluster["centroid"]
        if len(centroid) != dimensions:
            raise ValueError(
                f"cluster {cluster['id']} centroid has {len(centroid)} values, "
                f"expected {dimensions} (embedding.dimensions)"
            )
        if not all(math.isfinite(v) for v in centroid):
            raise ValueError(f"cluster {cluster['id']} centroid contains a non-finite value (NaN/Infinity)")


def check_profiles_invariants(artifact: dict) -> None:
    """The invariants model-profiles.schema.json's own description says it cannot express: every
    cluster key falls within the referenced cluster map's 0..cluster_count-1 range, and
    number_succeeded + number_failed == number_of_tasks for every global/per-cluster block.
    `cluster_count` is read with `.get()` — see docs/engineering-notes.md, "profiles.json's
    cluster_count is optional".
    """
    cluster_count = artifact.get("cluster_count")
    for model in artifact["models"]:
        if cluster_count is not None:
            for cluster_id_str in model["clusters"]:
                cid = int(cluster_id_str)
                if not (0 <= cid < cluster_count):
                    raise ValueError(f"model {model['model_id']!r} has cluster id {cid}, outside 0..{cluster_count - 1}")
        for scope_name in ("global", *model["clusters"].keys()):
            stats = model["global"] if scope_name == "global" else model["clusters"][scope_name]
            total = stats["number_succeeded"] + stats["number_failed"]
            if total != stats["number_of_tasks"]:
                raise ValueError(
                    f"model {model['model_id']!r} scope {scope_name}: "
                    f"number_succeeded+number_failed={total} != number_of_tasks={stats['number_of_tasks']}"
                )


def validate_cluster_map(artifact: dict) -> None:
    validate_against_schema(artifact, CLUSTER_MAP_SCHEMA_PATH, "cluster-map.json")
    check_cluster_map_invariants(artifact)


def validate_profiles(artifact: dict) -> None:
    validate_against_schema(artifact, MODEL_PROFILES_SCHEMA_PATH, "model-profiles.json")
    check_profiles_invariants(artifact)
