"""Incrementally embedding and cluster-assigning a source added via corpus.py's
INCREMENTAL_SOURCES / add-corpus-source — the embedding+clustering counterpart to corpus.py's
incremental-add mechanism. Same philosophy throughout: dry-run by default, and an existing
artifact is either never touched at all (embeddings.npz, cluster-map.json — see below for why
that's safe) or backed up before any real write (task-cluster-map.json).

Deliberately NOT a refit: this always assigns new vectors to the EXISTING cluster-map.json's
centroids via common/assign.py::assign_cluster (nearest-centroid, no re-fit). cluster-map.json is
never read back in and rewritten — common/assign.py::load_cluster_map only ever reads
`clusters`/`embedding.dimensions`/`embedding.model_id`/`artifact_id`, and nothing else in the repo
(schema validation, runtime, calibration) cross-checks its `kmeans.corpus_size`/`sources`/
`corpus_digest` provenance fields against real corpus state — so leaving them referring to the
pre-incremental corpus is inert, not a correctness bug. This also means `cluster_map_id` never
changes, so every existing model-profiles.json entry stays valid.
"""
from __future__ import annotations

import dataclasses
import shutil
from datetime import UTC, datetime
from pathlib import Path

from ...common import embedding as embedding_mod
from ...common.assign import assign_cluster, load_cluster_map
from ...common.config import EmbeddingConfig
from ..corpus import read_corpus_jsonl
from . import task_cluster_map as task_cluster_map_mod


@dataclasses.dataclass(frozen=True)
class IncrementalEmbedReport:
    source: str
    loaded: int
    embeddings_path: Path
    applied: bool


def embed_incremental_source(
    name: str, corpus_path: Path, embeddings_path: Path, embedding_config: EmbeddingConfig, apply: bool = False,
) -> IncrementalEmbedReport:
    """Reads ONLY `name`'s rows from corpus_path (read-only — the shared corpus.jsonl the corpus
    stage already grew) and embeds them with `embedding_config`. Only writes, and only to
    `embeddings_path` (a source-specific path, e.g. .cache/embeddings-<name>.npz — never the
    shared base embeddings.npz), when apply=True. Safe to re-run: always re-embeds this source's
    current full row set and overwrites only its own file."""
    rows = [r for r in read_corpus_jsonl(corpus_path) if r.source == name]
    if rows:
        vectors = embedding_mod.embed_texts([r.text for r in rows], embedding_config)
        if apply:
            embedding_mod.save_embeddings(embeddings_path, [r.id for r in rows], vectors)

    return IncrementalEmbedReport(
        source=name, loaded=len(rows), embeddings_path=embeddings_path, applied=bool(apply and rows),
    )


@dataclasses.dataclass(frozen=True)
class IncrementalAssignResult:
    accepted: list[dict]  # {"task_id": str, "source": str, "cluster_id": int}
    skipped_id_collision: list[str]  # task_ids already present in task-cluster-map.json


@dataclasses.dataclass(frozen=True)
class IncrementalAssignReport:
    source: str
    assigned: int
    cluster_distribution: dict[int, int]  # cluster_id -> count of newly-assigned rows landing there
    skipped_id_collision: int
    backup_path: Path | None
    task_cluster_map_path: Path
    applied: bool


def assign_incremental_clusters(
    name: str,
    cluster_map_path: Path,
    embeddings_path: Path,
    task_cluster_map_path: Path,
    embedding_config: EmbeddingConfig,
    apply: bool = False,
) -> tuple[IncrementalAssignReport, IncrementalAssignResult]:
    """Loads cluster-map.json read-only (never rewritten — see module docstring). Verifies its
    recorded embedding model/dimensions match `embedding_config` — a hard error, not a warning,
    since assigning a different embedding space's vectors against these centroids would be
    meaningless. Loads `embeddings_path` (the source-specific npz `embed_incremental_source`
    wrote), assigns each vector via the existing assign_cluster, and skips (never reassigns or
    overwrites) any task_id already present in task_cluster_map_path. Only on apply=True does it
    back up task_cluster_map_path and write the merged (existing + newly accepted) tasks list."""
    if not embeddings_path.exists():
        raise ValueError(f"{embeddings_path} not found — run embed-incremental-source for {name!r} first.")

    cluster_map = load_cluster_map(cluster_map_path)
    if (
        cluster_map.embedding_model_id != embedding_config.model_id
        or cluster_map.dimensions != embedding_config.dimensions
    ):
        raise ValueError(
            f"cluster-map.json was built with embedding model {cluster_map.embedding_model_id!r} "
            f"({cluster_map.dimensions}-dim), but the current config/embedding.yaml is "
            f"{embedding_config.model_id!r} ({embedding_config.dimensions}-dim) — assigning these "
            "vectors to these centroids would not be meaningful."
        )

    ids, vectors = embedding_mod.load_embeddings(embeddings_path)

    existing = (
        task_cluster_map_mod.load_task_cluster_map(task_cluster_map_path)
        if task_cluster_map_path.exists()
        else None
    )
    existing_tasks: list[dict] = existing["tasks"] if existing else []
    existing_ids = {t["task_id"] for t in existing_tasks}

    accepted: list[dict] = []
    skipped: list[str] = []
    cluster_distribution: dict[int, int] = {}
    for task_id, vector in zip(ids, vectors):
        if task_id in existing_ids:
            skipped.append(task_id)
            continue
        assignment = assign_cluster(vector, cluster_map)
        accepted.append({"task_id": task_id, "source": name, "cluster_id": assignment.cluster_id})
        cluster_distribution[assignment.cluster_id] = cluster_distribution.get(assignment.cluster_id, 0) + 1

    backup_path = None
    applied = False
    if apply and accepted:
        if task_cluster_map_path.exists():
            timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
            backup_path = task_cluster_map_path.parent / f"task-cluster-map-backup-{timestamp}.json"
            shutil.copy2(task_cluster_map_path, backup_path)
        merged = (
            dict(existing)
            if existing
            else {
                "schema_version": 1,
                "artifact_id": f"taskclustermap-incremental-{cluster_map.artifact_id}",
                "created_at": datetime.now(UTC).isoformat(),
                "cluster_map_id": cluster_map.artifact_id,
                "tasks": [],
            }
        )
        merged["tasks"] = existing_tasks + accepted
        task_cluster_map_mod.write_task_cluster_map(merged, task_cluster_map_path)
        applied = True

    report = IncrementalAssignReport(
        source=name,
        assigned=len(accepted),
        cluster_distribution=cluster_distribution,
        skipped_id_collision=len(skipped),
        backup_path=backup_path,
        task_cluster_map_path=task_cluster_map_path,
        applied=applied,
    )
    return report, IncrementalAssignResult(accepted=accepted, skipped_id_collision=skipped)
