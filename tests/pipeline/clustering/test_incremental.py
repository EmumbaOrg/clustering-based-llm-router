import json

import numpy as np

from router.common import embedding as embedding_mod
from router.common.config import EmbeddingConfig, EmbeddingInputConfig, TruncationConfig
from router.pipeline.clustering.incremental import (
    assign_incremental_clusters,
    embed_incremental_source,
)
from router.pipeline.corpus import CorpusRow, write_corpus_jsonl


def _fake_embedding_config(dimensions: int = 2, model_id: str = "test-model") -> EmbeddingConfig:
    return EmbeddingConfig(
        model_id=model_id,
        dimensions=dimensions,
        normalisation="l2",
        distance="euclidean",
        input=EmbeddingInputConfig(
            normalisation="crlf-lf+trim",
            truncation=TruncationConfig(unit="chars", max=8000, strategy="head"),
        ),
    )


def _write_cluster_map(path, centroids: list[list[float]], model_id="test-model", dimensions=2):
    clusters = [{"id": i, "size": 1, "centroid": c} for i, c in enumerate(centroids)]
    data = {
        "artifact_id": "clustermap-test",
        "embedding": {"model_id": model_id, "dimensions": dimensions},
        "clusters": clusters,
    }
    path.write_text(json.dumps(data))


# --- embed_incremental_source --------------------------------------------------------------


def test_embed_incremental_source_only_embeds_the_named_source(tmp_path, monkeypatch):
    corpus_path = tmp_path / "corpus.jsonl"
    write_corpus_jsonl(
        [
            CorpusRow(id="a:0", source="a", text="unrelated task"),
            CorpusRow(id="new:0", source="new-source", text="task one"),
            CorpusRow(id="new:1", source="new-source", text="task two"),
        ],
        corpus_path,
    )
    embeddings_path = tmp_path / "embeddings-new-source.npz"
    config = _fake_embedding_config()

    seen_texts = []

    def fake_embed_texts(texts, cfg, **kwargs):
        seen_texts.extend(texts)
        return np.array([[float(i), 0.0] for i in range(len(texts))])

    monkeypatch.setattr(embedding_mod, "embed_texts", fake_embed_texts)

    report = embed_incremental_source("new-source", corpus_path, embeddings_path, config, apply=True)

    assert seen_texts == ["task one", "task two"]
    assert report.loaded == 2
    assert report.applied is True
    ids, vectors = embedding_mod.load_embeddings(embeddings_path)
    assert ids == ["new:0", "new:1"]
    assert vectors.shape == (2, 2)


def test_embed_incremental_source_dry_run_writes_nothing(tmp_path, monkeypatch):
    corpus_path = tmp_path / "corpus.jsonl"
    write_corpus_jsonl([CorpusRow(id="new:0", source="new-source", text="task one")], corpus_path)
    embeddings_path = tmp_path / "embeddings-new-source.npz"
    monkeypatch.setattr(embedding_mod, "embed_texts", lambda texts, cfg, **kw: np.array([[0.0, 0.0]]))

    report = embed_incremental_source("new-source", corpus_path, embeddings_path, _fake_embedding_config(), apply=False)

    assert report.loaded == 1
    assert report.applied is False
    assert not embeddings_path.exists()


def test_embed_incremental_source_never_touches_the_shared_base_embeddings_file(tmp_path, monkeypatch):
    corpus_path = tmp_path / "corpus.jsonl"
    write_corpus_jsonl([CorpusRow(id="new:0", source="new-source", text="task one")], corpus_path)
    base_embeddings_path = tmp_path / "embeddings.npz"
    embedding_mod.save_embeddings(base_embeddings_path, ["a:0"], np.array([[1.0, 2.0]]))
    original_bytes = base_embeddings_path.read_bytes()

    monkeypatch.setattr(embedding_mod, "embed_texts", lambda texts, cfg, **kw: np.array([[0.0, 0.0]]))
    embed_incremental_source("new-source", corpus_path, tmp_path / "embeddings-new-source.npz", _fake_embedding_config(), apply=True)

    assert base_embeddings_path.read_bytes() == original_bytes


# --- assign_incremental_clusters -----------------------------------------------------------


def test_assign_incremental_clusters_assigns_to_nearest_centroid_and_reports_distribution(tmp_path):
    cluster_map_path = tmp_path / "cluster-map.json"
    _write_cluster_map(cluster_map_path, [[0.0, 0.0], [10.0, 10.0]])
    embeddings_path = tmp_path / "embeddings-new-source.npz"
    embedding_mod.save_embeddings(
        embeddings_path, ["new:0", "new:1", "new:2"],
        np.array([[0.1, 0.1], [9.9, 9.9], [0.2, 0.2]]),
    )
    task_cluster_map_path = tmp_path / "task-cluster-map.json"

    report, result = assign_incremental_clusters(
        "new-source", cluster_map_path, embeddings_path, task_cluster_map_path, _fake_embedding_config(), apply=False,
    )

    assert report.assigned == 3
    assert report.cluster_distribution == {0: 2, 1: 1}
    assert {r["task_id"]: r["cluster_id"] for r in result.accepted} == {"new:0": 0, "new:1": 1, "new:2": 0}


def test_assign_incremental_clusters_raises_on_embedding_config_mismatch(tmp_path):
    cluster_map_path = tmp_path / "cluster-map.json"
    _write_cluster_map(cluster_map_path, [[0.0, 0.0]], model_id="original-model", dimensions=2)
    embeddings_path = tmp_path / "embeddings-new-source.npz"
    embedding_mod.save_embeddings(embeddings_path, ["new:0"], np.array([[0.0, 0.0]]))

    try:
        assign_incremental_clusters(
            "new-source", cluster_map_path, embeddings_path, tmp_path / "task-cluster-map.json",
            _fake_embedding_config(model_id="different-model"), apply=False,
        )
        assert False, "expected a ValueError"
    except ValueError:
        pass


def test_assign_incremental_clusters_raises_when_embeddings_file_missing(tmp_path):
    cluster_map_path = tmp_path / "cluster-map.json"
    _write_cluster_map(cluster_map_path, [[0.0, 0.0]])

    try:
        assign_incremental_clusters(
            "new-source", cluster_map_path, tmp_path / "no-such-embeddings.npz",
            tmp_path / "task-cluster-map.json", _fake_embedding_config(), apply=False,
        )
        assert False, "expected a ValueError"
    except ValueError:
        pass


def test_assign_incremental_clusters_skips_a_task_id_already_in_task_cluster_map(tmp_path):
    cluster_map_path = tmp_path / "cluster-map.json"
    _write_cluster_map(cluster_map_path, [[0.0, 0.0]])
    embeddings_path = tmp_path / "embeddings-new-source.npz"
    embedding_mod.save_embeddings(embeddings_path, ["existing:0", "new:0"], np.array([[0.0, 0.0], [0.0, 0.0]]))
    task_cluster_map_path = tmp_path / "task-cluster-map.json"
    task_cluster_map_path.write_text(json.dumps({
        "schema_version": 1, "artifact_id": "x", "created_at": "now",
        "cluster_map_id": "clustermap-test",
        "tasks": [{"task_id": "existing:0", "source": "a", "cluster_id": 0}],
    }))

    report, result = assign_incremental_clusters(
        "new-source", cluster_map_path, embeddings_path, task_cluster_map_path, _fake_embedding_config(), apply=False,
    )

    assert report.assigned == 1
    assert report.skipped_id_collision == 1
    assert result.accepted[0]["task_id"] == "new:0"


def test_assign_incremental_clusters_dry_run_makes_zero_writes_and_creates_no_backup(tmp_path):
    cluster_map_path = tmp_path / "cluster-map.json"
    _write_cluster_map(cluster_map_path, [[0.0, 0.0]])
    cluster_map_bytes = cluster_map_path.read_bytes()
    embeddings_path = tmp_path / "embeddings-new-source.npz"
    embedding_mod.save_embeddings(embeddings_path, ["new:0"], np.array([[0.0, 0.0]]))
    task_cluster_map_path = tmp_path / "task-cluster-map.json"
    task_cluster_map_path.write_text(json.dumps({
        "schema_version": 1, "artifact_id": "x", "created_at": "now",
        "cluster_map_id": "clustermap-test", "tasks": [],
    }))
    original_task_map_bytes = task_cluster_map_path.read_bytes()

    report, _ = assign_incremental_clusters(
        "new-source", cluster_map_path, embeddings_path, task_cluster_map_path, _fake_embedding_config(), apply=False,
    )

    assert report.applied is False
    assert report.backup_path is None
    assert cluster_map_path.read_bytes() == cluster_map_bytes  # never touched, dry-run or not
    assert task_cluster_map_path.read_bytes() == original_task_map_bytes
    assert list(tmp_path.glob("task-cluster-map-backup-*.json")) == []


def test_assign_incremental_clusters_apply_backs_up_then_appends_preserving_existing_tasks(tmp_path):
    cluster_map_path = tmp_path / "cluster-map.json"
    _write_cluster_map(cluster_map_path, [[0.0, 0.0], [10.0, 10.0]])
    cluster_map_bytes = cluster_map_path.read_bytes()
    embeddings_path = tmp_path / "embeddings-new-source.npz"
    embedding_mod.save_embeddings(embeddings_path, ["new:0"], np.array([[9.9, 9.9]]))
    task_cluster_map_path = tmp_path / "task-cluster-map.json"
    existing_artifact = {
        "schema_version": 1, "artifact_id": "x", "created_at": "now",
        "cluster_map_id": "clustermap-test",
        "tasks": [{"task_id": "existing:0", "source": "a", "cluster_id": 0}],
    }
    task_cluster_map_path.write_text(json.dumps(existing_artifact))
    original_task_map_bytes = task_cluster_map_path.read_bytes()

    report, _ = assign_incremental_clusters(
        "new-source", cluster_map_path, embeddings_path, task_cluster_map_path, _fake_embedding_config(), apply=True,
    )

    assert report.applied is True
    assert cluster_map_path.read_bytes() == cluster_map_bytes  # NEVER rewritten, even on apply

    assert report.backup_path is not None
    assert report.backup_path.read_bytes() == original_task_map_bytes

    new_artifact = json.loads(task_cluster_map_path.read_text())
    assert new_artifact["tasks"] == [
        {"task_id": "existing:0", "source": "a", "cluster_id": 0},
        {"task_id": "new:0", "source": "new-source", "cluster_id": 1},
    ]


def test_assign_incremental_clusters_second_apply_run_is_a_no_op(tmp_path):
    cluster_map_path = tmp_path / "cluster-map.json"
    _write_cluster_map(cluster_map_path, [[0.0, 0.0]])
    embeddings_path = tmp_path / "embeddings-new-source.npz"
    embedding_mod.save_embeddings(embeddings_path, ["new:0"], np.array([[0.0, 0.0]]))
    task_cluster_map_path = tmp_path / "task-cluster-map.json"

    first, _ = assign_incremental_clusters(
        "new-source", cluster_map_path, embeddings_path, task_cluster_map_path, _fake_embedding_config(), apply=True,
    )
    assert first.applied is True
    # task_cluster_map_path didn't exist before this call, so there was nothing to back up yet —
    # mirrors backup_corpus_jsonl's own "no-op when the file doesn't exist" semantics.
    assert first.backup_path is None

    second, _ = assign_incremental_clusters(
        "new-source", cluster_map_path, embeddings_path, task_cluster_map_path, _fake_embedding_config(), apply=True,
    )
    assert second.assigned == 0
    assert second.applied is False
    # The file exists now (created by the first call) but the second run is a pure no-op (nothing
    # accepted), so no backup is made for it either — 0 backups total, not 1.
    assert list(tmp_path.glob("task-cluster-map-backup-*.json")) == []
