import json

import numpy as np
import pytest

from router.common.config import EmbeddingConfig, EmbeddingInputConfig, TruncationConfig
from router.pipeline.clustering.cluster import ClusterDiagnostics, ClusterResult
from router.pipeline.clustering.cluster_map import (
    SCHEMA_PATH,
    build_cluster_map_dict,
    compute_corpus_digest,
    validate_cluster_map,
)
from router.pipeline.corpus import CorpusRow, SourceProvenance


def _fake_embedding_config(dimensions: int = 4) -> EmbeddingConfig:
    return EmbeddingConfig(
        model_id="test-model",
        dimensions=dimensions,
        normalisation="l2",
        distance="euclidean",
        input=EmbeddingInputConfig(
            normalisation="crlf-lf+trim",
            truncation=TruncationConfig(unit="chars", max=8000, strategy="head"),
        ),
    )


def _fake_cluster_result(k: int = 2, dimensions: int = 4) -> ClusterResult:
    centroids = np.zeros((k, dimensions), dtype=np.float64)
    sizes = [5] * k
    diagnostics = ClusterDiagnostics(k=k, inertia=1.0, sizes=sizes, min_size=5, max_size=5, mean_size=5.0)
    labels = np.array([i % k for i in range(sum(sizes))])
    return ClusterResult(k=k, labels=labels, centroids=centroids, diagnostics=diagnostics)


def _fake_rows(n: int = 10) -> list[CorpusRow]:
    return [CorpusRow(id=f"a:{i}", source="a", text=f"task {i}") for i in range(n)]


def _fake_sources(rows: int = 10) -> list[SourceProvenance]:
    return [SourceProvenance(name="a", hf_id="org/a", split="train", field="prompt", rows=rows, license="MIT")]


def test_build_cluster_map_dict_validates_against_the_real_schema():
    artifact = build_cluster_map_dict(
        _fake_embedding_config(), _fake_cluster_result(), _fake_rows(), _fake_sources(), seed=42, n_init=10,
    )
    validate_cluster_map(artifact)  # must not raise


def test_schema_file_exists_and_is_valid_json():
    assert SCHEMA_PATH.exists()
    json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def test_compute_corpus_digest_is_order_independent():
    rows_a = [CorpusRow(id="a:0", source="a", text="x"), CorpusRow(id="a:1", source="a", text="y")]
    rows_b = list(reversed(rows_a))
    assert compute_corpus_digest(rows_a) == compute_corpus_digest(rows_b)


def test_compute_corpus_digest_changes_with_content():
    rows_a = [CorpusRow(id="a:0", source="a", text="x")]
    rows_b = [CorpusRow(id="a:0", source="a", text="y")]
    assert compute_corpus_digest(rows_a) != compute_corpus_digest(rows_b)


def test_compute_corpus_digest_has_the_declared_prefix_and_length():
    digest = compute_corpus_digest(_fake_rows())
    assert digest.startswith("sha256:")
    assert len(digest) == len("sha256:") + 64


def test_validate_cluster_map_rejects_cluster_count_mismatching_k():
    artifact = build_cluster_map_dict(
        _fake_embedding_config(), _fake_cluster_result(k=2), _fake_rows(1), _fake_sources(1), seed=1, n_init=1,
    )
    artifact["kmeans"]["k"] = 3  # now disagrees with len(clusters) == 2
    with pytest.raises(ValueError):
        validate_cluster_map(artifact)


def test_validate_cluster_map_rejects_centroid_length_mismatching_dimensions():
    artifact = build_cluster_map_dict(
        _fake_embedding_config(dimensions=4), _fake_cluster_result(dimensions=4), _fake_rows(1), _fake_sources(1),
        seed=1, n_init=1,
    )
    artifact["clusters"][0]["centroid"] = artifact["clusters"][0]["centroid"][:2]
    with pytest.raises(ValueError):
        validate_cluster_map(artifact)


def test_validate_cluster_map_rejects_non_contiguous_cluster_ids():
    artifact = build_cluster_map_dict(
        _fake_embedding_config(), _fake_cluster_result(k=2), _fake_rows(1), _fake_sources(1), seed=1, n_init=1,
    )
    artifact["clusters"][1]["id"] = 5
    with pytest.raises(ValueError):
        validate_cluster_map(artifact)


def test_validate_cluster_map_rejects_unknown_additional_fields():
    artifact = build_cluster_map_dict(
        _fake_embedding_config(), _fake_cluster_result(), _fake_rows(), _fake_sources(), seed=42, n_init=10,
    )
    artifact["unexpected_field"] = "should not be allowed"
    with pytest.raises(ValueError):
        validate_cluster_map(artifact)
