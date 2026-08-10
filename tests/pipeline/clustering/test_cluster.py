import numpy as np

from router.pipeline.clustering.cluster import (
    ClusterDiagnostics,
    format_diagnostics,
    run_kmeans,
)


def _two_blobs() -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.vstack(
        [
            rng.normal(loc=0, scale=0.01, size=(20, 4)),
            rng.normal(loc=5, scale=0.01, size=(20, 4)),
        ]
    )


def test_run_kmeans_is_deterministic_with_a_fixed_seed():
    vectors = _two_blobs()
    result_a = run_kmeans(vectors, k=2, seed=42, n_init=10)
    result_b = run_kmeans(vectors, k=2, seed=42, n_init=10)
    assert np.allclose(result_a.centroids, result_b.centroids)


def test_run_kmeans_centroids_are_cast_to_float64():
    vectors = np.random.default_rng(0).normal(size=(10, 3)).astype(np.float32)
    result = run_kmeans(vectors, k=2, seed=1, n_init=5)
    assert result.centroids.dtype == np.float64


def test_diagnostics_cluster_sizes_sum_to_corpus_size():
    vectors = np.random.default_rng(0).normal(size=(30, 3))
    result = run_kmeans(vectors, k=3, seed=1, n_init=5)
    assert sum(result.diagnostics.sizes) == 30


def test_format_diagnostics_flags_a_suspiciously_small_cluster():
    diag = ClusterDiagnostics(k=2, inertia=1.0, sizes=[1, 999], min_size=1, max_size=999, mean_size=500.0)
    assert "WARNING" in format_diagnostics(diag, corpus_size=1000)


def test_format_diagnostics_no_warning_when_sizes_are_balanced():
    diag = ClusterDiagnostics(k=2, inertia=1.0, sizes=[500, 500], min_size=500, max_size=500, mean_size=500.0)
    assert "WARNING" not in format_diagnostics(diag, corpus_size=1000)
