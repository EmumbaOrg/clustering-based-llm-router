import pytest

from router.common.artifacts import (
    check_cluster_map_invariants,
    check_profiles_invariants,
)


def _cluster_map(k: int = 2, dimensions: int = 2) -> dict:
    return {
        "embedding": {"dimensions": dimensions},
        "kmeans": {"k": k},
        "clusters": [{"id": i, "size": 1, "centroid": [0.0] * dimensions} for i in range(k)],
    }


def test_check_cluster_map_invariants_accepts_a_well_formed_map():
    check_cluster_map_invariants(_cluster_map())  # must not raise


def test_rejects_cluster_count_mismatching_k():
    artifact = _cluster_map(k=2)
    artifact["kmeans"]["k"] = 3
    with pytest.raises(ValueError):
        check_cluster_map_invariants(artifact)


def test_rejects_non_contiguous_cluster_ids():
    artifact = _cluster_map(k=2)
    artifact["clusters"][1]["id"] = 5
    with pytest.raises(ValueError):
        check_cluster_map_invariants(artifact)


def test_rejects_centroid_length_mismatching_dimensions():
    artifact = _cluster_map(dimensions=4)
    artifact["clusters"][0]["centroid"] = [0.0, 0.0]
    with pytest.raises(ValueError):
        check_cluster_map_invariants(artifact)


def test_rejects_a_nan_centroid_value():
    artifact = _cluster_map()
    artifact["clusters"][0]["centroid"] = [float("nan"), 0.0]
    with pytest.raises(ValueError):
        check_cluster_map_invariants(artifact)


def test_rejects_an_infinite_centroid_value():
    artifact = _cluster_map()
    artifact["clusters"][0]["centroid"] = [float("inf"), 0.0]
    with pytest.raises(ValueError):
        check_cluster_map_invariants(artifact)


def _stats(n=4, succeeded=3) -> dict:
    failed = n - succeeded
    rate = failed / n if n else 0.0
    return {"number_of_tasks": n, "number_succeeded": succeeded, "number_failed": failed, "raw_error_rate": rate, "smoothed_error_rate": rate}


def _profiles(cluster_count: int | None = 3) -> dict:
    artifact = {"models": [{"model_id": "m1", "global": _stats(), "clusters": {"0": _stats()}}]}
    if cluster_count is not None:
        artifact["cluster_count"] = cluster_count
    return artifact


def test_check_profiles_invariants_accepts_a_well_formed_artifact():
    check_profiles_invariants(_profiles())  # must not raise


def test_rejects_a_cluster_id_outside_the_cluster_map_range():
    artifact = _profiles(cluster_count=3)
    artifact["models"][0]["clusters"]["99"] = _stats()
    with pytest.raises(ValueError):
        check_profiles_invariants(artifact)


def test_rejects_mismatched_success_failure_counts():
    artifact = _profiles()
    artifact["models"][0]["global"]["number_of_tasks"] = 999
    with pytest.raises(ValueError):
        check_profiles_invariants(artifact)


def test_tolerates_a_missing_optional_cluster_count():
    # cluster_count is optional per the schema even though profiles.py always writes it — a
    # reader must not KeyError on an otherwise-valid artifact that omits it.
    check_profiles_invariants(_profiles(cluster_count=None))  # must not raise
