import numpy as np

from router.common.assign import ClusterMap, assign_cluster, cluster_map_from_dict


def _map(centroids: list[list[float]]) -> ClusterMap:
    arr = np.array(centroids, dtype=np.float64)
    return ClusterMap(artifact_id="test", embedding_model_id="test-model", dimensions=arr.shape[1], centroids=arr)


def test_assigns_to_the_nearest_centroid():
    cm = _map([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
    assignment = assign_cluster(np.array([0.5, 0.5]), cm)
    assert assignment.cluster_id == 0


def test_exact_tie_resolves_to_the_lower_id():
    cm = _map([[0.0, 0.0], [2.0, 0.0]])
    assignment = assign_cluster(np.array([1.0, 0.0]), cm)  # equidistant from both
    assert assignment.cluster_id == 0
    assert assignment.distance == assignment.runner_up_distance


def test_self_assign_a_centroid_has_zero_distance():
    cm = _map([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assignment = assign_cluster(np.array([1.0, 2.0, 3.0]), cm)
    assert assignment.cluster_id == 0
    assert assignment.distance == 0.0


def test_runner_up_is_the_second_nearest():
    cm = _map([[0.0], [1.0], [5.0]])
    assignment = assign_cluster(np.array([0.9]), cm)
    assert assignment.cluster_id == 1
    assert assignment.runner_up_cluster_id == 0


def test_dimension_mismatch_raises():
    cm = _map([[0.0, 0.0], [1.0, 1.0]])
    try:
        assign_cluster(np.array([0.0, 0.0, 0.0]), cm)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_cluster_map_from_dict_builds_centroids_indexed_by_id():
    data = {
        "artifact_id": "clustermap-test",
        "embedding": {"model_id": "test-model", "dimensions": 2},
        "clusters": [
            {"id": 1, "size": 1, "centroid": [1.0, 1.0]},
            {"id": 0, "size": 1, "centroid": [0.0, 0.0]},
        ],
    }
    cm = cluster_map_from_dict(data)
    assert cm.artifact_id == "clustermap-test"
    assert cm.embedding_model_id == "test-model"
    assert cm.dimensions == 2
    assert cm.centroids.tolist() == [[0.0, 0.0], [1.0, 1.0]]


def test_cluster_map_from_dict_rejects_duplicate_ids():
    data = {
        "artifact_id": "t",
        "embedding": {"model_id": "m", "dimensions": 1},
        "clusters": [{"id": 0, "size": 1, "centroid": [0.0]}, {"id": 0, "size": 1, "centroid": [1.0]}],
    }
    try:
        cluster_map_from_dict(data)
        assert False, "expected ValueError"
    except ValueError:
        pass
