import json

import numpy as np

from router.pipeline.clustering.task_cluster_map import build_task_cluster_map_dict, write_task_cluster_map
from router.pipeline.corpus import CorpusRow


def test_build_task_cluster_map_dict_pairs_each_id_with_its_label_and_source():
    ids = ["BigCodeBench/1", "ds1000:2"]
    labels = np.array([3, 7])
    rows_by_id = {
        "BigCodeBench/1": CorpusRow(id="BigCodeBench/1", source="bigcodebench", text="t"),
        "ds1000:2": CorpusRow(id="ds1000:2", source="ds1000", text="t"),
    }

    artifact = build_task_cluster_map_dict(ids, labels, rows_by_id, cluster_map_id="clustermap-x")

    assert artifact["cluster_map_id"] == "clustermap-x"
    assert artifact["tasks"] == [
        {"task_id": "BigCodeBench/1", "source": "bigcodebench", "cluster_id": 3},
        {"task_id": "ds1000:2", "source": "ds1000", "cluster_id": 7},
    ]


def test_build_task_cluster_map_dict_cluster_id_is_a_plain_int_not_a_numpy_scalar():
    # json.dumps chokes on numpy int64 — this must be caught here, not at write time.
    artifact = build_task_cluster_map_dict(
        ["a"], np.array([5]), {"a": CorpusRow(id="a", source="s", text="t")}, cluster_map_id="cm",
    )
    assert type(artifact["tasks"][0]["cluster_id"]) is int


def test_build_task_cluster_map_dict_skips_an_embedded_id_with_no_matching_corpus_row():
    # Regression guard for a corpus.jsonl/embeddings.npz mismatch — must not crash, and must not
    # silently misalign a later id/label with this row's position (see the function's own
    # docstring on why this is looked up by id, not zipped against an already-filtered list).
    ids = ["missing", "present"]
    labels = np.array([1, 2])
    rows_by_id = {"present": CorpusRow(id="present", source="s", text="t")}

    artifact = build_task_cluster_map_dict(ids, labels, rows_by_id, cluster_map_id="cm")

    assert artifact["tasks"] == [{"task_id": "present", "source": "s", "cluster_id": 2}]


def test_write_task_cluster_map_writes_valid_json_to_the_given_path(tmp_path):
    artifact = build_task_cluster_map_dict(
        ["a"], np.array([0]), {"a": CorpusRow(id="a", source="s", text="t")}, cluster_map_id="cm",
    )
    path = write_task_cluster_map(artifact, tmp_path / "task-cluster-map.json")

    assert json.loads(path.read_text(encoding="utf-8")) == artifact
