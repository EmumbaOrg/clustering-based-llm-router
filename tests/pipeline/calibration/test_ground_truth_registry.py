from router.pipeline.calibration import ground_truth_registry as registry_module
from router.pipeline.calibration.grading.base import Task


def _task(prompt: str = "fix the bug") -> Task:
    return Task(task_id="t1", source="swe-gym", prompt=prompt, reference_solution="", row={})


def test_load_registry_returns_an_empty_shell_when_no_file_exists(tmp_path):
    registry = registry_module.load_registry(tmp_path / "does-not-exist.json")
    assert registry == {"schema_version": registry_module.SCHEMA_VERSION, "entries": {}}


def test_write_then_load_round_trips(tmp_path):
    path = tmp_path / "registry.json"
    registry = registry_module.load_registry(path)
    registry_module.upsert(
        registry, task_id="t1", source="swe-gym", prompt_digest="sha256:abc",
        reference_outcome="pass", reference_detail="", null_outcome="fail", null_detail="",
        verdict="valid",
    )
    registry_module.write_registry(registry, path)

    reloaded = registry_module.load_registry(path)
    assert reloaded["entries"]["t1"]["verdict"] == "valid"
    assert reloaded["entries"]["t1"]["source"] == "swe-gym"


def test_lookup_returns_none_for_an_unseen_task():
    registry = {"schema_version": 1, "entries": {}}
    assert registry_module.lookup(registry, "unseen", "sha256:x") is None


def test_lookup_returns_the_cached_entry_when_the_digest_matches():
    registry = {"schema_version": 1, "entries": {}}
    registry_module.upsert(
        registry, task_id="t1", source="swe-gym", prompt_digest="sha256:abc",
        reference_outcome="pass", reference_detail="", null_outcome="fail", null_detail="",
        verdict="valid",
    )
    entry = registry_module.lookup(registry, "t1", "sha256:abc")
    assert entry is not None
    assert entry["verdict"] == "valid"


def test_lookup_treats_a_digest_mismatch_as_unknown_not_stale_trust():
    # The underlying task's prompt changed since verification (or, in practice, a hash collision
    # never happens) — a stale verdict must not be trusted as if it still applies.
    registry = {"schema_version": 1, "entries": {}}
    registry_module.upsert(
        registry, task_id="t1", source="swe-gym", prompt_digest="sha256:old",
        reference_outcome="pass", reference_detail="", null_outcome="fail", null_detail="",
        verdict="valid",
    )
    assert registry_module.lookup(registry, "t1", "sha256:new") is None


def test_compute_task_digest_is_stable_for_the_same_prompt_and_differs_for_a_different_one():
    assert registry_module.compute_task_digest(_task("a")) == registry_module.compute_task_digest(_task("a"))
    assert registry_module.compute_task_digest(_task("a")) != registry_module.compute_task_digest(_task("b"))
