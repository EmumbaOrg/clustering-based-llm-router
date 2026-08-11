import json
import random

from router.pipeline import corpus as corpus_mod
from router.pipeline.calibration import tasks as tasks_mod
from router.pipeline.corpus import (
    SOURCE_METADATA,
    CorpusRow,
    _extract_multi_swe_rl_text,
    _multi_swe_rl_ordered_paths,
    _multi_swe_rl_row_id,
    _multi_swe_rl_rows_from_lines,
    dedup_exact,
    provenance_for,
)


def test_dedup_exact_keeps_first_occurrence_and_drops_exact_repeats():
    rows = [
        CorpusRow(id="a:0", source="a", text="fix the bug"),
        CorpusRow(id="a:1", source="a", text="fix the bug"),
        CorpusRow(id="a:2", source="a", text="a different task"),
    ]
    deduped = dedup_exact(rows)
    assert [r.id for r in deduped] == ["a:0", "a:2"]


def test_dedup_exact_normalizes_surrounding_whitespace_before_hashing():
    rows = [
        CorpusRow(id="a:0", source="a", text="fix the bug"),
        CorpusRow(id="a:1", source="a", text="  fix the bug  "),
    ]
    assert len(dedup_exact(rows)) == 1


def test_dedup_exact_is_a_noop_on_already_unique_rows():
    rows = [CorpusRow(id=f"a:{i}", source="a", text=f"task {i}") for i in range(5)]
    assert dedup_exact(rows) == rows


def test_provenance_for_known_source_uses_registered_metadata():
    provenance = provenance_for("ds1000", rows=42)
    assert provenance.hf_id == SOURCE_METADATA["ds1000"]["hf_id"]
    assert provenance.split == SOURCE_METADATA["ds1000"]["split"]
    assert provenance.rows == 42
    assert provenance.license == "CC-BY-SA-4.0"


def test_multi_swe_rl_is_registered_with_a_pinned_batch_and_unverified_license():
    # load_dataset can't read this dataset at all (see corpus.py's module docstring), so `split`
    # intentionally carries the release batch directory rather than an HF split name.
    meta = SOURCE_METADATA["multi-swe-rl"]
    assert meta["hf_id"] == "ByteDance-Seed/Multi-SWE-RL"
    assert meta["split"] == corpus_mod.MULTI_SWE_RL_BATCH
    assert meta["license"].startswith("unverified")


def test_multi_swe_rl_provenance_reports_all_six_fields():
    provenance = provenance_for("multi-swe-rl", rows=4723)
    assert provenance.rows == 4723
    assert "resolved_issues" in provenance.field
    assert "title+body" in provenance.field


def test_multi_swe_rl_is_registered_last_so_dedup_never_evicts_an_incumbent_row():
    # build_corpus() iterates SOURCE_METADATA in insertion order and dedup_exact() keeps the FIRST
    # occurrence of a duplicate, so a newcomer registered anywhere but last could evict an existing
    # row and churn its id (and thus the corpus digest).
    assert list(SOURCE_METADATA)[-1] == "multi-swe-rl"


def test_multi_swe_rl_is_a_corpus_source_only_not_a_calibration_source():
    # No grader exists for it (would need a per-repo Docker image per instance) — this documents
    # that boundary so it stays a conscious decision, not an accidental omission.
    assert "multi-swe-rl" not in tasks_mod._LOADERS


def test_multi_swe_rl_text_prefers_resolved_issue_over_pull_request_text():
    record = {
        "title": "short PR title",
        "body": "short PR body",
        "resolved_issues": [
            {"title": "A" * 50, "body": "B" * 50, "number": 1},
        ],
    }
    text = _extract_multi_swe_rl_text(record)
    assert "short PR title" not in text
    assert "A" * 50 in text
    assert "B" * 50 in text


def test_multi_swe_rl_text_concatenates_multiple_resolved_issues():
    record = {
        "resolved_issues": [
            {"title": "first issue " * 5, "body": "first body " * 5},
            {"title": "second issue " * 5, "body": "second body " * 5},
        ],
    }
    text = _extract_multi_swe_rl_text(record)
    assert "first issue" in text
    assert "second issue" in text


def test_multi_swe_rl_text_falls_back_to_pull_request_text_when_issues_missing():
    record = {"title": "fix the null pointer crash", "body": "details " * 20}
    text = _extract_multi_swe_rl_text(record)
    assert text is not None
    assert "fix the null pointer crash" in text


def test_multi_swe_rl_text_falls_back_when_resolved_issues_is_empty_list():
    record = {"title": "fix it", "body": "the actual fix description " * 5, "resolved_issues": []}
    assert "fix it" in _extract_multi_swe_rl_text(record)


def test_multi_swe_rl_text_falls_back_when_issue_text_is_too_thin():
    record = {
        "title": "a real pull request title with enough content",
        "body": "a real pull request body with enough content " * 3,
        "resolved_issues": [{"title": "x", "body": ""}],  # well under the threshold
    }
    text = _extract_multi_swe_rl_text(record)
    assert "pull request" in text


def test_multi_swe_rl_text_returns_none_when_nothing_usable():
    assert _extract_multi_swe_rl_text({}) is None
    assert _extract_multi_swe_rl_text({"resolved_issues": []}) is None
    assert _extract_multi_swe_rl_text({"title": "", "body": None}) is None


def test_multi_swe_rl_text_never_emits_the_literal_string_none_for_a_null_body():
    record = {"title": "a title with plenty of characters here", "body": None}
    text = _extract_multi_swe_rl_text(record)
    assert text is None or "None" not in text


def test_multi_swe_rl_text_survives_malformed_resolved_issues_shapes():
    assert _extract_multi_swe_rl_text({"resolved_issues": "oops"}) is None
    record = {
        "resolved_issues": [None, 5, {"title": "x" * 100, "body": ""}],
        "title": "",
        "body": "",
    }
    text = _extract_multi_swe_rl_text(record)
    assert text is not None
    assert "x" * 100 in text


def test_multi_swe_rl_row_id_uses_instance_id_when_present():
    assert _multi_swe_rl_row_id({"instance_id": "helix-editor__helix-6675"}) == "multi-swe-rl:helix-editor__helix-6675"


def test_multi_swe_rl_row_id_reconstructs_from_org_repo_number_when_instance_id_absent():
    record = {"org": "helix-editor", "repo": "helix", "number": 6675}
    assert _multi_swe_rl_row_id(record) == "multi-swe-rl:helix-editor__helix-6675"


def test_multi_swe_rl_row_id_is_none_when_neither_is_available():
    assert _multi_swe_rl_row_id({}) is None
    assert _multi_swe_rl_row_id({"org": "helix-editor"}) is None


def test_multi_swe_rl_ordered_paths_excludes_non_dataset_files():
    files = [
        ("data_20240601_20250331/rust/a__b_dataset.jsonl", 100),
        ("data_20240601_20250331/multi_swe_bench_discarded_instances.jsonl", 50),
    ]
    paths = _multi_swe_rl_ordered_paths(files)
    assert paths == ["data_20240601_20250331/rust/a__b_dataset.jsonl"]


def test_multi_swe_rl_ordered_paths_is_smallest_first_per_language_then_interleaved():
    files = [
        ("data/rust/big_dataset.jsonl", 900),
        ("data/rust/small_dataset.jsonl", 10),
        ("data/go/big_dataset.jsonl", 800),
        ("data/go/small_dataset.jsonl", 20),
        ("data/java/big_dataset.jsonl", 700),
        ("data/java/small_dataset.jsonl", 30),
    ]
    paths = _multi_swe_rl_ordered_paths(files)
    # First 3 results must be one smallest file per language (3 distinct languages) — proves a
    # capped/truncated run stays both cheap and multilingual, not cheap-but-single-language.
    first_three = paths[:3]
    assert {p.split("/")[1] for p in first_three} == {"rust", "go", "java"}
    assert all(p.endswith("small_dataset.jsonl") for p in first_three)


def test_multi_swe_rl_ordered_paths_is_deterministic_regardless_of_input_order():
    files = [
        ("data/rust/a_dataset.jsonl", 5),
        ("data/go/a_dataset.jsonl", 3),
        ("data/rust/b_dataset.jsonl", 1),
    ]
    shuffled = files.copy()
    random.Random(0).shuffle(shuffled)
    assert _multi_swe_rl_ordered_paths(files) == _multi_swe_rl_ordered_paths(shuffled)


def test_multi_swe_rl_rows_from_lines_skips_blank_and_malformed_lines():
    good_a = json.dumps({"instance_id": "a__b-1", "title": "x", "body": "y" * 20})
    good_b = json.dumps({"instance_id": "a__b-2", "title": "fix bug", "body": "a real body " * 5})
    lines = ["", "not json at all {{{", good_a, "   ", good_b]
    rows = list(_multi_swe_rl_rows_from_lines(lines, seen_ids=set()))
    assert [r.id for r in rows] == ["multi-swe-rl:a__b-1", "multi-swe-rl:a__b-2"]
    assert all(r.source == "multi-swe-rl" for r in rows)


def test_multi_swe_rl_rows_from_lines_skips_non_object_json():
    lines = ["[]", '"just a string"', "42"]
    assert list(_multi_swe_rl_rows_from_lines(lines, seen_ids=set())) == []


def test_multi_swe_rl_rows_from_lines_deduplicates_against_a_shared_seen_ids_set():
    seen_ids: set[str] = set()
    line = json.dumps({"instance_id": "a__b-1", "title": "fix bug", "body": "a real body " * 5})
    first = list(_multi_swe_rl_rows_from_lines([line], seen_ids))
    second = list(_multi_swe_rl_rows_from_lines([line], seen_ids))
    assert len(first) == 1
    assert len(second) == 0
