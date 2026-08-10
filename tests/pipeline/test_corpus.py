from router.pipeline.corpus import (
    SOURCE_METADATA,
    CorpusRow,
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


def test_multi_swe_rl_is_not_a_registered_source():
    # Deliberately excluded this pass (unverified license, 23.8GB) — this test documents that
    # exclusion so re-adding it is a conscious decision, not an accidental one.
    assert "multi-swe-rl" not in SOURCE_METADATA
