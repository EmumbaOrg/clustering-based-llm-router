"""Loads the corpus: SWE-smith (sampled 20,000 of ~59,136), SWE-Gym, BigCodeBench-Instruct, and
DS-1000 (all rows from each). Multi-SWE-RL is deliberately excluded from this pass — a 23.8GB
download with no independently-verifiable row count and a license conflict between its README
(CC0) and its repo metadata tag ("other"). See ../README.md and
docs/superpowers/specs/2026-08-04-cluster-routing-implementation-plan.md for the full reasoning.
Adding it later is a new entry in SOURCE_METADATA plus a loader for its non-parquet JSONL layout,
not a redesign of this module.

Row ids are pipeline-local (`<source>:<position-after-sampling>`), not each dataset's own id field
— not every source is guaranteed to expose one under a consistent name, and a positional id is
sufficient for this pipeline's own traceability (corpus.jsonl <-> embeddings.npz <-> artifact).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from pathlib import Path

from datasets import load_dataset

logger = logging.getLogger(__name__)

# hf_id/split/field are exactly what's needed to reproduce this pull: `load_dataset(hf_id,
# split=split)`, then read `field` for the natural-language task text. Nothing else from these
# datasets (patches, test files, repo snapshots) is used.
SOURCE_METADATA: dict[str, dict[str, str]] = {
    "swe-smith": {
        "hf_id": "SWE-bench/SWE-smith",
        "split": "train",
        "field": "problem_statement",
        "license": "MIT",
    },
    "swe-gym": {
        "hf_id": "SWE-Gym/SWE-Gym",
        "split": "train",
        "field": "problem_statement",
        "license": "MIT",
    },
    "bigcodebench": {
        "hf_id": "bigcode/bigcodebench",
        "split": "v0.1.4",  # latest versioned split; the unversioned default merges all 5 (5,700 rows)
        "field": "instruct_prompt",  # not complete_prompt, which is docstring-style not instruction-style
        "license": "Apache-2.0",
    },
    "ds1000": {
        "hf_id": "xlangai/DS-1000",
        "split": "test",
        "field": "prompt",
        "license": "CC-BY-SA-4.0",
    },
}

# SWE-smith is the one source the spec asks us to cap (20,000 of ~59,136 available) so synthetic
# Python tasks don't dominate the corpus. Every other source contributes everything it has.
SWE_SMITH_SAMPLE_SIZE = 20_000
CORPUS_SAMPLE_SEED = 42


@dataclasses.dataclass(frozen=True)
class CorpusRow:
    id: str
    source: str
    text: str


@dataclasses.dataclass(frozen=True)
class SourceProvenance:
    name: str
    hf_id: str
    split: str
    field: str
    rows: int
    license: str


def provenance_for(name: str, rows: int) -> SourceProvenance:
    meta = SOURCE_METADATA[name]
    return SourceProvenance(
        name=name, hf_id=meta["hf_id"], split=meta["split"], field=meta["field"], rows=rows, license=meta["license"],
    )


def _load_source(name: str, cap: int | None) -> list[CorpusRow]:
    meta = SOURCE_METADATA[name]
    logger.info(f"loading corpus source {name} ({meta['hf_id']}, split={meta['split']})")
    ds = load_dataset(meta["hf_id"], split=meta["split"])

    effective_cap = cap
    if name == "swe-smith" and cap is None:
        effective_cap = SWE_SMITH_SAMPLE_SIZE
    if effective_cap is not None and effective_cap < len(ds):
        ds = ds.shuffle(seed=CORPUS_SAMPLE_SEED)

    # Shuffle-then-filter-then-stop, rather than shuffle-then-cap-then-filter: a meaningful
    # fraction of SWE-smith rows have a genuinely empty problem_statement (confirmed by direct
    # inspection — synthetic mutation tasks with no generated NL description), so capping BEFORE
    # filtering would silently under-sample. This guarantees up to `effective_cap` non-empty rows
    # whenever that many exist in the dataset, at the cost of a full pass when a cap is set.
    rows: list[CorpusRow] = []
    for i, example in enumerate(ds):
        if effective_cap is not None and len(rows) >= effective_cap:
            break
        text = example.get(meta["field"])
        if not text or not str(text).strip():
            continue
        rows.append(CorpusRow(id=f"{name}:{i}", source=name, text=str(text)))
    logger.info(f"loaded {len(rows)} rows from {name} ({meta['license']})")
    return rows


def dedup_exact(rows: list[CorpusRow]) -> list[CorpusRow]:
    """Near-duplicate detection (MinHash/LSH or an embedding-similarity pass) is deferred — every
    prompt gets embedded anyway, so a cheap cosine-similarity follow-up is possible later without
    touching this function.
    """
    seen: set[str] = set()
    deduped: list[CorpusRow] = []
    for row in rows:
        key = hashlib.sha256(row.text.strip().encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def build_corpus(sample: int | None = None) -> tuple[list[CorpusRow], list[SourceProvenance]]:
    """`sample`, when given, caps EACH source at that many rows (a dry-run knob, not a global cap)
    — so a small sample still exercises every loader rather than only the first source's data.
    """
    all_rows: list[CorpusRow] = []
    provenances: list[SourceProvenance] = []
    for name in SOURCE_METADATA:
        rows = _load_source(name, sample)
        all_rows.extend(rows)
        provenances.append(provenance_for(name, len(rows)))
    deduped = dedup_exact(all_rows)
    logger.info(f"corpus build complete: {len(deduped)} rows after dedup (from {len(all_rows)})")
    return deduped, provenances


def write_corpus_jsonl(rows: list[CorpusRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dataclasses.asdict(row), ensure_ascii=False) + "\n")


def read_corpus_jsonl(path: Path) -> list[CorpusRow]:
    rows: list[CorpusRow] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(CorpusRow(**json.loads(line)))
    return rows
