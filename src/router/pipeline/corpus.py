"""Loads the corpus: SWE-smith (sampled 20,000 of ~59,136), SWE-Gym, BigCodeBench-Instruct,
DS-1000 (all rows from each), and Multi-SWE-RL (batch 1 only, ~4,723 multilingual instances — see
../README.md). Multi-SWE-RL spans 7 languages (C, C++, Go, Java, JS, Rust, TS); only its Go slice
is gradeable so far (`calibration/grading/multiswerl.py`, `calibration/tasks.py`'s Go-only loader)
— the other 6 remain corpus-only here, same as before.

Multi-SWE-RL can't use the generic `load_dataset(hf_id, split=split)` path below — its 74
batch-1 JSONL files have per-repo-heterogeneous nested fields, and Arrow schema unification across
them fails with "Couldn't cast array of type string to null" (the same reason the HF dataset
viewer is broken for this dataset). It's fetched file-by-file via `huggingface_hub.hf_hub_download`
and parsed with plain `json.loads` instead; see `_load_multi_swe_rl`.

Row ids are pipeline-local (`<source>:<position-after-sampling>`) for every source EXCEPT
multi-swe-rl, which uses the dataset's own stable `instance_id`. That source's files are read
smallest-first (see `_multi_swe_rl_ordered_paths`) and that order is truncated by `--sample`, so a
positional id would not be stable across runs — and `compute_corpus_digest`
(clustering/cluster_map.py) sorts rows by id, so an unstable id would make the corpus digest
non-reproducible for identical content.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from posixpath import basename, dirname

from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download

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
    # Registered LAST deliberately: build_corpus() iterates SOURCE_METADATA in insertion order and
    # dedup_exact() keeps the FIRST occurrence of a duplicate, so a Multi-SWE-RL row that happens
    # to duplicate an existing row can never evict it and churn an existing row's id (and thus the
    # corpus digest). `split` has no HF meaning here — load_dataset can't read this dataset at all
    # (see module docstring) — so it instead records which release batch was pulled, which is what
    # actually matters for reproducing this source. `field` is a composite description rather than
    # one dataset field name; both are plain minLength-1 strings in the schema, so this needs no
    # schema change.
    "multi-swe-rl": {
        "hf_id": "ByteDance-Seed/Multi-SWE-RL",
        "split": "data_20240601_20250331",
        "field": "resolved_issues[].title+body (fallback: title+body)",
        "license": "unverified (card: CC0-1.0; HF tag: other)",
    },
}

# SWE-smith is the one source the spec asks us to cap (20,000 of ~59,136 available) so synthetic
# Python tasks don't dominate the corpus. Every other source contributes everything it has.
SWE_SMITH_SAMPLE_SIZE = 20_000
CORPUS_SAMPLE_SEED = 42

# Multi-SWE-RL's initial ~4,723-instance release. A second batch (data_20250401_20250631) exists
# upstream and is deliberately not pulled — pinning one batch keeps this source reproducible as
# upstream adds more, matching docs/routing-poc-background-research.md's original ~4,723 figure.
MULTI_SWE_RL_BATCH = SOURCE_METADATA["multi-swe-rl"]["split"]

# Instance files are `<language>/<org>__<repo>_dataset.jsonl`. Matching the suffix rather than
# blocklisting a name is what excludes multi_swe_bench_discarded_instances.jsonl (a list of
# dropped ids, not instances) and any future sibling metadata file in the same batch directory.
_MULTI_SWE_RL_FILE_SUFFIX = "_dataset.jsonl"

# Measured directly over 39 records spanning all 7 batch-1 languages: resolved_issues title+body
# was present in 39/39, median 750 chars, minimum 99. This threshold should never fire on today's
# data — it exists to catch upstream format drift, not to route normal rows to the weaker
# title/body (pull request) fallback.
_MULTI_SWE_RL_MIN_ISSUE_CHARS = 80


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


def _extract_multi_swe_rl_text(record: dict) -> str | None:
    # This dataset has no problem_statement field. `resolved_issues` is the GitHub ISSUE (the
    # problem) and is the true analogue of the other sources' problem_statement; the top-level
    # title/body is the PULL REQUEST, i.e. a description of the solution — embedding that would
    # cluster on how a fix was written, not on what was asked. Hence PR text is only a fallback,
    # used when the issue text is missing or too thin to be a real substitute (see
    # _MULTI_SWE_RL_MIN_ISSUE_CHARS).
    issues = record.get("resolved_issues")
    parts: list[str] = []
    if isinstance(issues, list):
        for issue in issues:
            if not isinstance(issue, dict):
                continue
            parts += [str(issue.get(k) or "").strip() for k in ("title", "body")]
    issue_text = "\n\n".join(p for p in parts if p)

    pr_text = "\n\n".join(
        p for p in (str(record.get("title") or "").strip(), str(record.get("body") or "").strip()) if p
    )
    if len(issue_text) >= _MULTI_SWE_RL_MIN_ISSUE_CHARS:
        return issue_text
    # str(x or "") rather than str(x) throughout: a JSON null body must never become the literal
    # string "None" in the corpus.
    best = max((issue_text, pr_text), key=len)
    return best or None


def _multi_swe_rl_row_id(record: dict) -> str | None:
    instance_id = str(record.get("instance_id") or "").strip()
    if not instance_id:
        # Reconstruct the dataset's own `<org>__<repo>-<number>` form rather than falling back to
        # a positional index — see the module docstring for why this source can't use positional
        # ids.
        org, repo, number = (str(record.get(k) or "").strip() for k in ("org", "repo", "number"))
        if not (org and repo and number):
            return None
        instance_id = f"{org}__{repo}-{number}"
    return f"multi-swe-rl:{instance_id}"


def _multi_swe_rl_ordered_paths(files: list[tuple[str, int]]) -> list[str]:
    """Smallest-first WITHIN each language, then round-robin ACROSS languages.

    File sizes are extremely skewed — a handful of files are 86MB-973MB (bloat is per-instance
    test-log fields, not more instances) while most of the rest total well under that. `--sample N`
    stops as soon as N rows exist, so read order decides whether a dry run costs megabytes or
    gigabytes. Ranking by size within a language and interleaving keeps a dry run both cheap AND
    multilingual, rather than cheap-but-single-language.

    TRADEOFF: unlike every other source, a CAPPED multi-swe-rl sample is not random — it is biased
    toward small repos. Fine for a dry run that just needs to exercise the loader; never treat a
    --sample multi-swe-rl subset as representative.
    """
    by_language: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for path, size in files:
        if basename(path).endswith(_MULTI_SWE_RL_FILE_SUFFIX):
            by_language[basename(dirname(path))].append((size, path))
    ranked = [
        (rank, size, path)
        for _, entries in sorted(by_language.items())
        for rank, (size, path) in enumerate(sorted(entries))
    ]
    return [path for _, _, path in sorted(ranked)]


def _multi_swe_rl_rows_from_lines(lines: Iterable[str], seen_ids: set[str]) -> Iterator[CorpusRow]:
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue  # a single truncated/malformed line must not abort a 74-file, multi-GB read
        if not isinstance(record, dict):
            continue
        row_id = _multi_swe_rl_row_id(record)
        text = _extract_multi_swe_rl_text(record)
        if row_id is None or text is None or row_id in seen_ids:
            continue
        seen_ids.add(row_id)
        yield CorpusRow(id=row_id, source="multi-swe-rl", text=text)


def _multi_swe_rl_repo_files() -> list[tuple[str, int]]:
    hf_id = SOURCE_METADATA["multi-swe-rl"]["hf_id"]
    entries = HfApi().list_repo_tree(hf_id, MULTI_SWE_RL_BATCH, repo_type="dataset", recursive=True)
    return [(e.path, e.size) for e in entries if getattr(e, "size", None) is not None]


def _load_multi_swe_rl(cap: int | None) -> list[CorpusRow]:
    hf_id = SOURCE_METADATA["multi-swe-rl"]["hf_id"]
    paths = _multi_swe_rl_ordered_paths(_multi_swe_rl_repo_files())
    logger.info(f"loading corpus source multi-swe-rl ({hf_id}, batch={MULTI_SWE_RL_BATCH}, {len(paths)} files)")

    # No default cap, unlike swe-smith: all ~4,723 instances are wanted for the language/ecosystem
    # coverage this source exists to add (docs/routing-poc-background-research.md).
    rows: list[CorpusRow] = []
    seen_ids: set[str] = set()
    for i, path in enumerate(paths, start=1):
        if cap is not None and len(rows) >= cap:
            break  # checked BEFORE hf_hub_download, so a capped run never pays for an unread file
        local_path = hf_hub_download(hf_id, filename=path, repo_type="dataset")
        before = len(rows)
        # errors="replace": a stray non-UTF-8 byte in an embedded test log must not kill the run.
        with open(local_path, encoding="utf-8", errors="replace") as f:
            for row in _multi_swe_rl_rows_from_lines(f, seen_ids):
                rows.append(row)
                if cap is not None and len(rows) >= cap:
                    break
        logger.info(f"multi-swe-rl [{i}/{len(paths)}] {path}: +{len(rows) - before} rows (total {len(rows)})")
    logger.info(f"loaded {len(rows)} rows from multi-swe-rl ({SOURCE_METADATA['multi-swe-rl']['license']})")
    return rows


# Sources needing something other than the generic load_dataset(hf_id, split=split) path below —
# same shape as calibration/tasks.py's _LOADERS.
_CUSTOM_LOADERS: dict[str, Callable[[int | None], list[CorpusRow]]] = {
    "multi-swe-rl": _load_multi_swe_rl,
}


def _load_source(name: str, cap: int | None) -> list[CorpusRow]:
    custom_loader = _CUSTOM_LOADERS.get(name)
    if custom_loader is not None:
        return custom_loader(cap)

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
