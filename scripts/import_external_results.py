"""One-off: imports `external-results/<source>/<model>.json` files (real per-task pass/fail data
pulled from public result datasets -- bigcode/bigcodebench-perf, SWE-bench/SWE-smith-trajectories,
SWE-Gym/OpenHands-Sampled-Trajectories) and merges a real per-model calibration result into
artifacts/model-profiles.json, using the pipeline's own aggregation/smoothing code
(calibrate.py::_aggregate_outcomes, profiles.py::build_profiles_dict/merge_profiles_dict) rather
than a hand-rolled reimplementation of the smoothing formula.

Per model, every task row across every source file for that model_id is combined into ONE
aggregation pass before merging -- merge_profiles_dict REPLACES a model's entry wholesale rather
than combining multiple calls, so importing e.g. gpt-4o-2024-08-06 from swe-smith/ and swe-gym/
as two separate merges would silently drop whichever source's evidence went in first.

These external outcomes only ever carry "pass"/"fail" -- none of the 3 source datasets report
error_missing_dep/error_timeout/error_harness granularity, so a real calibration run's failure-mode
detail is not reproduced here (see each external-results file's own "notes" field).

Run: .venv/bin/python scripts/import_external_results.py [--dry-run] [--model MODEL_ID ...]
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from router.common.assign import load_cluster_map
from router.common.config import ModelConfig, load_calibration_config, load_embedding_config, load_models_config
from router.pipeline.calibration import profiles as profiles_mod
from router.pipeline.calibration.calibrate import SelectedTask, _aggregate_outcomes
from router.pipeline.calibration.grading.base import GradeResult, Task

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_RESULTS_DIR = REPO_ROOT / "external-results"
PROFILES_PATH = profiles_mod.ARTIFACTS_DIR / "model-profiles.json"
CLUSTER_MAP_PATH = profiles_mod.ARTIFACTS_DIR / "cluster-map.json"

# Models with real result data under external-results/ but not (yet) in config/models.yaml.
# Filled in with real published specs so the profile entry -- and any future live use of this
# model_id -- stays accurate. Only consulted when config/models.yaml doesn't already define it.
FALLBACK_MODEL_CONFIGS: dict[str, ModelConfig] = {
    "gpt-4o-2024-08-06": ModelConfig(
        model_id="gpt-4o-2024-08-06",
        provider="openai",
        runner="pi",
        cost_input=2.50e-6,
        cost_output=10.00e-6,
        context_window=128_000,
        max_tokens=16_384,
    ),
}


def _load_external_result_files() -> dict[str, list[Path]]:
    """model_id -> every external-results/<source>/<model_id>.json file found for it."""
    by_model: dict[str, list[Path]] = {}
    for source_dir in sorted(p for p in EXTERNAL_RESULTS_DIR.iterdir() if p.is_dir()):
        for f in sorted(source_dir.glob("*.json")):
            by_model.setdefault(f.stem, []).append(f)
    return by_model


def _resolve_model_config(model_id: str, known_models: dict[str, ModelConfig]) -> ModelConfig:
    if model_id in known_models:
        return known_models[model_id]
    if model_id in FALLBACK_MODEL_CONFIGS:
        return FALLBACK_MODEL_CONFIGS[model_id]
    raise SystemExit(
        f"model_id {model_id!r} is in external-results/ but not in config/models.yaml and has no "
        f"FALLBACK_MODEL_CONFIGS entry in this script -- add one before importing it."
    )


def _selected_tasks_and_outcomes(files: list[Path]) -> tuple[list[tuple[SelectedTask, GradeResult]], list[str]]:
    per_task_outcomes: list[tuple[SelectedTask, GradeResult]] = []
    seen_task_ids: set[str] = set()
    sources_used: list[str] = []
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        sources_used.append(data["source"]["dataset"])
        for row in data["tasks"]:
            task_id = row["task_id"]
            if task_id in seen_task_ids:
                raise SystemExit(
                    f"task_id {task_id!r} appears in more than one external-results file for the "
                    f"same model ({f.parent.name}) -- expected disjoint task ids across sources."
                )
            seen_task_ids.add(task_id)
            task = Task(task_id=task_id, source=f.parent.name, prompt="", reference_solution="", row={})
            selected = SelectedTask(task=task, cluster_id=row["cluster_id"], split="calibration")
            grade = GradeResult(outcome=row["outcome"], detail=f"imported from {data['source']['dataset']}")
            per_task_outcomes.append((selected, grade))
    return per_task_outcomes, sources_used


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print what would be merged, don't write model-profiles.json")
    parser.add_argument(
        "--model", action="append",
        help="Only import this model_id (repeatable). Default: every model found under external-results/.",
    )
    args = parser.parse_args()

    by_model = _load_external_result_files()
    if not by_model:
        raise SystemExit(f"no external-results/*/*.json files found under {EXTERNAL_RESULTS_DIR}")

    target_models = args.model or sorted(by_model)
    unknown = [m for m in target_models if m not in by_model]
    if unknown:
        raise SystemExit(f"--model {unknown} not found under {EXTERNAL_RESULTS_DIR} (available: {sorted(by_model)})")

    known_models = {m.model_id: m for m in load_models_config()}
    cluster_map = load_cluster_map(CLUSTER_MAP_PATH)
    embedding_config = load_embedding_config()
    calibration_config = load_calibration_config()

    results = []
    for model_id in target_models:
        files = by_model[model_id]
        model_config = _resolve_model_config(model_id, known_models)
        per_task_outcomes, sources_used = _selected_tasks_and_outcomes(files)
        result = _aggregate_outcomes(model_config, per_task_outcomes, calibration_config)
        results.append(result)
        stats = result.global_stats
        print(
            f"{model_id}: {stats.number_succeeded}/{stats.number_of_tasks} pass "
            f"(smoothed_error_rate={stats.smoothed_error_rate:.3f}), "
            f"{len(result.cluster_stats)}/{cluster_map.centroids.shape[0]} clusters touched, "
            f"sources={sources_used}"
        )

    calibration_run_id = f"external-import-{datetime.now(UTC).strftime('%Y-%m-%d-%H%M%S')}"
    new_artifact = profiles_mod.build_profiles_dict(
        results, cluster_map, embedding_config, calibration_config, calibration_run_id,
    )

    if not PROFILES_PATH.exists():
        raise SystemExit(f"{PROFILES_PATH} doesn't exist -- run a full calibration first.")
    existing_artifact = json.loads(PROFILES_PATH.read_text(encoding="utf-8"))
    merged = profiles_mod.merge_profiles_dict(existing_artifact, new_artifact)
    profiles_mod.validate_profiles(merged)

    if args.dry_run:
        print(
            f"\n--dry-run: would merge {target_models} into {PROFILES_PATH} "
            f"({len(existing_artifact['models'])} -> {len(merged['models'])} models). Not written."
        )
        return

    path = profiles_mod.write_profiles(merged, PROFILES_PATH)
    print(f"\nWrote merged artifact to {path} ({len(merged['models'])} models total)")


if __name__ == "__main__":
    main()
