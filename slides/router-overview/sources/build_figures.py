"""Regenerates the data-driven figures for router-overview-slides.md from the real artifacts.
Run from the clustering-based-llm-router repo root: uv run python <this folder>/build_figures.py
Images are written to the img/ folder next to this sources/ folder."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from router.common.config import load_models_config
from router.common.scoring import score_candidates

OUT = Path(__file__).resolve().parents[1] / "img"
ART = Path("artifacts")
K = 24

# The current candidate set, dearest first, with its price tier and a fixed colour per model
# (the same colours the router UI uses).
CANDIDATES = {
    "gpt-6.1-sol": ("high", "#eb6834"),
    "gpt-5.4-mini": ("mid", "#1baf7a"),
    "gpt-5.6-luna": ("low", "#2a78d6"),
    "gpt-5-nano": ("lowest", "#eda100"),
}

SOURCES = ["swe-smith", "swe-gym", "multi-swe-rl", "swe-bench-pro", "bigcodebench", "ds1000"]
COLOR = {
    "swe-smith": "#1f6f78", "swe-gym": "#3a9ca6", "multi-swe-rl": "#8fd0d6",
    "swe-bench-pro": "#e08a2c",
    "bigcodebench": "#3b5ba9", "ds1000": "#8aa3dc",
}
plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})

tasks = json.load(open(ART / "task-cluster-map.json"))["tasks"]
profiles = json.load(open(ART / "model-profiles.json"))["models"]


def corpus_mix() -> None:
    counts = Counter(t["source"] for t in tasks)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ys = list(range(len(SOURCES)))[::-1]
    vals = [counts[s] for s in SOURCES]
    ax.barh(ys, vals, color=[COLOR[s] for s in SOURCES])
    ax.set_yticks(ys, SOURCES)
    for y, v in zip(ys, vals):
        ax.text(v + 250, y, f"{v:,}", va="center")
    ax.set_xlim(0, max(vals) * 1.15)
    ax.set_xlabel(f"tasks in corpus  (total {sum(vals):,})")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ("#1f6f78", "#e08a2c", "#3b5ba9")]
    ax.legend(handles, ["real repo, graded in Docker", "real repo, prompts only (not graded)", "self-contained, graded by exec"],
              loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "corpus-mix.png", dpi=160)


def sources_per_cluster() -> None:
    by = defaultdict(Counter)
    for t in tasks:
        by[t["cluster_id"]][t["source"]] += 1
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(K)
    bottom = np.zeros(K)
    for s in SOURCES:
        share = np.array([by[c][s] / max(1, sum(by[c].values())) for c in range(K)])
        ax.bar(x, share, bottom=bottom, color=COLOR[s], label=s, width=0.85)
        bottom += share
    ax.set_xticks(x, [str(c) for c in range(K)])
    ax.set_xlabel("cluster id")
    ax.set_ylabel("share of cluster's tasks")
    ax.set_ylim(0, 1)
    ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.13), frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "sources-per-cluster.png", dpi=160)


def profile_heatmap() -> None:
    by_id = {m["model_id"]: m for m in profiles}
    order = [by_id[mid] for mid in CANDIDATES]
    labels, grid, overall = [], [], []
    for m in order:
        grid.append([m["clusters"].get(str(c), {}).get("smoothed_error_rate", np.nan) for c in range(K)])
        labels.append(f"{m['model_id']}  ({CANDIDATES[m['model_id']][0]})")
        g = m["global"]
        overall.append((g["number_succeeded"], g["number_of_tasks"]))
    grid = np.array(grid, dtype=float)

    fig, (ax, axb) = plt.subplots(1, 2, figsize=(15, 3.6), gridspec_kw={"width_ratios": [5, 1.2]})
    cmap = matplotlib.colormaps["RdYlGn_r"].copy()
    cmap.set_bad("#d9d9d9")
    im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(K), [str(c) for c in range(K)], fontsize=10)
    ax.set_yticks(range(len(labels)), labels, fontsize=11)
    ax.set_xlabel("cluster id   (cell = smoothed error rate)")
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("error rate")

    rates = [s / n for s, n in overall]
    axb.barh(range(len(labels)), rates, color="#4c9a5a")
    axb.set_ylim(len(labels) - 0.5, -0.5)
    axb.set_yticks([])
    axb.set_xlim(0, 1.15)
    axb.set_xticks([0, 0.5, 1])
    axb.set_xlabel("overall pass rate")
    for y, (s, n) in enumerate(overall):
        axb.text(s / n + 0.02, y, f"{s}/{n}", va="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "profile-heatmap.png", dpi=160)


def lambda_crossover(cluster_id: int = 3) -> None:
    # Cluster 3 (python-docx / python-pptx tasks) is the one cluster where all three winning
    # tiers appear inside lambda 0-0.5: sol, then luna, then nano.
    roster = [m for m in load_models_config() if m.model_id in CANDIDATES]
    by_model = {m["model_id"]: m for m in profiles}
    lams = np.linspace(0, 0.5, 501)
    series = defaultdict(list)
    for lam in lams:
        for s in score_candidates(cluster_id, roster, by_model, float(lam)):
            series[s.model_id].append(s.routing_score)
    fig, ax = plt.subplots(figsize=(11, 5))
    for mid in CANDIDATES:
        ax.plot(lams, series[mid], label=f"{mid} ({CANDIDATES[mid][0]})", color=CANDIDATES[mid][1], linewidth=2.5)
    winner = [min(series, key=lambda m: series[m][i]) for i in range(len(lams))]
    flips = [lams[i] for i in range(1, len(lams)) if winner[i] != winner[i - 1]]
    top = ax.get_ylim()[1]
    for f in flips:
        ax.axvline(f, color="#888", linestyle="--", linewidth=1)
        ax.text(f + 0.005, top * 0.97, f"pick flips at λ≈{f:.2f}", va="top", fontsize=11, color="#555")
    for lam in (0, 0.02, 0.05, 0.10, 0.20, 0.40):
        ax.axvline(lam, color="#ddd", linewidth=0.8, zorder=0)
    ax.set_xlabel("λ  (cost weight)   — light lines = calibration sweep values")
    ax.set_ylabel("routing score  (lower wins)")
    ax.set_title(f"Cluster {cluster_id}: routing score = error + λ · normalised cost", fontsize=13)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "lambda-crossover.png", dpi=160)


if __name__ == "__main__":
    corpus_mix()
    sources_per_cluster()
    profile_heatmap()
    lambda_crossover()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
