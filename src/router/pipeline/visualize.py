"""Projects the corpus embedding space down to 2D (PCA) and plots where a sample of tasks landed
relative to the K-means cluster centroids from artifacts/cluster-map.json — a sanity-check view of
the cluster map, not a routing artifact itself. Cluster assignment uses the exact same
`assign_cluster` the runtime/calibration use, so what's plotted matches real routing decisions
rather than a fresh, possibly-inconsistent re-clustering.
"""
from __future__ import annotations

import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — this runs from a CLI, never a GUI session
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from ..common.assign import ClusterMap, assign_cluster


def plot_clusters(
    ids: list[str],
    vectors: np.ndarray,
    cluster_map: ClusterMap,
    output_path: Path,
    sample_size: int = 500,
    highlight_ids: list[str] | None = None,
    seed: int = 42,
) -> tuple[Path, list[str]]:
    """Samples `sample_size` tasks (plus any `highlight_ids`, even if not sampled), assigns each to
    its cluster, projects everything into 2D with one shared PCA fit (so centroids and points land
    in the same coordinate system), and saves a scatter plot to `output_path`.

    Returns (output_path, missing_highlight_ids) — the latter so the caller can warn about any
    `highlight_ids` not found in `ids` rather than silently dropping them.
    """
    id_to_idx = {tid: i for i, tid in enumerate(ids)}
    highlight_ids = highlight_ids or []
    missing = [tid for tid in highlight_ids if tid not in id_to_idx]
    highlight_positions = [id_to_idx[tid] for tid in highlight_ids if tid in id_to_idx]

    rng = random.Random(seed)
    n = len(ids)
    sample_positions = rng.sample(range(n), min(sample_size, n))

    combined_positions = sorted(set(sample_positions) | set(highlight_positions))
    combined_ids = [ids[i] for i in combined_positions]
    combined_vectors = vectors[combined_positions]

    cluster_ids = np.array([assign_cluster(v, cluster_map).cluster_id for v in combined_vectors])
    k = cluster_map.centroids.shape[0]

    pca = PCA(n_components=2, random_state=seed)
    points_2d = pca.fit_transform(combined_vectors)
    centroids_2d = pca.transform(cluster_map.centroids)
    variance_retained = pca.explained_variance_ratio_.sum()

    fig, ax = plt.subplots(figsize=(11, 8))
    cmap = plt.get_cmap("tab20" if k <= 20 else "hsv", k)
    for cid in range(k):
        mask = cluster_ids == cid
        if mask.any():
            ax.scatter(
                points_2d[mask, 0], points_2d[mask, 1],
                s=18, color=cmap(cid), alpha=0.65, label=f"cluster {cid} (n={mask.sum()})",
            )

    ax.scatter(
        centroids_2d[:, 0], centroids_2d[:, 1],
        marker="X", s=220, color="black", edgecolors="white", linewidths=1.2, zorder=5,
    )
    for cid in range(k):
        ax.annotate(
            str(cid), (centroids_2d[cid, 0], centroids_2d[cid, 1]),
            color="white", fontsize=8, fontweight="bold", ha="center", va="center", zorder=6,
        )

    highlight_id_set = set(highlight_ids)
    for i, tid in enumerate(combined_ids):
        if tid in highlight_id_set:
            x, y = points_2d[i]
            ax.scatter([x], [y], s=70, facecolors="none", edgecolors="red", linewidths=1.6, zorder=7)
            ax.annotate(tid, (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points", zorder=8)

    ax.set_title(
        f"Cluster map projection (PCA, {variance_retained:.0%} variance retained) — "
        f"{len(combined_ids)} tasks across k={k} clusters"
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path, missing
