"""K-means clustering over corpus embeddings. k is config/clustering.yaml's default_k unless
`build-artifact --k <value>` overrides it."""
from __future__ import annotations

import dataclasses
import logging

import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

# Below this fraction of the corpus, a cluster is flagged in diagnostics as suspiciously small —
# informational only, doesn't block anything.
SMALL_CLUSTER_WARNING_FRACTION = 0.01


@dataclasses.dataclass(frozen=True)
class ClusterDiagnostics:
    k: int
    inertia: float
    sizes: list[int]
    min_size: int
    max_size: int
    mean_size: float


@dataclasses.dataclass(frozen=True)
class ClusterResult:
    k: int
    labels: np.ndarray
    centroids: np.ndarray  # float64, shape (k, dimensions)
    diagnostics: ClusterDiagnostics


def run_kmeans(vectors: np.ndarray, k: int, seed: int, n_init: int) -> ClusterResult:
    # n_init is passed explicitly (never "auto") — sklearn's "auto" resolves to a single run for
    # the default k-means++ init, silently defeating the spec's "at least 10 initializations."
    model = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
    labels = model.fit_predict(vectors)
    centroids = model.cluster_centers_.astype(np.float64)  # inherits input dtype otherwise
    sizes = np.bincount(labels, minlength=k).tolist()
    diagnostics = ClusterDiagnostics(
        k=k,
        inertia=float(model.inertia_),
        sizes=sizes,
        min_size=min(sizes),
        max_size=max(sizes),
        mean_size=sum(sizes) / k,
    )
    logger.info(
        f"k={k} evaluated: inertia={diagnostics.inertia:.2f}, "
        f"cluster sizes min={diagnostics.min_size} max={diagnostics.max_size} mean={diagnostics.mean_size:.1f}"
    )
    return ClusterResult(k=k, labels=labels, centroids=centroids, diagnostics=diagnostics)


def format_diagnostics(diag: ClusterDiagnostics, corpus_size: int) -> str:
    threshold = max(1, int(corpus_size * SMALL_CLUSTER_WARNING_FRACTION))
    warning = ""
    if diag.min_size < threshold:
        warning = f" [WARNING: min cluster size {diag.min_size} is below {threshold} ({SMALL_CLUSTER_WARNING_FRACTION:.0%} of corpus)]"
    return (
        f"k={diag.k}: inertia={diag.inertia:.2f}, "
        f"cluster size min={diag.min_size} max={diag.max_size} mean={diag.mean_size:.1f}{warning}"
    )
