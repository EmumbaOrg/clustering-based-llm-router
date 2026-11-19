"""A small local web UI over the routing decision: type a prompt, move lambda, see which cluster the
prompt lands in and which model wins. Stdlib HTTP only (no new dependency), bound to localhost.

One long-lived process, so the encoder loads once instead of on every `decide` call, and each
prompt's embedding is cached — moving the lambda slider re-scores without re-embedding.
"""
from __future__ import annotations

import collections
import dataclasses
import json
import logging
import threading
import time
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

from ..common.artifacts import ARTIFACTS_DIR
from ..common.embedding import embed_one
from ..common.scoring import (
    ScoredCandidate,
    normalise_costs,
    select_model,
    static_price_per_1m,
)
from .context import RoutingContext, _compute_digest
from .decide import RoutingDecision, decide_from_vector

logger = logging.getLogger(__name__)

# The four GPT models with full 240-task calibrations on the same pin — the comparable set.
DEFAULT_UI_MODELS = ("gpt-5.6-luna", "gpt-6.1-sol", "gpt-5.4-mini", "gpt-5-nano")

DEFAULT_TASK_CLUSTER_MAP_PATH = ARTIFACTS_DIR / "task-cluster-map.json"
LAMBDA_SWEEP = (0.0, 0.02, 0.05, 0.10, 0.20, 0.40)  # mirrors config/calibration.yaml's lambda_sweep
_PAGE_PATH = Path(__file__).parent / "ui.html"
_MAX_BODY_BYTES = 1_000_000
_VECTOR_CACHE_SIZE = 64


def lambda_segments(scores: list[ScoredCandidate]) -> list[dict]:
    """Which model wins over the whole lambda >= 0 axis for one cluster, as contiguous segments.
    Scores are linear in lambda, so the winner can only change where two score lines cross."""
    crossings = {
        (b.predicted_error - a.predicted_error) / (a.normalised_cost - b.normalised_cost)
        for a in scores for b in scores
        if a.normalised_cost != b.normalised_cost
    }
    edges = [0.0] + sorted(x for x in crossings if x > 0)

    def winner_at(lambda_: float) -> str:
        rescored = [dataclasses.replace(s, routing_score=s.predicted_error + lambda_ * s.normalised_cost) for s in scores]
        return select_model(rescored).model_id

    segments: list[dict] = []
    for i, lo in enumerate(edges):
        hi = edges[i + 1] if i + 1 < len(edges) else None
        model_id = winner_at(lo + 1.0 if hi is None else (lo + hi) / 2)
        if segments and segments[-1]["model_id"] == model_id:
            segments[-1]["to"] = hi
        else:
            segments.append({"from": lo, "to": hi, "model_id": model_id})
    return segments


def price_tier(normalised_cost: float) -> str:
    """Price band relative to the offered candidates (the same min-max scale routing uses), so the
    labels stay meaningful for any --model set: the dearest is "high", the cheapest "lowest"."""
    if normalised_cost >= 0.5:
        return "high"
    if normalised_cost >= 0.2:
        return "mid"
    return "low" if normalised_cost > 0 else "lowest"


def load_cluster_sources(path: Path) -> dict[int, dict[str, int]]:
    """cluster_id -> {source: task count} from task-cluster-map.json; empty if the file is absent
    (it's descriptive only — routing never reads it)."""
    if not path.exists():
        return {}
    counts: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    for task in json.loads(path.read_text(encoding="utf-8"))["tasks"]:
        counts[task["cluster_id"]][task["source"]] += 1
    return {cid: dict(c.most_common()) for cid, c in counts.items()}


class RouterUI:
    """The UI's state: one routing context (re-pinned per lambda) plus a small prompt -> vector
    cache. `embed` is injectable so tests never load the real encoder."""

    def __init__(
        self, ctx: RoutingContext, cluster_sizes: dict[int, int], cluster_sources: dict[int, dict[str, int]],
        embed: Callable[[str], np.ndarray] | None = None,
    ):
        self._ctx = ctx
        self._cluster_sizes = cluster_sizes
        self._cluster_sources = cluster_sources
        self._embed = embed or (lambda prompt: embed_one(prompt, ctx.embedding))
        self._vectors: collections.OrderedDict[str, tuple[np.ndarray, float]] = collections.OrderedDict()
        self._lock = threading.Lock()  # the encoder isn't safe to call from several threads at once

    def info(self) -> dict:
        candidates = self._ctx.candidates
        prices = [static_price_per_1m(m) for m in candidates]
        models = []
        for m, price, norm in zip(candidates, prices, normalise_costs(prices), strict=True):
            overall = self._ctx.profiles_by_model[m.model_id].get("global", {})
            models.append({
                "model_id": m.model_id,
                "tier": price_tier(norm),
                "cost_input_per_1m": m.cost_input * 1000,
                "cost_output_per_1m": m.cost_output * 1000,
                "static_price": price,
                "normalised_cost": norm,
                "global_passed": overall.get("number_succeeded"),
                "global_graded": overall.get("number_of_tasks"),
                "context_window": m.context_window,
                "thinking": m.thinking,
            })
        return {
            # Configured order, not sorted: the page assigns each model a fixed colour by position.
            "candidates": [m.model_id for m in candidates],
            "models": models,
            "cluster_count": int(self._ctx.cluster_map.centroids.shape[0]),
            "cluster_map_id": self._ctx.cluster_map_id,
            "profiles_id": self._ctx.profiles_id,
            "embedding_model_id": self._ctx.embedding.model_id,
            "lambda_sweep": list(LAMBDA_SWEEP),
        }

    def _vector(self, prompt: str) -> tuple[np.ndarray, float, bool]:
        with self._lock:
            if prompt in self._vectors:
                self._vectors.move_to_end(prompt)
                vector, embed_ms = self._vectors[prompt]
                return vector, embed_ms, True
            started = time.monotonic()
            vector = self._embed(prompt)
            embed_ms = (time.monotonic() - started) * 1000
            self._vectors[prompt] = (vector, embed_ms)
            if len(self._vectors) > _VECTOR_CACHE_SIZE:
                self._vectors.popitem(last=False)
            return vector, embed_ms, False

    def _context_at(self, lambda_: float) -> RoutingContext:
        # Same validation as load_routing_context; the digest pins lambda, so recompute it.
        if not np.isfinite(lambda_) or lambda_ < 0:
            raise ValueError(f"lambda must be finite and >= 0, got {lambda_}")
        ctx = self._ctx
        digest = _compute_digest(lambda_, ctx.embedding.model_id, ctx.cluster_map_id, ctx.profiles_id, ctx.candidates)
        return dataclasses.replace(ctx, lambda_=lambda_, digest=digest)

    def decide(self, prompt: str, lambda_: float) -> dict:
        if not prompt.strip():
            raise ValueError("prompt is empty")
        ctx = self._context_at(lambda_)
        vector, embed_ms, cached = self._vector(prompt)
        decision = decide_from_vector(vector, ctx, embed_ms=embed_ms, prompt_chars=len(prompt))
        return self._render(decision, cached)

    def _render(self, decision: RoutingDecision, cached: bool) -> dict:
        cid = decision.cluster.cluster_id
        rows = []
        for s in sorted(decision.scores, key=lambda s: (s.routing_score, s.model_id)):
            cluster_stats = self._ctx.profiles_by_model[s.model_id].get("clusters", {}).get(str(cid), {})
            rows.append({
                **dataclasses.asdict(s),
                "cluster_passed": cluster_stats.get("number_succeeded"),
                "cluster_graded": cluster_stats.get("number_of_tasks"),
            })
        return {
            "selected": decision.selected.model_id,
            "lambda": decision.lambda_,
            "cluster": {
                **dataclasses.asdict(decision.cluster),
                "size": self._cluster_sizes.get(cid),
                "sources": self._cluster_sources.get(cid, {}),
            },
            "scores": rows,
            "segments": lambda_segments(decision.scores),
            "excluded": decision.excluded,
            "embed_ms": decision.embed_ms,
            "embedding_cached": cached,
            "score_ms": decision.score_ms,
            "digest": decision.digest,
        }


def _make_handler(ui: RouterUI) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: HTTPStatus, payload: dict) -> None:
            self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

        def do_GET(self) -> None:
            path = urlsplit(self.path).path  # the page's ?prompt=/?example= params are read client-side
            if path in ("/", "/index.html"):
                # Read per request (it is tiny) so edits to ui.html show up on refresh.
                self._send(HTTPStatus.OK, _PAGE_PATH.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/info":
                self._send_json(HTTPStatus.OK, ui.info())
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/api/decide":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length > _MAX_BODY_BYTES:
                self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "prompt too large"})
                return
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                result = ui.decide(str(body.get("prompt", "")), float(body.get("lambda", 0.0)))
            except (ValueError, TypeError) as e:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(e)})
                return
            self._send_json(HTTPStatus.OK, result)

        def log_message(self, format: str, *args) -> None:
            logger.debug("%s - %s", self.address_string(), format % args)

    return Handler


def make_server(ui: RouterUI, host: str, port: int) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), _make_handler(ui))
