import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from router.runtime import ui as ui_mod
from router.runtime.decide import decide_from_vector

V0 = np.array([1.0, 0.0, 0.0, 0.0])  # nearest centroid: cluster 0


@pytest.fixture
def router_ui(load_context):
    calls = []

    def fake_embed(prompt):
        calls.append(prompt)
        return V0

    ui = ui_mod.RouterUI(load_context(), {0: 100, 1: 50, 2: 25}, {0: {"swe-smith": 70, "ds1000": 30}}, embed=fake_embed)
    ui.embed_calls = calls
    return ui


def test_segments_cover_every_lambda_and_switch_at_the_score_crossings(load_context):
    # Cluster 0: strong (0.15, cost 1.0) -> mid (0.45, cost ~0.135) -> cheap (0.70, cost 0).
    scores = decide_from_vector(V0, load_context()).scores
    segments = ui_mod.lambda_segments(scores)

    assert [s["model_id"] for s in segments] == ["strong-model", "mid-model", "cheap-model"]
    assert segments[0]["from"] == 0.0 and segments[-1]["to"] is None
    assert segments[0]["to"] == pytest.approx(0.30 / (1 - 4.7 / 34.7))
    assert segments[1]["to"] == pytest.approx(0.25 / (4.7 / 34.7))
    for seg in segments:  # every segment's winner agrees with a real decision inside it
        inside = seg["from"] + 1.0 if seg["to"] is None else (seg["from"] + seg["to"]) / 2
        assert decide_from_vector(V0, load_context(lambda_=inside)).selected.model_id == seg["model_id"]


def test_decide_reports_cluster_and_per_cluster_stats(router_ui):
    result = router_ui.decide("fix the bug", 0.0)

    assert result["selected"] == "strong-model"
    assert result["cluster"]["cluster_id"] == 0
    assert result["cluster"]["size"] == 100
    assert result["cluster"]["sources"] == {"swe-smith": 70, "ds1000": 30}
    assert result["scores"][0]["model_id"] == "strong-model"
    assert all(r["model_id"] != "reference-oracle" for r in result["scores"])


def test_changing_lambda_reuses_the_cached_embedding(router_ui):
    first = router_ui.decide("fix the bug", 0.0)
    second = router_ui.decide("fix the bug", 0.5)

    assert router_ui.embed_calls == ["fix the bug"]
    assert not first["embedding_cached"] and second["embedding_cached"]
    assert second["selected"] == "mid-model"
    assert first["digest"] != second["digest"]  # the digest pins lambda


@pytest.mark.parametrize("norm, tier", [(1.0, "high"), (0.5, "high"), (0.415, "mid"), (0.2, "mid"), (0.08, "low"), (0.0, "lowest")])
def test_price_tier_bands_the_normalised_cost(norm, tier):
    assert ui_mod.price_tier(norm) == tier


def test_info_lists_each_candidate_with_its_tier_and_price(router_ui):
    models = {m["model_id"]: m for m in router_ui.info()["models"]}

    assert {k: m["tier"] for k, m in models.items()} == {"cheap-model": "lowest", "mid-model": "low", "strong-model": "high"}
    assert models["strong-model"]["static_price"] == pytest.approx(35.0)
    assert models["mid-model"]["cost_input_per_1m"] == pytest.approx(2.5)


@pytest.mark.parametrize("prompt, lambda_", [("   ", 0.0), ("fix", -1.0), ("fix", float("nan"))])
def test_decide_rejects_bad_input(router_ui, prompt, lambda_):
    with pytest.raises(ValueError):
        router_ui.decide(prompt, lambda_)


def test_http_api_serves_page_info_and_decisions(router_ui):
    server = ui_mod.make_server(router_ui, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        for page in ("/", "/?example=2&lambda=0.1"):
            with urllib.request.urlopen(f"{base}{page}") as res:
                assert b"<title>" in res.read()
        with urllib.request.urlopen(f"{base}/api/info") as res:
            assert json.load(res)["candidates"] == ["cheap-model", "mid-model", "strong-model"]

        request = urllib.request.Request(
            f"{base}/api/decide", data=json.dumps({"prompt": "fix", "lambda": 100}).encode(), method="POST",
        )
        with urllib.request.urlopen(request) as res:
            assert json.load(res)["selected"] == "cheap-model"

        bad = urllib.request.Request(f"{base}/api/decide", data=b'{"prompt": ""}', method="POST")
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(bad)
        assert err.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
