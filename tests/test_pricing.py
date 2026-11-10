"""The prices shown on the nodes: worked examples, and the rates pinned to
the rate cards in config/serving-profiles.yaml while the pack lives in the
anex.sh repository."""
import asyncio
import os
import sys

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from anex_comfy import pricing  # noqa: E402

PROFILES = os.path.join(os.path.dirname(__file__), "..", "..", "config",
                        "serving-profiles.yaml")
# MiniMax-compatible name -> catalog model id.
CATALOG_IDS = {"MiniMax-H3": "minimax-h3",
               "MiniMax-H3-Ref2VA-SMD-4step": "minimax-h3-ref2va",
               "FastH3-8-Step-V2": "fasth3-8-step-v2"}


def test_worked_examples():
    assert pricing.clip_usd("MiniMax-H3", 6) == pytest.approx(0.408)
    assert pricing.clip_usd("FastH3-8-Step-V2", 10) == pytest.approx(0.17)
    # 2K is $0.11 a second on both models.
    assert pricing.clip_usd("MiniMax-H3", 6, "2K") == pytest.approx(0.66)
    # 7 reference images: 2 past the free 5. A 4 s reference video in a 6 s
    # clip bills its 4 s.
    assert pricing.clip_usd("MiniMax-H3-Ref2VA-SMD-4step", 6, images=7,
                            reference_video_seconds=4) == pytest.approx(0.578)
    # At 2K the reference seconds bill at the 2K rate too.
    assert pricing.clip_usd("MiniMax-H3-Ref2VA-SMD-4step", 6, "2K",
                            reference_video_seconds=4) == pytest.approx(1.10)
    # The reference seconds are capped at 15.
    assert pricing.clip_usd("MiniMax-H3-Ref2VA-SMD-4step", 15,
                            reference_video_seconds=40) == pytest.approx(1.53)
    assert pricing.upscale_usd(6) == pytest.approx(0.255)


def test_label():
    assert pricing.label(0.48) == "$0.48"
    assert pricing.label(1.3) == "$1.30"
    assert pricing.rate_label(0.0425) == "$0.0425"
    assert pricing.rate_label(0.068) == "$0.068"
    assert pricing.rate_label(0.05) == "$0.05"


def test_rates_match_the_rate_cards():
    yaml = pytest.importorskip("yaml")
    if not os.path.exists(PROFILES):
        pytest.skip("not inside the anex.sh repository")
    with open(PROFILES) as f:
        doc = yaml.safe_load(f)
    cards = {}
    for value in doc.values():
        if isinstance(value, list):
            for row in value:
                if isinstance(row, dict) and "video_output_usd_per_second" in row:
                    cards[row["model"]] = row
    for name, model_id in CATALOG_IDS.items():
        card = cards[model_id]
        assert float(card["video_output_usd_per_second"]) == \
            pricing.OUTPUT_USD_PER_SECOND[name], name
        if name in pricing.TWO_K_MODELS:
            assert float(card["video_upscale_usd_per_second"]) == \
                pricing.TWO_K_ADD_USD_PER_SECOND[name], name
            assert float(card["video_input_image_usd"]) == pricing.IMAGE_USD
    assert float(cards["seedvr2-3b"]["video_output_usd_per_second"]) == \
        pricing.UPSCALE_USD_PER_SECOND


# The prices read from the catalog: a node shows what anex.sh charges now,
# not what this version of the pack shipped with.
def _catalog(**overrides):
    def model(model_id, **fees):
        return {"id": model_id, "media_fees": fees}
    rows = {
        "minimax-h3": model("minimax-h3", video_output_usd_per_second="0.05",
                            video_upscale_usd_per_second="0.05",
                            video_input_image_usd="0.03", video_free_images=4),
        "minimax-h3-ref2va": model(
            "minimax-h3-ref2va", video_output_usd_per_second="0.04",
            video_upscale_usd_per_second="0.06", video_input_image_usd="0.03",
            video_free_images=4),
        "fasth3-8-step-v2": model("fasth3-8-step-v2",
                                  video_output_usd_per_second="0.01"),
        "seedvr2-3b": model("seedvr2-3b", video_output_usd_per_second="0.03"),
        "gemma4-31b": model("gemma4-31b", image_usd="0.0001"),
    }
    rows.update(overrides)
    return {"models": [r for r in rows.values() if r is not None]}


@pytest.fixture(autouse=True)
def _builtin_prices():
    pricing.reset()
    yield
    pricing.reset()


def test_catalog_prices_replace_the_builtin_ones():
    assert pricing.apply_catalog(_catalog())
    assert pricing.clip_usd("MiniMax-H3", 10) == pytest.approx(0.50)
    assert pricing.clip_usd("MiniMax-H3", 10, "2K") == pytest.approx(1.00)
    # 6 images, 4 free: 2 billed at $0.03.
    assert pricing.clip_usd("MiniMax-H3", 10, images=6) == pytest.approx(0.56)
    assert pricing.clip_usd("MiniMax-H3-Ref2VA-SMD-4step", 5,
                            reference_video_seconds=5) == pytest.approx(0.40)
    assert pricing.clip_usd("FastH3-8-Step-V2", 10) == pytest.approx(0.10)
    assert pricing.upscale_usd(10) == pytest.approx(0.30)
    assert pricing.upscale_rate() == 0.03


def test_a_model_missing_from_the_catalog_keeps_its_price():
    assert pricing.apply_catalog(_catalog(**{"fasth3-8-step-v2": None,
                                             "seedvr2-3b": None}))
    assert pricing.clip_usd("FastH3-8-Step-V2", 10) == pytest.approx(0.17)
    assert pricing.upscale_rate() == pricing.UPSCALE_USD_PER_SECOND
    assert pricing.clip_usd("MiniMax-H3", 10) == pytest.approx(0.50)


@pytest.mark.parametrize("doc", [
    {},
    {"models": "nope"},
    {"models": [{"id": "gemma4-31b", "media_fees": {}}]},
    _catalog(**{"minimax-h3": {"id": "minimax-h3", "media_fees": {
        "video_output_usd_per_second": "-1"}}}),
    _catalog(**{"minimax-h3": {"id": "minimax-h3", "media_fees": {
        "video_output_usd_per_second": "free"}}}),
])
def test_a_bad_catalog_changes_nothing(doc):
    assert not pricing.apply_catalog(doc)
    assert pricing.clip_usd("MiniMax-H3", 10) == pytest.approx(0.68)
    assert pricing.upscale_rate() == pricing.UPSCALE_USD_PER_SECOND


class _CatalogServer:
    def __init__(self, status=200, body=None):
        self.status, self.body, self.hits = status, body, 0
        self.app = web.Application()
        self.app.router.add_get("/api/estimate/catalog", self.catalog)

    async def catalog(self, request):
        self.hits += 1
        return web.json_response(self.body, status=self.status)


def _refresh(server, monkeypatch, *steps):
    """Run the steps against the fake catalog server, in one event loop: a
    number is a pricing.refresh at that monotonic time, a callable is run
    as it is (to check the prices, or change what the server answers)."""
    async def go():
        async with TestServer(server.app) as ts:
            monkeypatch.setattr(pricing, "CATALOG_URL",
                                str(ts.make_url("/api/estimate/catalog")))
            for step in steps:
                if callable(step):
                    step()
                else:
                    await pricing.refresh(now=step)
    asyncio.run(go())


def test_refresh_reads_the_catalog_and_keeps_it_for_a_while(monkeypatch):
    server = _CatalogServer(body=_catalog())

    def read_once():
        assert server.hits == 1
        assert pricing.clip_usd("MiniMax-H3", 10) == pytest.approx(0.50)
        assert pricing.source.endswith("/api/estimate/catalog")
    _refresh(server, monkeypatch, 1000.0,
             1000.0 + pricing.REFRESH_SECONDS - 1, read_once,
             1000.0 + pricing.REFRESH_SECONDS)
    assert server.hits == 2


def test_refresh_failure_keeps_the_prices_and_retries_later(monkeypatch):
    server = _CatalogServer(status=503, body={})

    def failed_once():
        assert server.hits == 1
        assert pricing.clip_usd("MiniMax-H3", 10) == pytest.approx(0.68)
        assert pricing.source == "built-in"
        server.status, server.body = 200, _catalog()
    _refresh(server, monkeypatch, 1000.0,
             1000.0 + pricing.RETRY_SECONDS - 1, failed_once,
             1000.0 + pricing.RETRY_SECONDS)
    assert server.hits == 2
    assert pricing.clip_usd("MiniMax-H3", 10) == pytest.approx(0.50)


def test_refresh_never_raises_when_the_host_is_down(monkeypatch):
    monkeypatch.setattr(pricing, "CATALOG_URL", "http://127.0.0.1:9/catalog")
    asyncio.run(pricing.refresh(now=1.0))
    assert pricing.clip_usd("MiniMax-H3", 10) == pytest.approx(0.68)
