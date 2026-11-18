"""The prices shown on the nodes: worked examples, and the built-in rates
pinned to the rate cards in config/serving-profiles.yaml while the pack
lives in the anex.sh repository."""
import os
import sys
from decimal import Decimal

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from anex_comfy import catalog, pricing  # noqa: E402

PROFILES = os.path.join(os.path.dirname(__file__), "..", "..", "config",
                        "serving-profiles.yaml")


@pytest.fixture(autouse=True)
def _builtin_models(monkeypatch):
    monkeypatch.setattr(catalog, "_loaded", list(catalog.BUILTIN))
    monkeypatch.setattr(catalog, "_upscale_usd", None)


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
    assert pricing.label(None) == "price not listed"
    assert pricing.rate_label(0.0425) == "$0.0425"
    assert pricing.rate_label(0.068) == "$0.068"
    assert pricing.rate_label(0.05) == "$0.05"


def test_builtin_rates_match_the_rate_cards():
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

    def usd(card, key):
        return None if card.get(key) is None else Decimal(str(card[key]))
    for model in catalog.BUILTIN:
        # The first id is the model's own; the others are older names.
        card = cards[catalog.BUILTIN_IDS[model.name][0]]
        assert usd(card, "video_output_usd_per_second") == model.output_usd, \
            model.name
        assert usd(card, "video_upscale_usd_per_second") == model.upscale_usd, \
            model.name
        assert usd(card, "video_input_image_usd") == model.image_usd, model.name
        assert (card.get("video_free_images") or 0) == model.free_images, \
            model.name
        assert ("2K" in model.resolutions) == (model.upscale_usd is not None)
    assert float(cards[catalog.UPSCALER_ID]["video_output_usd_per_second"]) \
        == pricing.UPSCALE_USD_PER_SECOND


def _catalog():
    """A catalog with prices unlike the built-in ones, two models sharing
    the name MiniMax-H3 at different prices."""
    def video(name, kinds, seed=False):
        return {"name": name, "names": [name], "kinds": kinds,
                "resolutions": ["768P", "2K"], "supports_seed": seed}
    return {"models": [
        {"id": "minimax-h3", "video": video("MiniMax-H3", ["text", "frames"]),
         "media_fees": {"video_output_usd_per_second": "0.05",
                        "video_upscale_usd_per_second": "0.05",
                        "video_input_image_usd": "0.03",
                        "video_free_images": 4}},
        {"id": "minimax-h3-ref2va",
         "video": video("MiniMax-H3", ["reference"], seed=True),
         "media_fees": {"video_output_usd_per_second": "0.07",
                        "video_upscale_usd_per_second": "0.04",
                        "video_input_image_usd": "0.03",
                        "video_free_images": 4}},
        {"id": "minimax-h3-ref2va-smd-4step",
         "video": video("MiniMax-H3-Ref2VA-SMD-4step", ["reference"]),
         "media_fees": {"video_output_usd_per_second": "0.04",
                        "video_upscale_usd_per_second": "0.06",
                        "video_input_image_usd": "0.03",
                        "video_free_images": 4}},
        {"id": "seedvr2-3b",
         "media_fees": {"video_output_usd_per_second": "0.03"}},
    ]}


def test_catalog_prices_replace_the_builtin_ones(tmp_path):
    catalog.load(fetch=_catalog, cache_path=str(tmp_path / "c.json"))
    assert pricing.clip_usd("MiniMax-H3", 10) == pytest.approx(0.50)
    assert pricing.clip_usd("MiniMax-H3", 10, "2K") == pytest.approx(1.00)
    # 6 images, 4 free: 2 billed at $0.03.
    assert pricing.clip_usd("MiniMax-H3", 10, kind="frames",
                            images=6) == pytest.approx(0.56)
    # The same name on a reference request is the other model.
    assert pricing.clip_usd("MiniMax-H3", 10,
                            kind="reference") == pytest.approx(0.70)
    assert pricing.clip_usd("MiniMax-H3-Ref2VA-SMD-4step", 5,
                            reference_video_seconds=5) == pytest.approx(0.40)
    assert pricing.upscale_usd(10) == pytest.approx(0.30)
    assert pricing.upscale_rate() == 0.03
    assert pricing.two_k_models() == ("MiniMax-H3",
                                      "MiniMax-H3-Ref2VA-SMD-4step")


def test_a_model_the_catalog_lacks(tmp_path):
    catalog.load(fetch=_catalog, cache_path=str(tmp_path / "c.json"))
    # A built-in model keeps its built-in price; an unknown one has none.
    assert pricing.clip_usd("FastH3-8-Step-V2", 10) == pytest.approx(0.17)
    assert pricing.clip_usd("Not-A-Model", 10, kind="text") is None
