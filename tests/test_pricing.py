"""The prices shown on the nodes: worked examples, and the rates pinned to
the rate cards in config/serving-profiles.yaml while the pack lives in the
anex.sh repository."""
import os
import sys

import pytest

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
