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
    assert pricing.clip_usd("MiniMax-H3", 6) == 0.48
    assert pricing.clip_usd("FastH3-8-Step-V2", 10) == 0.40
    assert pricing.clip_usd("MiniMax-H3", 6, "2K") == 0.78
    # 7 reference images: 2 past the free 5. A 4 s reference video in a 6 s
    # clip bills its 4 s.
    assert pricing.clip_usd("MiniMax-H3-Ref2VA-SMD-4step", 6, images=7,
                            reference_video_seconds=4) == pytest.approx(0.88)
    # At 2K the reference seconds bill at the 2K rate too.
    assert pricing.clip_usd("MiniMax-H3-Ref2VA-SMD-4step", 6, "2K",
                            reference_video_seconds=4) == pytest.approx(1.30)
    # The reference seconds are capped at 15.
    assert pricing.clip_usd("MiniMax-H3-Ref2VA-SMD-4step", 15,
                            reference_video_seconds=40) == pytest.approx(2.40)
    assert pricing.upscale_usd(6) == 0.30


def test_label():
    assert pricing.label(0.48) == "$0.48"
    assert pricing.label(1.3) == "$1.30"


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
                pricing.UPSCALE_USD_PER_SECOND
            assert float(card["video_input_image_usd"]) == pricing.IMAGE_USD
    assert float(cards["seedvr2-3b"]["video_output_usd_per_second"]) == \
        pricing.UPSCALE_USD_PER_SECOND
