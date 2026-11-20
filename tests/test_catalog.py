"""The model list the nodes' menus and prices are built from: read from the
anex.sh catalog, else the last copy kept on disk, else the built-in list."""
import asyncio
import json
import logging
import os
import sys
import time
from decimal import Decimal

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from anex_comfy import catalog  # noqa: E402


def entry(model_id, video=None, **fees):
    out = {"id": model_id, "media_fees": fees}
    if video is not None:
        out["video"] = video
    return out


def block(name, kinds, resolutions=("768P", "2K"), seed=False):
    return {"name": name, "names": [name], "kinds": list(kinds),
            "resolutions": list(resolutions), "supports_seed": seed}


def this_release(text_h3="0.068", reference_h3="0.068"):
    """The catalog of this release: four video models with a `video` block,
    two of them under the name MiniMax-H3 (told apart by kind)."""
    return {"models": [
        entry("gemma4-31b", image_usd="0.0001"),
        entry("minimax-h3", block("MiniMax-H3", ["text", "frames"]),
              video_output_usd_per_second=text_h3,
              video_upscale_usd_per_second="0.042",
              video_input_image_usd="0.034", video_free_images=5),
        entry("fasth3-8-step-v2",
              block("FastH3-8-Step-V2", ["text"], ["768P"], seed=True),
              video_output_usd_per_second="0.017"),
        entry("minimax-h3-ref2va",
              block("MiniMax-H3", ["reference"], seed=True),
              video_output_usd_per_second=reference_h3,
              video_upscale_usd_per_second="0.042",
              video_input_image_usd="0.034", video_free_images=5),
        entry("minimax-h3-ref2va-smd-4step",
              block("MiniMax-H3-Ref2VA-SMD-4step", ["reference"]),
              video_output_usd_per_second="0.051",
              video_upscale_usd_per_second="0.059",
              video_input_image_usd="0.034", video_free_images=5),
        entry("seedvr2-3b", video_output_usd_per_second="0.0425"),
    ]}


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(catalog, "_loaded", None)
    monkeypatch.setattr(catalog, "_live", None)
    monkeypatch.setattr(catalog, "_upscale_usd", None)
    monkeypatch.setattr(catalog, "_upscale_4k_usd", None)
    monkeypatch.setattr(catalog, "_next_refresh", 0.0)


def use(payload, tmp_path):
    return catalog.load(fetch=lambda: payload,
                        cache_path=str(tmp_path / "catalog.json"))


def test_this_releases_catalog(tmp_path):
    use(this_release(text_h3="0.07"), tmp_path)
    assert [m.name for m in catalog.models("reference")] == [
        "MiniMax-H3", "MiniMax-H3-Ref2VA-SMD-4step"]
    assert [m.name for m in catalog.models("frames")] == ["MiniMax-H3"]
    assert [m.name for m in catalog.models("text")] == [
        "MiniMax-H3", "FastH3-8-Step-V2"]
    # Two entries share the name; the kind picks the one a request reaches.
    reference = catalog.find("MiniMax-H3", "reference")
    assert reference.output_usd == Decimal("0.068")
    assert reference.supports_seed is True
    text = catalog.find("MiniMax-H3", "text")
    assert text.output_usd == Decimal("0.07")
    assert text.supports_seed is False
    assert catalog.find("FastH3-8-Step-V2", "text").resolutions == ("768P",)
    assert catalog.find("FastH3-8-Step-V2", "reference") is None
    assert catalog.default("reference") == "MiniMax-H3-Ref2VA-SMD-4step"
    assert catalog.default("text") == catalog.default("frames") == "MiniMax-H3"
    assert catalog.upscale_usd() == Decimal("0.0425")


def test_a_catalog_without_video_blocks_prices_the_builtin_models():
    # anex.sh before this release: the 4-step model under the old id.
    old = {"models": [
        entry("minimax-h3", video_output_usd_per_second="0.068",
              video_upscale_usd_per_second="0.042",
              video_input_image_usd="0.034", video_free_images=5),
        entry("minimax-h3-ref2va", video_output_usd_per_second="0.051",
              video_upscale_usd_per_second="0.059",
              video_input_image_usd="0.034", video_free_images=5),
    ]}
    models = catalog.parse(old)
    assert [m.name for m in models] == [m.name for m in catalog.BUILTIN]
    four_step = next(m for m in models
                     if m.name == "MiniMax-H3-Ref2VA-SMD-4step")
    assert four_step.output_usd == Decimal("0.051")
    assert four_step.kinds == ("reference",)
    # Listed in neither form: the built-in price stays.
    fast = next(m for m in models if m.name == "FastH3-8-Step-V2")
    assert fast.output_usd == Decimal("0.017")


def test_without_blocks_the_4_step_id_wins_over_the_old_one():
    both = {"models": [
        entry("minimax-h3-ref2va", video_output_usd_per_second="0.068"),
        entry("minimax-h3-ref2va-smd-4step",
              video_output_usd_per_second="0.051",
              video_upscale_usd_per_second="0.059"),
    ]}
    four_step = next(m for m in catalog.parse(both)
                     if m.name == "MiniMax-H3-Ref2VA-SMD-4step")
    assert four_step.output_usd == Decimal("0.051")
    assert four_step.upscale_usd == Decimal("0.059")


def test_a_new_model_appears_with_no_change_to_the_pack(tmp_path):
    doc = this_release()
    doc["models"].append(entry(
        "some-new-model", block("Some-New-Model", ["text"], ["768P"]),
        video_output_usd_per_second="0.02"))
    use(doc, tmp_path)
    assert "Some-New-Model" in [m.name for m in catalog.models("text")]
    assert catalog.find("Some-New-Model", "text").output_usd == Decimal("0.02")


@pytest.mark.parametrize("bad", [
    {"name": "Bad", "names": [], "kinds": ["text", "sing"],
     "resolutions": ["768P"], "supports_seed": False},
    {"name": "Bad", "names": [], "kinds": ["text"], "supports_seed": False},
    {"name": "Bad", "names": [], "kinds": ["text"], "resolutions": ["768P"]},
])
def test_a_bad_entry_is_skipped_and_the_rest_load(bad):
    doc = this_release()
    doc["models"].append(entry("bad", bad, video_output_usd_per_second="0.02"))
    doc["models"].append(entry("unpriced", block("Unpriced", ["text"])))
    names = [m.name for m in catalog.parse(doc)]
    assert "Bad" not in names and "Unpriced" not in names
    assert len(names) == 4


@pytest.mark.parametrize("doc", [{}, {"models": "nope"}, {"models": [
    entry("x", block("X", ["sing"]), video_output_usd_per_second="0.1")]}])
def test_an_answer_with_no_usable_model_is_refused(doc):
    with pytest.raises(ValueError):
        catalog.parse(doc)


def down():
    raise OSError("anex.sh unreachable")


def test_load_falls_back_to_the_last_copy_then_the_builtin_list(tmp_path):
    cache = str(tmp_path / "sub" / "catalog.json")
    assert catalog.load(fetch=down, cache_path=cache) == catalog.BUILTIN
    # A good read is kept on disk...
    catalog.load(fetch=this_release, cache_path=cache)
    with open(cache) as f:
        assert json.load(f) == this_release()
    # ...and used when anex.sh cannot be reached.
    models = catalog.load(fetch=down, cache_path=cache)
    assert [m.name for m in models] == [
        "MiniMax-H3", "FastH3-8-Step-V2", "MiniMax-H3",
        "MiniMax-H3-Ref2VA-SMD-4step"]
    assert catalog.upscale_usd() == Decimal("0.0425")


def test_a_corrupt_copy_falls_through_to_the_builtin_list(tmp_path, caplog):
    cache = tmp_path / "catalog.json"
    cache.write_text("{not json")
    with caplog.at_level(logging.WARNING, logger="comfyui-anex"):
        assert catalog.load(fetch=down, cache_path=str(cache)) \
            == catalog.BUILTIN
    assert catalog.upscale_usd() is None
    assert "using the built-in list" in caplog.text
    caplog.clear()
    catalog.load(fetch=this_release, cache_path=str(cache))
    with caplog.at_level(logging.WARNING, logger="comfyui-anex"):
        catalog.load(fetch=down, cache_path=str(cache))
    assert f"using the copy kept at {cache}" in caplog.text


@pytest.mark.parametrize("price", ["1e999999999", "100.01", "-0.01", "NaN"])
def test_an_absurd_price_skips_the_entry(price):
    doc = this_release()
    doc["models"].append(entry("odd", block("Odd", ["text"]),
                               video_output_usd_per_second="0.02",
                               video_input_image_usd=price))
    doc["models"].append(entry("odder", block("Odder", ["text"]),
                               video_output_usd_per_second=price))
    names = [m.name for m in catalog.parse(doc)]
    assert "Odd" not in names and "Odder" not in names
    assert len(names) == 4


def test_the_cache_is_written_whole_and_leaves_no_temporary_file(tmp_path):
    cache = tmp_path / "catalog.json"
    catalog.load(fetch=this_release, cache_path=str(cache))
    assert os.listdir(tmp_path) == ["catalog.json"]
    # A copy that cannot be moved into place (a directory is in the way)
    # leaves nothing behind.
    blocked = tmp_path / "blocked"
    (blocked / "catalog.json").mkdir(parents=True)
    (blocked / "catalog.json" / "x").write_text("")
    catalog.load(fetch=this_release,
                 cache_path=str(blocked / "catalog.json"))
    assert os.listdir(blocked) == ["catalog.json"]


def test_a_bad_answer_is_not_kept(tmp_path):
    cache = tmp_path / "catalog.json"
    catalog.load(fetch=this_release, cache_path=str(cache))
    catalog.load(fetch=lambda: {"models": "nope"}, cache_path=str(cache))
    assert json.loads(cache.read_text()) == this_release()
    assert len(catalog.models("reference")) == 2


def test_a_slow_answer_is_given_up_on(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "FETCH_TIMEOUT_SECONDS", 0.2)

    def slow():
        time.sleep(2)
        return this_release()
    started = time.monotonic()
    assert catalog.load(fetch=slow, cache_path=str(tmp_path / "c.json")) \
        == catalog.BUILTIN
    assert time.monotonic() - started < 1


def test_an_unwritable_cache_never_raises(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("")
    models = catalog.load(fetch=this_release,
                          cache_path=str(blocker / "catalog.json"))
    assert len(models) == 4


def test_the_cache_lives_outside_the_pack():
    pack = os.path.dirname(os.path.dirname(os.path.abspath(catalog.__file__)))
    path = os.path.abspath(catalog.default_cache_path())
    assert not path.startswith(pack + os.sep)
    assert path.endswith(os.path.join("comfyui-anex", "catalog.json"))


def test_the_models_are_read_once(monkeypatch, tmp_path):
    reads = []

    def fetch():
        reads.append(1)
        return this_release()
    monkeypatch.setattr(catalog, "_fetch", fetch)
    monkeypatch.setattr(catalog, "default_cache_path",
                        lambda: str(tmp_path / "catalog.json"))
    catalog.models("text")
    catalog.models("reference")
    catalog.default("frames")
    assert reads == [1]


def test_a_kind_with_no_model_still_has_a_menu(monkeypatch):
    monkeypatch.setattr(catalog, "_loaded", [catalog.BUILTIN[1]])
    assert catalog.names("reference") == ["MiniMax-H3-Ref2VA-SMD-4step"]
    assert catalog.default("reference") == "MiniMax-H3-Ref2VA-SMD-4step"
    assert catalog.find("MiniMax-H3-Ref2VA-SMD-4step", "reference") is None
    assert catalog.default("text") == "FastH3-8-Step-V2"


# --------------------------------------------------------------------------- #
# Refresh while ComfyUI runs: prices follow anex.sh, menus stay as built
# --------------------------------------------------------------------------- #
class Fetcher:
    def __init__(self, payload):
        self.payload, self.calls = payload, 0

    def __call__(self):
        self.calls += 1
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


def refresh(fetch, tmp_path, now):
    asyncio.run(catalog.refresh(fetch, str(tmp_path / "catalog.json"),
                                now=now))


def test_a_price_change_reaches_the_next_run_after_the_interval(tmp_path):
    use(this_release(), tmp_path)
    menus = {k: catalog.names(k) for k in catalog.KINDS}
    fetch = Fetcher(this_release(text_h3="0.09"))
    catalog._next_refresh = 1000.0
    refresh(fetch, tmp_path, 999.0)  # inside the interval: no read
    assert fetch.calls == 0
    assert catalog.find("MiniMax-H3", "text").output_usd == Decimal("0.068")
    refresh(fetch, tmp_path, 1000.0)
    assert fetch.calls == 1
    assert catalog.find("MiniMax-H3", "text").output_usd == Decimal("0.09")
    refresh(fetch, tmp_path, 1000.0 + catalog.REFRESH_SECONDS - 1)
    assert fetch.calls == 1
    refresh(fetch, tmp_path, 1000.0 + catalog.REFRESH_SECONDS)
    assert fetch.calls == 2
    assert {k: catalog.names(k) for k in catalog.KINDS} == menus


def test_a_failed_refresh_keeps_the_prices_and_retries_after_a_minute(tmp_path):
    use(this_release(text_h3="0.07"), tmp_path)
    catalog._next_refresh = 1000.0
    fetch = Fetcher(OSError("down"))
    refresh(fetch, tmp_path, 1000.0)
    assert fetch.calls == 1
    assert catalog.find("MiniMax-H3", "text").output_usd == Decimal("0.07")
    assert catalog.upscale_usd() == Decimal("0.0425")
    refresh(fetch, tmp_path, 1000.0 + catalog.RETRY_SECONDS - 1)
    assert fetch.calls == 1
    fetch.payload = this_release(text_h3="0.09")
    refresh(fetch, tmp_path, 1000.0 + catalog.RETRY_SECONDS)
    assert fetch.calls == 2
    assert catalog.find("MiniMax-H3", "text").output_usd == Decimal("0.09")


def test_a_failed_read_at_start_is_retried_by_a_run(tmp_path):
    catalog.load(fetch=down, cache_path=str(tmp_path / "catalog.json"))
    assert catalog.names("reference") == ["MiniMax-H3-Ref2VA-SMD-4step"]
    started = catalog._next_refresh
    refresh(Fetcher(this_release()), tmp_path, started)
    # Runs now use the catalog; the menus stay as built.
    assert catalog.find("MiniMax-H3", "reference") is not None
    assert catalog.names("reference") == ["MiniMax-H3-Ref2VA-SMD-4step"]


def test_a_model_the_refresh_no_longer_lists(tmp_path):
    use(this_release(), tmp_path)
    doc = this_release()
    doc["models"] = [m for m in doc["models"]
                     if m["id"] != "fasth3-8-step-v2"]
    refresh(Fetcher(doc), tmp_path, catalog._next_refresh)
    assert "FastH3-8-Step-V2" in catalog.names("text")
    assert catalog.find("FastH3-8-Step-V2", "text") is None


def with_4k(payload):
    """This release's catalog, with the 4K rates and 4K listed on the three
    H3 models that render it."""
    for e in payload["models"]:
        fees = e["media_fees"]
        if e["id"] in ("minimax-h3", "minimax-h3-ref2va",
                       "minimax-h3-ref2va-smd-4step"):
            e["video"]["resolutions"] = ["768P", "2K", "4K"]
            fees["video_upscale_4k_usd_per_second"] = "0.068"
        if e["id"] == "seedvr2-3b":
            fees["video_output_4k_usd_per_second"] = "0.068"
    return payload


def test_catalog_with_4k(tmp_path):
    from anex_comfy import pricing
    use(with_4k(this_release()), tmp_path)
    assert catalog.resolutions() == ("768P", "2K", "4K")
    assert catalog.find("MiniMax-H3", "text").resolutions == (
        "768P", "2K", "4K")
    assert catalog.find("MiniMax-H3", "text").upscale_4k_usd == Decimal("0.068")
    assert catalog.find("FastH3-8-Step-V2", "text").resolutions == ("768P",)
    assert catalog.upscale_usd("4K") == Decimal("0.068")
    assert catalog.upscale_usd("2K") == Decimal("0.0425")
    assert catalog.upscale_resolutions() == ("2K", "4K")
    assert pricing.clip_usd("MiniMax-H3", 10, "4K") == 1.36
    assert pricing.clip_usd("MiniMax-H3-Ref2VA-SMD-4step", 10, "4K") == 1.19
    assert pricing.upscale_usd(10, "4K") == 0.68


def test_catalog_without_4k_fields_offers_no_4k(tmp_path):
    from anex_comfy import pricing
    use(this_release(), tmp_path)
    assert catalog.resolutions() == ("768P", "2K")
    assert catalog.upscale_usd("4K") is None
    assert catalog.upscale_resolutions() == ("2K",)
    assert pricing.clip_usd("MiniMax-H3", 10, "2K") == 1.1
    assert pricing.upscale_usd(10) == 0.425
    with pytest.raises(ValueError):
        pricing.upscale_rate("4K")


def test_a_model_listing_4k_without_its_rate_does_not_offer_it(tmp_path):
    payload = with_4k(this_release())
    for e in payload["models"]:
        if e["id"] == "minimax-h3":
            del e["media_fees"]["video_upscale_4k_usd_per_second"]
    use(payload, tmp_path)
    assert catalog.find("MiniMax-H3", "text").resolutions == ("768P", "2K")
    assert catalog.find("MiniMax-H3", "reference").resolutions == (
        "768P", "2K", "4K")


def test_builtin_models_offer_no_4k(tmp_path):
    def offline():
        raise OSError("no network")
    catalog.load(fetch=offline, cache_path=str(tmp_path / "none.json"))
    assert catalog.resolutions() == ("768P", "2K")
    assert catalog.upscale_resolutions() == ("2K",)
    assert all(m.upscale_4k_usd is None for m in catalog.BUILTIN)


def test_a_refresh_without_4k_withdraws_the_4k_upscale(tmp_path):
    use(with_4k(this_release()), tmp_path)
    assert catalog.upscale_resolutions() == ("2K", "4K")
    refresh(Fetcher(this_release()), tmp_path, catalog._next_refresh)
    assert catalog.upscale_resolutions() == ("2K",)
    assert catalog.upscale_usd("4K") is None
    assert catalog.upscale_usd("2K") == Decimal("0.0425")


def test_the_upscale_node_is_worded_for_every_resolution():
    """The upscale node offers 2K and 4K and takes any accepted clip, so
    neither its texts nor the pack's describe it as 2K-only or 768P-only.
    Its node id stays, so saved workflows keep working."""
    root = os.path.join(os.path.dirname(__file__), "..")

    def read(*parts):
        with open(os.path.join(root, *parts), encoding="utf-8") as f:
            return f.read()
    nodes_src = read("anex_comfy", "nodes.py")
    start = nodes_src.index("class AnexUpscaleTo2K")
    node = nodes_src[start:nodes_src.index("async def execute", start)]
    assert 'node_id="AnexUpscaleTo2K"' in node
    assert 'display_name="anex.sh Upscale Video"' in node
    for text in (node, read("README.md"), read("pyproject.toml")):
        assert "Upscale Video to 2K" not in text
        assert "768P video" not in text
        assert "finished 768P clip" not in text
    assert "2K upscaling" not in read("pyproject.toml")
