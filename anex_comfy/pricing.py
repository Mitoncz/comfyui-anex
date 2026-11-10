"""List prices of anex.sh video generation, for the price shown on a node.

The prices come from anex.sh's public price catalog (the one its price
calculator reads), fetched when a node runs and kept for a while, so a price
change reaches the node without a new release of the pack. Until the catalog
has been read, or when it cannot be, the built-in prices below are used: the
published rates at https://anex.sh/docs/video#pricing when this version was
released. What a request actually costs is settled by anex.sh from the
measured units, never above the amount held when the task is created.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, replace

# The built-in prices. Per second of video made, by MiniMax-compatible name.
OUTPUT_USD_PER_SECOND = {
    "MiniMax-H3": 0.068,
    "MiniMax-H3-Ref2VA-SMD-4step": 0.051,
    "FastH3-8-Step-V2": 0.017,
}
# Added per second of video at 2K (the 768p render, upscaled): 2K costs
# $0.11 a second on both models, so each adds $0.11 less its own rate.
TWO_K_ADD_USD_PER_SECOND = {
    "MiniMax-H3": 0.042,
    "MiniMax-H3-Ref2VA-SMD-4step": 0.059,
}
# Per second of 2K video when a finished clip is upscaled.
UPSCALE_USD_PER_SECOND = 0.0425
# Per image (frames and reference images together), past the free ones.
IMAGE_USD = 0.034
FREE_IMAGES = 5
# 2K is offered on these models only.
TWO_K_MODELS = ("MiniMax-H3", "MiniMax-H3-Ref2VA-SMD-4step")
# The part of the reference videos that bills is capped at this many seconds.
MAX_REFERENCE_SECONDS = 15

# Where the prices are read. ANEX_PRICES_URL replaces it, for a test
# deployment. The catalog names models by their anex.sh ids.
CATALOG_URL = "https://anex.sh/api/estimate/catalog"
PRICES_URL_ENV = "ANEX_PRICES_URL"
CATALOG_IDS = {"MiniMax-H3": "minimax-h3",
               "MiniMax-H3-Ref2VA-SMD-4step": "minimax-h3-ref2va",
               "FastH3-8-Step-V2": "fasth3-8-step-v2"}
UPSCALER_ID = "seedvr2-3b"
# How long a read catalog is kept, how soon a failed read is tried again,
# and how long a read may take before the node goes on without it.
REFRESH_SECONDS = 10 * 60
RETRY_SECONDS = 60
FETCH_TIMEOUT_SECONDS = 5


@dataclass(frozen=True)
class ModelRates:
    output: float
    two_k_add: float | None
    image: float | None
    free_images: int


def _builtin() -> tuple[dict[str, ModelRates], float]:
    return ({name: ModelRates(
        output=rate,
        two_k_add=TWO_K_ADD_USD_PER_SECOND.get(name),
        image=IMAGE_USD if name in TWO_K_MODELS else None,
        free_images=FREE_IMAGES if name in TWO_K_MODELS else 0)
        for name, rate in OUTPUT_USD_PER_SECOND.items()},
        UPSCALE_USD_PER_SECOND)


_rates, _upscale = _builtin()
_next_fetch = 0.0
# Where the prices in use came from: "built-in" or the catalog's URL.
source = "built-in"


def reset() -> None:
    """Back to the built-in prices, with the catalog not yet read."""
    global _rates, _upscale, _next_fetch, source
    _rates, _upscale = _builtin()
    _next_fetch = 0.0
    source = "built-in"


def _usd(value) -> float | None:
    """A price from the catalog (a decimal string), or None if absent. A
    value that is not a non-negative number refuses the whole catalog."""
    if value is None:
        return None
    usd = float(value)
    if not usd >= 0:
        raise ValueError(f"bad price {value!r}")
    return usd


def apply_catalog(doc: dict) -> bool:
    """Take the prices of a catalog (the JSON of CATALOG_URL). A model the
    catalog does not list keeps its price; a catalog that lists none of
    them, or carries a malformed price, changes nothing. Returns whether
    any price was taken."""
    global _rates, _upscale
    try:
        fees = {m["id"]: m.get("media_fees") or {} for m in doc["models"]}
        rates = dict(_rates)
        found = False
        for name, model_id in CATALOG_IDS.items():
            fee = fees.get(model_id)
            output = _usd((fee or {}).get("video_output_usd_per_second"))
            if output is None:
                continue
            found = True
            rates[name] = replace(
                rates[name], output=output,
                two_k_add=_usd(fee.get("video_upscale_usd_per_second")),
                image=_usd(fee.get("video_input_image_usd")),
                free_images=int(fee.get("video_free_images") or 0))
        upscale = _usd((fees.get(UPSCALER_ID) or {}).get(
            "video_output_usd_per_second"))
    except (KeyError, TypeError, ValueError, AttributeError):
        return False
    if not found and upscale is None:
        return False
    _rates = rates
    if upscale is not None:
        _upscale = upscale
    return True


async def refresh(session=None, *, now: float | None = None) -> None:
    """Read the catalog if the prices in use are older than REFRESH_SECONDS
    (or a failed read is older than RETRY_SECONDS). Never raises: on any
    failure the node shows the prices it already has."""
    global _next_fetch, source
    now = time.monotonic() if now is None else now
    if now < _next_fetch:
        return
    url = os.environ.get(PRICES_URL_ENV) or CATALOG_URL
    _next_fetch = now + RETRY_SECONDS
    try:
        import aiohttp
        from .client import USER_AGENT
        timeout = aiohttp.ClientTimeout(total=FETCH_TIMEOUT_SECONDS)
        own = session is None
        session = session or aiohttp.ClientSession(timeout=timeout)
        try:
            async with session.get(url, timeout=timeout,
                                   headers={"User-Agent": USER_AGENT}) as resp:
                if resp.status != 200:
                    return
                doc = await resp.json(content_type=None)
        finally:
            if own:
                await session.close()
    except Exception:  # noqa: BLE001 - a price display must never fail a node
        return
    if apply_catalog(doc):
        source = url
        _next_fetch = now + REFRESH_SECONDS


def two_k_models() -> tuple[str, ...]:
    """The models that render 2K."""
    return tuple(n for n, r in _rates.items() if r.two_k_add is not None)


def upscale_rate() -> float:
    """Per second of 2K video when a finished clip is upscaled."""
    return _upscale


def clip_usd(model: str, duration: int, resolution: str = "768P", *,
             images: int = 0, reference_video_seconds: float = 0) -> float:
    """The list price of one clip. `reference_video_seconds` is the summed
    length of the reference videos, each counted up to the clip's length;
    they bill at the clip's own rate per second, 2K included."""
    r = _rates[model]
    rate = r.output
    if resolution == "2K":
        rate += r.two_k_add or 0.0
    usd = duration * rate + max(images - r.free_images, 0) * (r.image or 0.0)
    billed_ref = min(round(reference_video_seconds), MAX_REFERENCE_SECONDS)
    return round(usd + billed_ref * rate, 4)


def upscale_usd(duration: float) -> float:
    """The list price of upscaling a finished clip of this length to 2K."""
    return round(round(duration) * _upscale, 4)


def label(usd: float) -> str:
    """A price as the node shows it."""
    return f"${usd:.2f}"


def rate_label(usd: float) -> str:
    """A per-second rate as the node shows it: to the cent where that is
    exact, else with the digits the rate has (at most four)."""
    text = f"{usd:.4f}".rstrip("0")
    if len(text.split(".")[1]) < 2:
        text = f"{usd:.2f}"
    return f"${text}"
