"""List prices of anex.sh video generation, for the price shown on a node.

The prices are those of the models catalog.py read from anex.sh when
ComfyUI started (or its fallbacks, down to the published rates at
https://anex.sh/docs/video#pricing when this version was released). What a
request actually costs is settled by anex.sh from the measured units, never
above the amount held when the task is created.
"""
from __future__ import annotations

from decimal import Decimal

from . import catalog

# Per second of 2K video when a finished clip is upscaled, until the catalog
# says otherwise.
UPSCALE_USD_PER_SECOND = 0.0425
# The part of the reference videos that bills is capped at this many seconds.
MAX_REFERENCE_SECONDS = 15


def _model(name: str, kind: str | None) -> catalog.VideoModel | None:
    """The model of this name: for this kind of request when one is given,
    else the first listed under the name. A name the loaded list lacks is
    priced at its built-in rates when it is one of the built-in models."""
    listed = catalog.live()
    for models in (listed, catalog.BUILTIN):
        for model in models:
            if model.name == name and (kind is None or kind in model.kinds):
                return model
    return None


def two_k_models() -> tuple[str, ...]:
    """The models that render 2K."""
    return tuple(dict.fromkeys(m.name for m in catalog.live()
                               if "2K" in m.resolutions))


def upscale_rate(resolution: str = "2K") -> float:
    """Per second of video when a finished clip is upscaled to this
    resolution. Raises ValueError for one the catalog gives no price for."""
    usd = catalog.upscale_usd(resolution)
    if usd is None:
        if resolution == "2K":
            return UPSCALE_USD_PER_SECOND
        raise ValueError(f"no price for upscaling to {resolution}")
    return float(usd)


def clip_usd(model: str, duration: int, resolution: str = "768P", *,
             kind: str | None = None, images: int = 0,
             reference_video_seconds: float = 0) -> float | None:
    """The list price of one clip, or None for a model with no known price.
    `reference_video_seconds` is the summed length of the reference videos,
    each counted up to the clip's length; they bill at the clip's own rate
    per second, 2K and 4K included. Reference audio is free, so it takes no part."""
    m = _model(model, kind)
    if m is None:
        return None
    rate = m.output_usd
    if resolution == "2K":
        rate += m.upscale_usd or Decimal(0)
    elif resolution == "4K":
        rate += m.upscale_4k_usd or Decimal(0)
    usd = (duration * rate
           + max(images - m.free_images, 0) * (m.image_usd or Decimal(0)))
    billed_ref = min(round(reference_video_seconds), MAX_REFERENCE_SECONDS)
    return round(float(usd + billed_ref * rate), 4)


def upscale_usd(duration: float, resolution: str = "2K") -> float:
    """The list price of upscaling a finished clip of this length."""
    return round(round(duration) * upscale_rate(resolution), 4)


def label(usd: float | None) -> str:
    """A price as the node shows it."""
    return "price not listed" if usd is None else f"${usd:.2f}"


def rate_label(usd: float) -> str:
    """A per-second rate as the node shows it: to the cent where that is
    exact, else with the digits the rate has (at most four)."""
    text = f"{usd:.4f}".rstrip("0")
    if len(text.split(".")[1]) < 2:
        text = f"{usd:.2f}"
    return f"${text}"
