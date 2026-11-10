"""List prices of anex.sh video generation, for the price shown on a node.

These mirror the published rates at https://anex.sh/docs/video#pricing.
What a request actually costs is settled by anex.sh from the measured units,
never above the amount held when the task is created.
"""
from __future__ import annotations

# Per second of video made, by MiniMax-compatible model name.
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


def clip_usd(model: str, duration: int, resolution: str = "768P", *,
             images: int = 0, reference_video_seconds: float = 0) -> float:
    """The list price of one clip. `reference_video_seconds` is the summed
    length of the reference videos, each counted up to the clip's length;
    they bill at the clip's own rate per second, 2K included."""
    rate = OUTPUT_USD_PER_SECOND[model]
    if resolution == "2K":
        rate += TWO_K_ADD_USD_PER_SECOND[model]
    usd = duration * rate + max(images - FREE_IMAGES, 0) * IMAGE_USD
    billed_ref = min(round(reference_video_seconds), MAX_REFERENCE_SECONDS)
    return round(usd + billed_ref * rate, 4)


def upscale_usd(duration: float) -> float:
    """The list price of upscaling a finished clip of this length to 2K."""
    return round(round(duration) * UPSCALE_USD_PER_SECOND, 4)


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
