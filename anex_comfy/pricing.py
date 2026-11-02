"""List prices of anex.sh video generation, for the price shown on a node.

These mirror the published rates at https://anex.sh/docs/video#pricing.
What a request actually costs is settled by anex.sh from the measured units,
never above the amount held when the task is created.
"""
from __future__ import annotations

# Per second of video made, by MiniMax-compatible model name.
OUTPUT_USD_PER_SECOND = {
    "MiniMax-H3": 0.08,
    "MiniMax-H3-Ref2VA-SMD-4step": 0.08,
    "FastH3-8-Step-V2": 0.04,
}
# Added per second of video at 2K (the 768p render, upscaled), and the rate
# of an upscale of a finished clip.
UPSCALE_USD_PER_SECOND = 0.05
# Per image (frames and reference images together), past the free ones.
IMAGE_USD = 0.04
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
        rate += UPSCALE_USD_PER_SECOND
    usd = duration * rate + max(images - FREE_IMAGES, 0) * IMAGE_USD
    billed_ref = min(round(reference_video_seconds), MAX_REFERENCE_SECONDS)
    return round(usd + billed_ref * rate, 4)


def upscale_usd(duration: float) -> float:
    """The list price of upscaling a finished clip of this length to 2K."""
    return round(round(duration) * UPSCALE_USD_PER_SECOND, 4)


def label(usd: float) -> str:
    """A price as the node shows it."""
    return f"${usd:.2f}"
