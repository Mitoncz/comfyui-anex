"""The video models anex.sh offers, for the nodes' model menus and prices.

Read once, when ComfyUI defines the nodes, from anex.sh's public catalog
(the one its price calculator reads), so a model or price added on anex.sh
reaches the menus after a restart of ComfyUI, with no new release of the
pack. The read waits at most FETCH_TIMEOUT_SECONDS. When it fails, the last
good answer kept on disk is used, and without one the list built into the
pack (BUILTIN): the models and published rates of
https://anex.sh/docs/video#pricing when this version was released. Nothing
here raises: a menu or a price must never keep ComfyUI from starting.

Each video model in the catalog carries a `video` block: the name a request
sends as `model`, the kinds of request it answers (text, frames,
reference), its resolutions and whether it honours a seed. A catalog with
no such block anywhere (one older than this pack) yields the built-in
models, priced from the catalog by BUILTIN_IDS.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from typing import Callable

log = logging.getLogger("comfyui-anex")

# Fixed, like the API hosts in client.py, so nothing in the environment can
# point the pack elsewhere.
CATALOG_URL = "https://anex.sh/api/estimate/catalog"
FETCH_TIMEOUT_SECONDS = 3
MAX_CATALOG_BYTES = 4 * 2**20

KINDS = ("text", "frames", "reference")
RESOLUTIONS = ("768P", "2K")
# The model a node starts on, when the loaded list has it for that kind:
# the one the node used before it had a menu, so a saved workflow renders
# as it did.
PREFERRED = {"text": "MiniMax-H3", "frames": "MiniMax-H3",
             "reference": "MiniMax-H3-Ref2VA-SMD-4step"}
# The upscaler's catalog id: its price is that of the Upscale to 2K node.
UPSCALER_ID = "seedvr2-3b"


@dataclass(frozen=True)
class VideoModel:
    name: str
    kinds: tuple[str, ...]
    resolutions: tuple[str, ...]
    supports_seed: bool
    output_usd: Decimal
    upscale_usd: Decimal | None
    image_usd: Decimal | None
    free_images: int


BUILTIN: list[VideoModel] = [
    VideoModel("MiniMax-H3", ("text", "frames"), ("768P", "2K"), True,
               Decimal("0.068"), Decimal("0.042"), Decimal("0.034"), 5),
    VideoModel("FastH3-8-Step-V2", ("text",), ("768P",), True,
               Decimal("0.017"), None, None, 0),
    VideoModel("MiniMax-H3-Ref2VA-SMD-4step", ("reference",), ("768P", "2K"),
               True, Decimal("0.051"), Decimal("0.059"), Decimal("0.034"), 5),
]
# The catalog ids that price each built-in model, first match wins. The
# 4-step Ref2VA model is listed under its own id once anex.sh lists the full
# Ref2VA model too, and under minimax-h3-ref2va before that.
BUILTIN_IDS = {
    "MiniMax-H3": ("minimax-h3",),
    "FastH3-8-Step-V2": ("fasth3-8-step-v2",),
    "MiniMax-H3-Ref2VA-SMD-4step": ("minimax-h3-ref2va-smd-4step",
                                    "minimax-h3-ref2va"),
}

# What load() found: the models, and the upscaler's price per second (None
# when the catalog did not give it).
_loaded: list[VideoModel] | None = None
_upscale_usd: Decimal | None = None


def _usd(value) -> Decimal | None:
    """A price from the catalog (a decimal string), or None if absent."""
    if value is None:
        return None
    try:
        usd = Decimal(str(value))
    except InvalidOperation:
        raise ValueError(f"bad price {value!r}") from None
    if not usd.is_finite() or usd < 0:
        raise ValueError(f"bad price {value!r}")
    return usd


def _priced(fees: dict, **fields) -> dict:
    """The price fields of a VideoModel from a catalog entry's media_fees."""
    output = _usd(fees.get("video_output_usd_per_second"))
    if output is None:
        raise ValueError("no price per second")
    free = fees.get("video_free_images") or 0
    if not isinstance(free, int) or free < 0:
        raise ValueError(f"bad free image count {free!r}")
    return dict(fields, output_usd=output,
                upscale_usd=_usd(fees.get("video_upscale_usd_per_second")),
                image_usd=_usd(fees.get("video_input_image_usd")),
                free_images=free)


def _from_block(entry: dict) -> VideoModel:
    block = entry["video"]
    name, kinds = block["name"], tuple(block["kinds"])
    resolutions = tuple(block["resolutions"])
    seed = block["supports_seed"]
    if not isinstance(name, str) or not name:
        raise ValueError(f"bad name {name!r}")
    if not kinds or not set(kinds) <= set(KINDS):
        raise ValueError(f"unknown kinds {kinds!r}")
    if not resolutions or not all(isinstance(r, str) for r in resolutions):
        raise ValueError(f"bad resolutions {resolutions!r}")
    if not isinstance(seed, bool):
        raise ValueError(f"bad supports_seed {seed!r}")
    return VideoModel(**_priced(entry.get("media_fees") or {}, name=name,
                                kinds=kinds, resolutions=resolutions,
                                supports_seed=seed))


def _read(payload: dict) -> tuple[list[VideoModel], Decimal | None]:
    """The models of a catalog answer, and the upscaler's price. Raises
    ValueError when the answer is not a catalog or yields no model."""
    listed = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(listed, list):
        raise ValueError("not a catalog: no list of models")
    entries = [e for e in listed if isinstance(e, dict)]
    fees = {e.get("id"): e.get("media_fees") or {} for e in entries}
    try:
        upscale = _usd(fees.get(UPSCALER_ID, {}).get(
            "video_output_usd_per_second"))
    except (ValueError, AttributeError):
        upscale = None
    if any("video" in e for e in entries):
        found = []
        for entry in entries:
            if "video" not in entry:
                continue
            try:
                found.append(_from_block(entry))
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                log.warning("anex.sh catalog: skipped %s: %s",
                            entry.get("id"), exc)
        if not found:
            raise ValueError("no usable video model in the catalog")
        return found, upscale
    found = []
    for model in BUILTIN:
        for model_id in BUILTIN_IDS[model.name]:
            try:
                model = replace(model, **_priced(fees.get(model_id) or {}))
            except (TypeError, ValueError, AttributeError):
                continue
            break
        found.append(model)
    return found, upscale


def parse(payload: dict) -> list[VideoModel]:
    """The video models of a catalog answer (the JSON of CATALOG_URL)."""
    return _read(payload)[0]


def _fetch() -> dict:
    import urllib.request
    from .client import USER_AGENT
    request = urllib.request.Request(CATALOG_URL,
                                     headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request,
                                timeout=FETCH_TIMEOUT_SECONDS) as resp:
        return json.loads(resp.read(MAX_CATALOG_BYTES))


def _within_timeout(fetch: Callable[[], dict]) -> dict:
    """fetch(), given up on after FETCH_TIMEOUT_SECONDS in all (a socket
    timeout alone does not bound a slow name lookup or a trickling answer)."""
    result: dict = {}

    def run():
        try:
            result["payload"] = fetch()
        except BaseException as exc:  # noqa: BLE001 - reported below
            result["error"] = exc
    worker = threading.Thread(target=run, name="anex-catalog", daemon=True)
    worker.start()
    worker.join(FETCH_TIMEOUT_SECONDS)
    if "payload" in result:
        return result["payload"]
    raise result.get("error") or TimeoutError(
        f"no answer in {FETCH_TIMEOUT_SECONDS} s")


def default_cache_path() -> str:
    """Where the last good catalog is kept: ComfyUI's user directory, or a
    per-user cache directory where ComfyUI's paths are not importable."""
    try:
        import folder_paths
        base = folder_paths.get_user_directory()
    except Exception:  # noqa: BLE001 - not running inside ComfyUI
        base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
            os.path.expanduser("~"), ".cache")
    return os.path.join(base, "comfyui-anex", "catalog.json")


def _write_cache(path: str, payload: dict) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    except Exception as exc:  # noqa: BLE001 - the cache is a convenience
        log.warning("anex.sh catalog: could not keep a copy at %s: %s",
                    path, exc)


def load(fetch: Callable[[], dict] | None = None,
         cache_path: str | None = None) -> list[VideoModel]:
    """Read the catalog and make it the one the nodes use: from anex.sh
    (by `fetch`, CATALOG_URL by default), else the copy kept on disk, else
    BUILTIN. Never raises."""
    global _loaded, _upscale_usd
    models, upscale = list(BUILTIN), None
    fetch = fetch or _fetch
    try:
        path = cache_path or default_cache_path()
        try:
            payload = _within_timeout(fetch)
            models, upscale = _read(payload)
            _write_cache(path, payload)
        except Exception as exc:  # noqa: BLE001 - fall back to the copy
            log.warning("anex.sh catalog not read (%s); using the %s", exc,
                        "last copy" if os.path.exists(path) else
                        "built-in list")
            try:
                with open(path, encoding="utf-8") as f:
                    models, upscale = _read(json.load(f))
            except Exception:  # noqa: BLE001 - fall back to BUILTIN
                models, upscale = list(BUILTIN), None
    except Exception:  # noqa: BLE001 - never keep ComfyUI from starting
        models, upscale = list(BUILTIN), None
    _loaded, _upscale_usd = models, upscale
    return models


def loaded() -> list[VideoModel]:
    """The models in use, read on first use."""
    return _loaded if _loaded is not None else load()


def upscale_usd() -> Decimal | None:
    """The catalog's price per second of the Upscale to 2K node, if read."""
    loaded()
    return _upscale_usd


def models(kind: str) -> list[VideoModel]:
    """The models that answer this kind of request, in catalog order."""
    return [m for m in loaded() if kind in m.kinds]


def find(name: str, kind: str) -> VideoModel | None:
    """The model a request of this kind reaches under this name."""
    return next((m for m in models(kind) if m.name == name), None)


def names(kind: str) -> list[str]:
    """A node's model menu: never empty, so ComfyUI can load the node even
    when the list has no model of its kind (anex.sh then answers)."""
    found = list(dict.fromkeys(m.name for m in models(kind)))
    return found or [PREFERRED[kind]]


def default(kind: str) -> str:
    """The model a node's menu starts on."""
    options = names(kind)
    return PREFERRED[kind] if PREFERRED[kind] in options else options[0]
