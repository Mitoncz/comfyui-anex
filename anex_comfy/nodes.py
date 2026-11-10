"""ComfyUI nodes for anex.sh video generation (MiniMax H3, FastH3 and 2K
upscaling), on the MiniMax-compatible API with the user's anex.sh key."""
from __future__ import annotations

import io
import logging
import os

import numpy as np
import torch
from PIL import Image
from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input, InputImpl, Types

from . import pricing
from .client import (REGIONS, DEFAULT_REGION, KEY_ENV, AnexClient, AnexError,
                     Interrupted, TaskEnded, key_source, resolve_key)

log = logging.getLogger("comfyui-anex")

CATEGORY = "anex.sh/video"
RATIOS = ["16:9", "21:9", "4:3", "1:1", "3:4", "9:16"]
MAX_PROMPT_CHARS = 7000
MIN_SECONDS, MAX_SECONDS = 4, 15
MAX_REFERENCE_IMAGES = 9
MAX_REFERENCE_VIDEOS = 3
REFERENCE_VIDEO_SECONDS = (2, 15)


# --------------------------------------------------------------------------- #
# ComfyUI plumbing
# --------------------------------------------------------------------------- #
def _interrupted() -> bool:
    try:
        from comfy.model_management import processing_interrupted
    except ImportError:
        return False
    return processing_interrupted()


def _status_fn(node_id, price: str):
    """Writes the task's progress under the node, after the clip's list
    price (or "no charge")."""
    logged = {"last": None}

    def show(text: str) -> None:
        # The log gets a line per change of state, not per poll.
        state = text.split(" (")[0]
        if state != logged["last"]:
            logged["last"] = state
            log.info("%s: %s", price, text)
        text = f"{price}{'' if price == 'no charge' else ' at list price'}\n{text}"
        try:
            from server import PromptServer
            if node_id is not None and PromptServer.instance is not None:
                PromptServer.instance.send_progress_text(text, node_id)
        except Exception:  # the progress line is cosmetic
            pass
    return show


def _client(cls, region: str, api_key: str, price: str) -> AnexClient:
    return AnexClient(resolve_key(api_key), region,
                      on_status=_status_fn(getattr(cls.hidden, "unique_id", None),
                                           price),
                      is_interrupted=_interrupted)


def _image_png(image: torch.Tensor) -> bytes:
    """One ComfyUI IMAGE (H, W, C in 0..1) as PNG bytes."""
    array = (image.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
    if array.shape[-1] == 1:
        array = array[..., 0]
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="PNG")
    return buf.getvalue()


def _images(batch: torch.Tensor | None) -> list[torch.Tensor]:
    if batch is None:
        return []
    return [batch] if batch.dim() == 3 else list(batch)


def _check_image(image: torch.Tensor, label: str) -> None:
    height, width = int(image.shape[0]), int(image.shape[1])
    if min(width, height) < 256 or max(width, height) > 5760:
        raise ValueError(f"{label} is {width}x{height}; each side must be "
                         "256 to 5,760 pixels")
    if not 0.4 <= width / height <= 2.5:
        raise ValueError(f"{label} is {width}x{height}; width / height must "
                         "be 0.4 to 2.5")


def _video_mp4(video: Input.Video) -> bytes:
    """A VIDEO as MP4 bytes. A clip that is already H.264 MP4 (as ours are)
    is copied, anything else re-encoded."""
    buf = io.BytesIO()
    video.save_to(buf, format=Types.VideoContainer.MP4,
                  codec=Types.VideoCodec.H264)
    return buf.getvalue()


def _check_prompt(prompt: str) -> str:
    if not prompt or not prompt.strip():
        raise ValueError("the prompt is empty")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError(f"the prompt is {len(prompt)} characters; at most "
                         f"{MAX_PROMPT_CHARS}")
    return prompt


async def _generate(cls, region: str, api_key: str, body: dict, price: str,
                    *, uploads=(), upscale: bool = False) -> IO.NodeOutput:
    """Upload the request's files, run the task and return (VIDEO, task id).

    `price` is the clip's list price, shown with every status line.
    `uploads` are (label, bytes, content type, content item builder): each
    file is uploaded and its `input:<id>` passed to the builder, whose item
    joins `body["content"]`."""
    client = _client(cls, region, api_key, price)
    try:
        async with client:
            client.status(f"Starting (key from {key_source(api_key)})")
            for label, data, content_type, build in uploads:
                client.status(f"Uploading {label}")
                body.setdefault("content", []).append(
                    build(await client.upload(data, content_type,
                                              filename=label)))
            task_id, task, clip = await client.run(body, upscale=upscale)
    except Interrupted:
        raise Exception(
            "Interrupted: the anex.sh task was cancelled if it had not "
            "started yet" + _fetch_hint(client.task_id)) from None
    except TaskEnded as exc:
        raise Exception(f"anex.sh: {exc}") from None
    except AnexError as exc:
        raise Exception(f"anex.sh: {exc}{_fetch_hint(client.task_id)}") from None
    usage = task.get("usage") or {}
    log.info("anex.sh task %s done: %s s of video", task_id,
             usage.get("output_seconds"))
    return IO.NodeOutput(InputImpl.VideoFromFile(io.BytesIO(clip)), task_id)


def _fetch_hint(task_id: str | None) -> str:
    """How to get a clip the node stopped waiting for."""
    if not task_id:
        return ""
    return (f". If it was already running it still finishes: fetch it with "
            f"the anex.sh Get Task node, task id {task_id}")


# --------------------------------------------------------------------------- #
# Shared inputs
# --------------------------------------------------------------------------- #
def _prompt_input(tooltip: str = "What the clip shows and sounds like. Up "
                  "to 7,000 characters.") -> IO.String.Input:
    return IO.String.Input("prompt", multiline=True, default="",
                           tooltip=tooltip)


def _duration_input() -> IO.Int.Input:
    return IO.Int.Input("duration", default=6, min=MIN_SECONDS,
                        max=MAX_SECONDS, step=1,
                        display_mode=IO.NumberDisplay.slider,
                        tooltip="Length of the clip in seconds (4 to 15). "
                        "Billed per second.")


def _resolution_input() -> IO.Combo.Input:
    return IO.Combo.Input("resolution", options=["768P", "2K"],
                          default="768P",
                          tooltip="768P is the model's own output. 2K is the "
                          "768p clip upscaled, at an extra rate per second.")


def _seed_input() -> IO.Int.Input:
    return IO.Int.Input("seed", default=0, min=0, max=2**31 - 1, step=1,
                        display_mode=IO.NumberDisplay.number,
                        control_after_generate=True,
                        tooltip="A new seed makes ComfyUI ask for a new clip "
                        "instead of reusing the last one.")


def _key_hint() -> tuple[str, str]:
    """The api_key field's label and placeholder: whether ComfyUI was started
    with ANEX_API_KEY (read once, when the nodes load; the environment of a
    running ComfyUI does not change). The canvas draws the label, never the
    placeholder, of a one-line field, so the label carries the hint."""
    if os.environ.get(KEY_ENV, "").strip():
        return (f"api_key (empty: {KEY_ENV} is set)",
                f"empty: using {KEY_ENV} from the environment")
    return (f"api_key ({KEY_ENV} not set)",
            f"paste your key, or set {KEY_ENV} and restart ComfyUI")


def _account_inputs() -> list:
    return [
        IO.Combo.Input("region", options=list(REGIONS), default=DEFAULT_REGION,
                       advanced=True,
                       tooltip="eu for organizations that keep their data in "
                       "the EEA: the clip is made and stored in the EU."),
        IO.String.Input("api_key", default="", advanced=True,
                        display_name=_key_hint()[0], placeholder=_key_hint()[1],
                        tooltip=f"Leave empty to use the {KEY_ENV} "
                        "environment variable (recommended). A key typed here "
                        "is saved inside the workflow and in the metadata of "
                        "files ComfyUI writes."),
    ]


_OUTPUTS = [
    IO.Video.Output(display_name="video"),
    IO.String.Output(display_name="task_id",
                     tooltip="The anex.sh task id; connect it to Upscale to "
                     "2K to upscale this clip without uploading it again."),
]
_HIDDEN = [IO.Hidden.unique_id]


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
class AnexTextToVideo(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="AnexTextToVideo",
            display_name="anex.sh Text to Video",
            category=CATEGORY,
            description="A clip with sound from a text prompt, made by "
            "MiniMax H3 (50 steps) or FastH3-8-Step-V2 (8 steps, faster and "
            "cheaper, 768P only) on anex.sh.",
            inputs=[
                IO.Combo.Input("model", options=["MiniMax-H3",
                                                 "FastH3-8-Step-V2"],
                               default="MiniMax-H3"),
                _prompt_input(),
                _duration_input(),
                IO.Combo.Input("ratio", options=RATIOS, default="16:9",
                               tooltip="Aspect ratio of the clip."),
                _resolution_input(),
                _seed_input(),
                *_account_inputs(),
            ],
            outputs=_OUTPUTS,
            hidden=_HIDDEN,
        )

    @classmethod
    async def execute(cls, model: str, prompt: str, duration: int, ratio: str,
                      resolution: str, seed: int, region: str = DEFAULT_REGION,
                      api_key: str = "") -> IO.NodeOutput:
        if resolution == "2K" and model not in pricing.TWO_K_MODELS:
            raise ValueError(f"{model} makes 768P only")
        body = {"model": model,
                "content": [{"type": "text", "text": _check_prompt(prompt)}],
                "resolution": resolution, "duration": duration,
                "ratio": ratio, "seed": seed}
        await pricing.refresh()
        price = pricing.label(pricing.clip_usd(model, duration, resolution))
        return await _generate(cls, region, api_key, body, price)


class AnexFirstLastFrameToVideo(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="AnexFirstLastFrameToVideo",
            display_name="anex.sh First/Last Frame to Video",
            category=CATEGORY,
            description="A MiniMax H3 clip that starts on one image, ends on "
            "another, or both. The frame sets the aspect ratio.",
            inputs=[
                IO.Image.Input("first_frame", optional=True,
                               tooltip="The clip starts on this image."),
                IO.Image.Input("last_frame", optional=True,
                               tooltip="The clip ends on this image."),
                _prompt_input(),
                _duration_input(),
                _resolution_input(),
                _seed_input(),
                *_account_inputs(),
            ],
            outputs=_OUTPUTS,
            hidden=_HIDDEN,
        )

    @classmethod
    async def execute(cls, prompt: str, duration: int, resolution: str,
                      seed: int, first_frame: torch.Tensor | None = None,
                      last_frame: torch.Tensor | None = None,
                      region: str = DEFAULT_REGION,
                      api_key: str = "") -> IO.NodeOutput:
        if first_frame is None and last_frame is None:
            raise ValueError("connect a first frame, a last frame, or both")
        uploads = []
        for role, batch in (("first_frame", first_frame),
                            ("last_frame", last_frame)):
            if batch is None:
                continue
            image = _images(batch)[0]
            _check_image(image, role.replace("_", " "))
            uploads.append((f"{role}.png", _image_png(image), "image/png",
                            lambda src, role=role: {
                                "type": "image_url", "role": role,
                                "image_url": {"url": src}}))
        body = {"model": "MiniMax-H3",
                "content": [{"type": "text", "text": _check_prompt(prompt)}],
                "resolution": resolution, "duration": duration, "seed": seed}
        await pricing.refresh()
        price = pricing.label(pricing.clip_usd("MiniMax-H3", duration,
                                               resolution, images=len(uploads)))
        return await _generate(cls, region, api_key, body, price,
                               uploads=uploads)


class AnexReferenceToVideo(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="AnexReferenceToVideo",
            display_name="anex.sh Reference to Video / Edit Video",
            category=CATEGORY,
            description="MiniMax H3 Ref2VA (4-step): people, products or "
            "places from up to 9 reference images appear in the clip, or up "
            "to 3 reference videos are edited as the prompt describes (an "
            "object added or replaced, a new background, a new style). The "
            "clip is made again, so framing and audio are not frame-exact.",
            inputs=[
                IO.Autogrow.Input(
                    "reference_images",
                    template=IO.Autogrow.TemplateNames(
                        IO.Image.Input("reference_image"),
                        names=[f"image_{i}" for i in
                               range(1, MAX_REFERENCE_IMAGES + 1)],
                        min=0),
                    tooltip="Up to 9 images, sent in connection order: "
                    "Image 1, Image 2 and so on."),
                IO.Autogrow.Input(
                    "reference_videos",
                    template=IO.Autogrow.TemplateNames(
                        IO.Video.Input("reference_video"),
                        names=[f"video_{i}" for i in
                               range(1, MAX_REFERENCE_VIDEOS + 1)],
                        min=0),
                    tooltip="Up to 3 videos, 2 to 15 s each and 15 s in "
                    "total, sent in connection order: Video 1, Video 2, "
                    "Video 3."),
                _prompt_input("What to make or change, naming the references "
                              "as Image 1 or Video 1. Up to 7,000 characters."),
                _duration_input(),
                IO.Combo.Input("ratio", options=["adaptive", *RATIOS],
                               default="adaptive",
                               tooltip="adaptive picks the ratio nearest the "
                               "first reference."),
                _resolution_input(),
                _seed_input(),
                *_account_inputs(),
            ],
            outputs=_OUTPUTS,
            hidden=_HIDDEN,
        )

    @classmethod
    async def execute(cls, prompt: str, duration: int, ratio: str,
                      resolution: str, seed: int,
                      reference_images: dict | None = None,
                      reference_videos: dict | None = None,
                      region: str = DEFAULT_REGION,
                      api_key: str = "") -> IO.NodeOutput:
        images = [image for batch in (reference_images or {}).values()
                  for image in _images(batch)]
        videos = [v for v in (reference_videos or {}).values() if v is not None]
        if not images and not videos:
            raise ValueError("connect at least one reference image or video")
        if len(images) > MAX_REFERENCE_IMAGES:
            raise ValueError(f"{len(images)} reference images; at most "
                             f"{MAX_REFERENCE_IMAGES}")
        if len(videos) > MAX_REFERENCE_VIDEOS:
            raise ValueError(f"{len(videos)} reference videos; at most "
                             f"{MAX_REFERENCE_VIDEOS}")
        lengths = [float(video.get_duration()) for video in videos]
        low, high = REFERENCE_VIDEO_SECONDS
        for n, seconds in enumerate(lengths, 1):
            if not low <= seconds <= high:
                raise ValueError(f"Video {n} is {seconds:.1f} s; each "
                                 f"reference video must be {low} to {high} s")
        total = sum(lengths)
        if total > high:
            raise ValueError(f"the reference videos total {total:.1f} s; at "
                             f"most {REFERENCE_VIDEO_SECONDS[1]} s")
        uploads = []
        for n, image in enumerate(images, 1):
            _check_image(image, f"Image {n}")
            uploads.append((f"image_{n}.png", _image_png(image), "image/png",
                            lambda src: {"type": "image_url",
                                         "role": "reference_image",
                                         "image_url": {"url": src}}))
        for n, video in enumerate(videos, 1):
            uploads.append((f"video_{n}.mp4", _video_mp4(video), "video/mp4",
                            lambda src: {"type": "video_url",
                                         "role": "reference_video",
                                         "video_url": {"url": src}}))
        body = {"model": "MiniMax-H3-Ref2VA-SMD-4step",
                "content": [{"type": "text", "text": _check_prompt(prompt)}],
                "resolution": resolution, "duration": duration,
                "ratio": ratio, "seed": seed}
        # Each reference video bills the part the model reads: its length or
        # the clip's, whichever is shorter.
        await pricing.refresh()
        price = pricing.label(pricing.clip_usd(
            "MiniMax-H3-Ref2VA-SMD-4step", duration, resolution,
            images=len(images),
            reference_video_seconds=sum(min(s, duration) for s in lengths)))
        return await _generate(cls, region, api_key, body, price,
                               uploads=uploads)


class AnexUpscaleTo2K(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="AnexUpscaleTo2K",
            display_name="anex.sh Upscale Video to 2K",
            category=CATEGORY,
            description="Upscales a finished 768P clip to 2K, "
            "keeping it as it is otherwise. Connect the task_id of an anex.sh "
            "node (nothing is uploaded), or any 768P video.",
            inputs=[
                IO.String.Input("task_id", optional=True, force_input=True,
                                tooltip="The task_id output of an anex.sh "
                                "video node."),
                IO.Video.Input("video", optional=True,
                               tooltip="A 768P clip, 4 to 15 s, used when no "
                               "task_id is connected."),
                *_account_inputs(),
            ],
            outputs=_OUTPUTS,
            hidden=_HIDDEN,
        )

    @classmethod
    async def execute(cls, task_id: str | None = None,
                      video: Input.Video | None = None,
                      region: str = DEFAULT_REGION,
                      api_key: str = "") -> IO.NodeOutput:
        body = {"model": "MiniMax-H3", "resolution": "2K"}
        uploads = []
        await pricing.refresh()
        if task_id:
            body["source_task_id"] = task_id
            price = f"{pricing.rate_label(pricing.upscale_rate())} a second"
        elif video is not None:
            price = pricing.label(pricing.upscale_usd(video.get_duration()))
            uploads.append(("base_video.mp4", _video_mp4(video), "video/mp4",
                            lambda src: {"type": "video_url",
                                         "role": "base_video",
                                         "video_url": {"url": src}}))
        else:
            raise ValueError("connect a task_id or a video")
        return await _generate(cls, region, api_key, body, price,
                               uploads=uploads, upscale=True)


class AnexGetTask(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="AnexGetTask",
            display_name="anex.sh Get Task",
            category=CATEGORY,
            description="The clip of a task made earlier: one a node stopped "
            "waiting for (ComfyUI was closed or restarted, or the run was "
            "cancelled once the task was rendering), or one made through the "
            "API. Waits for the task to finish if it has not yet. Costs "
            "nothing; the clip was billed when it was made. Clips are kept "
            "30 days.",
            inputs=[
                IO.String.Input("task_id", default="",
                                tooltip="The task id, from the node's status "
                                "line or error message, or its task_id "
                                "output."),
                *_account_inputs(),
            ],
            outputs=_OUTPUTS,
            hidden=_HIDDEN,
        )

    @classmethod
    async def execute(cls, task_id: str, region: str = DEFAULT_REGION,
                      api_key: str = "") -> IO.NodeOutput:
        task_id = (task_id or "").strip()
        if not task_id:
            raise ValueError("enter a task id")
        client = _client(cls, region, api_key, "no charge")
        try:
            async with client:
                client.status(f"Task {task_id}: fetching (key from "
                              f"{key_source(api_key)})")
                _task, clip = await client.fetch(task_id)
        except Interrupted:
            raise Exception(f"Interrupted: task {task_id} was left as it "
                            "is") from None
        except AnexError as exc:
            raise Exception(f"anex.sh: {exc}") from None
        return IO.NodeOutput(InputImpl.VideoFromFile(io.BytesIO(clip)), task_id)


NODES = [AnexTextToVideo, AnexFirstLastFrameToVideo, AnexReferenceToVideo,
         AnexUpscaleTo2K, AnexGetTask]


class AnexExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return NODES


async def comfy_entrypoint() -> AnexExtension:
    return AnexExtension()
