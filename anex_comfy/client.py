"""A small async client for anex.sh video generation, free of ComfyUI.

It speaks the MiniMax-compatible API (create a task, poll it, cancel it) on
the `tasks` host and uploads images and videos through the batch API's media
uploads, so a reference never passes through third-party storage. The nodes
in nodes.py wrap it; the tests drive it against a fake server.
"""
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Callable

import aiohttp

# The hosts of each region. An organization that keeps its data in the EEA
# uses the EU hosts; the other region's host answers 421.
REGIONS = {
    "world": ("https://tasks.anex.sh", "https://api.anex.sh"),
    "eu": ("https://tasks-eu.anex.sh", "https://api-eu.anex.sh"),
}
DEFAULT_REGION = "world"

# Environment overrides, for a test deployment: ANEX_TASKS_URL and
# ANEX_API_URL replace both hosts of whichever region is picked.
KEY_ENV = "ANEX_API_KEY"
TASKS_URL_ENV = "ANEX_TASKS_URL"
API_URL_ENV = "ANEX_API_URL"

TERMINAL = ("succeeded", "failed", "cancelled")
POLL_SECONDS = 5.0
# A cold start rents and boots a GPU box before the clip renders, which can
# take a while; past this the node gives up (and cancels the task if it is
# still queued).
MAX_WAIT_SECONDS = 60 * 60
# Creates answer 429 while the organization has 30 requests in flight; the
# node waits for a slot this long before giving up.
MAX_BUSY_WAIT_SECONDS = 30 * 60
# A poll that fails on the network or with a 5xx is retried; this many in a
# row ends the wait.
MAX_POLL_FAILURES = 12

USER_AGENT = "comfyui-anex/0.1.0"


class AnexError(Exception):
    """A refusal or failure from anex.sh, with a message meant for the user."""

    def __init__(self, message: str, status: int | None = None,
                 code: str | None = None):
        super().__init__(message)
        self.status = status
        self.code = code


class TaskEnded(AnexError):
    """The task failed or was cancelled on our side."""


class Interrupted(Exception):
    """The user interrupted the run while the task was in flight."""


@dataclass(frozen=True)
class Endpoints:
    tasks_url: str
    api_url: str


def endpoints(region: str = DEFAULT_REGION) -> Endpoints:
    """The tasks and API hosts for a region, with the environment overrides
    applied."""
    if region not in REGIONS:
        raise AnexError(f"unknown region {region!r}; use "
                        + " or ".join(REGIONS))
    tasks_url, api_url = REGIONS[region]
    return Endpoints(
        (os.environ.get(TASKS_URL_ENV) or tasks_url).rstrip("/"),
        (os.environ.get(API_URL_ENV) or api_url).rstrip("/"))


def resolve_key(api_key: str | None) -> str:
    """The key typed on the node, or else ANEX_API_KEY."""
    key = (api_key or "").strip() or os.environ.get(KEY_ENV, "").strip()
    if not key:
        raise AnexError(
            f"no anex.sh API key: set the {KEY_ENV} environment variable "
            "before starting ComfyUI, or fill in the node's api_key field. "
            "Keys are on the Keys page of the console at https://anex.sh")
    return key


async def _error_from(resp: aiohttp.ClientResponse) -> AnexError:
    """An AnexError from a non-2xx answer: MiniMax's error body on the tasks
    host, `{"detail": ...}` on the API host."""
    try:
        body = await resp.json(content_type=None)
    except (aiohttp.ContentTypeError, ValueError):
        body = None
    message = None
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            message = err.get("message")
        elif isinstance(body.get("detail"), str):
            message = body["detail"]
        elif isinstance(body.get("detail"), list) and body["detail"]:
            message = "; ".join(str(d.get("msg", d)) if isinstance(d, dict)
                                else str(d) for d in body["detail"])
    message = message or f"HTTP {resp.status}"
    if resp.status == 401:
        message = f"the API key was not accepted ({message})"
    elif resp.status in (402, 403) and "balance" in message:
        message = (f"{message}: top up your credits in the console at "
                   "https://anex.sh")
    return AnexError(message, status=resp.status)


def _retry_after(resp: aiohttp.ClientResponse, default: float) -> float:
    try:
        return max(1.0, float(resp.headers.get("Retry-After", default)))
    except ValueError:
        return default


StatusFn = Callable[[str], None]
InterruptFn = Callable[[], bool]


class AnexClient:
    """One session's worth of calls. Use as `async with AnexClient(...)`."""

    def __init__(self, api_key: str, region: str = DEFAULT_REGION, *,
                 on_status: StatusFn | None = None,
                 is_interrupted: InterruptFn | None = None,
                 poll_seconds: float = POLL_SECONDS,
                 max_wait_seconds: float = MAX_WAIT_SECONDS,
                 max_busy_wait_seconds: float = MAX_BUSY_WAIT_SECONDS):
        self.ep = endpoints(region)
        self._headers = {"Authorization": f"Bearer {api_key}",
                         "User-Agent": USER_AGENT}
        self._status = on_status or (lambda _text: None)
        self._interrupted = is_interrupted or (lambda: False)
        self.poll_seconds = poll_seconds
        self.max_wait_seconds = max_wait_seconds
        self.max_busy_wait_seconds = max_busy_wait_seconds
        self._session: aiohttp.ClientSession | None = None
        # The task this client created or fetched last, for messages.
        self.task_id: str | None = None

    async def __aenter__(self) -> "AnexClient":
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=30,
                                          sock_read=300),
            trust_env=True)
        return self

    async def __aexit__(self, *exc) -> None:
        if self._session is not None:
            await self._session.close()

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None:
            raise RuntimeError("use AnexClient as an async context manager")
        return self._session

    def status(self, text: str) -> None:
        """Report progress to the caller (the line under a ComfyUI node)."""
        self._status(text)

    async def _sleep(self, seconds: float) -> None:
        """Sleep in short steps, raising Interrupted as soon as the user
        interrupts."""
        end = time.monotonic() + seconds
        while True:
            if self._interrupted():
                raise Interrupted()
            left = end - time.monotonic()
            if left <= 0:
                return
            await asyncio.sleep(min(left, 0.5))

    # ------------------------------------------------------------------ #
    # Uploads
    # ------------------------------------------------------------------ #
    async def upload(self, data: bytes, content_type: str,
                     filename: str = "input") -> str:
        """Upload one image or video and return the `input:<id>` source a
        request names it by (valid for 24 hours)."""
        async with self.session.post(
                f"{self.ep.api_url}/batch/v1/inputs", headers=self._headers,
                json={"kind": "media", "content_type": content_type}) as resp:
            if resp.status >= 300:
                raise await _error_from(resp)
            minted = await resp.json()
        form = aiohttp.FormData()
        for name, value in (minted.get("fields") or {}).items():
            form.add_field(name, value)
        form.add_field("file", data, filename=filename,
                       content_type=content_type)
        async with self.session.post(minted["upload_url"], data=form) as resp:
            if resp.status >= 300:
                raise AnexError(f"the upload was refused (HTTP {resp.status}); "
                                "the file may be over the size limit",
                                status=resp.status)
        return f"input:{minted['id']}"

    # ------------------------------------------------------------------ #
    # Tasks
    # ------------------------------------------------------------------ #
    async def _create(self, path: str, body: dict) -> str:
        """POST a create, waiting out the in-flight limit (429)."""
        deadline = time.monotonic() + self.max_busy_wait_seconds
        while True:
            async with self.session.post(f"{self.ep.tasks_url}{path}",
                                         headers=self._headers,
                                         json=body) as resp:
                if resp.status == 429 and time.monotonic() < deadline:
                    wait = _retry_after(resp, 15.0)
                    self._status("Waiting for a free slot: the organization "
                                 "has 30 requests in flight")
                elif resp.status >= 300:
                    raise await _error_from(resp)
                else:
                    return str((await resp.json())["task_id"])
            await self._sleep(wait)

    async def create(self, body: dict) -> str:
        """Create a generation task and return its id."""
        return await self._create("/v2/video_generation", body)

    async def create_upscale(self, body: dict) -> str:
        """Create a 2K upscale of a finished clip and return the task id."""
        return await self._create("/v2/video_regeneration", body)

    async def get(self, task_id: str) -> dict:
        async with self.session.get(
                f"{self.ep.tasks_url}/v2/query/video_generation/{task_id}",
                headers=self._headers) as resp:
            if resp.status >= 300:
                raise await _error_from(resp)
            return (await resp.json())["task"]

    async def cancel(self, task_id: str) -> bool:
        """Cancel a task: True when it was still queued and so costs
        nothing. A running task cannot be stopped; it finishes and is
        billed."""
        try:
            async with self.session.delete(
                    f"{self.ep.tasks_url}/v2/video_generation/{task_id}",
                    headers=self._headers) as resp:
                return resp.status < 300
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False

    async def wait(self, task_id: str, *, cancel_on_stop: bool = True) -> dict:
        """Poll a task until it ends and return it when it succeeded.

        A failed or cancelled task raises TaskEnded with the task's own
        message. On an interrupt, a timeout or lost contact the task is
        cancelled first (free while it is still queued), unless
        `cancel_on_stop` is false: a task this client only fetches is left
        to finish."""
        started = time.monotonic()
        failures = 0
        try:
            while True:
                try:
                    task = await self.get(task_id)
                    failures = 0
                except AnexError as exc:
                    if exc.status is None or exc.status < 500:
                        raise
                    task, failures = None, failures + 1
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    task, failures = None, failures + 1
                if failures >= MAX_POLL_FAILURES:
                    raise AnexError(f"lost contact with task {task_id}")
                if task is not None:
                    status = task.get("status")
                    if status == "succeeded":
                        return task
                    if status in TERMINAL:
                        err = task.get("error") or {}
                        raise TaskEnded(
                            f"task {task_id} {status}: "
                            f"{err.get('message') or err.get('code') or status}",
                            code=err.get("code"))
                    elapsed = int(time.monotonic() - started)
                    self._status(f"Task {task_id}: {status} ({elapsed} s)")
                if time.monotonic() - started > self.max_wait_seconds:
                    raise AnexError(
                        f"task {task_id} did not finish within "
                        f"{int(self.max_wait_seconds // 60)} minutes")
                await self._sleep(self.poll_seconds)
        except TaskEnded:
            raise
        except (Interrupted, AnexError):
            if not cancel_on_stop:
                raise
            cancelled = await self.cancel(task_id)
            self._status(f"Task {task_id} cancelled at no charge" if cancelled
                         else f"Task {task_id} was already running: it "
                              "finishes and is billed")
            raise

    async def download(self, url: str) -> bytes:
        """The clip at a task's signed link (valid for an hour)."""
        for attempt in range(3):
            try:
                async with self.session.get(url) as resp:
                    if resp.status >= 300:
                        raise AnexError(f"the clip download failed "
                                        f"(HTTP {resp.status})",
                                        status=resp.status)
                    return await resp.read()
            except (aiohttp.ClientError, asyncio.TimeoutError):
                if attempt == 2:
                    raise
                await self._sleep(2.0 * (attempt + 1))
        raise AssertionError("unreachable")

    async def run(self, body: dict, *, upscale: bool = False
                  ) -> tuple[str, dict, bytes]:
        """Create a task, wait for it and download its clip: (task id, the
        task, the MP4 bytes)."""
        task_id = await (self.create_upscale(body) if upscale
                         else self.create(body))
        self.task_id = task_id
        self._status(f"Task {task_id}: created")
        task = await self.wait(task_id)
        return task_id, task, await self._clip(task_id, task)

    async def fetch(self, task_id: str) -> tuple[dict, bytes]:
        """An existing task's clip: wait for the task to end (never
        cancelling it) and download the clip. (the task, the MP4 bytes)"""
        self.task_id = task_id
        task = await self.wait(task_id, cancel_on_stop=False)
        return task, await self._clip(task_id, task)

    async def _clip(self, task_id: str, task: dict) -> bytes:
        url = (task.get("content") or {}).get("url")
        if not url:
            raise AnexError(f"task {task_id} succeeded without a clip link")
        self._status(f"Task {task_id}: downloading the clip")
        return await self.download(url)

