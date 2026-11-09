"""The client against a fake anex.sh: create, poll, upload, cancel, errors."""
import asyncio
import os
import sys

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from anex_comfy import client as anex  # noqa: E402

KEY = "test-key"
CLIP = b"\x00\x00\x00\x18ftypmp42fake-clip"


class Fake:
    """A fake tasks host and API host in one app. `statuses` is the sequence
    a task reports on successive polls (the last one repeats)."""

    def __init__(self, statuses=("queued", "running", "succeeded"),
                 busy_creates=0, fail_message=None, delete_status=200):
        self.statuses = list(statuses)
        self.busy_creates = busy_creates
        self.fail_message = fail_message
        self.delete_status = delete_status
        self.creates, self.deletes, self.uploads, self.polls = [], [], [], 0
        self.app = web.Application()
        r = self.app.router
        r.add_post("/v2/video_generation", self.create)
        r.add_post("/v2/video_regeneration", self.create)
        r.add_get("/v2/query/video_generation/{task_id}", self.query)
        r.add_delete("/v2/video_generation/{task_id}", self.delete)
        r.add_post("/batch/v1/inputs", self.mint)
        r.add_post("/upload", self.upload)
        r.add_get("/clip.mp4", self.clip)

    def _authorized(self, request):
        return request.headers.get("Authorization") == f"Bearer {KEY}"

    async def create(self, request):
        if not self._authorized(request):
            return web.json_response(
                {"type": "error", "error": {"type": "authorized_error",
                 "message": "invalid api key (1004)", "http_code": "401"}},
                status=401)
        if self.busy_creates:
            self.busy_creates -= 1
            return web.json_response({}, status=429,
                                     headers={"Retry-After": "0"})
        body = await request.json()
        self.creates.append((request.path, body))
        if not body.get("content") and "source_task_id" not in body:
            return web.json_response(
                {"type": "error", "error": {
                    "type": "bad_request_error",
                    "message": "invalid params, content is required (2013)",
                    "http_code": "400"}}, status=400)
        return web.json_response({"task_id": "318407566120873452"})

    async def query(self, request):
        status = self.statuses[min(self.polls, len(self.statuses) - 1)]
        self.polls += 1
        task = {"id": request.match_info["task_id"], "status": status}
        if status == "succeeded":
            task["content"] = {"url": str(request.url.with_path("/clip.mp4"))}
            task["usage"] = {"output_seconds": 6}
        if status == "failed":
            task["error"] = {"code": "content_policy_violation",
                             "message": self.fail_message}
        return web.json_response({"task": task})

    async def delete(self, request):
        self.deletes.append(request.match_info["task_id"])
        return web.json_response({}, status=self.delete_status)

    async def mint(self, request):
        body = await request.json()
        assert body["kind"] == "media"
        return web.json_response(
            {"id": f"up{len(self.uploads)}", "status": "awaiting_upload",
             "upload_url": str(request.url.with_path("/upload")),
             "fields": {"key": "k", "policy": "p"}}, status=201)

    async def upload(self, request):
        form = await request.post()
        self.uploads.append((dict((k, v) for k, v in form.items()
                                  if k != "file"),
                             form["file"].file.read(),
                             form["file"].content_type))
        return web.Response(status=204)

    async def clip(self, request):
        return web.Response(body=CLIP, content_type="video/mp4")


def run(fake, coro_fn, **kwargs):
    """Start the fake, point the client at it and run coro_fn(client). The
    hosts are fixed in the module, so the test swaps the region table."""
    async def main():
        server = TestServer(fake.app)
        await server.start_server()
        base = str(server.make_url("")).rstrip("/")
        saved = anex.REGIONS[anex.DEFAULT_REGION]
        anex.REGIONS[anex.DEFAULT_REGION] = (base, base)
        try:
            async with anex.AnexClient(KEY, poll_seconds=0.01, **kwargs) as c:
                return await coro_fn(c)
        finally:
            anex.REGIONS[anex.DEFAULT_REGION] = saved
            await server.close()
    return asyncio.run(main())


def test_run_creates_polls_and_downloads():
    fake = Fake()
    seen = []
    body = {"model": "MiniMax-H3", "content": [{"type": "text", "text": "x"}],
            "resolution": "768P", "duration": 6}
    task_id, task, clip = run(fake, lambda c: c.run(body),
                              on_status=seen.append)
    assert task_id == "318407566120873452"
    assert clip == CLIP
    assert task["status"] == "succeeded"
    assert fake.creates == [("/v2/video_generation", body)]
    assert any("running" in line for line in seen)
    assert fake.deletes == []


def test_upscale_goes_to_the_regeneration_endpoint():
    fake = Fake(statuses=("succeeded",))
    body = {"model": "MiniMax-H3", "resolution": "2K",
            "source_task_id": "318407566120873452"}
    run(fake, lambda c: c.run(body, upscale=True))
    assert fake.creates[0][0] == "/v2/video_regeneration"


def test_busy_create_waits_for_a_slot():
    fake = Fake(statuses=("succeeded",), busy_creates=2)
    seen = []
    run(fake, lambda c: c.run({"content": [1]}), on_status=seen.append)
    assert len(fake.creates) == 1
    assert any("30 requests in flight" in line for line in seen)


def test_upload_posts_fields_then_file_and_returns_input_source():
    fake = Fake()
    source = run(fake, lambda c: c.upload(b"PNGDATA", "image/png", "a.png"))
    assert source == "input:up0"
    fields, data, content_type = fake.uploads[0]
    assert fields == {"key": "k", "policy": "p"}
    assert data == b"PNGDATA" and content_type == "image/png"


def test_a_failed_task_raises_its_message_and_is_not_cancelled():
    fake = Fake(statuses=("queued", "failed"),
                fail_message="the prompt breaks the use rules")
    with pytest.raises(anex.TaskEnded) as exc:
        run(fake, lambda c: c.run({"content": [1]}))
    assert "the prompt breaks the use rules" in str(exc.value)
    assert exc.value.code == "content_policy_violation"
    assert fake.deletes == []


def test_an_interrupt_cancels_the_task():
    fake = Fake(statuses=("queued",))
    polls = {"n": 0}

    def interrupted():
        polls["n"] += 1
        return polls["n"] > 3

    seen = []
    with pytest.raises(anex.Interrupted):
        run(fake, lambda c: c.run({"content": [1]}),
            is_interrupted=interrupted, on_status=seen.append)
    assert fake.deletes == ["318407566120873452"]
    assert "cancelled at no charge" in seen[-1]


def test_an_interrupt_while_running_says_it_is_billed():
    fake = Fake(statuses=("running",), delete_status=400)
    polls = {"n": 0}

    def interrupted():
        polls["n"] += 1
        return polls["n"] > 3

    seen = []
    with pytest.raises(anex.Interrupted):
        run(fake, lambda c: c.run({"content": [1]}),
            is_interrupted=interrupted, on_status=seen.append)
    assert "finishes and is billed" in seen[-1]


def test_a_timeout_cancels_the_task():
    fake = Fake(statuses=("queued",))
    with pytest.raises(anex.AnexError, match="did not finish"):
        run(fake, lambda c: c.run({"content": [1]}), max_wait_seconds=0.05)
    assert fake.deletes == ["318407566120873452"]


def test_a_refused_create_shows_the_servers_message():
    fake = Fake()
    with pytest.raises(anex.AnexError, match="content is required") as exc:
        run(fake, lambda c: c.run({"model": "MiniMax-H3"}))
    assert exc.value.status == 400


def test_a_bad_key_says_so():
    fake = Fake()

    async def go(c):
        c._headers["Authorization"] = "Bearer wrong"
        return await c.create({"content": [1]})

    with pytest.raises(anex.AnexError, match="API key was not accepted"):
        run(fake, go)


def test_regions_are_fixed_hosts(monkeypatch):
    # The variables older versions read change nothing: the key only ever
    # goes to the region's own anex.sh hosts.
    monkeypatch.setenv("ANEX_TASKS_URL", "https://example.com")
    monkeypatch.setenv("ANEX_API_URL", "https://example.com")
    assert anex.endpoints("world") == anex.Endpoints(
        "https://tasks.anex.sh", "https://api.anex.sh")
    assert anex.endpoints("eu") == anex.Endpoints(
        "https://tasks-eu.anex.sh", "https://api-eu.anex.sh")
    with pytest.raises(anex.AnexError):
        anex.endpoints("mars")


def test_key_comes_from_the_node_or_the_environment(monkeypatch):
    monkeypatch.setenv(anex.KEY_ENV, "from-env")
    assert anex.resolve_key("") == "from-env"
    assert anex.resolve_key(" typed ") == "typed"
    monkeypatch.delenv(anex.KEY_ENV)
    with pytest.raises(anex.AnexError, match="ANEX_API_KEY"):
        anex.resolve_key(None)


def test_fetch_waits_for_an_existing_task_and_downloads_it():
    fake = Fake(statuses=("running", "succeeded"))
    task, clip = run(fake, lambda c: c.fetch("318407566120873452"))
    assert clip == CLIP and task["status"] == "succeeded"
    assert fake.creates == [] and fake.deletes == []


def test_stopping_a_fetch_never_cancels_the_task():
    fake = Fake(statuses=("queued",))
    polls = {"n": 0}

    def interrupted():
        polls["n"] += 1
        return polls["n"] > 3

    with pytest.raises(anex.Interrupted):
        run(fake, lambda c: c.fetch("318407566120873452"),
            is_interrupted=interrupted)
    assert fake.deletes == []
    fake = Fake(statuses=("queued",))
    with pytest.raises(anex.AnexError, match="did not finish"):
        run(fake, lambda c: c.fetch("318407566120873452"),
            max_wait_seconds=0.05)
    assert fake.deletes == []


def test_the_client_remembers_its_task():
    fake = Fake(statuses=("succeeded",))

    async def go(c):
        await c.run({"content": [1]})
        return c.task_id

    assert run(fake, go) == "318407566120873452"


def test_key_source_names_where_the_key_comes_from(monkeypatch):
    monkeypatch.setenv(anex.KEY_ENV, "from-env")
    assert anex.key_source("typed") == "the node's api_key field"
    assert anex.key_source("") == "ANEX_API_KEY"
    monkeypatch.delenv(anex.KEY_ENV)
    assert anex.key_source("  ") is None
