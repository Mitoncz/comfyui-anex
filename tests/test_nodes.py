"""The nodes' requests, against ComfyUI's own node API. Skipped where
ComfyUI is not importable; run them with ComfyUI on the path:

    PYTHONPATH=/path/to/ComfyUI /path/to/ComfyUI/.venv/bin/python -m pytest tests
"""
import asyncio
import importlib
import io
import os
import sys

import pytest

pytest.importorskip("comfy_api.latest")
torch = pytest.importorskip("torch")

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(PACK))
pack = importlib.import_module(os.path.basename(PACK))  # as ComfyUI loads it
nodes = importlib.import_module(f"{pack.__name__}.anex_comfy.nodes")

from comfy_api.latest import IO, InputImpl  # noqa: E402

CLIP = b"fake-mp4"


class FakeClient:
    """Stands in for AnexClient: records uploads and the request run."""
    calls = []

    def __init__(self, key, region, on_status=None, is_interrupted=None):
        self.region = region
        self.status = on_status or (lambda _t: None)
        self.uploads = []
        FakeClient.calls.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def upload(self, data, content_type, filename="input"):
        self.uploads.append((filename, content_type, len(data)))
        return f"input:up{len(self.uploads)}"

    task_id = None
    fail_with = None

    async def run(self, body, upscale=False):
        self.body, self.upscale = body, upscale
        self.task_id = "task-1"
        if FakeClient.fail_with is not None:
            raise FakeClient.fail_with
        return "task-1", {"status": "succeeded"}, CLIP

    async def fetch(self, task_id):
        self.fetched = task_id
        return {"status": "succeeded"}, CLIP


@pytest.fixture(autouse=True)
def fake_client(monkeypatch):
    FakeClient.calls = []
    FakeClient.fail_with = None
    monkeypatch.setattr(nodes, "AnexClient", FakeClient)
    monkeypatch.setenv("ANEX_API_KEY", "k")
    for node in nodes.NODES:
        node.GET_SCHEMA()
    yield


def run(node, **kwargs):
    out = asyncio.run(node.execute(**kwargs))
    assert isinstance(out, IO.NodeOutput)
    return FakeClient.calls[-1], out


def image(width=512, height=512):
    return torch.rand(1, height, width, 3)


def a_video(seconds):
    """A real H.264 clip of the given length, as a ComfyUI VIDEO."""
    import av
    buf = io.BytesIO()
    with av.open(buf, mode="w", format="mp4") as out:
        stream = out.add_stream("h264", rate=24)
        stream.width, stream.height, stream.pix_fmt = 320, 240, "yuv420p"
        for _ in range(int(seconds * 24)):
            frame = av.VideoFrame.from_ndarray(
                (torch.rand(240, 320, 3) * 255).byte().numpy(), format="rgb24")
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    buf.seek(0)
    return InputImpl.VideoFromFile(buf)


def test_every_schema_is_valid_and_no_node_is_a_partner_node():
    for node in nodes.NODES:
        schema = node.define_schema()
        schema.validate()
        assert not schema.is_api_node
        assert schema.node_id.startswith("Anex")


def test_text_to_video_body():
    client, out = run(nodes.AnexTextToVideo, model="FastH3-8-Step-V2",
                      prompt="a fox", duration=4, ratio="9:16",
                      resolution="768P", seed=3)
    assert client.body == {
        "model": "FastH3-8-Step-V2",
        "content": [{"type": "text", "text": "a fox"}],
        "resolution": "768P", "duration": 4, "ratio": "9:16", "seed": 3}
    assert client.upscale is False and client.region == "world"
    assert out.result[1] == "task-1"


def test_fasth3_refuses_2k_before_spending_anything():
    with pytest.raises(ValueError, match="768P only"):
        run(nodes.AnexTextToVideo, model="FastH3-8-Step-V2", prompt="a fox",
            duration=4, ratio="16:9", resolution="2K", seed=0)
    assert FakeClient.calls == []


def test_empty_prompt_is_refused():
    with pytest.raises(ValueError, match="prompt is empty"):
        run(nodes.AnexTextToVideo, model="MiniMax-H3", prompt="  ",
            duration=4, ratio="16:9", resolution="768P", seed=0)


def test_frames_upload_and_name_their_roles():
    client, _ = run(nodes.AnexFirstLastFrameToVideo, prompt="p", duration=5,
                    resolution="2K", seed=0, last_frame=image())
    assert [u[:2] for u in client.uploads] == [("last_frame.png", "image/png")]
    assert client.body["model"] == "MiniMax-H3"
    assert client.body["resolution"] == "2K"
    assert client.body["content"][1] == {
        "type": "image_url", "role": "last_frame",
        "image_url": {"url": "input:up1"}}
    assert "ratio" not in client.body


def test_frames_need_one_frame_and_a_valid_size():
    with pytest.raises(ValueError, match="first frame, a last frame"):
        run(nodes.AnexFirstLastFrameToVideo, prompt="p", duration=5,
            resolution="768P", seed=0)
    with pytest.raises(ValueError, match="width / height"):
        run(nodes.AnexFirstLastFrameToVideo, prompt="p", duration=5,
            resolution="768P", seed=0, first_frame=image(1600, 512))


def test_references_keep_their_order_images_then_videos():
    client, _ = run(nodes.AnexReferenceToVideo, prompt="p", duration=6,
                    ratio="adaptive", resolution="768P", seed=0,
                    reference_images={"image_1": image(), "image_2": image()},
                    reference_videos={"video_1": a_video(3)})
    content = client.body["content"]
    assert client.body["model"] == "MiniMax-H3-Ref2VA-SMD-4step"
    assert client.body["ratio"] == "adaptive"
    assert [(c["type"], c.get("role")) for c in content] == [
        ("text", None), ("image_url", "reference_image"),
        ("image_url", "reference_image"), ("video_url", "reference_video")]
    assert content[3]["video_url"]["url"] == "input:up3"
    assert client.uploads[2][:2] == ("video_1.mp4", "video/mp4")


def test_references_are_checked_before_upload():
    with pytest.raises(ValueError, match="at least one reference"):
        run(nodes.AnexReferenceToVideo, prompt="p", duration=6,
            ratio="adaptive", resolution="768P", seed=0)
    with pytest.raises(ValueError, match="must be 2 to 15 s"):
        run(nodes.AnexReferenceToVideo, prompt="p", duration=6,
            ratio="adaptive", resolution="768P", seed=0,
            reference_videos={"video_1": a_video(1)})
    assert FakeClient.calls == []


def test_upscale_by_task_id_uploads_nothing():
    client, _ = run(nodes.AnexUpscaleTo2K, task_id="458154338194580776")
    assert client.upscale is True
    assert client.uploads == []
    assert client.body == {"model": "MiniMax-H3", "resolution": "2K",
                           "source_task_id": "458154338194580776"}


def test_upscale_of_a_video_uploads_it_as_the_base_video():
    client, _ = run(nodes.AnexUpscaleTo2K, video=a_video(4), region="eu")
    assert client.region == "eu"
    assert client.body["content"] == [{
        "type": "video_url", "role": "base_video",
        "video_url": {"url": "input:up1"}}]


def test_upscale_needs_a_source():
    with pytest.raises(ValueError, match="task_id or a video"):
        run(nodes.AnexUpscaleTo2K)


def test_get_task_fetches_and_costs_nothing():
    client, out = run(nodes.AnexGetTask, task_id=" 458154338194580776 ")
    assert client.fetched == "458154338194580776"
    assert out.result[1] == "458154338194580776"
    with pytest.raises(ValueError, match="task id"):
        run(nodes.AnexGetTask, task_id="  ")


def test_a_node_that_stops_waiting_names_the_task_to_fetch():
    client_mod = importlib.import_module(f"{pack.__name__}.anex_comfy.client")
    FakeClient.fail_with = client_mod.AnexError("lost contact with task task-1")
    with pytest.raises(Exception, match="Get Task node, task id task-1"):
        run(nodes.AnexTextToVideo, model="MiniMax-H3", prompt="a fox",
            duration=4, ratio="16:9", resolution="768P", seed=0)
    # A task that failed on our side has no clip to fetch.
    FakeClient.fail_with = client_mod.TaskEnded("task task-1 failed: refused")
    with pytest.raises(Exception) as exc:
        run(nodes.AnexTextToVideo, model="MiniMax-H3", prompt="a fox",
            duration=4, ratio="16:9", resolution="768P", seed=0)
    assert "Get Task" not in str(exc.value)


def test_the_empty_key_field_says_whether_the_environment_has_the_key(monkeypatch):
    def field():
        inputs = nodes.AnexTextToVideo.define_schema().inputs
        return next(i for i in inputs if i.id == "api_key")

    monkeypatch.setenv("ANEX_API_KEY", "k")
    assert field().display_name == "api_key (empty: ANEX_API_KEY is set)"
    assert "using ANEX_API_KEY" in field().placeholder
    monkeypatch.delenv("ANEX_API_KEY")
    assert field().display_name == "api_key (ANEX_API_KEY not set)"
    assert "set ANEX_API_KEY and restart" in field().placeholder


def test_the_status_line_names_the_key_source():
    seen = []
    original = FakeClient.__init__

    def init(self, key, region, on_status=None, is_interrupted=None):
        original(self, key, region, on_status=seen.append,
                 is_interrupted=is_interrupted)

    FakeClient.__init__ = init
    try:
        run(nodes.AnexTextToVideo, model="MiniMax-H3", prompt="a fox",
            duration=4, ratio="16:9", resolution="768P", seed=0)
    finally:
        FakeClient.__init__ = original
    assert any("key from ANEX_API_KEY" in line for line in seen)
