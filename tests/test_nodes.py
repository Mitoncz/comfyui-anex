"""The nodes' requests, against ComfyUI's own node API. Skipped where
ComfyUI is not importable; run them with ComfyUI on the path:

    PYTHONPATH=/path/to/ComfyUI /path/to/ComfyUI/.venv/bin/python -m pytest tests
"""
import asyncio
import importlib
import io
import json
import os
import sys

import pytest

pytest.importorskip("comfy_api.latest")
torch = pytest.importorskip("torch")

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(PACK))
pack = importlib.import_module(os.path.basename(PACK))  # as ComfyUI loads it
nodes = importlib.import_module(f"{pack.__name__}.anex_comfy.nodes")
catalog = importlib.import_module(f"{pack.__name__}.anex_comfy.catalog")

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
    # The built-in list, as for a pack that cannot reach anex.sh; never the
    # network.
    monkeypatch.setattr(catalog, "_loaded", list(catalog.BUILTIN))
    monkeypatch.setattr(catalog, "_live", None)
    monkeypatch.setattr(catalog, "_upscale_usd", None)
    monkeypatch.setattr(catalog, "_upscale_4k_usd", None)
    monkeypatch.setattr(catalog, "_next_refresh", float("inf"))

    def offline():
        raise OSError("the tests never reach anex.sh")
    monkeypatch.setattr(catalog, "_fetch", offline)
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
    with pytest.raises(ValueError, match="FastH3-8-Step-V2 makes 768P only"):
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
    with pytest.raises(ValueError, match="at least one reference image, audio"):
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


def an_audio(seconds, rate=16000):
    return {"waveform": torch.zeros(1, 1, int(seconds * rate)),
            "sample_rate": rate}


def test_reference_audio_sent_as_wav_between_images_and_videos():
    client, _ = run(nodes.AnexReferenceToVideo, prompt="p", duration=6,
                    ratio="adaptive", resolution="768P", seed=0,
                    reference_images={"image_1": image()},
                    reference_videos={"video_1": a_video(3)},
                    reference_audios={"audio_1": an_audio(4),
                                      "audio_2": an_audio(2)})
    content = client.body["content"]
    assert [(c["type"], c.get("role")) for c in content] == [
        ("text", None), ("image_url", "reference_image"),
        ("audio_url", "reference_audio"), ("audio_url", "reference_audio"),
        ("video_url", "reference_video")]
    assert [u[:2] for u in client.uploads] == [
        ("image_1.png", "image/png"), ("audio_1.wav", "audio/wav"),
        ("audio_2.wav", "audio/wav"), ("video_1.mp4", "video/mp4")]
    assert content[2]["audio_url"]["url"] == "input:up2"


def test_audio_alone_is_a_reference_request():
    client, _ = run(nodes.AnexReferenceToVideo, prompt="p", duration=6,
                    ratio="adaptive", resolution="768P", seed=0,
                    reference_audios={"audio_1": an_audio(5)})
    assert [c["type"] for c in client.body["content"]] == ["text", "audio_url"]


def test_reference_audio_checked_before_upload():
    with pytest.raises(ValueError, match="Audio 1 is 1.0 s"):
        run(nodes.AnexReferenceToVideo, prompt="p", duration=6,
            ratio="adaptive", resolution="768P", seed=0,
            reference_audios={"audio_1": an_audio(1)})
    with pytest.raises(ValueError, match="total"):
        run(nodes.AnexReferenceToVideo, prompt="p", duration=6,
            ratio="adaptive", resolution="768P", seed=0,
            reference_audios={"audio_1": an_audio(8), "audio_2": an_audio(8)})
    assert FakeClient.calls == []


# --------------------------------------------------------------------------- #
# Model menus read from the anex.sh catalog
# --------------------------------------------------------------------------- #
GENERATORS = {"text": nodes.AnexTextToVideo,
              "frames": nodes.AnexFirstLastFrameToVideo,
              "reference": nodes.AnexReferenceToVideo}


def _video(name, kinds, resolutions=("768P", "2K"), seed=False):
    return {"name": name, "names": [name], "kinds": list(kinds),
            "resolutions": list(resolutions), "supports_seed": seed}


def release_catalog():
    """The catalog of this release, plus a new text model; MiniMax-H3 is two
    models, told apart by the kind of request."""
    def fees(output, upscale=None, images=False):
        return {"video_output_usd_per_second": output,
                "video_upscale_usd_per_second": upscale,
                "video_input_image_usd": "0.034" if images else None,
                "video_free_images": 5 if images else 0}
    return {"models": [
        {"id": "minimax-h3", "video": _video("MiniMax-H3", ["text", "frames"]),
         "media_fees": fees("0.068", "0.042", images=True)},
        {"id": "minimax-h3-ref2va",
         "video": _video("MiniMax-H3", ["reference"], seed=True),
         "media_fees": fees("0.068", "0.042", images=True)},
        {"id": "minimax-h3-ref2va-smd-4step",
         "video": _video("MiniMax-H3-Ref2VA-SMD-4step", ["reference"]),
         "media_fees": fees("0.051", "0.059", images=True)},
        {"id": "fasth3-8-step-v2",
         "video": _video("FastH3-8-Step-V2", ["text"], ["768P"]),
         "media_fees": fees("0.017")},
        {"id": "some-new-model",
         "video": _video("Some-New-Model", ["text"], ["768P"], seed=True),
         "media_fees": fees("0.02")},
    ]}


@pytest.fixture
def release(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "default_cache_path",
                        lambda: str(tmp_path / "c.json"))
    catalog.load(fetch=release_catalog)
    # No refresh unless a test asks for one.
    monkeypatch.setattr(catalog, "_next_refresh", float("inf"))


def model_input(node):
    return next(i for i in node.define_schema().inputs if i.id == "model")


def run_seen(node, **kwargs):
    """Run the node; returns the client and the list price it showed."""
    prices = []
    original = nodes._status_fn

    def status_fn(node_id, price):
        prices.append(price)
        return original(node_id, price)
    nodes._status_fn = status_fn
    try:
        client, _ = run(node, **kwargs)
    finally:
        nodes._status_fn = original
    return client, prices[-1]


REQUESTS = {
    "text": dict(prompt="a fox", duration=4, ratio="16:9", seed=7),
    "frames": dict(prompt="p", duration=4, seed=7, first_frame=image()),
    "reference": dict(prompt="p", duration=4, ratio="adaptive", seed=7,
                      reference_images={"image_1": image()}),
}


def test_each_generator_offers_the_catalogs_models(release):
    for kind, node in GENERATORS.items():
        menu = model_input(node)
        assert menu.options == [m.name for m in catalog.models(kind)], kind
        assert menu.default == catalog.default(kind), kind
    assert model_input(nodes.AnexTextToVideo).options == [
        "MiniMax-H3", "FastH3-8-Step-V2", "Some-New-Model"]
    assert model_input(nodes.AnexReferenceToVideo).options == [
        "MiniMax-H3", "MiniMax-H3-Ref2VA-SMD-4step"]
    assert model_input(nodes.AnexReferenceToVideo).default == \
        "MiniMax-H3-Ref2VA-SMD-4step"
    for node in GENERATORS.values():
        node.define_schema().validate()


@pytest.mark.parametrize("kind,model,price,seed", [
    ("text", "MiniMax-H3", "$0.27", False),
    ("text", "Some-New-Model", "$0.08", True),
    ("frames", "MiniMax-H3", "$0.27", False),
    ("reference", "MiniMax-H3", "$0.27", True),
    ("reference", "MiniMax-H3-Ref2VA-SMD-4step", "$0.20", False),
])
def test_the_request_and_price_follow_the_model(release, kind, model, price,
                                                seed):
    client, seen = run_seen(GENERATORS[kind], model=model, resolution="768P",
                            **REQUESTS[kind])
    assert client.body["model"] == model
    assert ("seed" in client.body) is seed
    if seed:
        assert client.body["seed"] == 7
    assert seen == price


def test_2k_on_a_model_without_it_is_refused_before_anything_is_sent(release):
    with pytest.raises(ValueError, match="Some-New-Model makes 768P only"):
        run(nodes.AnexTextToVideo, model="Some-New-Model", resolution="2K",
            **REQUESTS["text"])
    assert FakeClient.calls == []
    client, _ = run(nodes.AnexReferenceToVideo, model="MiniMax-H3",
                    resolution="2K", **REQUESTS["reference"])
    assert client.body["resolution"] == "2K"


def test_the_builtin_list_behaves_as_before():
    """A pack that cannot reach anex.sh: the menus hold the models the nodes
    always sent, and every request carries its seed, as before."""
    assert model_input(nodes.AnexTextToVideo).options == [
        "MiniMax-H3", "FastH3-8-Step-V2"]
    assert model_input(nodes.AnexFirstLastFrameToVideo).options == [
        "MiniMax-H3"]
    assert model_input(nodes.AnexReferenceToVideo).options == [
        "MiniMax-H3-Ref2VA-SMD-4step"]
    for kind, node in GENERATORS.items():
        assert model_input(node).default == catalog.default(kind)
        client, seen = run_seen(node, model=catalog.default(kind),
                                resolution="768P", **REQUESTS[kind])
        assert client.body["seed"] == 7
        assert seen.startswith("$")


def test_a_node_saved_without_a_model_takes_the_default(release):
    for kind in ("frames", "reference"):
        client, _ = run(GENERATORS[kind], resolution="768P", **REQUESTS[kind])
        assert client.body["model"] == catalog.default(kind)


def test_a_kind_with_no_model_still_loads_and_lets_anex_sh_answer(monkeypatch):
    monkeypatch.setattr(catalog, "_loaded", [catalog.BUILTIN[1]])
    menu = model_input(nodes.AnexReferenceToVideo)
    assert menu.options == ["MiniMax-H3-Ref2VA-SMD-4step"]
    nodes.AnexReferenceToVideo.define_schema().validate()
    client, seen = run_seen(nodes.AnexReferenceToVideo, resolution="2K",
                            **REQUESTS["reference"])
    assert client.body["model"] == "MiniMax-H3-Ref2VA-SMD-4step"
    assert client.body["seed"] == 7
    assert seen == "$0.44"  # 4 s at 2K, priced at its built-in rates


def test_a_model_with_no_known_price_says_so(monkeypatch):
    monkeypatch.setattr(catalog, "_loaded", [])
    client, seen = run_seen(nodes.AnexTextToVideo, model="Some-Other-Model",
                            resolution="2K", **REQUESTS["text"])
    assert client.body["model"] == "Some-Other-Model"
    assert seen == "price not listed"


def _widget_names(node):
    """The node's widgets in the order the ComfyUI frontend creates them and
    restores a saved workflow's positional `widgets_values` onto them: the
    required inputs, then the optional ones, and after a widget with
    control_after_generate the value of that control."""
    widget_types = (IO.String.Input, IO.Int.Input, IO.Float.Input,
                    IO.Boolean.Input, IO.Combo.Input)
    by_id = {i.id: i for i in node.define_schema().inputs}
    names = []
    for group in ("required", "optional"):
        for name in node.INPUT_TYPES().get(group, {}):
            spec = by_id.get(name)
            if isinstance(spec, widget_types):
                names.append(name)
                if getattr(spec, "control_after_generate", None):
                    names.append(f"{name}:control")
    return names


def test_workflows_saved_before_the_menu_keep_their_values(release):
    """The example workflows were saved by the pack before the First/Last
    Frame and Reference nodes had a model menu. Their values must land on
    the same inputs, and the menu, past the end of the saved values, keeps
    its default."""
    folder = os.path.join(PACK, "example_workflows")
    seen = set()
    for name in sorted(os.listdir(folder)):
        with open(os.path.join(folder, name)) as f:
            workflow = json.load(f)
        for saved in workflow["nodes"]:
            node = getattr(nodes, saved["type"], None)
            if node not in GENERATORS.values():
                continue
            seen.add(saved["type"])
            widgets = _widget_names(node)
            values = dict(zip(widgets, saved["widgets_values"]))
            assert values["resolution"] in ("768P", "2K")
            assert values["seed:control"] == "randomize"
            assert values["region"] in ("world", "eu")
            if node is nodes.AnexTextToVideo:
                assert widgets[0] == "model"
                assert values["model"] == "FastH3-8-Step-V2"
            else:
                assert widgets[-1] == "model"
                assert len(saved["widgets_values"]) == len(widgets) - 1
    assert seen == {"AnexTextToVideo", "AnexFirstLastFrameToVideo",
                    "AnexReferenceToVideo"}


def test_a_run_refreshes_prices_but_not_the_menus(release, monkeypatch):
    doc = release_catalog()
    for m in doc["models"]:
        if m["id"] == "minimax-h3":
            m["media_fees"]["video_output_usd_per_second"] = "0.1"
    doc["models"] = [m for m in doc["models"] if m["id"] != "some-new-model"]
    doc["models"].append({"id": "seedvr2-3b", "media_fees": {
        "video_output_usd_per_second": "0.03"}})
    monkeypatch.setattr(catalog, "_fetch", lambda: doc)
    monkeypatch.setattr(catalog, "_next_refresh", 0.0)
    menu = model_input(nodes.AnexTextToVideo).options
    client, seen = run_seen(nodes.AnexTextToVideo, model="MiniMax-H3",
                            resolution="768P", **REQUESTS["text"])
    assert seen == "$0.40"  # 4 s at the refreshed $0.10
    assert model_input(nodes.AnexTextToVideo).options == menu
    # A model the refreshed catalog dropped still runs, with no price.
    client, seen = run_seen(nodes.AnexTextToVideo, model="Some-New-Model",
                            resolution="768P", **REQUESTS["text"])
    assert client.body["model"] == "Some-New-Model"
    assert client.body["seed"] == 7
    assert seen == "price not listed"
    # The Upscale node shows the refreshed upscaler price too.
    _, seen = run_seen(nodes.AnexUpscaleTo2K, task_id="458154338194580776")
    assert seen == "$0.03 a second"


def test_a_failed_refresh_never_fails_a_run(release, monkeypatch):
    monkeypatch.setattr(catalog, "_next_refresh", 0.0)
    client, seen = run_seen(nodes.AnexReferenceToVideo, model="MiniMax-H3",
                            resolution="768P", **REQUESTS["reference"])
    assert seen == "$0.27"
    _, seen = run_seen(nodes.AnexUpscaleTo2K, task_id="458154338194580776")
    assert seen == "$0.0425 a second"


def test_resolution_options_follow_the_catalog(monkeypatch):
    def options(node):
        schema = node.GET_SCHEMA()
        return next(i.options for i in schema.inputs
                    if i.id == "resolution")
    assert options(nodes.AnexTextToVideo) == list(catalog.resolutions())
    assert options(nodes.AnexUpscaleTo2K) == ["2K"]
    monkeypatch.setattr(catalog, "_upscale_4k_usd", __import__(
        "decimal").Decimal("0.068"))
    assert options(nodes.AnexUpscaleTo2K) == ["2K", "4K"]
    client, _ = run(nodes.AnexUpscaleTo2K, task_id="t", resolution="4K")
    assert client.body["resolution"] == "4K"
    monkeypatch.setattr(catalog, "_upscale_4k_usd", None)
    with pytest.raises(ValueError, match="4K"):
        run(nodes.AnexUpscaleTo2K, task_id="t", resolution="4K")
