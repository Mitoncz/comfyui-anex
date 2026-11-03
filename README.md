# anex.sh video for ComfyUI

Nodes that make video with sound on [anex.sh](https://anex.sh) from inside
ComfyUI, billed to your anex.sh credits. Nothing runs on your GPU: each node
sends a request to anex.sh, waits for the clip and hands it on as a ComfyUI
`VIDEO`, audio included.

| node | model | does |
|---|---|---|
| anex.sh Text to Video | MiniMax H3 (50 steps) or FastH3-8-Step-V2 (8 steps) | a clip from a prompt |
| anex.sh First/Last Frame to Video | MiniMax H3 | a clip that starts on one image, ends on another, or both |
| anex.sh Reference to Video / Edit Video | MiniMax H3 Ref2VA (4-step) | people, products or places from up to 9 images in a new clip, or up to 3 videos edited as the prompt describes |
| anex.sh Upscale Video to 2K | | a finished 768P clip at 2K |
| anex.sh Get Task | | the clip of a task made earlier, at no charge |

Clips are 768P (768 pixels on the short side) or 2K, 24 frames per second,
4 to 15 seconds. Every node also outputs the anex.sh `task_id`: connect it to
Upscale Video to 2K to upscale that clip without sending it back.

## Install

Search for "anex.sh video" in ComfyUI-Manager, or run
`comfy node install comfyui-anex`. To install by hand, put this folder in
`ComfyUI/custom_nodes/` (for example
`git clone https://github.com/Mitoncz/comfyui-anex ComfyUI/custom_nodes/comfyui-anex`)
and restart ComfyUI. It needs nothing beyond ComfyUI's own packages.

The nodes are under "anex.sh/video" in the node library. Three ready
workflows open from the Templates panel under "comfyui-anex", or by dragging
a file from `example_workflows/` onto the canvas:

- `anex_text_to_video`: a 4 s FastH3 clip, then the same clip upscaled to 2K.
- `anex_first_last_frame`: a clip that starts on an image you load.
- `anex_edit_video`: a video you load, edited with an image as a reference.

## Your API key

Create a key on the Keys page of the [console](https://anex.sh) and start
ComfyUI with it in the environment:

```bash
export ANEX_API_KEY=<your key>
python main.py
```

Each node also has an `api_key` field under its advanced inputs. A key typed
there is saved inside the workflow and in the metadata of the files ComfyUI
writes, so anyone you share either with can use it. The environment variable
keeps it out of both.

Organizations that keep their data in the EEA (European Economic Area) set
`region` to `eu` on each node: the clip is then made and stored in the EU.

## Prices

The node's status line shows the clip's list price before it starts:

| | per second of video | per reference image | per second of reference video |
|---|---|---|---|
| MiniMax H3 | $0.08 | first 5 free, then $0.04 | |
| MiniMax H3 Ref2VA (4-step) | $0.08 | first 5 free, then $0.04 | $0.08 |
| FastH3-8-Step-V2 | $0.04 | | |
| 2K | $0.05 on top of the rate above | | $0.05 on top |
| Upscale Video to 2K | $0.05 | | |

A reference video bills the part the model reads: its length or the clip's,
whichever is shorter. Failed, refused and cancelled-while-queued requests
cost nothing. Full details: [anex.sh/docs/video](https://anex.sh/docs/video#pricing).

## Cancelling

Cancel the run in ComfyUI and the node cancels its anex.sh task. A task that
is still queued is dropped at no charge. One that has started rendering
finishes and is billed, because the render cannot be stopped part way.

## Closing ComfyUI mid-run

The clip keeps rendering on anex.sh. If ComfyUI is still running when you
come back, the node carries on and picks the clip up. If ComfyUI was closed
or restarted, put the task id (shown under the node and in its error
message) into an **anex.sh Get Task** node: it waits for the task if it has
not finished and loads its clip, at no charge. Clips are kept 30 days.

## Inputs you supply

Images and videos are uploaded to anex.sh storage (never to a third party)
and kept for 24 hours.

- Images: 256 to 5,760 pixels a side, width / height from 0.4 to 2.5.
- Reference videos: 2 to 15 seconds each, 15 seconds in total, 50 MB each.
  A video that is not H.264 MP4 is re-encoded before upload.

Every request is checked against the use rules of the MiniMax H3 licence
before anything is generated; a refused request fails the node with the
reason and is not billed. See the [terms](https://anex.sh/terms).

## Environment variables

| variable | for |
|---|---|
| `ANEX_API_KEY` | your API key |
| `ANEX_TASKS_URL`, `ANEX_API_URL` | other hosts in place of the region's own (testing) |

## Licence

MIT. The models' own licences apply to the video you make with them; see
[anex.sh/docs/video](https://anex.sh/docs/video).
