"""anex.sh video generation nodes for ComfyUI.

ComfyUI imports this directory as a package and calls comfy_entrypoint.
Imported any other way (pytest collecting the tests) it defines nothing.
"""
if __package__:
    from .anex_comfy.nodes import comfy_entrypoint

    __all__ = ["comfy_entrypoint"]
