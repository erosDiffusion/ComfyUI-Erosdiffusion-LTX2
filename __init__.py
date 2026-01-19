"""
ComfyUI-Erosdiffusion-LTX2: Custom nodes for LTXV scene extension with audio.

Provides nodes for extending video scenes with synchronized audio generation,
timestamped prompts, and image guides.
"""

from .scene_extender import LTXVSceneExtender
from .scene_extender_mvp import LTXVSceneExtenderMVP

# V3 API: Export nodes using NODES class variable
NODES = [
    LTXVSceneExtender,
    LTXVSceneExtenderMVP,
]

# Node class mappings for ComfyUI discovery (legacy compatibility)
NODE_CLASS_MAPPINGS = {
    "LTXVSceneExtender": LTXVSceneExtender,
    "LTXVSceneExtenderMVP": LTXVSceneExtenderMVP,
}

# Display name mappings (legacy compatibility)
NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXVSceneExtender": "💜 LTXV Scene Extender ErosDiffusion",
    "LTXVSceneExtenderMVP": "💜 LTXV Scene Extender (MVP) ErosDiffusion",
}

# Web directory for frontend components
WEB_DIRECTORY = "./js"

__all__ = [
    "NODES",
    "NODE_CLASS_MAPPINGS", 
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

