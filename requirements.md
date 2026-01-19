# Requirements tracking for ComfyUI-Erosdiffusion-LTX2

## Initial Requirements (2026-01-19)

### REQ-001: LTXV Audio-Video Extension Node
**Branch:** feature/REQ-001-scene-extender
**Status:** In Progress

Create a new ComfyUI node that extends video AND audio simultaneously:
- Extends existing video and audio using LTXAVModel native generation
- Supports timestamped prompts with image guides at first/end/specific frames
- Handles memory constraints (10GB VRAM target)
- Provides smooth audio blending at chunk transitions
- Uses only seconds for user-facing time inputs (internal frame math abstracted)

### REQ-002: Timeline Editor Node (Secondary)
**Branch:** feature/REQ-002-timeline-editor
**Status:** Planned

Visual timeline interface for creating scene scripts:
- Drag-and-drop timeline with markers
- Image thumbnail previews
- Waveform visualization
- Dialogue text entry
- Export to script format
    
### REQ-003: Node Naming Convention
**Branch:** feature/REQ-001-scene-extender
**Status:** Completed

All custom nodes must follow the naming pattern:
- Prefix: `💜 ` (Purple Heart Emoji + Space)
- Suffix: ` ErosDiffusion` (Space + ErosDiffusion)
Example: `💜 LTXV Scene Extender ErosDiffusion`

## Script Format Specification

```
[MM:SS-MM:SS] SCENE_DESCRIPTION | AUDIO_SPEC | GUIDE_SPECS...

Audio specs:
  audio:silent              - No speech/sound
  audio:"dialogue text"     - Speech to generate
  audio:ambient             - Ambient sounds only

Guide specs:
  first:IMAGE_REF           - Guide at first frame
  end:IMAGE_REF             - Guide at last frame
  MM:SS:IMAGE_REF           - Guide at specific timestamp
  MM:SS.ms:IMAGE_REF        - With millisecond precision

IMAGE_REF:
  $0, $1, etc.              - Reference to guide_images batch
  filename.png              - File path
```
