> [!CAUTION]
> This is a **pre-alpha** node. 
> It is **not stable** and ** **in progress** and not meant for usage.
> At the moment it's also broken.
> I am only sharing upon request in the banodoco server and for timeline editor node for ttps://github.com/vrgamegirl19/comfyui-vrgamedevgirl to possibly integrate in her wf.

# ComfyUI-Erosdiffusion-LTX2

Custom ComfyUI nodes for extending video scenes with synchronized audio generation using LTXAVModel.

## Nodes

### LTXVSceneExtender
All-in-one node for extending video with synchronized audio, image guides, and timestamped prompts.

### LTXVTimelineEditor (Coming Soon)
Visual timeline editor for creating scene scripts.

## Installation

1. Clone or copy this folder to your ComfyUI `custom_nodes` directory
2. Restart ComfyUI

## Requirements

- ComfyUI (latest)
- LTXVideo model
- LTXAVModel (for audio-video generation)

## Usage

See the [analysis document](docs/analysis.md) for detailed usage examples and script format specification.

## Script Format

```
[MM:SS-MM:SS] Scene description | audio:spec | first:$0 | MM:SS:$1 | end:$2

# Audio specs:
#   audio:silent              - No speech/sound
#   audio:"dialogue text"     - Speech to generate
#   audio:ambient             - Ambient sounds only

# Image refs:
#   $0, $1, etc.              - Reference to guide_images batch by index
#   first:                    - Guide at first frame
#   end:                      - Guide at last frame
#   MM:SS:                    - Guide at specific timestamp
```

## Example

```
# === SHOT 1: WOMAN INTRODUCTION ===

[00:00-00:02] Closeup of woman's face, neutral expression | audio:silent | first:$0 | end:$1
[00:02-00:04] Cowboy shot of woman speaking | audio:"Hello, welcome" | first:$2 | 00:03:$3 | end:$4
[00:04-00:06] Side profile, nodding gently | audio:silent | first:$5 | end:$6
```

## License

MIT
