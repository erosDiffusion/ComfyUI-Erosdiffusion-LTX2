---
description: Automatically expand a prompt and generate an image with ComfyUI
---

// turbo-all

# Comfy-Create Workflow

Use this workflow to quickly generate an image from a prompt idea. It automatically expands your idea into a hyperdetailed technical prompt and runs it through ComfyUI.

## Steps

1. **Expand Prompt**: Take the user's prompt idea and expand it into a hyperdetailed, cinematic technical prompt suitable for high-end AI generation. Focus on lighting, textures, camera settings (e.g., 85mm, f/1.8), and specific artistic styles.

2. **Generate Image**:
```powershell
D:\ComfyUI7\python_embeded\python.exe "d:\ComfyUI7\ComfyUI\custom_nodes\ComfyUI-Erosdiffusion-LTX2\.agent\skills\generate-with-comfyui\scripts\comfyui_client.py" --workflow "movie 1 - zimage-quick-frame-production.json" --prompt "{{expanded_prompt}}"
```

3. **Display Result**: Parse the absolute path from the command output and display the image directly in the chat using the standard image syntax:
   `![Generated Image](/absolute/path/to/image.png)`

## Usage

Simply type `/comfy-create` followed by your image idea.
