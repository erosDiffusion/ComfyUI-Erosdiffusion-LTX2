---
name: edit-with-comfyui
description: Edit images in ComfyUI using Qwen 2.5VL with support for camera changes and image analysis
---

# Edit with ComfyUI Skill

## Purpose

Specialized skill for editing existing images using ComfyUI and Qwen 2.5VL. Supports prompt-based edits (e.g., transforming objects, backgrounds) and camera changes (zoom, angle shifts).

## Trigger

Use this skill when:
- User wants to modify an existing image (default: Flux Edit)
- User requests camera changes, stylization, or object manipulation
- User mentions "qwen" specifically (triggers Qwen Edit)

> [!NOTE]
> Trigger words like "qwen edit:" or "qwen:" are used only for template selection and are automatically stripped from the prompt before being submitted to ComfyUI.

## Workflow Templates

1.  **Flux Edit (Default)**: `antigravity - movie - flux2edit.json`
    -   Target Node for Prompt: `109` (CLIPTextEncode)
    -   Target Node for Image: `115` (LoadImageFromUrl)
2.  **Qwen Edit**: `movie 1.1 - qwen image edit antigravity.json`
    -   Target Node for Prompt: `68` (TextEncodeQwenImageEditPlus)
    -   Target Node for Image: `117` or `41` (LoadImageFromUrl / LoadImage)

## Prerequisites

- ComfyUI running at `http://127.0.0.1:8188`
- Workflow: `movie 1.1 - qwen image edit antigravity.json`
- Model: Qwen 2.5VL (installed in ComfyUI)

> [!IMPORTANT]
> **Performance**: This workflow uses heavy models (Qwen 2.5VL) and can take several minutes (2-5 mins) to complete a single edit. The client has an extended 20-minute timeout to accommodate this.

## How to Present Results

1.  **Images**: Display edited results directly using `![Edited Image](/absolute/path/to/image.png)`.
2.  **Context**: Explain the changes made (e.g., "Transformed turtle into a dragon, changed background to Rome").

## Workflow Steps

### Step 1: Analyze Image (Optional)
If the edit request is complex, first analyze the source image to identify key elements.

### Step 2: Prepare Edit Prompt
- **Subject Edits**: Describe the transformation clearly (e.g., "transform the turtle into a majestic red dragon"). Narrative prompts work as well (e.g., "change the camera angle to medium closeup from behind").
- **Background Edits**: Specify new environments (e.g., "set in the ruins of ancient Rome").
- **Camera Changes**: Use identifying tags or narrative descriptions. Use the `<sks>` prefix for specific camera commands listed in node 90 if absolute precision is needed.

### Step 3: Run Edit
Use the specialized client to run the edit workflow:
```powershell
D:\ComfyUI7\python_embeded\python.exe ".agent\skills\edit-with-comfyui\scripts\comfyui_edit_client.py" --image "/path/to/source.png" --prompt "your edit prompt"
```
The client will automatically:
1. Copy the source image to the ComfyUI `input` directory.
2. Target node 41 for the image reference.
3. Target node 68 for the edit prompt.

## Camera Changes Reference

Common triggers to include in prompts:
- `<sks> front view`
- `<sks> low-angle shot`
- `<sks> close-up` / `<sks> wide shot`
- `<sks> zoom in` / `<sks> zoom out`
