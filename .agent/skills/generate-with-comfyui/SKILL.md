---
name: generate-with-comfyui
description: Invoke ComfyUI workflows to generate images, video, audio, or other content
---

# Generate with ComfyUI Skill

## Purpose

Programmatically invoke ComfyUI workflows to generate content. Modify prompts, queue workflows, wait for completion, and retrieve results.

## Trigger

Use this skill when:
- User requests image/video/audio generation via ComfyUI
- A workflow needs to be executed with custom prompts
- Automated content generation is required

## Prerequisites

- ComfyUI running at `http://127.0.0.1:8188`
- Workflow JSON files in `D:\ComfyUI7\ComfyUI\user\default\workflows\`

## Quick Commands

### Check if ComfyUI is Running

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8188/system_stats" -Method Get
```

### Use Helper Script

```powershell
D:\ComfyUI7\python_embeded\python.exe ".agent\skills\generate-with-comfyui\scripts\comfyui_client.py" --help
```

## Workflow Steps

### Step 1: Verify ComfyUI Status

```powershell
$response = Invoke-RestMethod -Uri "http://127.0.0.1:8188/system_stats" -Method Get -ErrorAction SilentlyContinue
if ($response) { "ComfyUI is running" } else { "ComfyUI is NOT running" }
```

### Step 2: Load Workflow JSON

Load the workflow from `D:\ComfyUI7\ComfyUI\user\default\workflows\`. Example workflow for quick image generation:
- `movie 1 - zimage-quick-frame-production.json`

### Step 3: Modify Prompt

Find the prompt node (typically `CLIPTextEncode`) and modify `widgets_values[0]`:

```python
import json

# Load workflow
with open(workflow_path) as f:
    workflow = json.load(f)

# Find and modify prompt node
for node in workflow["nodes"]:
    if node["type"] == "CLIPTextEncode":
        node["widgets_values"][0] = "Your new prompt here"
        break
```

### Step 4: Queue Prompt

POST the workflow to `/prompt`:

```python
import requests
import json

# Convert workflow to API format (nodes by ID)
prompt = {}
for node in workflow["nodes"]:
    prompt[str(node["id"])] = {
        "class_type": node["type"],
        "inputs": {}  # Map inputs from links
    }

response = requests.post(
    "http://127.0.0.1:8188/prompt",
    json={"prompt": prompt}
)
prompt_id = response.json()["prompt_id"]
```

### Step 5: Wait for Completion

Poll `/history/{prompt_id}` or use WebSocket at `/ws`:

```python
import time

while True:
    history = requests.get(f"http://127.0.0.1:8188/history/{prompt_id}").json()
    if prompt_id in history:
        break
    time.sleep(1)
```

### Step 6: Retrieve Results

Get output files from history:

```python
outputs = history[prompt_id]["outputs"]
for node_id, output in outputs.items():
    if "images" in output:
        for img in output["images"]:
            filename = img["filename"]
            subfolder = img.get("subfolder", "")
            # File at: D:\ComfyUI7\ComfyUI\output\{subfolder}\{filename}
```

## How to Present Results

When the workflow completes, the helper script will provide absolute paths to the generated files. You MUST present these to the USER as follows:

1.  **Images**: Display them directly in the chat using the standard image syntax with absolute paths.
    - Example: `![Generated Image](file:///D:/ComfyUI7/ComfyUI/output/example.png)`
2.  **Videos/Other**: Link them using standard markdown links with absolute paths.
    - Example: `[Download Video](file:///D:/ComfyUI7/ComfyUI/output/example.mp4)`
3.  **Context**: Always mention which workflow was used and what prompt was applied.

## How to Use

To use this skill, follow these steps:

1.  **Select Workflow**: Use the helper script to list available workflows or use a known one.
    ```powershell
    D:\ComfyUI7\python_embeded\python.exe ".agent\skills\generate-with-comfyui\scripts\comfyui_client.py" --list-workflows
    ```
2.  **Execute**: Run the helper script with the chosen workflow and prompt.
    ```powershell
    D:\ComfyUI7\python_embeded\python.exe ".agent\skills\generate-with-comfyui\scripts\comfyui_client.py" --workflow "workflow_name.json" --prompt "your detailed prompt"
    ```
3.  **Display**: Capture the output path from the script's output and embed it in your response as described above.

## API Reference

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/system_stats` | GET | Check server status |
| `/prompt` | POST | Queue workflow execution |
| `/queue` | GET | Check queue status |
| `/history/{prompt_id}` | GET | Get execution results |
| `/view?filename=X&subfolder=Y` | GET | Retrieve output files |
| `/ws` | WebSocket | Real-time execution updates |

## WebSocket Events

Connect to `ws://127.0.0.1:8188/ws` for real-time updates:

- `status` - System status updates
- `execution_start` - Prompt execution begins
- `executing` - Node execution updates
- `progress` - Progress for long operations
- `executed` - Node completed

## Example Workflows

| Workflow | Purpose |
|----------|---------|
| `movie 1 - zimage-quick-frame-production.json` | Quick image generation with Z-Image Turbo |
| `video_ltx2_t2v.json` | Text-to-video with LTX2 |
| `video_ltx2_i2v.json` | Image-to-video with LTX2 |

## Output Location

Generated files saved to: `D:\ComfyUI7\ComfyUI\output\`
