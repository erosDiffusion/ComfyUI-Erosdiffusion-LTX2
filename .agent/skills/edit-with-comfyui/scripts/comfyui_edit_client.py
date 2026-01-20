import json
import urllib.request
import urllib.error
import urllib.parse
import time
import os
import argparse
import sys

# Configuration
COMFYUI_URL = "http://127.0.0.1:8188"
DEFAULT_WORKFLOW = r"D:\ComfyUI7\ComfyUI\user\default\workflows\antigravity - movie - flux2edit.json"
QWEN_WORKFLOW = r"D:\ComfyUI7\ComfyUI\user\default\workflows\movie 1.1 - qwen image edit antigravity.json"
OUTPUT_DIR = r"D:\ComfyUI7\ComfyUI\output"
INPUT_DIR = r"D:\ComfyUI7\ComfyUI\input"

def prepare_image(image_path):
    """Ensure the image exists in the ComfyUI input directory."""
    if not image_path:
        return None
    
    # If it's already in the input dir, just return the name
    if os.path.dirname(os.path.abspath(image_path)) == os.path.abspath(INPUT_DIR):
        return os.path.basename(image_path)
    
    # If it's an absolute path elsewhere, copy it
    if os.path.isabs(image_path) and os.path.exists(image_path):
        target_path = os.path.join(INPUT_DIR, os.path.basename(image_path))
        if not os.path.exists(target_path):
            import shutil
            print(f"Copying {image_path} to {target_path}")
            shutil.copy2(image_path, target_path)
        return os.path.basename(image_path)
    
    # If it's just a filename, assume it's in input or output
    filename = os.path.basename(image_path)
    if os.path.exists(os.path.join(INPUT_DIR, filename)):
        return filename
    
    # Check output dir (often used as input for next step)
    if os.path.exists(os.path.join(OUTPUT_DIR, filename)):
        target_path = os.path.join(INPUT_DIR, filename)
        import shutil
        print(f"Copying from output to input: {filename}")
        shutil.copy2(os.path.join(OUTPUT_DIR, filename), target_path)
        return filename
        
    return filename

def check_status():
    """Verify ComfyUI is running."""
    try:
        with urllib.request.urlopen(f"{COMFYUI_URL}/system_stats") as response:
            stats = json.loads(response.read().decode())
            print("ComfyUI is RUNNING")
            return True
    except Exception as e:
        print(f"ComfyUI is NOT reachable: {e}")
        return False

def get_node_info(node_type):
    """Fetch node input definitions from ComfyUI."""
    try:
        quoted_type = urllib.parse.quote(node_type)
        with urllib.request.urlopen(f"{COMFYUI_URL}/object_info/{quoted_type}") as response:
            return json.loads(response.read().decode()).get(node_type, {})
    except Exception as e:
        print(f"Warning: Could not fetch info for node {node_type}: {e}")
        return {}

LINK_TYPES = {"MODEL", "CLIP", "VAE", "CONDITIONING", "IMAGE", "LATENT", "MASK", "SIGMAS", "GUIDANCE", "CONTROL_NET", "STYLE_MODEL", "UPSCALE_MODEL", "FREEU_STRIDE", "BBOX", "SEGS", "SAM_MODEL", "CANVAS"}

def convert_to_api_format(workflow, prompt_text=None, image_filename=None):
    """Convert workflow JSON to ComfyUI API format specialized for editing."""
    prompt = {}
    link_map = {}
    for link in workflow.get("links", []):
        if link:
            link_id, src_node, src_slot, dst_node, dst_slot, link_type = link[:6]
            link_map[link_id] = (src_node, src_slot)
    
    node_info_cache = {}
    
    for node in workflow.get("nodes", []):
        node_id = str(node["id"])
        node_type = node["type"]
        
        # Skip UI-only nodes that aren't executable in the backend
        if node_type in ["Note", "MarkdownNote", "Reroute", "PrimitiveNode"]:
            continue
            
        inputs = {}
        if node_type not in node_info_cache:
            node_info_cache[node_type] = get_node_info(node_type)
        node_def = node_info_cache[node_type]
        
        # If we couldn't get node info, it might be a custom node or a group node.
        # ComfyUI doesn't execute group nodes in the backend; they must be skipped.
        # Custom group nodes often have UUID-like names or don't return type info.
        if not node_def:
             # If it's not a known executable class, skip it
             print(f"Warning: Skipping node {node_id} ({node_type}) - potentially a non-executable group or custom node.")
             continue
             
        required_inputs = node_def.get("input", {}).get("required", {})
        optional_inputs = node_def.get("input", {}).get("optional", {})
        all_input_defs = {**required_inputs, **optional_inputs}
        
        # Build list of widget input names
        widget_names = []
        for input_name, input_def in all_input_defs.items():
            # input_def is usually [type, extras]
            if isinstance(input_def, list) and len(input_def) > 0:
                input_type = input_def[0]
            else:
                input_type = input_def
            
            # If input_type is a list, it's a COMBO widget
            if isinstance(input_type, list):
                widget_names.append(input_name)
            elif isinstance(input_type, str) and input_type not in LINK_TYPES:
                widget_names.append(input_name)
        
        widget_values = node.get("widgets_values", [])
        if isinstance(widget_values, dict):
            # For nodes that save widgets as an object (e.g. LoadImageFromUrl)
            for k, v in widget_values.items():
                if k != "$$canvas-image-preview": # Skip UI-only fields
                   inputs[k] = v
        else:
            # Standard list-based widgets
            widget_idx = 0
            for input_name in widget_names:
                if widget_idx < len(widget_values):
                    value = widget_values[widget_idx]
                    if isinstance(value, str) and value in ["fixed", "randomize", "increment", "decrement"]:
                        widget_idx += 1
                        if widget_idx < len(widget_values):
                            value = widget_values[widget_idx]
                    inputs[input_name] = value
                    widget_idx += 1
        
        # Process linked inputs
        node_inputs = node.get("inputs", [])
        for inp in node_inputs:
            link_id = inp.get("link")
            if link_id is not None and link_id in link_map:
                src_node, src_slot = link_map[link_id]
                inputs[inp["name"]] = [str(src_node), src_slot]

        # Template-specific Logic for Editing
        # Flux Edit (Node 115 image, 109 prompt)
        if node_id == "109" and node_type == "CLIPTextEncode" and prompt_text:
            inputs["text"] = prompt_text
            print(f"Set Flux edit prompt on node 109: {prompt_text[:50]}...")
        
        if node_id == "115" and node_type == "LoadImageFromUrl" and image_filename:
            import random
            rand_val = random.random()
            url = f"http://127.0.0.1:8188/api/view?type=input&filename={image_filename}&subfolder=&rand={rand_val}"
            inputs["image"] = url
            print(f"Set Flux source image URL on node 115: {url}")

        # Qwen Edit (Node 68 prompt, 117/41 image)
        if node_id == "68" and node_type == "TextEncodeQwenImageEditPlus" and prompt_text:
            inputs["prompt"] = prompt_text
            print(f"Set Qwen edit prompt on node 68: {prompt_text[:50]}...")
            
        if node_id == "41" and node_type == "LoadImage" and image_filename:
            inputs["image"] = image_filename
            print(f"Set Qwen source image on node 41: {image_filename}")
            
        if node_id == "117" and node_type == "LoadImageFromUrl" and image_filename:
            import random
            rand_val = random.random()
            url = f"http://127.0.0.1:8188/api/view?type=input&filename={image_filename}&subfolder=&rand={rand_val}"
            inputs["image"] = url
            print(f"Set Qwen source image URL on node 117: {url}")

        # Seed Randomization to force execution
        if "seed" in inputs and node_type in ["KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced"]:
             import random
             inputs["seed"] = random.randint(1, 1125899906842624)
             print(f"Randomized seed for node {node_id} ({node_type})")
             
        # Support flux2 RandomNoise node
        if node_type == "RandomNoise" and "noise_seed" in inputs:
             import random
             inputs["noise_seed"] = random.randint(1, 1125899906842624)
             print(f"Randomized noise_seed for node {node_id} ({node_type})")

        prompt[node_id] = {
            "class_type": node_type,
            "inputs": inputs
        }
    
    return prompt

def queue_prompt(prompt):
    """Queue a prompt for execution."""
    data = json.dumps({"prompt": prompt}).encode("utf-8")
    req = urllib.request.Request(
        f"{COMFYUI_URL}/prompt",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode())
            return result.get("prompt_id")
    except urllib.error.HTTPError as e:
        error_body = e.read().decode()
        print(f"HTTP Error {e.code}: {e.reason}")
        with open("error_response.json", "w") as f:
            try:
                parsed_error = json.loads(error_body)
                json.dump(parsed_error, f, indent=2)
            except Exception:
                f.write(error_body)
        print("Error response saved to error_response.json")
        
        with open("debug_prompt.json", "w") as f:
            json.dump(prompt, f, indent=2)
        print("Full prompt saved to debug_prompt.json")
        raise

def wait_for_completion(prompt_id, timeout=1200):
    """Wait for a prompt to finish execution. Increased timeout for heavy models."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(f"{COMFYUI_URL}/history/{prompt_id}") as response:
                history = json.loads(response.read().decode())
                if prompt_id in history:
                    return history[prompt_id]
        except Exception:
            pass
        time.sleep(2)
        print(f"  Waiting... {int(time.time() - start_time)}s", end='\r')
    raise TimeoutError(f"Prompt {prompt_id} did not complete within {timeout}s")

def get_outputs(result):
    """Extract output file paths from execution result."""
    outputs = []
    if "outputs" in result:
        for node_id, node_output in result["outputs"].items():
            if "images" in node_output:
                for img in node_output["images"]:
                    filename = img["filename"]
                    subfolder = img.get("subfolder", "")
                    full_path = os.path.join(OUTPUT_DIR, subfolder, filename)
                    outputs.append({"node": node_id, "type": "image", "path": full_path})
    return outputs

def main():
    parser = argparse.ArgumentParser(description="ComfyUI Image Edit Client")
    parser.add_argument("--image", help="Path to the source image to edit")
    parser.add_argument("--prompt", help="Edit prompt (e.g., 'transform to dragon')")
    parser.add_argument("--check-status", action="store_true", help="Check ComfyUI status")
    
    args = parser.parse_args()
    
    if args.check_status:
        check_status()
        return
    
    if not check_status():
        sys.exit(1)
        
    if not args.prompt:
        print("Error: --prompt is required for editing")
        sys.exit(1)

    # Template selection logic
    prompt_raw = args.prompt
    is_qwen = "qwen" in prompt_raw.lower()
    
    # Clean the prompt if it contains the trigger
    prompt_clean = prompt_raw
    if is_qwen:
        # Remove common triggers like "qwen edit:", "qwen:", "qwen "
        import re
        prompt_clean = re.sub(r'^qwen\s*edit:\s*', '', prompt_clean, flags=re.IGNORECASE)
        prompt_clean = re.sub(r'^qwen:\s*', '', prompt_clean, flags=re.IGNORECASE)
        prompt_clean = re.sub(r'^qwen\s*', '', prompt_clean, flags=re.IGNORECASE)
        
    workflow_path = QWEN_WORKFLOW if is_qwen else DEFAULT_WORKFLOW
    
    print(f"Loading {'Qwen' if is_qwen else 'Flux'} edit workflow: {workflow_path}")
    if is_qwen:
        print(f"Cleaned prompt for Qwen: {prompt_clean[:50]}...")
        
    if not os.path.exists(workflow_path):
        print(f"Error: Workflow file not found at {workflow_path}")
        sys.exit(1)
        
    with open(workflow_path, 'r', encoding='utf-8') as f:
        workflow = json.load(f)
        
    image_filename = prepare_image(args.image)
    api_prompt = convert_to_api_format(workflow, prompt_clean, image_filename)
    
    print("Queueing edit prompt...")
    prompt_id = queue_prompt(api_prompt)
    print(f"  Prompt ID: {prompt_id}")
    
    print("Waiting for completion...")
    result = wait_for_completion(prompt_id)
    print("\n  Completed!")
    
    outputs = get_outputs(result)
    if outputs:
        print("\nOutputs:")
        for out in outputs:
            print(f"  [{out['type']}] {out['path']}")
    else:
        print("\nNo outputs found.")

if __name__ == "__main__":
    main()
