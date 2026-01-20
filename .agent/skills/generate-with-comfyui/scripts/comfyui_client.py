#!/usr/bin/env python
"""
ComfyUI Client - Helper script for invoking ComfyUI workflows.

Usage:
    python comfyui_client.py --check-status
    python comfyui_client.py --workflow "path/to/workflow.json" --prompt "your prompt"
    python comfyui_client.py --list-workflows
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

COMFYUI_URL = "http://127.0.0.1:8188"
WORKFLOWS_DIR = Path(r"D:\ComfyUI7\ComfyUI\user\default\workflows")
OUTPUT_DIR = Path(r"D:\ComfyUI7\ComfyUI\output")


def check_status():
    """Check if ComfyUI server is running."""
    try:
        req = urllib.request.Request(f"{COMFYUI_URL}/system_stats")
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            print("ComfyUI is RUNNING")
            print(f"  VRAM Used: {data.get('devices', [{}])[0].get('vram_used', 'N/A')}")
            print(f"  VRAM Total: {data.get('devices', [{}])[0].get('vram_total', 'N/A')}")
            return True
    except urllib.error.URLError:
        print("ComfyUI is NOT RUNNING")
        return False
    except Exception as e:
        print(f"Error checking status: {e}")
        return False


def list_workflows():
    """List available workflows."""
    if not WORKFLOWS_DIR.exists():
        print(f"Workflows directory not found: {WORKFLOWS_DIR}")
        return []
    
    workflows = sorted(WORKFLOWS_DIR.glob("*.json"))
    print(f"Available workflows in {WORKFLOWS_DIR}:\n")
    for i, wf in enumerate(workflows, 1):
        size_kb = wf.stat().st_size / 1024
        print(f"  {i:3}. {wf.name} ({size_kb:.1f} KB)")
    return workflows


def load_workflow(workflow_path):
    """Load workflow JSON from file."""
    path = Path(workflow_path)
    if not path.is_absolute():
        path = WORKFLOWS_DIR / path
    
    if not path.exists():
        raise FileNotFoundError(f"Workflow not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_api_format(workflow):
    """Detect if workflow is in API format vs UI format.
    
    API format: {"node_id": {"class_type": ..., "inputs": {...}}, ...}
    UI format: {"nodes": [...], "links": [...], ...}
    """
    if "nodes" in workflow and "links" in workflow:
        return False  # UI format
    # Check if any top-level key has class_type
    for key, value in workflow.items():
        if isinstance(value, dict) and "class_type" in value:
            return True  # API format
    return False


def find_prompt_node(workflow):
    """Find the prompt/text encoding node in the workflow."""
    prompt_types = ["CLIPTextEncode", "CLIPTextEncodeSDXL", "CLIPTextEncodeFlux"]
    
    for node in workflow.get("nodes", []):
        if node.get("type") in prompt_types:
            return node
    return None


def modify_prompt(workflow, new_prompt):
    """Modify the prompt text in the workflow."""
    node = find_prompt_node(workflow)
    if node:
        if "widgets_values" in node and len(node["widgets_values"]) > 0:
            old_prompt = node["widgets_values"][0]
            node["widgets_values"][0] = new_prompt
            print(f"Modified prompt in node {node['id']} ({node['type']})")
            print(f"  Old: {old_prompt[:50]}..." if len(old_prompt) > 50 else f"  Old: {old_prompt}")
            print(f"  New: {new_prompt[:50]}..." if len(new_prompt) > 50 else f"  New: {new_prompt}")
            return True
    print("Warning: No prompt node found to modify")
    return False


def get_node_info(node_type):
    """Get node input definitions from ComfyUI."""
    try:
        req = urllib.request.Request(f"{COMFYUI_URL}/object_info/{node_type}")
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            return data.get(node_type, {})
    except Exception:
        return {}


def convert_to_api_format(workflow, prompt_text=None):
    """Convert workflow JSON to ComfyUI API format."""
    prompt = {}
    
    # Build link map: link_id -> (source_node_id, source_slot)
    link_map = {}
    for link in workflow.get("links", []):
        link_id, src_node, src_slot, dst_node, dst_slot, link_type = link[:6]
        link_map[link_id] = (src_node, src_slot)
    
    # Cache node info
    node_info_cache = {}
    
    for node in workflow.get("nodes", []):
        node_id = str(node["id"])
        node_type = node["type"]
        
        # Note: We include muted nodes (mode=4) because they may be referenced by other nodes
        
        inputs = {}
        
        # Get node definition for proper input mapping
        if node_type not in node_info_cache:
            node_info_cache[node_type] = get_node_info(node_type)
        node_def = node_info_cache[node_type]
        
        # Get required and optional inputs from node definition
        required_inputs = node_def.get("input", {}).get("required", {})
        optional_inputs = node_def.get("input", {}).get("optional", {})
        all_input_defs = {**required_inputs, **optional_inputs}
        
        # Build list of widget input names (non-linked inputs)
        widget_names = []
        node_inputs = node.get("inputs", [])
        linked_input_names = {inp["name"] for inp in node_inputs if inp.get("link") is not None}
        
        for input_name in all_input_defs:
            if input_name not in linked_input_names:
                widget_names.append(input_name)
        
        # Map widget values to input names
        widget_values = node.get("widgets_values", [])
        widget_idx = 0
        for input_name in widget_names:
            if widget_idx < len(widget_values):
                value = widget_values[widget_idx]
                # If we encounter a seed control value (randomize/fixed/etc), 
                # we skip it and take the next value for the actual input
                if isinstance(value, str) and value in ["fixed", "increment", "decrement", "randomize"]:
                    widget_idx += 1
                    if widget_idx < len(widget_values):
                        value = widget_values[widget_idx]
                
                inputs[input_name] = value
                widget_idx += 1
        
        # Process linked inputs
        for inp in node_inputs:
            link_id = inp.get("link")
            if link_id is not None and link_id in link_map:
                src_node, src_slot = link_map[link_id]
                inputs[inp["name"]] = [str(src_node), src_slot]
        
        # Special case: User explicitly mentioned node 45 is the prompt node
        if node_id == "45" and node_type == "CLIPTextEncode" and prompt_text:
             inputs["text"] = prompt_text
             print(f"Explicitly set prompt for node 45: {prompt_text[:50]}...")
        
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
        print(f"Response: {error_body[:500]}")
        raise


def wait_for_completion(prompt_id, timeout=300):
    """Wait for prompt execution to complete."""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request(f"{COMFYUI_URL}/history/{prompt_id}")
            with urllib.request.urlopen(req) as response:
                history = json.loads(response.read().decode())
                if prompt_id in history:
                    return history[prompt_id]
        except urllib.error.URLError:
            pass
        
        time.sleep(1)
        elapsed = int(time.time() - start_time)
        print(f"\r  Waiting... {elapsed}s", end="", flush=True)
    
    print()
    raise TimeoutError(f"Prompt {prompt_id} did not complete within {timeout}s")


def get_outputs(result):
    """Extract output file paths from execution result."""
    outputs = []
    
    for node_id, output in result.get("outputs", {}).items():
        if "images" in output:
            for img in output["images"]:
                filename = img["filename"]
                subfolder = img.get("subfolder", "")
                file_path = OUTPUT_DIR / subfolder / filename if subfolder else OUTPUT_DIR / filename
                outputs.append({
                    "type": "image",
                    "filename": filename,
                    "path": str(file_path),
                    "node_id": node_id
                })
        
        if "gifs" in output:
            for gif in output["gifs"]:
                filename = gif["filename"]
                subfolder = gif.get("subfolder", "")
                file_path = OUTPUT_DIR / subfolder / filename if subfolder else OUTPUT_DIR / filename
                outputs.append({
                    "type": "video",
                    "filename": filename,
                    "path": str(file_path),
                    "node_id": node_id
                })
    
    return outputs


def run_workflow(workflow_path, prompt_text=None, timeout=300):
    """Run a complete workflow execution."""
    print(f"Loading workflow: {workflow_path}")
    workflow = load_workflow(workflow_path)
    
    # Detect format
    if is_api_format(workflow):
        print("  Detected: API format")
        api_prompt = workflow
        # Modify prompt in API format if needed
        if prompt_text:
            for node_id, node in api_prompt.items():
                if node.get("class_type") in ["CLIPTextEncode", "CLIPTextEncodeSDXL", "CLIPTextEncodeFlux"]:
                    if "text" in node.get("inputs", {}):
                        old = node["inputs"]["text"]
                        node["inputs"]["text"] = prompt_text
                        print(f"Modified prompt in node {node_id}")
                        print(f"  Old: {old[:50]}..." if len(str(old)) > 50 else f"  Old: {old}")
                        print(f"  New: {prompt_text[:50]}..." if len(prompt_text) > 50 else f"  New: {prompt_text}")
                        break
    else:
        print("  Detected: UI format (requires conversion)")
        if prompt_text:
            modify_prompt(workflow, prompt_text)
        print("Converting to API format...")
        api_prompt = convert_to_api_format(workflow, prompt_text)
    
    print("Queueing prompt...")
    prompt_id = queue_prompt(api_prompt)
    print(f"  Prompt ID: {prompt_id}")
    
    print("Waiting for completion...")
    result = wait_for_completion(prompt_id, timeout)
    print("\n  Completed!")
    
    outputs = get_outputs(result)
    
    if outputs:
        print("\nOutputs:")
        for out in outputs:
            print(f"  [{out['type']}] {out['path']}")
    else:
        print("\nNo outputs found in result")
    
    return outputs


def main():
    parser = argparse.ArgumentParser(description="ComfyUI Workflow Client")
    parser.add_argument("--check-status", action="store_true", help="Check if ComfyUI is running")
    parser.add_argument("--list-workflows", action="store_true", help="List available workflows")
    parser.add_argument("--workflow", "-w", type=str, help="Workflow file path or name")
    parser.add_argument("--prompt", "-p", type=str, help="Prompt text to use")
    parser.add_argument("--timeout", "-t", type=int, default=300, help="Timeout in seconds")
    
    args = parser.parse_args()
    
    if args.check_status:
        sys.exit(0 if check_status() else 1)
    
    if args.list_workflows:
        list_workflows()
        sys.exit(0)
    
    if args.workflow:
        if not check_status():
            print("Error: ComfyUI is not running")
            sys.exit(1)
        
        try:
            outputs = run_workflow(args.workflow, args.prompt, args.timeout)
            sys.exit(0 if outputs else 1)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    parser.print_help()


if __name__ == "__main__":
    main()
