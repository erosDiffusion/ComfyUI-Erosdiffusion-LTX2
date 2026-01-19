import re
import torch
import os
import folder_paths
import nodes
from PIL import Image, ImageOps
import numpy as np

class LTXVTimelineEditor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "script": ("STRING", {
                    "multiline": True, 
                    "default": "",
                    "tooltip": "Visual Timeline Script"
                })
            }
        }
    
    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("script_string", "guide_images")
    FUNCTION = "execute"
    CATEGORY = "ErosDiffusion/LTXV"
    OUTPUT_NODE = False

    DESCRIPTION = "Visually edit scene scripts. Outputs the script string and the batch of images referenced in the script (uploaded via the editor)."

    def execute(self, script):
        # Regex to find image refs: first:filename | end:filename | mid:time:filename
        # We need to extract them, load them, and replace them with $0, $1 etc.
        
        # 1. Parse all image references
        # Format in script: 
        #   first:filename.png
        #   end:filename.png
        #   mid:00:05.00:filename.png
        
        # We'll use a dict to map filename -> index to avoid duplicates? 
        # Or just sequential batch? 
        # User said "we want $number style variables".
        # If I reuse the same image in multiple places, I should probably reuse the index?
        # But for simplicity, let's just make a list of Unique images found.
        
        image_map = {} # val -> index (val could be just filename)
        images_loaded = []
        
        # Helper to load and process image
        def get_image_index(filename):
            filename = filename.strip()
            if not filename: return None
            
            if filename in image_map:
                return f"${image_map[filename]}"
            
            # Load Image
            try:
                # Use standard ComfyUI load logic
                image_path = folder_paths.get_annotated_filepath(filename)
                
                i = Image.open(image_path)
                i = ImageOps.exif_transpose(i)
                image = i.convert("RGB")
                image = np.array(image).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,] # [1, H, W, 3]
                
                images_loaded.append(image)
                idx = len(images_loaded) - 1
                image_map[filename] = idx
                return f"${idx}"
                
            except Exception as e:
                print(f"LTXVTimelineEditor: Failed to load image '{filename}': {e}")
                return None

        # 2. Process Script Line by Line
        new_lines = []
        lines = script.split('\n')
        
        for line in lines:
            # We look for our directives
            # Pattern: | first: (.*?) ($|\|)
            # Pattern: | end: (.*?) ($|\|)
            # Pattern: | mid:(\d+:\d+(?:\.\d+)?): (.*?) ($|\|)
            
            # Since regex is tricky with multiple pipes, let's split by pipe
            parts = [p.strip() for p in line.split('|')]
            new_parts = []
            new_parts.append(parts[0]) # The [Time] Prompt part
            
            for p in parts[1:]:
                if p.startswith("first:"):
                    fname = p[6:]
                    idx_str = get_image_index(fname)
                    if idx_str: new_parts.append(f"first:{idx_str}")
                
                elif p.startswith("end:"):
                    fname = p[4:]
                    idx_str = get_image_index(fname)
                    if idx_str: new_parts.append(f"end:{idx_str}")
                    
                elif p.startswith("mid:"):
                    # mid:time:filename
                    # Split by colon, max 2 splits? 
                    # mid:05:00:file.png -> problem if time has colons
                    # We expect MM:SS format usually.
                    # Let's extract time str and filename
                    # Regex for inner part
                    m = re.match(r"mid:(\d+:\d+(?:\.\d+)?):(.*)", p)
                    if m:
                        time_str = m.group(1)
                        fname = m.group(2)
                        idx_str = get_image_index(fname)
                        if idx_str: new_parts.append(f"mid:{time_str}:{idx_str}")
                    else:
                        new_parts.append(p) # Keep as is if cant parse
                else:
                    new_parts.append(p) # Other directives or unrecognized
            
            new_lines.append(" | ".join(new_parts))
            
        processed_script = "\n".join(new_lines)
        
        # 3. Batch Images
        if not images_loaded:
             # Return empty batch? Or None?
             # If no images, return a dummy small tensor to avoid errors if connected?
             # Or just None? If logic handles it.
             # LTXVSceneExtender "guide_images" is optional.
             empty_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32) # Dummy
             return (processed_script, None) # None is clearer for optional input
             
        # Resize logic for batching
        # We must resize all to match the first image dimensions (or max?)
        # Standard Comfy batching behavior: usually same size required.
        shape = images_loaded[0].shape # [1, H, W, 3]
        target_h, target_w = shape[1], shape[2]
        
        final_list = []
        for img in images_loaded:
            if img.shape[1] != target_h or img.shape[2] != target_w:
                # Resize
                # Permute to [1, 3, H, W] for interpolate
                img_p = img.movedim(-1, 1)
                img_p = nodes.common_upscale(img_p, target_w, target_h, "lanczos", "disabled")
                img_p = img_p.movedim(1, -1)
                final_list.append(img_p)
            else:
                final_list.append(img)
                
        batch = torch.cat(final_list, dim=0)
        
        return (processed_script, batch)
