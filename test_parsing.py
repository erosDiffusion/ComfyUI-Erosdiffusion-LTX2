import sys
sys.path.insert(0, r'D:\ComfyUI7\ComfyUI\custom_nodes\ComfyUI-Erosdiffusion-LTX2')

from script_parser import parse_scene_script

# Test script 1: User's format
script1 = """[00:00.00-00:04.04] the woman turns from left to right, static camera | audio:techno music | first:ComfyUI-zimage-diffusers-wrapper_00213_.png | end:ComfyUI-zimage-diffusers-wrapper_00213_.png
[00:04.04-00:08.00] the woman turns from right to left, static camera  | first:ComfyUI-zimage-diffusers-wrapper_00213_.png | audio:techno music  | end:ComfyUI-zimage-diffusers-wrapper_00212_.png"""

# Test script 2: Missing parts
script2 = """[00:00-00:04] no prompt no guides
[00:04-00:08] just a prompt | first:image.png
[00:08-00:12] | audio:silence"""

print("=== Test 1: User's script ===")
chunks = parse_scene_script(script1)
print(f"Parsed {len(chunks)} chunks:")
for i, c in enumerate(chunks):
    print(f"  Chunk {i}: {c.start_sec:.2f}s-{c.end_sec:.2f}s")
    print(f"    prompt: '{c.prompt[:50]}...'")
    print(f"    audio: '{c.audio_spec}'")
    print(f"    guides: {len(c.guides)}")
    for g in c.guides:
        print(f"      - {g.position}: {g.image_ref}")

print("\n=== Test 2: Missing parts ===")
chunks2 = parse_scene_script(script2)
print(f"Parsed {len(chunks2)} chunks:")
for i, c in enumerate(chunks2):
    print(f"  Chunk {i}: prompt='{c.prompt}', audio='{c.audio_spec}', guides={len(c.guides)}")
