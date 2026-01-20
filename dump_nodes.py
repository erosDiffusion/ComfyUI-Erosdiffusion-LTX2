import json
import urllib.request

COMFYUI_URL = "http://127.0.0.1:8188"

def dump_node_info(node_types):
    info = {
        "LINK_TYPES": ["MODEL", "CLIP", "VAE", "CONDITIONING", "IMAGE", "LATENT", "MASK", "SIGMAS", "GUIDANCE", "CONTROL_NET", "STYLE_MODEL", "UPSCALE_MODEL", "FREEU_STRIDE", "BBOX", "SEGS", "SAM_MODEL", "CANVAS"]
    }
    for nt in node_types:
        try:
            with urllib.request.urlopen(f"{COMFYUI_URL}/object_info/{nt}") as response:
                info[nt] = json.loads(response.read().decode()).get(nt, {})
        except Exception as e:
            info[nt] = {"error": str(e)}
            
    with open("node_info_dump.json", "w") as f:
        json.dump(info, f, indent=2)
    print("Dumped node info to node_info_dump.json")

if __name__ == "__main__":
    dump_node_info(["TextEncodeQwenImageEditPlus", "ImageResizeKJv2", "KSampler", "LoraLoaderModelOnly", "CLIPLoader", "UNETLoader"])
