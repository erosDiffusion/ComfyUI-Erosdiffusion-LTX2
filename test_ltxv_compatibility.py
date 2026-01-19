
import torch
import sys

# Mock LTXVAddGuide / comfy_extras/nodes_lt.py logic
def conditioning_get_any_value(cond, key, default):
    print(f"Checking cond of type {type(cond)}")
    if not cond:
        return default
    
    # Logic from nodes_lt.py that caused crash
    for i, t in enumerate(cond):
        print(f"  Item {i}: type {type(t)}")
        # The crash line:
        try:
            if key in t[1]: 
                return t[1][key]
        except KeyError as e:
            print(f"  CRASHED accessing t[1]: {e}")
            raise
        except TypeError as e:
             print(f"  CRASHED accessing t[1] (TypeError): {e}")
             raise
    return default

def test_structures():
    print("=== Testing Standard Conditioning ===")
    tensor = torch.zeros((1, 128, 768))
    meta = {"pooled_output": torch.zeros((1, 768))}
    std_cond = [[tensor, meta]]
    
    try:
        val = conditioning_get_any_value(std_cond, "keyframe_idxs", None)
        print("Standard Cond: Success (Found/Default)")
    except Exception as e:
        print(f"Standard Cond: FAILED {e}")

    print("\n=== Testing KJ-Style (List of Dicts) ===")
    kj_cond_item = {"t5xxl": tensor, "pooled": meta}
    kj_cond = [kj_cond_item]
    
    try:
        val = conditioning_get_any_value(kj_cond, "keyframe_idxs", None)
        print("KJ Cond: Success")
    except Exception as e:
        print(f"KJ Cond: FAILED as expected {e}")

    print("\n=== Testing My Fix (Explicit Wrap) ===")
    # My _encode_prompt wrapper
    fixed_cond = [[tensor, {"pooled_output": meta}]]
    try:
         val = conditioning_get_any_value(fixed_cond, "keyframe_idxs", None)
         print("Fixed Cond: Success")
    except Exception as e:
         print(f"Fixed Cond: FAILED {e}")

if __name__ == "__main__":
    test_structures()
