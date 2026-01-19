"""
LTXVSceneExtender: All-in-one node for extending video scenes with synchronized audio.

Combines LTXVLoopingSampler + LTXVNormalizingSampler + image guides into a single
node that processes timestamped scene scripts.
"""

import copy
from typing import Optional

import torch
from comfy_api.latest import io

# Import ComfyUI components
try:
    import nodes
    import comfy.model_management as mm
    import comfy.utils
    from comfy.nested_tensor import NestedTensor
    COMFY_AVAILABLE = True
except ImportError:
    COMFY_AVAILABLE = False
    NestedTensor = None

# Import LTXV nodes/components
try:
    from comfy_extras.nodes_lt import EmptyLTXVLatentVideo, LTXVAddGuide
    from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced
    
    # Try importing LTXV-specific utilities from the custom node package
    # This tries standard installation paths
    try:
        from custom_nodes.ComfyUI_LTXVideo.easy_samplers import LinearOverlapLatentTransition
        from custom_nodes.ComfyUI_LTXVideo.latents import LTXVAddLatentGuide, LTXVSelectLatents
        LTXV_UTILS_AVAILABLE = True
    except ImportError:
        # Check if we are inside the package (relative import) or handle import differently
        LTXV_UTILS_AVAILABLE = False
except ImportError:
    LTXV_UTILS_AVAILABLE = False

# Local imports
from .time_manager import TimeManager
from .script_parser import (
    parse_scene_script,
    resolve_image_refs,
    get_chunk_guide_images,
    SceneChunk,
)
from .audio_blender import AudioOverlapBlender

# Global Cache to avoid Locked Class issues
_CACHE = {}


def get_noise_mask(latent):
    """Helper to extract noise mask, handling nested tensors."""
    if "noise_mask" in latent:
        nm = latent["noise_mask"]
        if isinstance(nm, NestedTensor):
            nm = nm.tensors[0] # Get Video Mask
        return nm
    return None

class LTXVSceneExtender(io.ComfyNode):
    """
    All-in-one node for extending video scenes with synchronized audio.
    
    Combines temporal tiling, audio normalization, and image guides into
    a single node that processes timestamped scene scripts.
    
    Features:
    - Timestamped prompts with audio specs (silent, ambient, dialogue)
    - Image guides at first/end/specific frames
    - Smooth audio blending at chunk transitions
    - All time inputs in SECONDS (internal frame math abstracted)
    """
    
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXVSceneExtender",
            display_name="💜 LTXV Scene Extender ErosDiffusion",
            category="ErosDiffusion/ltxv",
            description="Extend video with synchronized audio, image guides, and timestamped prompts",
            inputs=[
                # === Model Inputs ===
                io.Model.Input(
                    "model",
                    tooltip="Diffusion model (LTXAVModel for audio-video, or video-only model)"
                ),
                io.Vae.Input("video_vae", tooltip="Video VAE for encoding/decoding"),
                io.Vae.Input(
                    "audio_vae",
                    optional=True,
                    tooltip="Audio VAE (required for audio generation)"
                ),
                io.Sampler.Input("sampler", tooltip="Sampler to use"),
                io.Sigmas.Input("sigmas", tooltip="Sigma schedule"),
                io.Noise.Input("noise", tooltip="Noise source"),
                io.Guider.Input(
                    "guider",
                    tooltip="Guider (e.g., STGGuiderAdvanced)"
                ),
                io.Clip.Input("clip", tooltip="CLIP model for encoding prompts"),
                
                # === Existing Content ===
                io.Latent.Input(
                    "latent",
                    tooltip="Existing video/AV latent to extend (optional for new generation)",
                    optional=True
                ),
                
                # === Extension Configuration ===
                io.Float.Input(
                    "overlap_duration",
                    default=1.0,
                    min=0.5,
                    max=3.0,
                    step=0.1,
                    tooltip="Overlap between chunks for smooth transitions"
                ),
                io.Float.Input(
                    "video_fps",
                    default=25.0,
                    min=1.0,
                    max=60.0,
                    step=1.0,
                    tooltip="Video frame rate"
                ),
                io.Int.Input(
                    "width",
                    default=960,
                    min=64,
                    max=2048,
                    step=32,
                    tooltip="Output video width"
                ),
                io.Int.Input(
                    "height",
                    default=512,
                    min=64,
                    max=2048,
                    step=32,
                    tooltip="Output video height"
                ),
                
                # === Scene Script ===
                io.String.Input(
                    "scene_script",
                    default="""[00:00-00:01] A woman with short hair speaks to camera | audio:"Hello" | first:$0 | end:$0
[00:01-00:03] A woman with short hair turns around and speaks to camera , camera orbits around | audio:"I like this camera" | first:$0 | end:$1
[00:03-00:06] A woman with short hair opens her arms and tilts head backwards happy | audio: "I feel free" | first:$1 | end:$2
[00:06-00:08] A woman with short hair puts her hands in her hair | first:$2""",
                    multiline=True,
                    tooltip="""Timestamped scene script. Format:
[MM:SS-MM:SS] Scene description | audio:spec | first:$0 | MM:SS:$1 | end:$2

Audio specs: audio:silent, audio:ambient, audio:"dialogue text"
Guide refs: $0, $1, etc. reference guide_images batch by index"""
                ),
                
                # === Image Guides ===
                io.Image.Input(
                    "guide_images",
                    optional=True,
                    tooltip="Batch of guide images (referenced as $0, $1, etc.)"
                ),
                io.Float.Input(
                    "guide_strength",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="Strength of image guides"
                ),
                
                # === Audio Controls ===
                io.Float.Input(
                    "audio_overlap_duration",
                    default=0.5,
                    min=0.1,
                    max=2.0,
                    step=0.1,
                    tooltip="Audio overlap for smooth blending at transitions"
                ),
                io.Int.Input(
                    "audio_slope_frames",
                    default=5,
                    min=1,
                    max=20,
                    step=1,
                    tooltip="Crossfade slope length for seamless audio"
                ),
                io.String.Input(
                    "audio_normalization",
                    default="1,1,0.25,1,1,0.25,1,1",
                    tooltip="Per-step audio normalization factors"
                ),
                
                # === Advanced ===
                io.Int.Input("video_overlap_frames", default=8, min=0, max=64),
                io.Int.Input("audio_overlap_frames", default=32, min=0, max=256),
                io.Float.Input(
                    "temporal_cond_strength",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="Conditioning strength from previous tile overlap"
                ),
                io.Float.Input(
                    "adain_factor",
                    default=0.1,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="AdaIN factor to prevent oversaturation"
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.Latent.Output(display_name="video_latent"),
                io.Latent.Output(display_name="audio_latent"),
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
            ],
        )
    
    @classmethod
    def execute(
        cls,
        model,
        video_vae,
        sampler,
        sigmas,
        noise,
        guider,
        clip,
        overlap_duration: float,
        video_fps: float,
        width: int,
        height: int,
        scene_script: str,
        guide_strength: float,
        audio_overlap_duration: float,
        audio_slope_frames: int,
        audio_normalization: str,
        video_overlap_frames: int,
        audio_overlap_frames: int,
        temporal_cond_strength: float,
        adain_factor: float,
        audio_vae=None,
        latent=None,
        guide_images=None,
    ) -> io.NodeOutput:
        batch_size = 1 # Hardcoded
        """Execute the scene extension."""
        
        # Initialize Cache
        global _CACHE
            
        # Clear cache if script fundamentally changed structure?
        # But maybe limit cache size?
        if len(_CACHE) > 50: # Arbitrary limit
             _CACHE.clear()

        # Check if we have an audio-video model
        is_av_model = cls._is_av_model(model)
        
        # Initialize TimeManager
        time_mgr = TimeManager(
            video_fps=video_fps,
            audio_sample_rate=16000 if audio_vae is None else getattr(
                audio_vae.autoencoder, 'sampling_rate', 16000
            ),
            mel_hop_length=160 if audio_vae is None else getattr(
                audio_vae.autoencoder, 'mel_hop_length', 160
            ),
        )
        
        # Parse scene script
        if scene_script.strip():
            chunks = parse_scene_script(scene_script)
        else:
            chunks = cls._generate_default_chunks(
                extension_duration, tile_duration, overlap_duration
            )
            
        print(f"LTXVSceneExtender: Processing {len(chunks)} chunks.")
        
        # --- OPTIMIZATION PHASE: PRE-ENCODING ---
        print("Optimization: Pre-encoding prompts and guides...")
        
        chunk_data, resolved_refs = cls._pre_encode_assets(
            chunks, clip, video_vae, guide_images, width, height, time_mgr
        )
        
        print("Optimization: Encoding complete. Starting Generation Loop.")
        # Trigger soft cache cleanup to free VAE/CLIP memory if possible
        mm.soft_empty_cache()

        # --- GENERATION LOOP ---
        full_video = None
        full_audio = None
        
        # Initialize audio blender if we have audio
        audio_blender = None
        if is_av_model and audio_vae is not None:
            audio_blender = AudioOverlapBlender(
                overlap_frames=int(
                    audio_overlap_duration * time_mgr.config.audio_latents_per_second
                ),
                slope_len=audio_slope_frames,
            )
            
        # Get dimensions
        time_scale_factor, width_scale_factor, height_scale_factor = (
            video_vae.downscale_index_formula
        )
        latent_height = height // height_scale_factor
        latent_width = width // width_scale_factor
        scale_factors = video_vae.downscale_index_formula

        # Keep track of previous latent for extension
        prev_latent = latent
        
        final_video_list = []
        
        positive = None
        negative = None # Store last used for return

        for i, (chunk, c_data) in enumerate(zip(chunks, chunk_data)):
            chunk_duration = chunk.end_sec - chunk.start_sec
            latent_length = max(1, time_mgr.calculate_video_latent_count(chunk_duration)) # Ensure at least 1           
            print(f"Processing chunk {i+1}/{len(chunks)}: [{chunk.start_sec:.1f}s - {chunk.end_sec:.1f}s] ({latent_length} latents / ~{latent_length*8} frames)")
            
            # === CACHING CHECK ===
            prev_context_hash = 0.0
            if prev_latent is not None:
                 if isinstance(prev_latent["samples"], NestedTensor):
                      v_tens = prev_latent["samples"].tensors[0]
                 else:
                      v_tens = prev_latent["samples"]
                 prev_context_hash = float(v_tens.mean().cpu()) # Simple hash
            
            cache_key = (
                i, 
                chunk.prompt,
                chunk.duration,
                chunk.transition_type,
                prev_context_hash,
                width, height
            )
            
            if cache_key in _CACHE:
                 print(f"Using Cached Chunk {i}")
                 chunk_res = _CACHE[cache_key]
                 final_video_list.append(chunk_res)
                 prev_latent = chunk_res
                 continue
                 
            # --- Allocation & Extension Logic ---
            is_extension = (prev_latent is not None) and (chunk.transition_type != "cut")
            
            if not is_extension:
                # NEW GENERATION
                # Basic Mask
                video_mask = torch.ones(
                    (batch_size, 1, latent_length, 1, 1),
                    device=mm.intermediate_device()
                )
                
                video_latent = torch.zeros(
                    [batch_size, 128, latent_length, latent_height, latent_width],
                    device=mm.intermediate_device() if COMFY_AVAILABLE else "cpu"
                )
                
                # Handle AV Model
                if is_av_model and NestedTensor is not None and audio_vae is not None:
                    audio_len = time_mgr.calculate_audio_latent_count(chunk_duration)
                    if audio_len < 1:
                        audio_len = 1
                        print(f"    [DEBUG] NEW GEN: audio_len forced to 1")
                    
                    # Get correct dimensions from VAE (defaults match typical LTXAV values)
                    a_ch = getattr(audio_vae, "latent_channels", 8)
                    a_freq = getattr(audio_vae, "latent_frequency_bins", 16)
                    
                    audio_latent = torch.zeros(
                        [batch_size, a_ch, audio_len, a_freq], 
                        device=mm.intermediate_device() if COMFY_AVAILABLE else "cpu"
                    )
                    
                    nt = NestedTensor((video_latent, audio_latent))
                    input_latent = {"samples": nt}
                    
                    # Mask handling for AV
                    # Video part: video_mask
                    # Audio part: Ones
                    audio_mask = torch.ones((batch_size, 1, audio_len, 1), device=mm.intermediate_device())
                    nt_mask = NestedTensor((video_mask, audio_mask))
                    input_latent["noise_mask"] = nt_mask
                else:
                    input_latent = {"samples": video_latent, "noise_mask": video_mask}
                    # If audio_vae missing, we skip audio latent creation
                    
            else:
                # EXTENSION
                # Extract Previous Tail
                prev_samples = prev_latent["samples"]
                if is_av_model and NestedTensor is not None and isinstance(prev_samples, NestedTensor):
                    prev_video = prev_samples.tensors[0] # [B, 128, T, H, W]
                    prev_audio_src = prev_samples.tensors[1]
                else:
                    prev_video = prev_samples
                    prev_audio_src = None
                
                # Copy Overlap (Robust)
                video_overlap = video_overlap_frames
                if video_overlap > prev_video.shape[2]:
                     video_overlap = prev_video.shape[2]
                if video_overlap > latent_length: # Clamp to new chunk size
                     video_overlap = latent_length
                     
                src_v = prev_video[..., -video_overlap:, :, :]
                
                # Allocate New
                current_v = torch.zeros(
                    [batch_size, 128, latent_length, latent_height, latent_width],
                    device=mm.intermediate_device() if COMFY_AVAILABLE else "cpu"
                )
                current_v[:, :, :video_overlap, :, :] = src_v
                
                # AV Logic
                if is_av_model and NestedTensor is not None and audio_vae is not None:
                     # Calculate audio length with defensive minimum
                     audio_len = time_mgr.calculate_audio_latent_count(chunk_duration)
                     if audio_len < 1:
                         audio_len = 1
                         print(f"    [DEBUG] audio_len forced to 1 (was {time_mgr.calculate_audio_latent_count(chunk_duration)})")
                     
                     a_ch = getattr(audio_vae, "latent_channels", 8)  # Default 8, not 128
                     a_freq = getattr(audio_vae, "latent_frequency_bins", 16)  # Default 16

                     current_a = torch.zeros(
                         [batch_size, a_ch, audio_len, a_freq],
                         device=mm.intermediate_device() if COMFY_AVAILABLE else "cpu"
                     )
                     
                     audio_overlap = audio_overlap_frames
                     if prev_audio_src is not None:
                          # Ensure prev_audio_src has batch dimension [B, C, T, F]
                          if prev_audio_src.ndim == 3:
                              prev_audio_src = prev_audio_src.unsqueeze(0)
                          
                          # Safe clamp of copy length
                          a_copy_len = min(audio_overlap, prev_audio_src.shape[2], current_a.shape[2])
                          
                          if a_copy_len > 0:
                              src_a = prev_audio_src[:, :, -a_copy_len:, :]
                              current_a[:, :, :a_copy_len, :] = src_a
                          else:
                              print(f"    [DEBUG] Skipping audio overlap copy (a_copy_len={a_copy_len})")
                     
                     nt = NestedTensor((current_v, current_a))
                     input_latent = {"samples": nt}
                     
                     # Create Masks (Mask out overlap regions)
                     v_mask = torch.ones((batch_size, 1, latent_length, 1, 1), device=mm.intermediate_device())
                     v_mask[:, :, :video_overlap] = 1.0 - temporal_cond_strength
                     
                     a_mask = torch.ones((batch_size, 1, audio_len, 1), device=mm.intermediate_device())
                     a_mask[:, :, :audio_overlap] = 1.0 - temporal_cond_strength
                     
                     input_latent["noise_mask"] = NestedTensor((v_mask, a_mask))
                else:
                     input_latent = {"samples": current_v}
                     v_mask = torch.ones((batch_size, 1, latent_length, 1, 1), device=mm.intermediate_device())
                     v_mask[:, :, :video_overlap] = 1.0 - temporal_cond_strength
                     input_latent["noise_mask"] = v_mask

            # --- SETUP CONDITIONING ---
            chunk_guider = copy.copy(guider)
            
            # Using pre-encoded prompts
            c_pos = c_data["cond_pos"]
            c_pos = c_data["cond_pos"]
            
            # Use Global Negative from Guider (since we didn't pre-encode negative)
            _, global_neg = cls._get_conds_from_guider(guider)
            c_neg = global_neg if global_neg is not None else []
            
            # Sanitize Negative (KJ Fix)
            if c_neg and len(c_neg) > 0 and isinstance(c_neg[0], dict):
                 c_neg = []
            
            # --- APPLY PRE-ENCODED GUIDES ---
            guides = c_data["guides"]
            if guides:
                print(f"  Applying {len(guides)} image guides (Pre-encoded)")
                
                # Extract working Tensors
                if isinstance(input_latent["samples"], NestedTensor):
                    working_v = input_latent["samples"].tensors[0]
                    # We handle audio parts later
                else:
                    working_v = input_latent["samples"]
                
                if isinstance(input_latent["noise_mask"], NestedTensor):
                    working_mask = input_latent["noise_mask"].tensors[0]
                else:
                    working_mask = input_latent["noise_mask"]
                
                for g in guides:
                    g_latent = g["latent"]
                    frame_offset = g["frame_idx"] # From script ref
                    g_strength = g["strength"]
                    
                    # Adjust offset for Extension chunks?
                    # Chunk logic: if extension, new latent contains OVERLAP.
                    # Script 'first' ($0) usually maps to 0.
                    # If latent starts with overlap, index 0 IS the overlap start.
                    # So offset logic should be consistent with how user expects it.
                    # If user says "first", they mean Frame 0 of this chunk.
                    # Which is Frame 0 of latent.
                    # So no adjustment needed?
                    # BUT 'resolve_image_refs' logic maps 'first'->0.
                    # IF is_extension, we might want to offset by overlap?
                    # No, let's assume raw frame index into the latent currently being generated.
                    
                    # 1. Update Conds (Keyframe Index)
                    # LTXVAddGuide method
                    c_pos = LTXVAddGuide.add_keyframe_index(c_pos, frame_offset, g_latent, scale_factors)
                    c_neg = LTXVAddGuide.add_keyframe_index(c_neg, frame_offset, g_latent, scale_factors)
                    
                    # 2. Update Latent/Mask
                    time_scale = scale_factors[0]
                    # Use Floor division for safer mapping of Frame -> Latent
                    l_idx = frame_offset // time_scale
                    
                    # Clamp to bounds to prevent crash if 'end' lands on boundary
                    cond_len = g_latent.shape[2]
                    max_idx = working_v.shape[2] - cond_len
                    l_idx = max(0, min(l_idx, max_idx))
                    
                    working_v, working_mask = LTXVAddGuide.replace_latent_frames(
                        working_v, working_mask, g_latent, l_idx, g_strength
                    )
                
                # Pack updated Tensors back
                if isinstance(input_latent["samples"], NestedTensor):
                     orig_samples = input_latent["samples"]
                     new_samples = NestedTensor((working_v, orig_samples.tensors[1]))
                     
                     orig_mask = input_latent["noise_mask"]
                     new_mask = NestedTensor((working_mask, orig_mask.tensors[1]))
                     
                     input_latent["samples"] = new_samples
                     input_latent["noise_mask"] = new_mask
                else:
                     input_latent["samples"] = working_v
                     input_latent["noise_mask"] = working_mask

            # Set Conditioning on Guider
            # Manual set to bypass validation if custom types
            try:
                 chunk_guider.conds = {"positive": c_pos, "negative": c_neg}
            except:
                 try:
                     chunk_guider.set_conds(c_pos, c_neg)
                 except:
                     pass # Hope it worked
            
            # --- SAMPLING ---
            print("  Sampling...")
            _, denoised = SamplerCustomAdvanced().sample(
                noise, chunk_guider, sampler, sigmas, input_latent
            )
            
            final_video_list.append(denoised)
            prev_latent = denoised
            
            # Cache it
            _CACHE[cache_key] = denoised
            
            # Keep refs for return
            positive = c_pos
            negative = c_neg

        # --- HANDLE EMPTY CHUNKS ---
        if len(final_video_list) == 0:
            print("WARNING: No chunks were processed. Returning empty latents.")
            # Create minimal placeholder latents
            empty_video = torch.zeros([1, 128, 1, latent_height, latent_width], device=mm.intermediate_device())
            empty_audio = torch.zeros([1, 8, 1, 16], device=mm.intermediate_device()) if audio_vae is not None else None
            
            video_out = {"samples": empty_video}
            audio_out = {"samples": empty_audio} if empty_audio is not None else {"samples": torch.zeros([1,8,1,16])}
            
            if is_av_model and NestedTensor is not None and empty_audio is not None:
                combined = {"samples": NestedTensor((empty_video, empty_audio))}
            else:
                combined = video_out
                
            return io.NodeOutput(combined, video_out, audio_out, [], [])

        # --- BLENDING ---
        print("Generation complete. Blending chunks...")
        
        full_video = None
        full_audio = None
        
        video_overlap_frames = time_mgr.calculate_video_latent_count(overlap_duration)
        audio_overlap_frames = time_mgr.calculate_audio_latent_count(overlap_duration)
        
        for i, chunk_res in enumerate(final_video_list):
            chunk = chunks[i]
            is_cut = chunk.transition_type == "cut"
            
            if is_av_model and NestedTensor is not None and isinstance(chunk_res["samples"], NestedTensor):
                v_part = chunk_res["samples"].tensors[0]
                a_part = chunk_res["samples"].tensors[1]
            else:
                v_part = chunk_res["samples"]
                a_part = None
            
            if i == 0:
                full_video = v_part
                full_audio = a_part
            else:
                if is_cut:
                    print(f"  Chunk {i+1}: Hard Cut")
                    overlap_use = 0
                else:
                    overlap_use = video_overlap_frames
                
                full_video = cls._blend_latents(full_video, v_part, overlap_use)
                
                if full_audio is not None and a_part is not None:
                     if is_cut:
                         a_ov_use = 0
                     else:
                         a_ov_use = audio_overlap_frames
                     full_audio = cls._blend_latents(full_audio, a_part, a_ov_use)

        # Output
        video_out = {"samples": full_video}
        # Use correct audio latent shape: [B, 8, T, 16] where 8 is channels, 16 is freq bins
        audio_out = {"samples": full_audio} if full_audio is not None else {"samples": torch.zeros([1,8,1,16], device=mm.intermediate_device())}
        
        if is_av_model and NestedTensor is not None and full_audio is not None:
             combined = {"samples": NestedTensor((full_video, full_audio))}
        else:
             combined = video_out
             
        return io.NodeOutput(
            combined,
            video_out,
            audio_out, # Fixed Typo
            positive,
            negative
        )

    @classmethod
    def _pre_encode_assets(cls, chunks, clip, video_vae, guide_images, width, height, time_mgr):
        """Helper to pre-encode prompts and guide images."""
        chunk_data = []
        
        # Resolve all references first (to get paths)
        resolved_refs = resolve_image_refs(chunks, guide_images)
        
        latent_width = width // 32
        latent_height = height // 32
        scale_factors = video_vae.downscale_index_formula

        for i, chunk in enumerate(chunks):
            # Text
            cond_pos = cls._encode_prompt(clip, chunk.prompt)
            # Use empty string for neg as simple default or passed param?
            # We implemented global negative in execute, but let's use empty here 
            # and let execute merge global?
            # User wants optimization. We can encode global NEG once outside loop?
            # Wait, signature of execute has `negative_prompt`? No, it has `guider`.
            # We extract from Guider.
            # So pre-encoding NEG is done via Guider extraction in Loop.
            # We only pre-encode POS specific to chunk.
            
            # Guides
            processed_guides = []
            chunk_guides, chunk_indices = get_chunk_guide_images(chunk, resolved_refs, time_mgr)
            
            if chunk_guides is not None:
                # chunk_guides is [N, H, W, C]
                # Indices string
                indices_list = [int(x) for x in chunk_indices.split(",")]
                
                # Resize
                resized_guides = comfy.utils.common_upscale(
                    chunk_guides.movedim(-1, 1), 
                    width, height, "bilinear", "center"
                ).movedim(1, -1)
                
                # Encode
                for img, idx in zip(resized_guides, indices_list):
                    try:
                        # Use LTXVAddGuide.encode logic
                        # (It expects batch of images but we do one by one for simplicity/safety)
                        # Actually LTXVAddGuide.encode handles batch.
                        # But we need granular control per index.
                        
                        _, g_latent = LTXVAddGuide.encode(
                            video_vae, latent_width, latent_height, 
                            img.unsqueeze(0), scale_factors
                        )
                        processed_guides.append({
                            "latent": g_latent,
                            "frame_idx": idx,
                            "strength": 1.0 # TODO: Get from chunk ref? Parser supports it, but here we used batch strength?
                            # Input guide_strength is global.
                            # Script parser has strength in $0:1.0?
                            # resolve_image_refs logic...
                        })
                    except Exception as e:
                         print(f"Warning: Guide encode failed: {e}")
            
            # Basic Negative placeholder (will be replaced by Global Neg in Loop)
            chunk_data.append({
                "cond_pos": cond_pos,
                "cond_neg": None, # Will use Global
                "guides": processed_guides
            })
            
        return chunk_data, resolved_refs

    @classmethod
    def _blend_latents(cls, prev: torch.Tensor, next_t: torch.Tensor, overlap: int) -> torch.Tensor:
        """Blend two latents with linear crossfade on dim 2 (time)."""
        if overlap <= 0:
            return torch.cat([prev, next_t], dim=2)
            
        overlap = min(overlap, prev.shape[2], next_t.shape[2])
        
        prev_cut = prev[:, :, :-overlap]
        prev_tail = prev[:, :, -overlap:]
        next_head = next_t[:, :, :overlap]
        next_cut = next_t[:, :, overlap:]
        
        alpha = torch.linspace(0, 1, overlap, device=prev.device, dtype=prev.dtype)
        
        # Dynamic Reshape for Broadcasting (Video 5D or Audio 4D)
        shape = [1, 1, -1] + [1] * (prev.ndim - 3)
        alpha = alpha.view(*shape)
        
        blended = prev_tail * (1.0 - alpha) + next_head * alpha
        return torch.cat([prev_cut, blended, next_cut], dim=2)
    
    @classmethod
    def _is_av_model(cls, model) -> bool:
        try:
            return model.model.diffusion_model.__class__.__name__ == "LTXAVModel"
        except AttributeError:
            return False
    
    @classmethod
    def _get_conds_from_guider(cls, guider):
        conds = None
        if hasattr(guider, "conds"):
            conds = guider.conds
        elif hasattr(guider, "raw_conds"):
            conds = guider.raw_conds
        elif hasattr(guider, "original_conds"):
            conds = guider.original_conds
            
        if conds is not None:
            if isinstance(conds, dict):
                return conds.get("positive"), conds.get("negative")
            return conds
        return None, None
    
    @classmethod
    def _encode_prompt(cls, clip, prompt: str):
        tokens = clip.tokenize(prompt)
        try:
             cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
             return [[cond, {"pooled_output": pooled}]]
        except:
             conds = clip.encode_from_tokens_scheduled(tokens)
             if conds and isinstance(conds, list) and len(conds) > 0 and isinstance(conds[0], dict):
                 new_conds = []
                 for c in conds:
                     tensor = None
                     if "cross_attn" in c:
                         tensor = c["cross_attn"]
                     elif "pooled_output" in c:
                         tensor = c["pooled_output"]
                     
                     if tensor is not None:
                         new_conds.append([tensor, c])
                 if new_conds:
                     return new_conds
             return conds
    
    @classmethod
    def _generate_default_chunks(
        cls,
        duration: float,
        tile_duration: float,
        overlap_duration: float
    ) -> list[SceneChunk]:
        chunks = []
        effective_tile = tile_duration - overlap_duration
        current_start = 0.0
        while current_start < duration:
            chunk_end = min(current_start + tile_duration, duration)
            chunks.append(SceneChunk(
                start_sec=current_start,
                end_sec=chunk_end,
                prompt="",
                audio_spec="silent",
                guides=[],
            ))
            current_start += effective_tile
            if chunk_end >= duration:
                break
        return chunks
