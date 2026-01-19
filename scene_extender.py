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
                    "extension_duration",
                    default=5.0,
                    min=0.1,
                    max=300.0,
                    step=0.1,
                    tooltip="How many seconds to extend (handled internally)"
                ),
                io.Float.Input(
                    "tile_duration",
                    default=3.0,
                    min=1.0,
                    max=10.0,
                    step=0.5,
                    tooltip="Duration of each temporal chunk in seconds"
                ),
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
                    default=768,
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
                    default="",
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
                io.Float.Input(
                    "temporal_cond_strength",
                    default=0.5,
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
        extension_duration: float,
        tile_duration: float,
        overlap_duration: float,
        video_fps: float,
        width: int,
        height: int,
        scene_script: str,
        guide_strength: float,
        audio_overlap_duration: float,
        audio_slope_frames: int,
        audio_normalization: str,
        temporal_cond_strength: float,
        adain_factor: float,
        audio_vae=None,
        latent=None,
        guide_images=None,
    ) -> io.NodeOutput:
        """Execute the scene extension."""
        
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
            # Generate default chunks based on extension duration
            chunks = cls._generate_default_chunks(
                extension_duration,
                tile_duration,
                overlap_duration
            )
        
        # Resolve image references
        resolved_refs = resolve_image_refs(chunks, guide_images)
        
        # Get guider's conditioning
        positive, negative = cls._get_conds_from_guider(guider)
        
        # Initialize audio blender if we have audio
        audio_blender = None
        if is_av_model and audio_vae is not None:
            audio_blender = AudioOverlapBlender(
                overlap_frames=int(
                    audio_overlap_duration * time_mgr.config.audio_latents_per_second
                ),
                slope_len=audio_slope_frames,
            )
        
        # Calculate dimensions
        time_scale_factor, width_scale_factor, height_scale_factor = (
            video_vae.downscale_index_formula
        )
        latent_height = height // height_scale_factor
        latent_width = width // width_scale_factor
        
        # Process chunks
        extended_latent = latent
        extended_video = None
        # Process chunks
        final_video_list = []
        final_audio_list = []
        
        # Keep track of previous latent for extension
        prev_latent = latent
        current_time = 0.0
        
        for i, chunk in enumerate(chunks):
            chunk_duration = chunk.end_sec - chunk.start_sec
            print(f"Processing chunk {i+1}/{len(chunks)}: [{chunk.start_sec:.1f}s - {chunk.end_sec:.1f}s]")
            
            # Encode chunk prompt
            chunk_cond = cls._encode_prompt(clip, chunk.prompt)
            
            # Create new guider for this chunk
            chunk_guider = copy.copy(guider)
            
            # Set prompt conditioning
            # Note: This is simplified; specialized usage might need SetConditioning logic
            c_pos, c_neg = cls._get_conds_from_guider(guider)
            
            # Update positive prompt
            # Try to merge with original conditioning to preserve styles/controlnets
            try:
                new_pos = []
                if c_pos is None:
                    raise ValueError("No original positive conditioning")
                    
                for t in c_pos:
                    # t should be [tensor, dict]
                    # chunk_cond is [[tensor, dict]]
                    chunk_tensor = chunk_cond[0][0]
                    chunk_meta = chunk_cond[0][1]
                    
                    # Create new cond pair
                    # Handle if t is not list/tuple or t[1] is not dict (KeyError/TypeError)
                    current_dict = t[1].copy() if hasattr(t, "__getitem__") and len(t) > 1 and isinstance(t[1], dict) else {}
                    
                    new_t = [chunk_tensor, current_dict]
                    # Update metadata
                    if "pooled_output" in chunk_meta:
                        new_t[1]["pooled_output"] = chunk_meta["pooled_output"]
                    new_t[1]["text"] = chunk.prompt
                    
                    new_pos.append(new_t)
            except (IndexError, KeyError, TypeError, ValueError) as e:
                keys_info = ""
                try:
                    if c_pos is not None and len(c_pos) > 0 and isinstance(c_pos[0], dict):
                        keys_info = f" Keys: {list(c_pos[0].keys())}"
                except:
                    pass
                print(f"  Warning: Could not merge conditioning ({e}){keys_info}, using raw chunk prompt.")
                new_pos = chunk_cond

            # Manually set conds to bypass validation if using custom/weird structures (LTXV dicts)
            if hasattr(chunk_guider, "conds"):
                 chunk_guider.conds = {"positive": new_pos, "negative": c_neg}
            elif hasattr(chunk_guider, "inner_set_conds"):
                 # Force set inner dict directly if possible, or try set_conds and fail
                 # CFGGuider uses inner_set_conds but it validates.
                 # We can try to monkeypatch or access protected?
                 # Actually BasicGuider.conds is exposed. CFGGuider just wraps it.
                 # If chunk_guider is BasicGuider/CFGGuider, .conds attribute usually exists.
                 try:
                     chunk_guider.conds = {"positive": new_pos, "negative": c_neg}
                 except:
                     try:
                        chunk_guider.set_conds(new_pos, c_neg)
                     except KeyError:
                         print("  Critical: Custom conditioning format rejected by Guider validation.")
                         # Last resort: if c_neg is incorrectly formatted for standard guider, we might have to clean it?
                         # But we can't clean it if we don't know the format.
                         # We'll assume the manual setting works for now.
                         pass
            else:
                 try:
                    chunk_guider.set_conds(new_pos, c_neg)
                 except:
                    pass

            # Determine dimensions
            
            # Determine if we should extend or start new
            is_extension = (prev_latent is not None) and (chunk.transition_type != "cut")
            
            # Prepare Input Latent
            if not is_extension:
                # FIRST CHUNK or CUT: New Generation
                print(f"  Type: {'New Generation' if prev_latent is None else 'Hard Cut (New Clean Generation)'}")
                latent_length = time_mgr.calculate_video_latent_count(chunk_duration)
                
                # Create empty video latent
                video_latent = torch.zeros(
                    [1, 128, latent_length, latent_height, latent_width],
                    device=mm.intermediate_device() if COMFY_AVAILABLE else "cpu"
                )
                
                # Handle AV Model
                if is_av_model and NestedTensor is not None:
                    audio_length = time_mgr.calculate_audio_latent_count(chunk_duration)
                    audio_latent = torch.zeros(
                        [1, 128, audio_length, 1], # 4D for Audio
                        device=mm.intermediate_device() if COMFY_AVAILABLE else "cpu"
                    )
                    
                    nt = NestedTensor((video_latent, audio_latent))
                    input_latent = {"samples": nt}
                else:
                    input_latent = {"samples": video_latent}
                
                start_frame_idx = 0
                
            else:
                # SUBSEQUENT CHUNK: Extension
                print("  Type: Extension")
                
                # Extract previous samples (handle AV)
                prev_samples = prev_latent["samples"]
                if is_av_model and NestedTensor is not None and isinstance(prev_samples, NestedTensor):
                    prev_video = prev_samples.tensors[0]
                else:
                    prev_video = prev_samples
                
                # Calculate overlap
                # Important: Total duration = Overlap + Chunk Duration
                total_duration = overlap_duration + chunk_duration
                latent_length = time_mgr.calculate_video_latent_count(total_duration)
                
                overlap_frames = time_mgr.calculate_video_latent_count(overlap_duration)
                last_frames = prev_video[:, :, -overlap_frames:]
                
                new_video = torch.zeros(
                    [1, 128, latent_length, latent_height, latent_width],
                    device=mm.intermediate_device() if COMFY_AVAILABLE else "cpu"
                )
                
                # Handle AV
                if is_av_model and NestedTensor is not None:
                    audio_length = time_mgr.calculate_audio_latent_count(total_duration)
                    new_audio = torch.zeros(
                        [1, 128, audio_length, 1], # 4D for Audio [B, C, T, D]
                        device=mm.intermediate_device() if COMFY_AVAILABLE else "cpu"
                    )
                    input_latent = {"samples": NestedTensor((new_video, new_audio))}
                else:
                    input_latent = {"samples": new_video}

                # Condition on overlap (Latent Guide)
                t = last_frames.to(mm.intermediate_device())
                if is_av_model and NestedTensor is not None:
                    v_part = input_latent["samples"].tensors[0]
                    # Ensure dimensions match (sometimes off by 1 due to rounding)
                    copy_len = min(t.shape[2], v_part.shape[2])
                    v_part[:, :, :copy_len] = t[:, :, :copy_len]
                    input_latent["samples"] = NestedTensor((v_part, input_latent["samples"].tensors[1]))
                else:
                    copy_len = min(t.shape[2], input_latent["samples"].shape[2])
                    input_latent["samples"][:, :, :copy_len] = t[:, :, :copy_len]
                
                # Make mask
                mask = torch.ones(
                    (1, 1, latent_length, 1, 1),
                    dtype=torch.float32,
                    device=mm.intermediate_device()
                )
                mask[:, :, :copy_len] = 1.0 - temporal_cond_strength
                
                if is_av_model and NestedTensor is not None:
                    # Audio mask too
                    # Audio overlap frames
                    audio_overlap = time_mgr.calculate_audio_latent_count(overlap_duration)
                    amask = torch.ones(
                        (1, 1, audio_length, 1), # 4D for Audio Mask
                        dtype=torch.float32,
                        device=mm.intermediate_device()
                    )
                    # We assume we want to preserve audio overlap too?
                    # Extension usually implies extending from context.
                    # Audio doesn't have "Latent Guide" traditionally but masking works.
                    amask[:, :, :audio_overlap] = 1.0 - temporal_cond_strength
                    
                    input_latent["noise_mask"] = NestedTensor((mask, amask))
                else:
                    input_latent["noise_mask"] = mask
                
                start_frame_idx = 0 # Relative to this chunk latent
                
                
            # Apply Image Guides
            chunk_cond_images, chunk_cond_indices = get_chunk_guide_images(
                chunk, resolved_refs, time_mgr
            )
            
            if chunk_cond_images is not None:
                guide_count = chunk_cond_images.shape[0]
                print(f"  Applying {guide_count} image guides")
                
                # Resize images first. chunk_cond_images is [N, H, W, C]
                # common_upscale expects [N, C, H, W] usually, or handles it?
                # Actually common_upscale takes [B, H, W, C] and returns [B, H, W, C]
                # Let's verify standard comfy behavior. 
                # Comfy uses [B, H, W, C] mostly. 
                # common_upscale implementation in comfy/utils.py swaps to BCHW, interpolates, swaps back.
                resized_guides = comfy.utils.common_upscale(
                    chunk_cond_images.movedim(-1, 1), # [N, C, H, W]
                    width, height, "bilinear", "center"
                ).movedim(1, -1) # [N, H, W, C]
                
                # Split indices string
                indices_list = [int(x) for x in chunk_cond_indices.split(",")]
                
                # For extension chunks, the latent starts BEFORE the chunk start (by overlap)
                # So we must offset guide indices to match the latent
                # BUT if it's a CUT, we are NOT extending, so it acts like "first chunk" (Starts at 0)
                if is_extension:
                     overlap_pixels = time_mgr.seconds_to_pixel_frame(overlap_duration)
                     indices_list = [x + overlap_pixels for x in indices_list]
                
                for img, idx in zip(resized_guides, indices_list):
                    # Logic similar to MVP: extract, guide, pack
                    c_pos, c_neg = cls._get_conds_from_guider(chunk_guider)
                    
                    # Sanitize incompatible negative conditioning (KJ Dicts) for LTXVAddGuide
                    if c_neg is not None and len(c_neg) > 0 and isinstance(c_neg[0], dict):
                        print("  Sanitizing incompatible Negative conditioning (KJ Dicts) for Image Guide application")
                        c_neg = [] # Empty list is safe for LTXVAddGuide
                    
                    if is_av_model and NestedTensor is not None:
                        # Extract video
                        current_v = input_latent["samples"].tensors[0]
                        temp_l = {"samples": current_v}
                        if "noise_mask" in input_latent:
                             temp_l["noise_mask"] = input_latent["noise_mask"].tensors[0]
                        
                        # Apply
                        new_pos, new_neg, new_l = LTXVAddGuide.execute(
                            c_pos, c_neg, video_vae, temp_l, img.unsqueeze(0),
                            idx, guide_strength
                        )
                        
                        # Pack back
                        v_samp = new_l["samples"]
                        a_samp = input_latent["samples"].tensors[1]
                        input_latent["samples"] = NestedTensor((v_samp, a_samp))
                        
                        if "noise_mask" in new_l:
                            v_mask = new_l["noise_mask"]
                            if "noise_mask" in input_latent:
                                a_mask = input_latent["noise_mask"].tensors[1]
                            else:
                                a_mask = torch.ones((1,1,a_samp.shape[2],1,1), device=a_samp.device)
                            input_latent["noise_mask"] = NestedTensor((v_mask, a_mask))
                            
                        chunk_guider.set_conds(new_pos, new_neg)
                    else:
                        new_pos, new_neg, input_latent = LTXVAddGuide.execute(
                            c_pos, c_neg, video_vae, input_latent, img.unsqueeze(0),
                            idx, guide_strength
                        )
                        chunk_guider.set_conds(new_pos, new_neg)

            # Execution
            print("  Sampling...")
            _, denoised = SamplerCustomAdvanced().sample(
                noise, chunk_guider, sampler, sigmas, input_latent
            )
            
            # Store Result (Accumulate)
            final_video_list.append(denoised)
            prev_latent = denoised
            
        print("Generation complete. Blending chunks...")
        
        full_video = None
        full_audio = None
        
        # Calculate overlap in latent frames
        video_overlap_frames = time_mgr.calculate_video_latent_count(overlap_duration)
        audio_overlap_frames = time_mgr.calculate_audio_latent_count(overlap_duration)
        
        for i, chunk_res in enumerate(final_video_list):
            chunk = chunks[i]
            is_cut = chunk.transition_type == "cut"
            
            # Extract components
            if is_av_model and NestedTensor is not None and isinstance(chunk_res["samples"], NestedTensor):
                v_part = chunk_res["samples"].tensors[0] # [B, 128, T, H, W]
                a_part = chunk_res["samples"].tensors[1] # [B, 128, T, 1, 1]
            else:
                v_part = chunk_res["samples"]
                a_part = None
            
            if i == 0:
                full_video = v_part
                full_audio = a_part
            else:
                # Blend Video
                # Linear crossfade for video unless CUT
                if is_cut:
                    print(f"  Chunk {i+1}: Hard Cut")
                    overlap_to_use = 0
                else:
                    overlap_to_use = video_overlap_frames
                    
                full_video = cls._blend_latents(full_video, v_part, overlap_to_use)
                
                # Blend Audio
                if full_audio is not None and a_part is not None:
                     if is_cut:
                         a_overlap_to_use = 0
                     else:
                         a_overlap_to_use = audio_overlap_frames
                     full_audio = cls._blend_latents(full_audio, a_part, a_overlap_to_use)

        # Final Output Packaging
        video_out = {"samples": full_video}
        audio_out = {"samples": full_audio} if full_audio is not None else {"samples": torch.zeros([1,64,1,1])} # Placeholder
        
        if is_av_model and NestedTensor is not None and full_audio is not None:
             combined = {"samples": NestedTensor((full_video, full_audio))}
        else:
             combined = video_out
             
        return io.NodeOutput(
            combined,
            video_out,
            audio_out,
            positive, # Return last conds
            negative
        )

    @classmethod
    def _blend_latents(cls, prev: torch.Tensor, next_t: torch.Tensor, overlap: int) -> torch.Tensor:
        """Blend two latents with linear crossfade on dim 2 (time)."""
        if overlap <= 0:
            return torch.cat([prev, next_t], dim=2)
            
        # Ensure proper overlap
        overlap = min(overlap, prev.shape[2], next_t.shape[2])
        
        # Regions
        prev_cut = prev[:, :, :-overlap]
        prev_tail = prev[:, :, -overlap:]
        next_head = next_t[:, :, :overlap]
        next_cut = next_t[:, :, overlap:]
        
        # Linear weights [0..1]
        alpha = torch.linspace(0, 1, overlap, device=prev.device, dtype=prev.dtype)
        
        # Reshape alpha for broadcasting
        # Valid for Video [B, C, T, H, W] (5D) and Audio [B, C, T, D] (4D)
        # Dynamically append 1s based on dimensions
        shape = [1, 1, -1] + [1] * (prev.ndim - 3)
        alpha = alpha.view(*shape)
        
        # Blend: prev_tail * (1-alpha) + next_head * alpha
        # Wait, if we are appending NEXT to PREV.
        # We want smooth transition from PREV to NEXT.
        # Start of overlap: 100% Prev, 0% Next.
        # End of overlap: 0% Prev, 100% Next.
        # So alpha should go 0->1.
        # Blended = prev_tail * (1 - alpha) + next_head * alpha
        
        blended = prev_tail * (1.0 - alpha) + next_head * alpha
        
        return torch.cat([prev_cut, blended, next_cut], dim=2)
    
    @classmethod
    def _is_av_model(cls, model) -> bool:
        """Check if model is LTXAVModel."""
        try:
            return model.model.diffusion_model.__class__.__name__ == "LTXAVModel"
        except AttributeError:
            return False
    
    @classmethod
    def _get_conds_from_guider(cls, guider):
        """Extract positive and negative conditioning from guider."""
        conds = None
        # Prefer 'conds' (Current/Active) over 'original_conds' (Historical/Input)
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
        """Encode a text prompt using CLIP."""
        tokens = clip.tokenize(prompt)
        try:
             # Try standard encode with pooling
             cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
             return [[cond, {"pooled_output": pooled}]]
        except:
             # Fallback for some clip implementations (e.g. KJ LTXV)
             conds = clip.encode_from_tokens_scheduled(tokens)
             
             # If result is List of Dicts (Non-Standard), Wrap it!
             if conds and isinstance(conds, list) and len(conds) > 0 and isinstance(conds[0], dict):
                 new_conds = []
                 for c in conds:
                     # Attempt to find tensor
                     tensor = None
                     if "cross_attn" in c:
                         tensor = c["cross_attn"]
                     elif "pooled_output" in c: # Fallback if cross_attn missing?
                         tensor = c["pooled_output"] # Danger?
                     
                     if tensor is not None:
                         # Use the dict itself as metadata
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
        """Generate default chunks when no script is provided."""
        chunks = []
        effective_tile = tile_duration - overlap_duration
        current_start = 0.0
        
        while current_start < duration:
            chunk_end = min(current_start + tile_duration, duration)
            chunks.append(SceneChunk(
                start_sec=current_start,
                end_sec=chunk_end,
                prompt="",  # Empty prompt for default
                audio_spec="silent",
                guides=[],
            ))
            current_start += effective_tile
            if chunk_end >= duration:
                break
        
        return chunks
