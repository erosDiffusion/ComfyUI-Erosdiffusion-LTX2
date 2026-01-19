"""
LTXVSceneExtender MVP: Wraps existing LTXVExtendSampler with script parsing.

This is a Minimum Viable Product that provides:
1. Script parsing for timestamped prompts with image guides
2. Single chunk video extension (uses LTXVExtendSampler internally)
3. Image guide resolution from batch

Full multi-chunk looping will be added in a future iteration.
"""

import copy
from typing import Optional, Tuple

import torch
import comfy.utils
from comfy_api.latest import io

# Import existing ComfyUI nodes
from comfy_extras.nodes_lt import EmptyLTXVLatentVideo, LTXVAddGuide
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced

# Try to import AV model support
try:
    from comfy.nested_tensor import NestedTensor
    from comfy.ldm.lightricks.av_model import LTXAVModel
    AV_MODEL_AVAILABLE = True
except ImportError:
    AV_MODEL_AVAILABLE = False
    NestedTensor = None
    LTXAVModel = None

# Local imports
from .time_manager import TimeManager
from .script_parser import (
    parse_scene_script,
    resolve_image_refs,
    SceneChunk,
    ImageGuide,
)


def is_av_model(guider) -> bool:
    """Check if the guider's model is LTXAVModel (audio-video)."""
    try:
        model_class = guider.model_patcher.model.diffusion_model.__class__.__name__
        return model_class == "LTXAVModel"
    except AttributeError:
        return False


def _get_raw_conds_from_guider(guider):
    """Extract raw conditions from guider (copied from Lightricks)."""
    if not hasattr(guider, "raw_conds"):
        if "negative" not in guider.original_conds:
            raise ValueError(
                "Guider does not have negative conds, cannot use it as a guider."
            )
        raw_pos = guider.original_conds["positive"]
        positive = [[raw_pos[0]["cross_attn"], copy.deepcopy(raw_pos[0])]]
        raw_neg = guider.original_conds["negative"]
        negative = [[raw_neg[0]["cross_attn"], copy.deepcopy(raw_neg[0])]]
        guider.raw_conds = (positive, negative)
    return guider.raw_conds


class LTXVSceneExtenderMVP(io.ComfyNode):
    """
    MVP Scene Extender: Single-chunk video extension with script parsing.
    
    Wraps LTXVBaseSampler/LTXVExtendSampler with timestamped script support.
    
    MVP Features:
    - Script parsing for prompts with audio specs and image guides
    - Image guide resolution from batch ($0, $1, etc.)
    - Single temporal chunk generation
    - Works with existing LTXVideo infrastructure
    
    Coming Soon:
    - Multi-chunk looping for long videos
    - Audio generation integration
    """
    
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXVSceneExtenderMVP",
            display_name="LTXV Scene Extender (MVP)",
            category="ErosDiffusion/ltxv",
            description="MVP: Extend video with timestamped prompts and image guides",
            inputs=[
                # === Model Inputs ===
                io.Model.Input("model", tooltip="LTX diffusion model"),
                io.Vae.Input("video_vae", tooltip="Video VAE"),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                io.Noise.Input("noise"),
                io.Guider.Input("guider", tooltip="STGGuider or similar"),
                
                # === Existing Content (optional) ===
                io.Latent.Input(
                    "latent",
                    optional=True,
                    tooltip="Existing video latent to extend (leave empty for new generation)"
                ),
                
                # === Generation Settings ===
                io.Int.Input("width", default=768, min=64, max=2048, step=32),
                io.Int.Input("height", default=512, min=64, max=2048, step=32),
                io.Int.Input(
                    "num_frames",
                    default=97,
                    min=1,
                    max=257,
                    step=8,
                    tooltip="Number of frames to generate (pixel frames)"
                ),
                io.Int.Input(
                    "frame_overlap",
                    default=24,
                    min=16,
                    max=80,
                    step=8,
                    tooltip="Overlap frames when extending (for continuity)"
                ),
                
                # === Scene Script ===
                io.String.Input(
                    "scene_script",
                    default="",
                    multiline=True,
                    tooltip="""Timestamped scene script (uses first chunk only in MVP).
Format: [MM:SS-MM:SS] prompt | audio:spec | first:$0 | end:$1

Example:
[00:00-00:03] A woman speaks | audio:"Hello" | first:$0 | end:$1"""
                ),
                
                # === Image Guides ===
                io.Image.Input(
                    "guide_images",
                    optional=True,
                    tooltip="Batch of guide images (referenced as $0, $1, etc.)"
                ),
                io.Float.Input(
                    "guide_strength",
                    default=0.9,
                    min=0.0,
                    max=1.0,
                    step=0.05
                ),
                
                # === Conditioning ===
                io.Float.Input(
                    "overlap_strength",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="Conditioning strength on overlap region (when extending)"
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
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
        width: int,
        height: int,
        num_frames: int,
        frame_overlap: int,
        scene_script: str,
        guide_strength: float,
        overlap_strength: float,
        latent=None,
        guide_images=None,
    ) -> io.NodeOutput:
        """Execute scene extension."""
        
        # Check if using AV model (audio-video)
        using_av_model = is_av_model(guider)
        if using_av_model:
            print("[SceneExtenderMVP] Detected LTXAVModel - audio will be generated!")
            print("  TIP: Connect output to LTXVSeparateAVLatent to split video and audio")
        else:
            print("[SceneExtenderMVP] Using video-only model (no audio generation)")
        
        # Parse script to get first chunk
        chunks = []
        if scene_script.strip():
            chunks = parse_scene_script(scene_script)
            if chunks:
                first_chunk = chunks[0]
                print(f"[SceneExtenderMVP] Using first chunk: [{first_chunk.start_sec:.1f}s - {first_chunk.end_sec:.1f}s]")
                print(f"  Prompt: {first_chunk.prompt[:60]}...")
                print(f"  Audio: {first_chunk.audio_spec}")
                print(f"  Guides: {len(first_chunk.guides)}")
        
        # Resolve image references
        resolved_refs = resolve_image_refs(chunks, guide_images)
        
        # Prepare guider
        guider = copy.copy(guider)
        guider.original_conds = copy.deepcopy(guider.original_conds)
        positive, negative = _get_raw_conds_from_guider(guider)
        
        # Get VAE scale factors
        time_scale_factor, width_scale_factor, height_scale_factor = (
            video_vae.downscale_index_formula
        )
        
        # Prepare guide images and indices
        cond_images = None
        cond_indices = None
        
        if chunks and chunks[0].guides and guide_images is not None:
            first_chunk = chunks[0]
            images_list = []
            indices_list = []
            
            time_mgr = TimeManager(video_fps=25.0)
            
            for guide in first_chunk.guides:
                if guide.image_ref in resolved_refs:
                    img = resolved_refs[guide.image_ref]
                    
                    # Calculate frame index
                    pos_sec = guide.get_position_seconds(
                        first_chunk.start_sec,
                        first_chunk.end_sec
                    )
                    # Convert to pixel frame (relative to chunk)
                    relative_sec = pos_sec - first_chunk.start_sec
                    pixel_frame = time_mgr.seconds_to_pixel_frame(relative_sec)
                    
                    images_list.append(img)
                    indices_list.append(pixel_frame)
            
            if images_list:
                cond_images = torch.cat(images_list, dim=0)
                cond_indices = ",".join(str(i) for i in indices_list)
                print(f"[SceneExtenderMVP] Guide images: {cond_images.shape[0]}, indices: {cond_indices}")
        
        # Resize guide images to match output dimensions
        if cond_images is not None:
            cond_images = (
                comfy.utils.common_upscale(
                    cond_images.movedim(-1, 1),
                    width,
                    height,
                    "bilinear",
                    crop="center",
                )
                .movedim(1, -1)
                .clamp(0, 1)
            )
        
        # === GENERATION LOGIC ===
        
        if latent is None:
            # New generation (like LTXVBaseSampler)
            print("[SceneExtenderMVP] Generating new video...")
            
            # Create empty audio and video latents
            # Calculate dimensions
            # Video: [batch, 128, frames, height//32, width//32]
            # Audio: [batch, 128, audio_frames, 1, 1]
            
            latent_height = height // height_scale_factor
            latent_width = width // width_scale_factor
            latent_length = (num_frames - 1) // time_scale_factor + 1
            
            video_latent = torch.zeros(
                [1, 128, latent_length, latent_height, latent_width],
                device=comfy.model_management.intermediate_device()
            )
            
            if using_av_model and NestedTensor is not None:
                # Calculate audio frames: 25 latents per second approx
                # For LTXV, audio is strictly tied to video length
                # 16000Hz / 160 hop / 4 downsample = 25 latents/sec
                # It usually matches video latent length if FPS=25
                audio_length = latent_length 
                
                audio_latent = torch.zeros(
                    [1, 128, audio_length, 1, 1],
                    device=comfy.model_management.intermediate_device()
                )
                
                # Create NestedTensor for AV
                samples = NestedTensor((video_latent, audio_latent))
                output_latent = {"samples": samples}
            else:
                output_latent = {"samples": video_latent}
            
            # Add guide conditioning if available
            if cond_images is not None and cond_indices is not None:
                indices = [int(i) for i in cond_indices.split(",")]
                
                for img, idx in zip(cond_images, indices):
                    if idx == 0:
                        # First frame: use I2V conditioning
                        encode_pixels = img.unsqueeze(0)[:, :, :, :3]
                        t = video_vae.encode(encode_pixels)
                        
                        if using_av_model and NestedTensor is not None:
                             # Update only video part of NestedTensor
                            v_samples = output_latent["samples"].tensors[0]
                            v_samples[:, :, :t.shape[2]] = t
                            # Pack back
                            output_latent["samples"] = NestedTensor((v_samples, output_latent["samples"].tensors[1]))
                        else:
                            output_latent["samples"][:, :, :t.shape[2]] = t
                        
                        # Create noise mask
                        if "noise_mask" not in output_latent:
                            video_mask = torch.ones(
                                (1, 1, video_latent.shape[2], 1, 1),
                                dtype=torch.float32,
                                device=video_latent.device,
                            )
                            video_mask[:, :, :t.shape[2]] = 1.0 - guide_strength
                            
                            if using_av_model and NestedTensor is not None:
                                # Create masks for both video and audio
                                audio_mask = torch.ones(
                                    (1, 1, audio_length, 1, 1),
                                    dtype=torch.float32,
                                    device=audio_latent.device,
                                )
                                output_latent["noise_mask"] = NestedTensor((video_mask, audio_mask))
                            else:
                                output_latent["noise_mask"] = video_mask
                    else:
                        # Other frames: add as guide
                        # LTXVAddGuide only accepts video latents, so we must unpack if AV
                        if using_av_model and NestedTensor is not None:
                            # Extract video component
                            current_video_samples = output_latent["samples"].tensors[0]
                            temp_latent = {"samples": current_video_samples}
                            if "noise_mask" in output_latent:
                                temp_latent["noise_mask"] = output_latent["noise_mask"].tensors[0]
                            
                            # Run AddGuide on video component
                            positive, negative, temp_latent = LTXVAddGuide.execute(
                                positive=positive,
                                negative=negative,
                                vae=video_vae,
                                latent=temp_latent,
                                image=img.unsqueeze(0),
                                frame_idx=idx,
                                strength=guide_strength,
                            )
                            
                            # Update video component in AV latent
                            v_samples = temp_latent["samples"]
                            a_samples = output_latent["samples"].tensors[1]
                            output_latent["samples"] = NestedTensor((v_samples, a_samples))
                            
                            if "noise_mask" in temp_latent:
                                v_mask = temp_latent["noise_mask"]
                                if "noise_mask" in output_latent:
                                    a_mask = output_latent["noise_mask"].tensors[1]
                                else:
                                    # Create default audio mask if missing
                                    a_mask = torch.ones(
                                        (1, 1, audio_length, 1, 1),
                                        dtype=torch.float32,
                                        device=a_samples.device,
                                    )
                                output_latent["noise_mask"] = NestedTensor((v_mask, a_mask))
                        else:
                            # Standard video-only case
                            positive, negative, output_latent = LTXVAddGuide.execute(
                                positive=positive,
                                negative=negative,
                                vae=video_vae,
                                latent=output_latent,
                                image=img.unsqueeze(0),
                                frame_idx=idx,
                                strength=guide_strength,
                            )
            
            # Set conditioning and sample
            guider.set_conds(positive, negative)
            
            _, denoised = SamplerCustomAdvanced().sample(
                noise=noise,
                guider=guider,
                sampler=sampler,
                sigmas=sigmas,
                latent_image=output_latent,
            )
            
            return io.NodeOutput(denoised, positive, negative)
        
        else:
            # Extend existing video (like LTXVExtendSampler)
            print("[SceneExtenderMVP] Extending existing video...")
            
            samples = latent["samples"]
            # Handle AV latent (NestedTensor)
            if NestedTensor is not None and isinstance(samples, NestedTensor):
                print("[SceneExtenderMVP] AV latent detected - extending video component only")
                samples = samples.tensors[0] # Video is index 0
                
            batch, channels, frames, lat_height, lat_width = samples.shape
            overlap = frame_overlap // time_scale_factor
            
            # Get last overlap frames as conditioning
            last_frames = samples[:, :, -overlap:]
            last_latent = {"samples": last_frames}
            
            # Create new latent for extension
            new_frame_count = overlap * time_scale_factor + num_frames
            new_latent = EmptyLTXVLatentVideo().execute(
                lat_width * width_scale_factor,
                lat_height * height_scale_factor,
                new_frame_count,
                1,
            )[0]
            
            # Import LTXVAddLatentGuide for overlap conditioning
            try:
                from custom_nodes.ComfyUI_LTXVideo.latents import LTXVAddLatentGuide
            except ImportError:
                # Fallback: just encode the overlap region directly
                t = last_frames.to(new_latent["samples"].device)
                new_latent["samples"][:, :, :t.shape[2]] = t
                
                # Create noise mask for overlap
                mask = torch.ones(
                    (1, 1, new_latent["samples"].shape[2], 1, 1),
                    dtype=torch.float32,
                    device=new_latent["samples"].device,
                )
                mask[:, :, :t.shape[2]] = 1.0 - overlap_strength
                new_latent["noise_mask"] = mask
                
                positive_ext = positive
                negative_ext = negative
            else:
                # Use proper latent guide
                positive_ext, negative_ext, new_latent = LTXVAddLatentGuide().generate(
                    vae=video_vae,
                    positive=positive,
                    negative=negative,
                    latent=new_latent,
                    guiding_latent=last_latent,
                    latent_idx=0,
                    strength=overlap_strength,
                )
            
            # Add image guide conditioning
            if cond_images is not None and cond_indices is not None:
                indices = [int(i) for i in cond_indices.split(",")]
                for img, idx in zip(cond_images, indices):
                    # Offset index by overlap
                    adjusted_idx = idx + (overlap * time_scale_factor)
                    positive_ext, negative_ext, new_latent = LTXVAddGuide.execute(
                        positive=positive_ext,
                        negative=negative_ext,
                        vae=video_vae,
                        latent=new_latent,
                        image=img.unsqueeze(0),
                        frame_idx=adjusted_idx,
                        strength=guide_strength,
                    )
            
            # Sample
            guider.set_conds(positive_ext, negative_ext)
            
            _, denoised = SamplerCustomAdvanced().sample(
                noise=noise,
                guider=guider,
                sampler=sampler,
                sigmas=sigmas,
                latent_image=new_latent,
            )
            
            # Blend with original using linear transition
            try:
                from custom_nodes.ComfyUI_LTXVideo.easy_samplers import LinearOverlapLatentTransition
                from custom_nodes.ComfyUI_LTXVideo.latents import LTXVSelectLatents
                
                # Drop first frame (reinterpreted overlap)
                truncated = LTXVSelectLatents().select_latents(denoised, 1, -1)[0]
                
                # Blend
                result = LinearOverlapLatentTransition().process(
                    latent, truncated, overlap - 1, axis=2
                )[0]
                
                return io.NodeOutput(result, positive_ext, negative_ext)
            except ImportError:
                # Fallback: simple concatenation
                result_samples = torch.cat([
                    samples[:, :, :-overlap],
                    denoised["samples"]
                ], dim=2)
                
                return io.NodeOutput(
                    {"samples": result_samples},
                    positive_ext,
                    negative_ext
                )
