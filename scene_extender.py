"""
LTXVSceneExtender: All-in-one node for extending video scenes with synchronized audio.

Combines LTXVLoopingSampler + LTXVNormalizingSampler + image guides into a single
node that processes timestamped scene scripts.
"""

import copy
from typing import Optional

import torch
from comfy_api.latest import io

# Try to import ComfyUI components - handle gracefully if not available
try:
    import comfy.model_management as mm
    import comfy.utils
    from comfy.nested_tensor import NestedTensor
    COMFY_AVAILABLE = True
except ImportError:
    COMFY_AVAILABLE = False
    NestedTensor = None

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
            display_name="LTXV Scene Extender",
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
        extended_audio = None
        
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}: [{chunk.start_sec:.1f}s - {chunk.end_sec:.1f}s]")
            print(f"  Prompt: {chunk.prompt[:50]}...")
            print(f"  Audio: {chunk.audio_spec}")
            print(f"  Guides: {len(chunk.guides)}")
            
            # Encode chunk prompt
            chunk_cond = cls._encode_prompt(clip, chunk.prompt)
            
            # Get guide images for this chunk
            chunk_images, chunk_indices = get_chunk_guide_images(
                chunk, resolved_refs, time_mgr
            )
            
            # Calculate frame count for this chunk
            chunk_duration = chunk.end_sec - chunk.start_sec
            chunk_frames = time_mgr.seconds_to_pixel_frame(chunk_duration)
            
            # For now, we'll create empty latents and return them
            # Full implementation would use LTXVExtendSampler/LTXVBaseSampler
            if extended_latent is None:
                # First chunk: create new latent
                latent_frames = time_mgr.calculate_video_latent_count(chunk_duration)
                extended_video = torch.zeros(
                    [1, 128, latent_frames, latent_height, latent_width],
                    device=mm.intermediate_device() if COMFY_AVAILABLE else "cpu"
                )
                extended_latent = {"samples": extended_video}
            
            # TODO: Implement actual sampling using LTXVExtendSampler patterns
            # This is a placeholder that shows the structure
        
        # Handle audio output
        if extended_audio is None:
            extended_audio = torch.zeros([1, 64, 1, 1])  # Placeholder
        
        # Combine outputs
        video_output = {"samples": extended_video}
        audio_output = {"samples": extended_audio}
        
        if is_av_model and COMFY_AVAILABLE and NestedTensor is not None:
            combined = {"samples": NestedTensor((extended_video, extended_audio))}
        else:
            combined = video_output
        
        return io.NodeOutput(
            combined,
            video_output,
            audio_output,
            positive,
            negative
        )
    
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
        try:
            return guider.raw_conds
        except AttributeError:
            try:
                return guider.original_conds
            except AttributeError:
                return None, None
    
    @classmethod
    def _encode_prompt(cls, clip, prompt: str):
        """Encode a text prompt using CLIP."""
        tokens = clip.tokenize(prompt)
        return clip.encode_from_tokens_scheduled(tokens)
    
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
