"""
Audio overlap blender for seamless chunk transitions.

Provides smooth crossfade blending at chunk boundaries to avoid
audible seams in generated audio.
"""

import torch
from typing import Optional


class AudioOverlapBlender:
    """
    Smooth audio blending for seamless chunk transitions.
    
    Uses linear crossfade with configurable slope length for
    gradual transitions that avoid audible artifacts.
    """
    
    def __init__(
        self,
        overlap_frames: int,
        slope_len: int = 5,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the audio blender.
        
        Args:
            overlap_frames: Number of audio latent frames to overlap
            slope_len: Additional frames for fade ramp (smoother = larger)
            device: Torch device for tensor operations
        """
        self.overlap_frames = overlap_frames
        self.slope_len = slope_len
        self.device = device or torch.device("cpu")
    
    def create_crossfade_mask(self, chunk_length: int) -> torch.Tensor:
        """
        Create smooth crossfade weights for audio overlap.
        
        Creates a mask that:
        - Fades in from 0 to 1 at the start (for blending with previous)
        - Stays at 1.0 in the middle
        - Fades out from 1 to 0 at the end (for blending with next)
        
        Args:
            chunk_length: Total length of the audio chunk in frames
            
        Returns:
            Weight tensor of shape [chunk_length]
        """
        weights = torch.ones(chunk_length, device=self.device)
        
        fade_len = self.overlap_frames + self.slope_len
        
        # Fade in at start
        if fade_len > 0 and fade_len <= chunk_length:
            fade_in = torch.linspace(0, 1, fade_len, device=self.device)
            weights[:fade_len] = fade_in
        
        # Fade out at end
        if fade_len > 0 and fade_len <= chunk_length:
            fade_out = torch.linspace(1, 0, fade_len, device=self.device)
            weights[-fade_len:] = fade_out
        
        return weights
    
    def blend_chunks(
        self,
        prev_audio: torch.Tensor,
        next_audio: torch.Tensor
    ) -> torch.Tensor:
        """
        Blend two audio chunks at overlap for seamless transition.
        
        The overlap region uses linear interpolation weighted by
        position to create a smooth crossfade.
        
        Args:
            prev_audio: Previous audio chunk [batch, channels, frames, ...]
            next_audio: Next audio chunk [batch, channels, frames, ...]
            
        Returns:
            Combined audio with blended overlap region
        """
        if self.overlap_frames <= 0:
            # No overlap, just concatenate
            return torch.cat([prev_audio, next_audio], dim=2)
        
        # Ensure we don't exceed chunk sizes
        actual_overlap = min(
            self.overlap_frames,
            prev_audio.shape[2],
            next_audio.shape[2]
        )
        
        if actual_overlap <= 0:
            return torch.cat([prev_audio, next_audio], dim=2)
        
        # Extract overlap regions
        prev_tail = prev_audio[:, :, -actual_overlap:]
        next_head = next_audio[:, :, :actual_overlap]
        
        # Create crossfade weights
        # Shape: [1, 1, overlap, 1...] to broadcast
        alpha = torch.linspace(
            1, 0, actual_overlap,
            device=prev_audio.device,
            dtype=prev_audio.dtype
        )
        
        # Reshape for broadcasting
        # Audio latent is [batch, channels, frames, freq_bins]
        # Alpha needs to be [1, 1, actual_overlap, 1] to broadcast
        if prev_tail.dim() == 4:
            # Audio: [B, C, T, F] -> alpha: [1, 1, overlap, 1]
            alpha = alpha.view(1, 1, actual_overlap, 1)
        elif prev_tail.dim() == 5:
            # Video: [B, C, T, H, W] -> alpha: [1, 1, overlap, 1, 1]
            alpha = alpha.view(1, 1, actual_overlap, 1, 1)
        else:
            # Fallback: add dims at front
            while alpha.dim() < prev_tail.dim():
                alpha = alpha.unsqueeze(0)
        
        alpha = alpha.expand_as(prev_tail)
        
        # Weighted blend
        blended = prev_tail * alpha + next_head * (1 - alpha)
        
        # Construct final audio
        result = torch.cat([
            prev_audio[:, :, :-actual_overlap],
            blended,
            next_audio[:, :, actual_overlap:]
        ], dim=2)
        
        return result
    
    def blend_multiple_chunks(
        self,
        chunks: list[torch.Tensor]
    ) -> torch.Tensor:
        """
        Blend multiple audio chunks sequentially.
        
        Args:
            chunks: List of audio chunk tensors
            
        Returns:
            Single combined audio tensor with all chunks blended
        """
        if not chunks:
            raise ValueError("No chunks to blend")
        
        if len(chunks) == 1:
            return chunks[0]
        
        result = chunks[0]
        for chunk in chunks[1:]:
            result = self.blend_chunks(result, chunk)
        
        return result


def get_audio_blend_coefficients(
    frame_index_start: int,
    frame_index_end: int,
    frame_count: int,
    slope_len: int = 3
) -> list[float]:
    """
    Create blend coefficients with smooth ramps.
    
    Based on Lightricks' get_video_latent_blend_coefficients pattern.
    
    Creates coefficients that:
    - Are 0.0 outside the range [start, end]
    - Ramp up from 0.0 to 1.0 over slope_len frames before start
    - Stay at 1.0 during [start, end]
    - Ramp down from 1.0 to 0.0 over slope_len frames after end
    
    Args:
        frame_index_start: Start frame of active region
        frame_index_end: End frame of active region
        frame_count: Total number of frames
        slope_len: Length of ramp in frames
        
    Returns:
        List of blend coefficients, one per frame
    """
    coeffs = [0.0] * frame_count
    
    # Clamp arguments to safe range
    frame_index_start = max(0, min(frame_count - 1, frame_index_start))
    frame_index_end = max(frame_index_start, min(frame_count - 1, frame_index_end))
    slope_len = max(1, slope_len)
    
    # Ramp up before start
    ramp_start = max(0, frame_index_start - slope_len)
    for i in range(ramp_start, frame_index_start):
        coeffs[i] = (i - ramp_start + 1) / slope_len
    
    # Plateau at 1.0
    for i in range(frame_index_start, frame_index_end + 1):
        coeffs[i] = 1.0
    
    # Ramp down after end
    ramp_end = min(frame_count, frame_index_end + slope_len + 1)
    for i in range(frame_index_end + 1, ramp_end):
        coeffs[i] = 1.0 - ((i - frame_index_end) / slope_len)
        coeffs[i] = max(0.0, coeffs[i])
    
    return coeffs


def normalize_audio_volume(
    audio: torch.Tensor,
    target_rms: float = 0.1
) -> torch.Tensor:
    """
    Normalize audio volume to target RMS level.
    
    This helps ensure consistent volume across chunks
    before blending.
    
    Args:
        audio: Audio tensor
        target_rms: Target RMS level
        
    Returns:
        Normalized audio tensor
    """
    # Calculate current RMS
    rms = torch.sqrt(torch.mean(audio ** 2))
    
    if rms > 0:
        # Scale to target RMS
        scale = target_rms / rms
        return audio * scale
    
    return audio
