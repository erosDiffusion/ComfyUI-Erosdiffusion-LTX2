"""
TimeManager: Abstracts all frame/latent/time conversions.

All user-facing inputs use SECONDS - this class handles the complex
internal conversions to/from pixel frames and latent indices.
"""

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class TimeConfig:
    """Configuration for time/frame conversions."""
    video_fps: float = 25.0
    time_scale_factor: int = 8  # Video: 8 pixel frames per latent frame
    audio_sample_rate: int = 16000
    mel_hop_length: int = 160
    latent_downsample_factor: int = 4
    
    @property
    def audio_latents_per_second(self) -> float:
        """Audio latent frames per second."""
        return self.audio_sample_rate / self.mel_hop_length / self.latent_downsample_factor
    
    @property
    def video_latents_per_second(self) -> float:
        """Approximate video latent frames per second."""
        return self.video_fps / self.time_scale_factor


class TimeManager:
    """
    Abstracts all frame/latent/time conversions internally.
    
    Users provide times in SECONDS, this class converts to:
    - Pixel frame indices (for video)
    - Video latent indices
    - Audio latent indices
    
    Key formulas:
    - Video: pixel_frames = (latent_frames - 1) * 8 + 1
    - Audio: audio_latent_frames = seconds * audio_latents_per_second
    """
    
    def __init__(
        self,
        video_fps: float = 25.0,
        audio_sample_rate: int = 16000,
        mel_hop_length: int = 160,
        latent_downsample_factor: int = 4,
        time_scale_factor: int = 8
    ):
        self.config = TimeConfig(
            video_fps=video_fps,
            time_scale_factor=time_scale_factor,
            audio_sample_rate=audio_sample_rate,
            mel_hop_length=mel_hop_length,
            latent_downsample_factor=latent_downsample_factor
        )
    
    # === Time to Frame Conversions ===
    
    def seconds_to_pixel_frame(self, seconds: float) -> int:
        """Convert seconds to pixel frame index."""
        return int(round(seconds * self.config.video_fps))
    
    def pixel_frame_to_seconds(self, pixel_frame: int) -> float:
        """Convert pixel frame index to seconds."""
        return pixel_frame / self.config.video_fps
    
    def seconds_to_video_latent_index(self, seconds: float) -> int:
        """
        Convert seconds to video latent frame index.
        
        Uses the formula: latent_idx = (pixel_frame + time_scale_factor - 1) // time_scale_factor
        With special handling for frame 0.
        """
        pixel_frame = self.seconds_to_pixel_frame(seconds)
        return self._pixel_to_video_latent_index(pixel_frame)
    
    def seconds_to_audio_latent_index(self, seconds: float) -> int:
        """Convert seconds to audio latent frame index."""
        return int(round(seconds * self.config.audio_latents_per_second))
    
    # === Range Conversions ===
    
    def seconds_to_video_latent_range(
        self, 
        start_sec: float, 
        end_sec: float
    ) -> Tuple[int, int]:
        """
        Convert time range (seconds) to video latent frame indices.
        
        Returns (start_latent_idx, end_latent_idx) as a closed interval.
        """
        start_pixel = self.seconds_to_pixel_frame(start_sec)
        end_pixel = self.seconds_to_pixel_frame(end_sec)
        
        start_latent = self._pixel_to_video_latent_index(start_pixel)
        end_latent = self._pixel_to_video_latent_index(end_pixel)
        
        return start_latent, end_latent
    
    def seconds_to_audio_latent_range(
        self, 
        start_sec: float, 
        end_sec: float
    ) -> Tuple[int, int]:
        """
        Convert time range (seconds) to audio latent frame indices.
        
        Returns (start_latent_idx, end_latent_idx) as a closed interval.
        """
        start = int(round(start_sec * self.config.audio_latents_per_second))
        end = int(round(end_sec * self.config.audio_latents_per_second))
        return start, end
    
    # === Chunk Calculations ===
    
    def duration_to_chunk_count(
        self, 
        duration_sec: float, 
        tile_size_sec: float, 
        overlap_sec: float
    ) -> int:
        """
        Calculate how many temporal chunks are needed.
        
        Args:
            duration_sec: Total duration to cover
            tile_size_sec: Size of each temporal tile
            overlap_sec: Overlap between tiles
            
        Returns:
            Number of chunks needed
        """
        if duration_sec <= 0:
            return 0
        if tile_size_sec <= overlap_sec:
            raise ValueError("tile_size_sec must be greater than overlap_sec")
        
        effective_tile = tile_size_sec - overlap_sec
        return max(1, int(math.ceil(duration_sec / effective_tile)))
    
    def get_chunk_time_ranges(
        self,
        total_duration_sec: float,
        tile_size_sec: float,
        overlap_sec: float
    ) -> list[Tuple[float, float]]:
        """
        Get time ranges for all chunks.
        
        Returns list of (start_sec, end_sec) tuples.
        """
        if total_duration_sec <= 0:
            return []
        
        chunks = []
        effective_tile = tile_size_sec - overlap_sec
        current_start = 0.0
        
        while current_start < total_duration_sec:
            chunk_end = min(current_start + tile_size_sec, total_duration_sec)
            chunks.append((current_start, chunk_end))
            current_start += effective_tile
            
            # Prevent infinite loop if we're at the end
            if chunk_end >= total_duration_sec:
                break
        
        return chunks
    
    def calculate_video_latent_count(self, duration_sec: float) -> int:
        """Calculate total video latent frames for a duration."""
        pixel_frames = self.seconds_to_pixel_frame(duration_sec)
        # Formula: latent_frames = (pixel_frames - 1) // time_scale_factor + 1
        if pixel_frames <= 0:
            return 0
        return (pixel_frames - 1) // self.config.time_scale_factor + 1
    
    def calculate_audio_latent_count(self, duration_sec: float) -> int:
        """Calculate total audio latent frames for a duration."""
        return int(round(duration_sec * self.config.audio_latents_per_second))
    
    # === Internal Helpers ===
    
    def _pixel_to_video_latent_index(self, pixel_frame: int) -> int:
        """
        Convert pixel frame to video latent index.
        
        Matches the logic in LTXVSetAudioVideoMaskByTime:
        - Frame 0 maps to latent 0
        - Frames 1-8 map to latent 1
        - Frames 9-16 map to latent 2
        - etc.
        """
        if pixel_frame <= 0:
            return 0
        
        # Build the xp array for searchsorted (same as Lightricks code)
        # xp = [0, 1, 9, 17, 25, ...] for time_scale_factor=8
        tsf = self.config.time_scale_factor
        max_latent = (pixel_frame + tsf - 1) // tsf + 1
        xp = np.array([0] + list(range(1, max_latent * tsf + 1, tsf)))
        
        # Use searchsorted to find the latent index
        latent_idx = np.searchsorted(xp, pixel_frame, side='left')
        return int(latent_idx)
    
    def video_latent_to_pixel_frame(self, latent_idx: int) -> int:
        """
        Convert video latent index to pixel frame.
        
        - Latent 0 -> frame 0
        - Latent 1 -> frame 1
        - Latent 2 -> frame 9
        - Latent n -> frame 1 + (n-1) * 8 for n > 0
        """
        if latent_idx <= 0:
            return 0
        if latent_idx == 1:
            return 1
        return 1 + (latent_idx - 1) * self.config.time_scale_factor
