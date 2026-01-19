"""
Unit tests for TimeManager.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from time_manager import TimeManager, TimeConfig


def test_time_config_defaults():
    """Test TimeConfig default values."""
    config = TimeConfig()
    assert config.video_fps == 25.0
    assert config.time_scale_factor == 8
    assert config.audio_sample_rate == 16000
    assert config.mel_hop_length == 160
    assert config.latent_downsample_factor == 4
    
    # Derived values
    assert config.audio_latents_per_second == 25.0  # 16000 / 160 / 4
    assert config.video_latents_per_second == 25.0 / 8  # ~3.125


def test_seconds_to_pixel_frame():
    """Test conversion from seconds to pixel frames."""
    tm = TimeManager(video_fps=25.0)
    
    assert tm.seconds_to_pixel_frame(0.0) == 0
    assert tm.seconds_to_pixel_frame(1.0) == 25
    assert tm.seconds_to_pixel_frame(2.0) == 50
    assert tm.seconds_to_pixel_frame(0.5) == 12  # round(0.5 * 25) = 12


def test_pixel_frame_to_seconds():
    """Test conversion from pixel frames to seconds."""
    tm = TimeManager(video_fps=25.0)
    
    assert tm.pixel_frame_to_seconds(0) == 0.0
    assert tm.pixel_frame_to_seconds(25) == 1.0
    assert tm.pixel_frame_to_seconds(50) == 2.0


def test_seconds_to_video_latent_index():
    """Test conversion from seconds to video latent indices."""
    tm = TimeManager(video_fps=25.0)
    
    # At 25fps with time_scale_factor=8:
    # 0s = frame 0 = latent 0
    # 1s = frame 25 = latent ~3-4
    assert tm.seconds_to_video_latent_index(0.0) == 0


def test_seconds_to_audio_latent_index():
    """Test conversion from seconds to audio latent indices."""
    tm = TimeManager()  # Default: 25 audio latents per second
    
    assert tm.seconds_to_audio_latent_index(0.0) == 0
    assert tm.seconds_to_audio_latent_index(1.0) == 25
    assert tm.seconds_to_audio_latent_index(2.0) == 50


def test_duration_to_chunk_count():
    """Test chunk count calculation."""
    tm = TimeManager()
    
    # 5 seconds, 3 second tiles, 1 second overlap
    # Effective tile = 3 - 1 = 2 seconds
    # Chunks needed = ceil(5 / 2) = 3
    assert tm.duration_to_chunk_count(5.0, 3.0, 1.0) == 3
    
    # 10 seconds, 4 second tiles, 1 second overlap
    # Effective tile = 3 seconds
    # Chunks needed = ceil(10 / 3) = 4
    assert tm.duration_to_chunk_count(10.0, 4.0, 1.0) == 4
    
    # Edge case: duration fits in one tile
    assert tm.duration_to_chunk_count(2.0, 3.0, 1.0) == 1


def test_get_chunk_time_ranges():
    """Test generation of chunk time ranges."""
    tm = TimeManager()
    
    # 6 seconds, 3 second tiles, 1 second overlap
    ranges = tm.get_chunk_time_ranges(6.0, 3.0, 1.0)
    
    assert len(ranges) == 3
    assert ranges[0] == (0.0, 3.0)
    assert ranges[1] == (2.0, 5.0)
    assert ranges[2] == (4.0, 6.0)


def test_calculate_video_latent_count():
    """Test video latent frame count calculation."""
    tm = TimeManager(video_fps=25.0)
    
    # 1 second at 25fps = 25 pixel frames
    # Latent frames = (25 - 1) // 8 + 1 = 24 // 8 + 1 = 4
    assert tm.calculate_video_latent_count(1.0) == 4


def test_calculate_audio_latent_count():
    """Test audio latent frame count calculation."""
    tm = TimeManager()
    
    assert tm.calculate_audio_latent_count(1.0) == 25
    assert tm.calculate_audio_latent_count(2.0) == 50


def test_video_latent_to_pixel_frame():
    """Test reverse conversion from latent to pixel frame."""
    tm = TimeManager()
    
    assert tm.video_latent_to_pixel_frame(0) == 0
    assert tm.video_latent_to_pixel_frame(1) == 1
    assert tm.video_latent_to_pixel_frame(2) == 9  # 1 + (2-1) * 8
    assert tm.video_latent_to_pixel_frame(3) == 17  # 1 + (3-1) * 8


if __name__ == "__main__":
    # Run tests
    test_time_config_defaults()
    print("[PASS] test_time_config_defaults")
    
    test_seconds_to_pixel_frame()
    print("[PASS] test_seconds_to_pixel_frame")
    
    test_pixel_frame_to_seconds()
    print("[PASS] test_pixel_frame_to_seconds")
    
    test_seconds_to_video_latent_index()
    print("[PASS] test_seconds_to_video_latent_index")
    
    test_seconds_to_audio_latent_index()
    print("[PASS] test_seconds_to_audio_latent_index")
    
    test_duration_to_chunk_count()
    print("[PASS] test_duration_to_chunk_count")
    
    test_get_chunk_time_ranges()
    print("[PASS] test_get_chunk_time_ranges")
    
    test_calculate_video_latent_count()
    print("[PASS] test_calculate_video_latent_count")
    
    test_calculate_audio_latent_count()
    print("[PASS] test_calculate_audio_latent_count")
    
    test_video_latent_to_pixel_frame()
    print("[PASS] test_video_latent_to_pixel_frame")
    
    print("\nAll tests passed!")
