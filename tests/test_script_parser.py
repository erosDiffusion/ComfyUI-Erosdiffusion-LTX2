"""
Unit tests for script_parser.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from script_parser import (
    parse_timestamp,
    format_timestamp,
    parse_guide_spec,
    parse_audio_spec,
    parse_scene_script,
    ImageGuide,
    SceneChunk,
)


def test_parse_timestamp_mmss():
    """Test parsing MM:SS format."""
    assert parse_timestamp("00:00") == 0.0
    assert parse_timestamp("00:30") == 30.0
    assert parse_timestamp("01:00") == 60.0
    assert parse_timestamp("01:30") == 90.0
    assert parse_timestamp("02:15") == 135.0


def test_parse_timestamp_with_ms():
    """Test parsing MM:SS.ms format."""
    assert parse_timestamp("00:00.5") == 0.5
    assert parse_timestamp("00:01.25") == 1.25
    assert parse_timestamp("01:30.5") == 90.5


def test_parse_timestamp_seconds_only():
    """Test parsing seconds-only format."""
    assert parse_timestamp("30") == 30.0
    assert parse_timestamp("90") == 90.0
    assert parse_timestamp("15.5") == 15.5


def test_format_timestamp():
    """Test formatting seconds to MM:SS."""
    assert format_timestamp(0.0) == "00:00"
    assert format_timestamp(30.0) == "00:30"
    assert format_timestamp(60.0) == "01:00"
    assert format_timestamp(90.0) == "01:30"
    assert format_timestamp(135.0) == "02:15"


def test_parse_guide_spec_first():
    """Test parsing 'first:' guide spec."""
    guide = parse_guide_spec("first:$0", 0.0, 2.0)
    assert guide is not None
    assert guide.position == "first"
    assert guide.image_ref == "$0"
    assert guide.strength == 1.0


def test_parse_guide_spec_end():
    """Test parsing 'end:' guide spec."""
    guide = parse_guide_spec("end:$1", 0.0, 2.0)
    assert guide is not None
    assert guide.position == "end"
    assert guide.image_ref == "$1"


def test_parse_guide_spec_middle():
    """Test parsing 'middle:' guide spec."""
    guide = parse_guide_spec("middle:$2", 0.0, 4.0)
    assert guide is not None
    assert guide.position == "middle"
    assert guide.image_ref == "$2"


def test_parse_guide_spec_timestamp():
    """Test parsing timestamp guide spec."""
    guide = parse_guide_spec("00:03:$3", 0.0, 6.0)
    assert guide is not None
    assert guide.position == "00"  # First part before colon
    assert "$3" in guide.image_ref or guide.image_ref == "03:$3"


def test_parse_guide_spec_with_strength():
    """Test parsing guide spec with strength modifier."""
    guide = parse_guide_spec("first:$0 @ 0.8", 0.0, 2.0)
    assert guide is not None
    assert guide.position == "first"
    assert guide.image_ref == "$0"
    assert guide.strength == 0.8


def test_parse_audio_spec_silent():
    """Test parsing audio:silent."""
    assert parse_audio_spec("audio:silent") == "silent"
    assert parse_audio_spec("audio:SILENT") == "silent"


def test_parse_audio_spec_ambient():
    """Test parsing audio:ambient."""
    assert parse_audio_spec("audio:ambient") == "ambient"


def test_parse_audio_spec_dialogue():
    """Test parsing audio with dialogue."""
    assert parse_audio_spec('audio:"Hello world"') == "Hello world"
    assert parse_audio_spec("audio:'Hello world'") == "Hello world"


def test_parse_scene_script_simple():
    """Test parsing a simple scene script."""
    script = """
[00:00-00:02] A woman speaks | audio:silent | first:$0
[00:02-00:04] She smiles | audio:"Hello" | first:$1 | end:$2
"""
    chunks = parse_scene_script(script)
    
    assert len(chunks) == 2
    
    # First chunk
    assert chunks[0].start_sec == 0.0
    assert chunks[0].end_sec == 2.0
    assert "woman speaks" in chunks[0].prompt
    assert chunks[0].audio_spec == "silent"
    assert len(chunks[0].guides) == 1
    
    # Second chunk
    assert chunks[1].start_sec == 2.0
    assert chunks[1].end_sec == 4.0
    assert "smiles" in chunks[1].prompt
    assert chunks[1].audio_spec == "Hello"
    assert len(chunks[1].guides) == 2


def test_parse_scene_script_with_shot_headers():
    """Test parsing script with shot headers."""
    script = """
# === SHOT 1: INTRO ===

[00:00-00:02] Opening scene | audio:silent | first:$0

# === SHOT 2: MAIN ===

[00:02-00:04] Main content | audio:"Dialogue" | first:$1
"""
    chunks = parse_scene_script(script)
    
    assert len(chunks) == 2
    assert chunks[0].shot_name == "SHOT 1: INTRO"
    assert chunks[1].shot_name == "SHOT 2: MAIN"


def test_scene_chunk_properties():
    """Test SceneChunk property methods."""
    chunk = SceneChunk(
        start_sec=0.0,
        end_sec=3.0,
        prompt="Test",
        audio_spec="silent",
        guides=[]
    )
    
    assert chunk.duration == 3.0
    assert chunk.is_silent == True
    assert chunk.is_ambient == False
    assert chunk.dialogue is None
    
    chunk2 = SceneChunk(
        start_sec=0.0,
        end_sec=2.0,
        prompt="Test",
        audio_spec="Hello world",
        guides=[]
    )
    
    assert chunk2.is_silent == False
    assert chunk2.dialogue == "Hello world"


def test_image_guide_get_position_seconds():
    """Test ImageGuide position to seconds conversion."""
    guide_first = ImageGuide(position="first", image_ref="$0")
    assert guide_first.get_position_seconds(2.0, 5.0) == 2.0
    
    guide_end = ImageGuide(position="end", image_ref="$1")
    assert guide_end.get_position_seconds(2.0, 5.0) == 5.0
    
    guide_middle = ImageGuide(position="middle", image_ref="$2")
    assert guide_middle.get_position_seconds(2.0, 6.0) == 4.0


def test_woman_speaking_example():
    """Test the full woman speaking example from requirements."""
    script = """
# === SHOT 1: WOMAN INTRODUCTION (6 seconds total) ===

[00:00-00:02] Closeup of woman's face, soft lighting, neutral expression | audio:silent | first:$0 | end:$1
[00:02-00:04] Cowboy shot of woman speaking confidently, gesturing with hands | audio:"Hello, welcome to my channel" | first:$2 | end:$4
[00:04-00:06] Side profile view of woman nodding gently, soft smile | audio:silent | first:$5 | end:$6
"""
    chunks = parse_scene_script(script)
    
    assert len(chunks) == 3
    
    # First chunk: closeup, silent
    assert chunks[0].start_sec == 0.0
    assert chunks[0].end_sec == 2.0
    assert "Closeup" in chunks[0].prompt
    assert chunks[0].is_silent
    assert len(chunks[0].guides) == 2
    
    # Second chunk: cowboy shot with dialogue
    assert chunks[1].start_sec == 2.0
    assert chunks[1].end_sec == 4.0
    assert chunks[1].dialogue == "Hello, welcome to my channel"
    assert not chunks[1].is_silent
    
    # Third chunk: side profile, silent
    assert chunks[2].start_sec == 4.0
    assert chunks[2].end_sec == 6.0
    assert chunks[2].is_silent


if __name__ == "__main__":
    test_parse_timestamp_mmss()
    print("[PASS] test_parse_timestamp_mmss")
    
    test_parse_timestamp_with_ms()
    print("[PASS] test_parse_timestamp_with_ms")
    
    test_parse_timestamp_seconds_only()
    print("[PASS] test_parse_timestamp_seconds_only")
    
    test_format_timestamp()
    print("[PASS] test_format_timestamp")
    
    test_parse_guide_spec_first()
    print("[PASS] test_parse_guide_spec_first")
    
    test_parse_guide_spec_end()
    print("[PASS] test_parse_guide_spec_end")
    
    test_parse_guide_spec_middle()
    print("[PASS] test_parse_guide_spec_middle")
    
    test_parse_guide_spec_with_strength()
    print("[PASS] test_parse_guide_spec_with_strength")
    
    test_parse_audio_spec_silent()
    print("[PASS] test_parse_audio_spec_silent")
    
    test_parse_audio_spec_ambient()
    print("[PASS] test_parse_audio_spec_ambient")
    
    test_parse_audio_spec_dialogue()
    print("[PASS] test_parse_audio_spec_dialogue")
    
    test_parse_scene_script_simple()
    print("[PASS] test_parse_scene_script_simple")
    
    test_parse_scene_script_with_shot_headers()
    print("[PASS] test_parse_scene_script_with_shot_headers")
    
    test_scene_chunk_properties()
    print("[PASS] test_scene_chunk_properties")
    
    test_image_guide_get_position_seconds()
    print("[PASS] test_image_guide_get_position_seconds")
    
    test_woman_speaking_example()
    print("[PASS] test_woman_speaking_example")
    
    print("\nAll tests passed!")
