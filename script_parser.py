"""
Script parser for timestamped scene scripts with audio specs and image guides.

Parses the scene script format:
[MM:SS-MM:SS] SCENE_DESCRIPTION | audio:SPEC | first:$0 | MM:SS:$1 | end:$2
"""

import re
from dataclasses import dataclass, field
from typing import Optional, Union

import torch


@dataclass
class ImageGuide:
    """Represents an image guide at a specific position."""
    position: str  # "first", "end", or timestamp string like "00:03" or "00:03.5"
    image_ref: str  # "$0", "$1", etc. or file path
    strength: float = 1.0
    
    def get_position_seconds(self, chunk_start: float, chunk_end: float) -> float:
        """
        Convert position to absolute seconds.
        
        Args:
            chunk_start: Start time of the chunk in seconds
            chunk_end: End time of the chunk in seconds
            
        Returns:
            Absolute position in seconds
        """
        if self.position == "first":
            return chunk_start
        elif self.position == "end":
            return chunk_end
        elif self.position == "middle":
            return (chunk_start + chunk_end) / 2
        else:
            # Parse timestamp MM:SS or MM:SS.ms
            return parse_timestamp(self.position)


@dataclass
class SceneChunk:
    """Represents a single scene chunk with timing, prompt, audio, and guides."""
    start_sec: float
    end_sec: float
    prompt: str
    audio_spec: str = "silent"  # "silent", "ambient", or dialogue text
    guides: list[ImageGuide] = field(default_factory=list)
    shot_name: Optional[str] = None  # Optional shot grouping
    transition_type: str = "blend" # "blend" or "cut"
    
    @property
    def duration(self) -> float:
        """Duration of this chunk in seconds."""
        return self.end_sec - self.start_sec
    
    @property
    def is_silent(self) -> bool:
        """Check if audio is silent."""
        return self.audio_spec.lower() == "silent"
    
    @property
    def is_ambient(self) -> bool:
        """Check if audio is ambient only."""
        return self.audio_spec.lower() == "ambient"
    
    @property
    def dialogue(self) -> Optional[str]:
        """Get dialogue text if present, None otherwise."""
        if self.is_silent or self.is_ambient:
            return None
        return self.audio_spec


def parse_timestamp(ts: str) -> float:
    """
    Parse a timestamp string to seconds.
    
    Supports formats:
    - "MM:SS" -> minutes and seconds
    - "MM:SS.ms" -> with milliseconds
    - "SS" -> seconds only
    - "SS.ms" -> seconds with milliseconds
    
    Args:
        ts: Timestamp string
        
    Returns:
        Time in seconds as float
    """
    ts = ts.strip()
    
    # Try MM:SS.ms or MM:SS format
    match = re.match(r"(\d{1,2}):(\d{2})(?:\.(\d+))?", ts)
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        ms = float(f"0.{match.group(3)}") if match.group(3) else 0.0
        return minutes * 60 + seconds + ms
    
    # Try SS.ms or SS format
    match = re.match(r"(\d+)(?:\.(\d+))?", ts)
    if match:
        seconds = int(match.group(1))
        ms = float(f"0.{match.group(2)}") if match.group(2) else 0.0
        return seconds + ms
    
    raise ValueError(f"Invalid timestamp format: {ts}")


def format_timestamp(seconds: float) -> str:
    """
    Format seconds to MM:SS timestamp.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted timestamp string
    """
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes:02d}:{secs:02d}"


def parse_guide_spec(spec: str, chunk_start: float, chunk_end: float) -> Optional[ImageGuide]:
    """
    Parse a guide specification string.
    
    Formats:
    - "first:$0" or "first:image.png"
    - "end:$1"
    - "middle:$2"
    - "00:03:$3" or "00:03.5:$4"
    - "00:03:image.png @ 0.8" (with strength)
    
    Args:
        spec: Guide specification string
        chunk_start: Start time of chunk in seconds
        chunk_end: End time of chunk in seconds
        
    Returns:
        ImageGuide or None if parsing fails
    """
    spec = spec.strip()
    if not spec:
        return None
    
    # Check for strength modifier (@ 0.8)
    strength = 1.0
    if " @ " in spec:
        spec, strength_str = spec.rsplit(" @ ", 1)
        try:
            strength = float(strength_str.strip())
        except ValueError:
            pass
    
    # Parse position:image_ref format
    if ":" in spec:
        parts = spec.split(":", 1)
        position_part = parts[0].strip().lower()
        image_ref = parts[1].strip() if len(parts) > 1 else ""
        
        # Check if position is a keyword or timestamp
        if position_part in ("first", "end", "middle"):
            return ImageGuide(
                position=position_part,
                image_ref=image_ref,
                strength=strength
            )
        else:
            # Assume it's a timestamp like "00:03"
            # Need to handle MM:SS:image format by rejoining
            if len(parts) > 1 and ":" in parts[1]:
                # Format is probably "MM:SS:image"
                ts_parts = spec.split(":")
                if len(ts_parts) >= 3:
                    timestamp = f"{ts_parts[0]}:{ts_parts[1]}"
                    image_ref = ":".join(ts_parts[2:])
                    return ImageGuide(
                        position=timestamp,
                        image_ref=image_ref.strip(),
                        strength=strength
                    )
            
            # Simple position:ref format
            return ImageGuide(
                position=position_part,
                image_ref=image_ref,
                strength=strength
            )
    
    return None


def parse_audio_spec(spec: str) -> str:
    """
    Parse audio specification.
    
    Formats:
    - "audio:silent"
    - "audio:ambient"
    - 'audio:"dialogue text"'
    
    Args:
        spec: Audio specification string
        
    Returns:
        Audio spec: "silent", "ambient", or dialogue text
    """
    spec = spec.strip()
    
    if not spec.lower().startswith("audio:"):
        return "silent"
    
    content = spec[6:].strip()  # Remove "audio:" prefix
    
    if content.lower() == "silent":
        return "silent"
    elif content.lower() == "ambient":
        return "ambient"
    elif content.startswith('"') and content.endswith('"'):
        return content[1:-1]  # Remove quotes
    elif content.startswith("'") and content.endswith("'"):
        return content[1:-1]  # Remove quotes
    else:
        return content


def parse_scene_script(text: str) -> list[SceneChunk]:
    """
    Parse a complete scene script into SceneChunk objects.
    
    Script format:
    ```
    # === SHOT NAME ===
    [MM:SS-MM:SS] Scene description | audio:spec | guide_specs... | transition:cut
    ```
    
    Args:
        text: Complete script text
        
    Returns:
        List of SceneChunk objects
    """
    chunks = []
    current_shot = None
    
    for line in text.strip().split("\n"):
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # Check for shot header: # === SHOT NAME ===
        shot_match = re.match(r"#\s*===\s*(.+?)\s*===", line)
        if shot_match:
            current_shot = shot_match.group(1).strip()
            continue
        
        # Skip other comments
        if line.startswith("#"):
            continue
        
        # Parse timestamped line: [MM:SS-MM:SS] content
        ts_match = re.match(
            r"\[(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\]\s*(.+)",
            line
        )
        if ts_match:
            start_min = int(ts_match.group(1))
            start_sec = int(ts_match.group(2))
            end_min = int(ts_match.group(3))
            end_sec = int(ts_match.group(4))
            content = ts_match.group(5)
            
            start_time = start_min * 60 + start_sec
            end_time = end_min * 60 + end_sec
            
            # Split content by | to get prompt, audio, and guides
            parts = [p.strip() for p in content.split("|")]
            prompt = parts[0] if parts else ""
            
            audio_spec = "silent"
            guides = []
            transition_type = "blend"
            
            for part in parts[1:]:
                part = part.strip()
                lower_part = part.lower()
                
                if lower_part.startswith("audio:"):
                    audio_spec = parse_audio_spec(part)
                elif lower_part.startswith("transition:"):
                    t_val = lower_part.split(":", 1)[1].strip()
                    if t_val in ("cut", "hard"):
                        transition_type = "cut"
                    else:
                        transition_type = "blend" # explicit blend
                else:
                    guide = parse_guide_spec(part, start_time, end_time)
                    if guide:
                        guides.append(guide)
            
            chunk = SceneChunk(
                start_sec=float(start_time),
                end_sec=float(end_time),
                prompt=prompt,
                audio_spec=audio_spec,
                guides=guides,
                shot_name=current_shot,
                transition_type=transition_type
            )
            chunks.append(chunk)
    
    return chunks


def resolve_image_refs(
    chunks: list[SceneChunk],
    guide_images: Optional[torch.Tensor]
) -> dict[str, torch.Tensor]:
    """
    Resolve image references ($0, $1, etc.) to actual tensors.
    
    Args:
        chunks: List of SceneChunk objects
        guide_images: Batch of guide images [N, H, W, C]
        
    Returns:
        Dict mapping image_ref to tensor
    """
    resolved = {}
    
    if guide_images is None:
        return resolved
    
    # Collect all unique refs
    all_refs = set()
    for chunk in chunks:
        for guide in chunk.guides:
            if guide.image_ref.startswith("$"):
                all_refs.add(guide.image_ref)
    
    # Resolve each ref
    for ref in all_refs:
        if ref.startswith("$"):
            try:
                idx = int(ref[1:])
                if 0 <= idx < guide_images.shape[0]:
                    resolved[ref] = guide_images[idx:idx+1]
            except ValueError:
                pass
    
    return resolved


def get_chunk_guide_images(
    chunk: SceneChunk,
    resolved_refs: dict[str, torch.Tensor],
    time_manager: "TimeManager"  # Forward reference
) -> tuple[Optional[torch.Tensor], Optional[str]]:
    """
    Get guide images and indices for a chunk.
    
    Args:
        chunk: SceneChunk to process
        resolved_refs: Dict of resolved image references
        time_manager: TimeManager for time conversions
        
    Returns:
        Tuple of (stacked images tensor, comma-separated indices string)
    """
    if not chunk.guides:
        return None, None
    
    images = []
    indices = []
    
    for guide in chunk.guides:
        # Get image tensor
        if guide.image_ref in resolved_refs:
            img = resolved_refs[guide.image_ref]
        else:
            # TODO: Load from file path
            continue
        
        # Get frame index
        pos_seconds = guide.get_position_seconds(chunk.start_sec, chunk.end_sec)
        # Convert to frame index relative to chunk start
        relative_seconds = pos_seconds - chunk.start_sec
        pixel_frame = time_manager.seconds_to_pixel_frame(relative_seconds)
        
        images.append(img)
        indices.append(str(pixel_frame))
    
    if not images:
        return None, None
    
    stacked = torch.cat(images, dim=0)
    indices_str = ",".join(indices)
    
    return stacked, indices_str
