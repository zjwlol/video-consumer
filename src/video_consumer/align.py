"""Stage 4: group transcript segments into sections anchored on frames."""
from __future__ import annotations
from dataclasses import dataclass, field
from .frames import Frame
from .transcribe import Segment

@dataclass
class Section:
  frame:Frame|None
  start_ms:int
  end_ms:int
  segments:list[Segment] = field(default_factory=list)

def build_sections(frames:list[Frame], segments:list[Segment], duration_ms:int) -> list[Section]:
  """Create frame-anchored sections and assign segments by start time."""
  if not frames: return [Section(frame=None, start_ms=0, end_ms=duration_ms, segments=list(segments))]
  # the opening capture is near 1s, but represents the section from t=0
  sections = [Section(frame=frame, start_ms=0 if i == 0 else frame.t_ms,
                      end_ms=frames[i+1].t_ms if i+1 < len(frames) else duration_ms) for i,frame in enumerate(frames)]
  section_i = 0
  for segment in segments:
    while section_i+1 < len(sections) and segment.start_ms >= sections[section_i+1].start_ms: section_i += 1
    sections[section_i].segments.append(segment)
  return sections
