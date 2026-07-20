"""Stage 4: group transcript segments into sections anchored on frames."""
from __future__ import annotations
from dataclasses import dataclass, field
from .frames import Frame
from .transcribe import Segment

@dataclass
class Section:
  frame:Frame|None; start_ms:int; end_ms:int  # noqa: E702
  segments:list[Segment] = field(default_factory=list)

def build_sections(frames:list[Frame], segments:list[Segment], duration_ms:int) -> list[Section]:
  """
  Groups transcript segments into sections anchored on frames.

  Each frame opens a section running until the next frame; segments are assigned
  by their start time. With no frames, everything lands in one big section.
  """
  if not frames: return [Section(frame=None, start_ms=0, end_ms=duration_ms, segments=list(segments))]
  # first section starts at 0 even though its frame is captured at ~1s, so speech
  # before the opening frame still lands in a section (the frame represents it)
  sections = [Section(frame=f, start_ms=0 if i == 0 else f.t_ms, end_ms=frames[i+1].t_ms if i+1 < len(frames) else duration_ms)
              for i,f in enumerate(frames)]
  si = 0
  for seg in segments:
    while si+1 < len(sections) and seg.start_ms >= sections[si+1].start_ms: si += 1
    sections[si].segments.append(seg)
  return sections
