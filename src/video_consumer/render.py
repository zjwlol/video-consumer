"""Stage 5: render transcript.md and manifest.json."""
from __future__ import annotations
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from .align import Section
from .frames import Frame, FrameSettings

def fmt_ts(ms:int) -> str:
  hours, remainder = divmod(ms//1000, 3600)
  minutes, seconds = divmod(remainder, 60)
  return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"

def render_markdown(title:str, source_name:str, duration_ms:int, model_name:str|None, sections:list[Section], out_md:Path) -> Path:
  transcribed = model_name or "no (video has no audio track)"
  lines = [f"# {title}", "", f"- Source: {source_name} | Duration: {fmt_ts(duration_ms)}",
           f"- Transcribed: {transcribed} | Frames: {sum(1 for section in sections if section.frame)}",
           f"- Generated: {datetime.now().isoformat(timespec='seconds')}", ""]
  for section in sections:
    lines += [f"## [{fmt_ts(section.start_ms)} – {fmt_ts(section.end_ms)}]", ""]
    if section.frame: lines += [f"![{section.frame.id}](frames/{section.frame.path.name})", ""]
    lines += [" ".join(segment.text for segment in section.segments) if section.segments else "*[no speech]*", ""]
  out_md.write_text("\n".join(lines))
  return out_md

def render_manifest(source_name:str, duration_ms:int, model_name:str|None, settings:FrameSettings,
                    frames:list[Frame], sections:list[Section], out_json:Path) -> Path:
  manifest = {
    "source": source_name, "duration_ms": duration_ms, "model": model_name,
    "generated": datetime.now().isoformat(timespec="seconds"), "settings": asdict(settings),
    "frames": [{"id":frame.id, "t_ms":frame.t_ms, "path":f"frames/{frame.path.name}", "trigger":frame.trigger,
                "scene_score":round(frame.scene_score, 3)} for frame in frames],
    "sections": [{"frame":section.frame.id if section.frame else None, "start_ms":section.start_ms, "end_ms":section.end_ms,
                  "segments":[{"start_ms":segment.start_ms, "end_ms":segment.end_ms, "text":segment.text}
                              for segment in section.segments]} for section in sections],
  }
  out_json.write_text(json.dumps(manifest, indent=2))
  return out_json
