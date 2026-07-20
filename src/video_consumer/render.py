"""Stage 5: render transcript.md and manifest.json."""
from __future__ import annotations
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from .align import Section
from .frames import Frame, FrameSettings

def fmt_ts(ms:int) -> str:
  h, rem = divmod(ms//1000, 3600)
  m, s = divmod(rem, 60)
  return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

def render_markdown(title:str, source_name:str, duration_ms:int, model_name:str|None, sections:list[Section], out_md:Path) -> Path:
  transcribed = model_name if model_name else "no (video has no audio track)"
  lines = [f"# {title}", "", f"- Source: {source_name} | Duration: {fmt_ts(duration_ms)}",
           f"- Transcribed: {transcribed} | Frames: {sum(1 for s in sections if s.frame)}",
           f"- Generated: {datetime.now().isoformat(timespec='seconds')}", ""]
  for sec in sections:
    lines += [f"## [{fmt_ts(sec.start_ms)} \u2013 {fmt_ts(sec.end_ms)}]", ""]
    if sec.frame: lines += [f"![{sec.frame.id}](frames/{sec.frame.path.name})", ""]
    lines += [" ".join(seg.text for seg in sec.segments) if sec.segments else "*[no speech]*", ""]
  out_md.write_text("\n".join(lines))
  return out_md

def render_manifest(source_name:str, duration_ms:int, model_name:str|None, settings:FrameSettings,
                    frames:list[Frame], sections:list[Section], out_json:Path) -> Path:
  manifest = {
    "source": source_name, "duration_ms": duration_ms,
    "model": model_name,  # null = no audio track, not transcribed
    "generated": datetime.now().isoformat(timespec="seconds"), "settings": asdict(settings),
    "frames": [{"id": f.id, "t_ms": f.t_ms, "path": f"frames/{f.path.name}", "trigger": f.trigger, "scene_score": round(f.scene_score, 3)}
               for f in frames],
    "sections": [{"frame": s.frame.id if s.frame else None, "start_ms": s.start_ms, "end_ms": s.end_ms,
                  "segments": [{"start_ms": g.start_ms, "end_ms": g.end_ms, "text": g.text} for g in s.segments]} for s in sections],
  }
  out_json.write_text(json.dumps(manifest, indent=2))
  return out_json
