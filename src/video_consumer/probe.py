"""Validate an input video and read its basic metadata with ffprobe."""
from __future__ import annotations
import json, shutil, subprocess
from dataclasses import dataclass
from pathlib import Path

class ProbeError(Exception): pass

@dataclass
class VideoInfo:
  path:Path
  duration_ms:int
  width:int
  height:int
  has_audio:bool

def require_binaries(names:tuple[str, ...] = ("ffmpeg", "ffprobe")):
  if missing:=[name for name in names if shutil.which(name) is None]: raise ProbeError(f"missing required binaries: {', '.join(missing)}")

def probe(path:Path) -> VideoInfo:
  if not path.is_file(): raise ProbeError(f"not a file: {path}")
  cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-show_entries", "stream=codec_type,width,height", "-of", "json", str(path)]
  try: result = subprocess.run(cmd, capture_output=True, text=True, check=True)
  except subprocess.CalledProcessError as error: raise ProbeError(f"ffprobe failed on {path.name}: {error.stderr.strip()}") from error
  data = json.loads(result.stdout)
  streams = data.get("streams", [])
  if not (video_streams:=[stream for stream in streams if stream.get("codec_type") == "video"]): raise ProbeError(f"no video stream in {path.name}")
  if (duration_s:=float(data.get("format", {}).get("duration", 0))) <= 0: raise ProbeError(f"could not determine duration of {path.name}")
  return VideoInfo(path=path, duration_ms=int(duration_s*1000), width=int(video_streams[0].get("width", 0)),
                   height=int(video_streams[0].get("height", 0)), has_audio=any(s.get("codec_type") == "audio" for s in streams))
