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
  audio_start_ms:int = 0  # where the first audio packet sits on the video timeline

def require_binaries(names:tuple[str, ...] = ("ffmpeg", "ffprobe")):
  if missing:=[name for name in names if shutil.which(name) is None]: raise ProbeError(f"missing required binaries: {', '.join(missing)}")

def _ffprobe(path:Path, *options:str) -> dict:
  cmd = ["ffprobe", "-v", "error", *options, "-of", "json", str(path)]
  try: result = subprocess.run(cmd, capture_output=True, text=True, check=True)
  except subprocess.CalledProcessError as error: raise ProbeError(f"ffprobe failed on {path.name}: {error.stderr.strip()}") from error
  return json.loads(result.stdout)

def probe(path:Path) -> VideoInfo:
  if not path.is_file(): raise ProbeError(f"not a file: {path}")
  data = _ffprobe(path, "-show_entries", "format=duration,start_time", "-show_entries", "stream=codec_type,width,height")
  streams, container = data.get("streams", []), data.get("format", {})
  if not (video_streams:=[stream for stream in streams if stream.get("codec_type") == "video"]): raise ProbeError(f"no video stream in {path.name}")
  if (duration_s:=float(container.get("duration", 0))) <= 0: raise ProbeError(f"could not determine duration of {path.name}")
  has_audio, audio_start_ms = any(s.get("codec_type") == "audio" for s in streams), 0
  if has_audio:
    packets = _ffprobe(path, "-select_streams", "a:0", "-show_entries", "packet=pts_time", "-read_intervals", "%+#1").get("packets", [])
    # the extracted wav starts at the first audio packet and that can be later than the start of the video
    if packets and "pts_time" in packets[0]:
      audio_start_ms = max(0, round((float(packets[0]["pts_time"])-float(container.get("start_time", 0)))*1000))
  return VideoInfo(path=path, duration_ms=int(duration_s*1000), width=int(video_streams[0].get("width", 0)),
                   height=int(video_streams[0].get("height", 0)), has_audio=has_audio, audio_start_ms=audio_start_ms)
