"""Stage 1: extract 16kHz mono PCM audio for whisper.cpp."""
from __future__ import annotations
import subprocess
from pathlib import Path

def extract_audio(video:Path, out_wav:Path, verbose=False) -> Path:
  out_wav.parent.mkdir(parents=True, exist_ok=True)
  cmd = ["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out_wav)]
  result = subprocess.run(cmd, capture_output=not verbose, text=True)
  if result.returncode != 0:
    raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr[-500:] if result.stderr else '(see output above)'}")
  return out_wav
