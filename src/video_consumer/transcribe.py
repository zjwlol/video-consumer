"""Stage 2: run whisper-cli and parse its JSON segment output."""
from __future__ import annotations
import json, subprocess
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Segment:
  start_ms:int
  end_ms:int
  text:str

def transcribe(wav:Path, model:Path, out_json:Path, language="en", verbose=False) -> Path:
  """Run whisper-cli and write its raw segment JSON to ``out_json``."""
  if not model.is_file(): raise RuntimeError(f"whisper model not found: {model}")
  out_json.parent.mkdir(parents=True, exist_ok=True)
  cmd = ["whisper-cli", "-m", str(model), "-f", str(wav), "-l", language, "-oj", "-of", str(out_json.with_suffix("")),
         "-pp" if verbose else "--no-prints"]
  result = subprocess.run(cmd, capture_output=not verbose, text=True)
  if result.returncode != 0: raise RuntimeError(f"whisper-cli failed: {result.stderr[-500:] if result.stderr else '(see output above)'}")
  if not out_json.is_file(): raise RuntimeError(f"whisper-cli produced no JSON at {out_json}")
  return out_json

def parse_segments(whisper_json:Path) -> list[Segment]:
  data = json.loads(whisper_json.read_text())
  return [Segment(start_ms=int(item["offsets"]["from"]), end_ms=int(item["offsets"]["to"]), text=text)
          for item in data.get("transcription", []) if (text:=item.get("text", "").strip())]
