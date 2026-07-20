"""Stage 2: run whisper-cli and parse its JSON output into segments."""
from __future__ import annotations
import json, subprocess
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Segment:
  start_ms:int; end_ms:int; text:str  # noqa: E702

def transcribe(wav:Path, model:Path, out_json:Path, language="en", verbose=False) -> Path:
  """
  Runs whisper-cli on a 16kHz mono wav, writing raw segment JSON to `out_json`.

  Returns `out_json`. Parse it with `parse_segments`.
  """
  if not model.is_file(): raise RuntimeError(f"whisper model not found: {model}")
  out_json.parent.mkdir(parents=True, exist_ok=True)
  # whisper-cli appends .json to -of value
  cmd = ["whisper-cli", "-m", str(model), "-f", str(wav), "-l", language, "-oj",
         "-of", str(out_json.with_suffix("")), "-pp" if verbose else "--no-prints"]
  result = subprocess.run(cmd, capture_output=not verbose, text=True)
  if result.returncode != 0:
    raise RuntimeError(f"whisper-cli failed: {(result.stderr.strip() if result.stderr else '(see output above)')[-500:]}")
  if not out_json.is_file(): raise RuntimeError(f"whisper-cli produced no JSON at {out_json}")
  return out_json

def parse_segments(whisper_json:Path) -> list[Segment]:
  data = json.loads(whisper_json.read_text())
  return [Segment(start_ms=int(i["offsets"]["from"]), end_ms=int(i["offsets"]["to"]), text=t)
          for i in data.get("transcription", []) if (t:=i.get("text", "").strip())]
