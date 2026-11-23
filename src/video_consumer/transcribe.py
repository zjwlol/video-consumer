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
  out_json.unlink(missing_ok=True)  # whisper-cli can exit 0 without writing and an old file would pass as the new one
  cmd = ["whisper-cli", "-m", str(model), "-f", str(wav), "-l", language, "-oj", "-of", str(out_json.with_suffix("")),
         "-pp" if verbose else "--no-prints"]
  result = subprocess.run(cmd, capture_output=not verbose, text=True)
  if result.returncode != 0: raise RuntimeError(f"whisper-cli failed: {result.stderr[-500:] if result.stderr else '(see output above)'}")
  if not out_json.is_file():
    errors = [line for line in (result.stderr or "").splitlines() if line.startswith("error:")]
    raise RuntimeError(f"whisper-cli wrote no JSON: {errors[-1] if errors else '(see its output)'}")
  return out_json

def parse_segments(whisper_json:Path, offset_ms=0) -> list[Segment]:
  """Read the whisper segments and move them by ``offset_ms`` onto the video timeline."""
  try:
    data = json.loads(whisper_json.read_text(encoding="utf-8"))
    return [Segment(start_ms=int(item["offsets"]["from"])+offset_ms, end_ms=int(item["offsets"]["to"])+offset_ms, text=text)
            for item in data["transcription"] if (text:=item["text"].strip())]
  except (ValueError, KeyError, TypeError, AttributeError) as error:
    raise RuntimeError(f"unreadable whisper JSON {whisper_json.name}: {error!r}") from error
