"""CLI entry point and stage orchestration with caching."""
from __future__ import annotations
import argparse, hashlib, json, shutil, sys, time
from dataclasses import asdict
from pathlib import Path
from . import __version__
from .align import build_sections
from .audio import extract_audio
from .frames import Frame, FrameSettings, build_frames
from .probe import ProbeError, probe, require_binaries
from .render import fmt_ts, render_manifest, render_markdown
from .transcribe import parse_segments, transcribe

DEFAULT_MODEL = Path.home()/"whisper-models"/"ggml-small.bin"
FORCE_CHOICES = ("all", "audio", "transcribe", "frames", "render")

def _arg_range(kind, lo=None, hi=None, lo_open=False):
  # argparse type factory with bounds checking
  def parse(raw:str):
    try: v = kind(raw)
    except ValueError: raise argparse.ArgumentTypeError(f"not a {kind.__name__}: {raw!r}")
    if lo is not None and (v <= lo if lo_open else v < lo):
      raise argparse.ArgumentTypeError(f"must be {'>' if lo_open else '>='} {lo}: {raw}")
    if hi is not None and v >= hi: raise argparse.ArgumentTypeError(f"must be < {hi}: {raw}")
    return v
  return parse

def build_parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(prog="video-consumer", description="Turn a video into a transcript + screenshot digest package.")
  p.add_argument("input", type=Path, help="video file (mp4/mov/mkv/webm/...)")
  p.add_argument("-o", "--out", type=Path, default=None, help="output dir (default ./output/<input-stem>/)")
  p.add_argument("-m", "--model", type=Path, default=DEFAULT_MODEL, help=f"whisper ggml model (default {DEFAULT_MODEL})")
  p.add_argument("--language", default="en", help="spoken language (default en)")
  p.add_argument("--scene-threshold", type=_arg_range(float, 0, 1, lo_open=True), default=0.10,
                 help="scene change score to trigger a frame (default 0.10; slide flips in screenshares score low, camera cuts score ~0.3+)")
  p.add_argument("--min-gap", type=_arg_range(float, 0), default=10.0, help="minimum seconds between frames (default 10)")
  p.add_argument("--max-gap", type=_arg_range(float, 0, lo_open=True), default=90.0,
                 help="max seconds without a frame before fallback fills (default 90)")
  p.add_argument("--frame-width", type=_arg_range(int, 16), default=640, help="frame JPEG width in px (default 640)")
  p.add_argument("--max-frames", type=_arg_range(int, 1), default=None, help="cap on scene frame count (default duration/20s)")
  p.add_argument("--force", nargs="?", const="all", choices=FORCE_CHOICES, default=None, help="redo all stages, or one of: %(choices)s")
  p.add_argument("-v", "--verbose", action="store_true")
  p.add_argument("--version", action="version", version=__version__)
  return p

def _file_fp(path:Path) -> str:
  st = path.stat()
  return f"{path.resolve()}:{st.st_size}:{st.st_mtime_ns}"

def _fp(*parts) -> str: return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]

class Stamps:
  """
  Per-stage cache fingerprints.

  A cached artifact is reused only when the inputs that produced it (source file,
  model, settings) are unchanged; `fresh` checks, `update` records after a rebuild.
  """
  def __init__(self, path:Path):
    self.path = path
    try: self.data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError): self.data = {}
  def fresh(self, stage:str, fingerprint:str, artifact:Path) -> bool: return self.data.get(stage) == fingerprint and artifact.is_file()
  def update(self, stage:str, fingerprint:str):
    self.data[stage] = fingerprint
    self.path.write_text(json.dumps(self.data, indent=2))

def _clear_cache(force:str|None, cache:Path, frames_dir:Path):
  wav, whisper_json, frames_json = cache/"audio.wav", cache/"whisper.json", cache/"frames.json"
  if force in ("all", "audio"):
    wav.unlink(missing_ok=True)
    whisper_json.unlink(missing_ok=True)  # transcript derives from audio
  if force in ("all", "transcribe"): whisper_json.unlink(missing_ok=True)
  if force in ("all", "frames"):
    frames_json.unlink(missing_ok=True)
    shutil.rmtree(frames_dir, ignore_errors=True)

def _load_cached_frames(frames_json:Path, frames_dir:Path) -> list[Frame]|None:
  if not frames_json.is_file(): return None
  frames = []
  for item in json.loads(frames_json.read_text()):
    if not (path:=frames_dir/Path(item["path"]).name).is_file(): return None  # cache out of sync with disk
    frames.append(Frame(id=item["id"], t_ms=item["t_ms"], path=path, trigger=item["trigger"], scene_score=item["scene_score"]))
  return frames

def _save_frames_cache(frames_json:Path, frames:list[Frame]):
  frames_json.write_text(json.dumps(
    [{"id": f.id, "t_ms": f.t_ms, "path": f.path.name, "trigger": f.trigger, "scene_score": f.scene_score} for f in frames], indent=2))

def run(args:argparse.Namespace) -> int:
  require_binaries()  # ffmpeg/ffprobe; whisper checked only when transcribing
  info = probe(args.input)

  out_dir = args.out or Path("output")/args.input.stem
  frames_dir, cache = out_dir/"frames", out_dir/".cache"
  cache.mkdir(parents=True, exist_ok=True)
  _clear_cache(args.force, cache, frames_dir)
  stamps = Stamps(cache/"stamps.json")
  input_fp = _file_fp(args.input)

  print(f"{args.input.name}: {fmt_ts(info.duration_ms)}, {info.width}x{info.height}, audio={'yes' if info.has_audio else 'NO'}")

  # stage 1+2: audio + transcription
  segments, model_name = [], None
  whisper_json = cache/"whisper.json"
  if not info.has_audio: print("[audio] no audio track, skipping transcription")
  else:
    require_binaries(("whisper-cli",))
    if not args.model.is_file(): raise ProbeError(f"whisper model not found: {args.model}")
    model_name = args.model.name
    transcribe_fp = _fp(input_fp, _file_fp(args.model), args.language)
    if stamps.fresh("transcribe", transcribe_fp, whisper_json):
      print("[transcribe] cached")
      segments = parse_segments(whisper_json)
    else:
      wav, audio_fp = cache/"audio.wav", _fp(input_fp)
      if stamps.fresh("audio", audio_fp, wav): print("[audio] cached")
      else:
        print("[audio]", flush=True)
        extract_audio(args.input, wav, args.verbose)
        stamps.update("audio", audio_fp)
      print("[transcribe]", flush=True)
      t0 = time.monotonic()
      transcribe(wav, args.model, whisper_json, args.language, args.verbose)
      segments = parse_segments(whisper_json)
      stamps.update("transcribe", transcribe_fp)
      print(f"  {len(segments)} segments in {time.monotonic()-t0:.0f}s")

  # stage 3: frames
  settings = FrameSettings(scene_threshold=args.scene_threshold, min_gap_s=args.min_gap, max_gap_s=args.max_gap,
                           frame_width=args.frame_width, max_frames=args.max_frames)
  frames_json, frames_fp = cache/"frames.json", _fp(input_fp, asdict(settings))
  frames = _load_cached_frames(frames_json, frames_dir) if stamps.fresh("frames", frames_fp, frames_json) else None
  if frames is not None: print(f"[frames] cached ({len(frames)})")
  else:
    print("[frames]", flush=True)
    shutil.rmtree(frames_dir, ignore_errors=True)  # no stale jpegs
    frames = build_frames(args.input, info.duration_ms, frames_dir, settings, args.verbose)
    _save_frames_cache(frames_json, frames)
    stamps.update("frames", frames_fp)
    print(f"  {len(frames)} frames")

  # stage 4+5: align + render
  print("[render]", flush=True)
  sections = build_sections(frames, segments, info.duration_ms)
  md = render_markdown(args.input.stem, args.input.name, info.duration_ms, model_name, sections, out_dir/"transcript.md")
  render_manifest(args.input.name, info.duration_ms, model_name, settings, frames, sections, out_dir/"manifest.json")

  print(f"\ndone: {md}")
  return 0

def main():
  args = build_parser().parse_args()
  try: sys.exit(run(args))
  except (ProbeError, RuntimeError) as e:
    print(f"error: {e}", file=sys.stderr)
    sys.exit(1)
  except KeyboardInterrupt: sys.exit(130)

if __name__ == "__main__": main()
