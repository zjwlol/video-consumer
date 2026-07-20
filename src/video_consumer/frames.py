"""Stage 3: pick frame timestamps (scene changes + interval fallback), extract small JPEGs, and drop near-duplicates."""
from __future__ import annotations
import math, re, subprocess
from dataclasses import dataclass
from pathlib import Path
import imagehash
from PIL import Image

@dataclass
class Frame:
  id:str; t_ms:int; path:Path  # noqa: E702
  trigger:str  # "scene" or "interval"
  scene_score:float = 0.0

@dataclass
class FrameSettings:
  scene_threshold:float = 0.10
  min_gap_s:float = 10.0
  max_gap_s:float = 90.0
  frame_width:int = 640
  max_frames:int|None = None  # None = auto (duration / 20s)
  phash_distance:int = 8

def detect_scene_changes(video:Path, threshold:float) -> list[tuple[int, float]]:
  """
  Detects visual scene changes in a single ffmpeg pass.

  Returns `[(t_ms, scene_score)]` for frames whose score exceeds `threshold`.
  """
  cmd = ["ffmpeg", "-i", str(video), "-vf", f"select='gt(scene,{threshold})',metadata=print:file=-", "-f", "null", "-"]
  result = subprocess.run(cmd, capture_output=True, text=True)
  if result.returncode != 0: raise RuntimeError(f"ffmpeg scene detection failed: {result.stderr[-500:]}")
  # output pairs lines like:  frame:12  pts:34816  pts_time:4.06  /  lavfi.scene_score=0.53
  candidates, current_t = [], None
  for line in result.stdout.splitlines():
    if m:=re.search(r"pts_time:([\d.]+)", line): current_t = int(float(m.group(1))*1000)
    elif (m:=re.search(r"lavfi\.scene_score=([\d.]+)", line)) and current_t is not None:
      candidates.append((current_t, float(m.group(1))))
      current_t = None
  return candidates

def select_timestamps(scene_candidates:list[tuple[int, float]], duration_ms:int, settings:FrameSettings) -> list[tuple[int, str, float]]:
  """
  Picks frame timestamps: min-gap filter, scene cap, then interval fallback.

  Returns sorted `[(t_ms, trigger, scene_score)]`. The cap limits scene frames only
  (`--max-frames 1` still yields the opening frame plus one scene); anchors (the
  opening frame and gap fills) are always kept on top of it, and fills are bounded
  by duration/max_gap anyway.
  """
  min_gap_ms, max_gap_ms = int(settings.min_gap_s*1000), int(settings.max_gap_s*1000)
  # opening frame at 1s, not 0: t=0 often lands on a black/blank first frame
  opening = (min(1000, duration_ms//2), "interval", 0.0)

  scenes:list[tuple[int, str, float]] = []
  last_t = opening[0]
  for t_ms, score in sorted(scene_candidates):
    if t_ms - last_t >= min_gap_ms: scenes.append((t_ms, "scene", score)); last_t = t_ms  # noqa: E702

  # cap applies to scene frames only; floor keeps auto mode sane for short videos
  budget = settings.max_frames or max(6, duration_ms//20_000)
  if len(scenes) > budget: scenes = sorted(sorted(scenes, key=lambda f: f[2], reverse=True)[:budget])

  # fill gaps strictly larger than max_gap with evenly spaced frames
  filled:list[tuple[int, str, float]] = []
  boundaries = [opening] + scenes + [(duration_ms, "end", 0.0)]
  for i, cur in enumerate(boundaries[:-1]):
    filled.append(cur)
    gap_start, gap_end = cur[0], boundaries[i+1][0]
    if (gap:=gap_end-gap_start) <= max_gap_ms: continue
    n_fill = math.ceil(gap/max_gap_ms) - 1
    filled += [(gap_start + k*gap//(n_fill+1), "interval", 0.0) for k in range(1, n_fill+1)]
  return sorted(filled)

def _frame_id(seq:int, t_ms:int) -> str:
  total_s = t_ms//1000
  return f"{seq:03d}_{total_s//60:02d}{total_s%60:02d}"

def extract_frames(video:Path, timestamps:list[tuple[int, str, float]], frames_dir:Path, settings:FrameSettings,
                   duration_ms:int) -> list[Frame]:
  frames_dir.mkdir(parents=True, exist_ok=True)
  frames = []
  for seq, (t_ms, trigger, score) in enumerate(timestamps, start=1):
    out = frames_dir/f"{(fid:=_frame_id(seq, t_ms))}.jpg"
    cmd = ["ffmpeg", "-y", "-ss", f"{t_ms/1000:.3f}", "-i", str(video), "-frames:v", "1",
           "-vf", f"scale={settings.frame_width}:-2", "-q:v", "4", str(out)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not out.is_file():
      # tolerate only seeks landing past the last decodable frame
      if t_ms >= duration_ms - 2000: out.unlink(missing_ok=True); continue  # noqa: E702
      raise RuntimeError(f"frame extraction failed at {t_ms/1000:.1f}s: {(result.stderr or '').strip()[-500:]}")
    frames.append(Frame(id=fid, t_ms=t_ms, path=out, trigger=trigger, scene_score=score))
  return frames

def _mean_rgb(img:Image.Image) -> tuple[float, float, float]:
  pixels = list(img.convert("RGB").resize((8, 8)).getdata())
  return tuple(sum(p[i] for p in pixels)/len(pixels) for i in range(3))

def dedupe_frames(frames:list[Frame], phash_distance:int) -> list[Frame]:
  """
  Drops frames perceptually identical to the previously kept one, deleting their files.

  phash is structure-only (luma DCT) so mean color must also match, otherwise flat
  slides in different colors collapse together. Interval fills are deduped too, on
  purpose: a fill exists to catch changes scene detection missed, so an identical
  fill carries no information and a static stretch keeps just one frame.
  """
  kept:list[Frame] = []
  prev_hash, prev_rgb = None, None
  for frame in frames:
    with Image.open(frame.path) as img: h, rgb = imagehash.phash(img), _mean_rgb(img)
    if prev_hash is not None and (h - prev_hash) <= phash_distance and max(abs(a-b) for a,b in zip(rgb, prev_rgb)) < 12:
      frame.path.unlink()
      continue
    prev_hash, prev_rgb = h, rgb
    kept.append(frame)
  return kept

def _renumber(frames:list[Frame]) -> list[Frame]:
  # renumber sequences after dedupe so filenames stay contiguous
  for seq, frame in enumerate(frames, start=1):
    if (new_id:=_frame_id(seq, frame.t_ms)) != frame.id:
      frame.path.rename(new_path:=frame.path.with_name(f"{new_id}.jpg"))
      frame.id, frame.path = new_id, new_path
  return frames

def build_frames(video:Path, duration_ms:int, frames_dir:Path, settings:FrameSettings, verbose=False) -> list[Frame]:
  candidates = detect_scene_changes(video, settings.scene_threshold)
  if verbose: print(f"  scene candidates: {len(candidates)}")
  timestamps = select_timestamps(candidates, duration_ms, settings)
  frames = extract_frames(video, timestamps, frames_dir, settings, duration_ms)
  before = len(frames)
  frames = dedupe_frames(frames, settings.phash_distance)
  if verbose and before != len(frames): print(f"  dedupe: {before} -> {len(frames)}")
  return _renumber(frames)
