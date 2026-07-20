"""Stage 3: select, extract, and deduplicate representative video frames."""
from __future__ import annotations
import math, re, subprocess
from dataclasses import dataclass
from pathlib import Path
import imagehash
from PIL import Image

@dataclass
class Frame:
  id:str
  t_ms:int
  path:Path
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

def detect_scene_changes(video:Path, threshold:float, verbose=False) -> list[tuple[int, float]]:
  """Return ``(timestamp_ms, score)`` for scene changes over the threshold."""
  cmd = ["ffmpeg", "-i", str(video), "-vf", f"select='gt(scene,{threshold})',metadata=print:file=-", "-f", "null", "-"]
  result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=None if verbose else subprocess.PIPE, text=True)
  if result.returncode != 0: raise RuntimeError(f"ffmpeg scene detection failed: {result.stderr[-500:] if result.stderr else '(see output above)'}")
  candidates, current_t = [], None
  for line in result.stdout.splitlines():
    if timestamp:=re.search(r"pts_time:([\d.]+)", line): current_t = int(float(timestamp.group(1))*1000)
    elif (score:=re.search(r"lavfi\.scene_score=([\d.]+)", line)) and current_t is not None:
      candidates.append((current_t, float(score.group(1))))
      current_t = None
  return candidates

def select_timestamps(scene_candidates:list[tuple[int, float]], duration_ms:int, settings:FrameSettings) -> list[tuple[int, str, float]]:
  """Apply the min-gap, scene cap, and interval fallback rules."""
  min_gap_ms, max_gap_ms = int(settings.min_gap_s*1000), int(settings.max_gap_s*1000)
  opening = (min(1000, duration_ms//2), "interval", 0.0)  # t=0 is often black or incomplete
  scenes:list[tuple[int, str, float]] = []
  last_t = opening[0]
  for t_ms,score in sorted(scene_candidates):
    if t_ms-last_t >= min_gap_ms:
      scenes.append((t_ms, "scene", score))
      last_t = t_ms
  budget = settings.max_frames or max(1, duration_ms//20_000)  # opening and gap-fill anchors are extra
  if len(scenes) > budget: scenes = sorted(sorted(scenes, key=lambda frame:frame[2], reverse=True)[:budget])
  filled:list[tuple[int, str, float]] = []
  boundaries = [opening] + scenes + [(duration_ms, "end", 0.0)]
  for i,cur in enumerate(boundaries[:-1]):
    filled.append(cur)
    gap_start, gap_end = cur[0], boundaries[i+1][0]
    if (gap:=gap_end-gap_start) <= max_gap_ms: continue
    n_fill = math.ceil(gap/max_gap_ms)-1
    filled += [(gap_start + k*gap//(n_fill+1), "interval", 0.0) for k in range(1, n_fill+1)]
  return sorted(filled)

def _frame_id(seq:int, t_ms:int) -> str:
  total_s = t_ms//1000
  return f"{seq:03d}_{total_s//60:02d}{total_s%60:02d}"

def extract_frames(video:Path, timestamps:list[tuple[int, str, float]], frames_dir:Path, settings:FrameSettings, duration_ms:int,
                   verbose=False) -> list[Frame]:
  """Extract one scaled JPEG for each selected timestamp."""
  frames_dir.mkdir(parents=True, exist_ok=True)
  frames = []
  for seq,(t_ms,trigger,score) in enumerate(timestamps, start=1):
    out = frames_dir/f"{(frame_id:=_frame_id(seq, t_ms))}.jpg"
    cmd = ["ffmpeg", "-y", "-ss", f"{t_ms/1000:.3f}", "-i", str(video), "-frames:v", "1", "-vf", f"scale={settings.frame_width}:-2",
           "-q:v", "4", "-update", "1", str(out)]
    result = subprocess.run(cmd, capture_output=not verbose, text=True)
    if result.returncode != 0 or not out.is_file():
      if t_ms >= duration_ms-2000:  # some containers have no decodable frame at their reported endpoint
        out.unlink(missing_ok=True)
        continue
      raise RuntimeError(f"frame extraction failed at {t_ms/1000:.1f}s: {result.stderr[-500:] if result.stderr else '(see output above)'}")
    frames.append(Frame(id=frame_id, t_ms=t_ms, path=out, trigger=trigger, scene_score=score))
  return frames

def _mean_rgb(img:Image.Image) -> tuple[float, float, float]:
  small = img.convert("RGB").resize((8, 8))
  pixels = list(small.get_flattened_data() if hasattr(small, "get_flattened_data") else small.getdata())
  return tuple(sum(pixel[channel] for pixel in pixels)/len(pixels) for channel in range(3))

def dedupe_frames(frames:list[Frame], phash_distance:int) -> list[Frame]:
  """Delete frames perceptually identical to the previously kept frame."""
  kept:list[Frame] = []
  prev_hash, prev_rgb = None, None
  for frame in frames:
    with Image.open(frame.path) as img: frame_hash, frame_rgb = imagehash.phash(img), _mean_rgb(img)
    if prev_hash is not None and (frame_hash-prev_hash) <= phash_distance and max(abs(a-b) for a,b in zip(frame_rgb, prev_rgb)) < 12:
      frame.path.unlink()
      continue
    prev_hash, prev_rgb = frame_hash, frame_rgb
    kept.append(frame)
  return kept

def _renumber(frames:list[Frame]) -> list[Frame]:
  for seq,frame in enumerate(frames, start=1):
    if (new_id:=_frame_id(seq, frame.t_ms)) == frame.id: continue
    frame.path.rename(new_path:=frame.path.with_name(f"{new_id}.jpg"))
    frame.id, frame.path = new_id, new_path
  return frames

def build_frames(video:Path, duration_ms:int, frames_dir:Path, settings:FrameSettings, verbose=False) -> list[Frame]:
  candidates = detect_scene_changes(video, settings.scene_threshold, verbose)
  if verbose: print(f"  scene candidates: {len(candidates)}")
  timestamps = select_timestamps(candidates, duration_ms, settings)
  frames = extract_frames(video, timestamps, frames_dir, settings, duration_ms, verbose)
  before = len(frames)
  frames = dedupe_frames(frames, settings.phash_distance)
  if verbose and before != len(frames): print(f"  dedupe: {before} -> {len(frames)}")
  return _renumber(frames)
