"""Pure-logic tests: no ffmpeg or whisper needed."""
from __future__ import annotations
import argparse, json, subprocess, tempfile, unittest
from pathlib import Path
from unittest.mock import Mock, patch
from PIL import Image
from video_consumer.align import build_sections
from video_consumer.cli import Stamps, _arg_range, _fp
from video_consumer.frames import Frame, FrameSettings, _frame_id, dedupe_frames, detect_scene_changes, select_timestamps
from video_consumer.render import fmt_ts
from video_consumer.transcribe import Segment, parse_segments

def seg(start_ms:int, end_ms:int, text="x") -> Segment: return Segment(start_ms=start_ms, end_ms=end_ms, text=text)
def frame(t_ms:int, frame_id="f") -> Frame: return Frame(id=frame_id, t_ms=t_ms, path=Path(f"{frame_id}.jpg"), trigger="scene")

# ***** frames: timestamp selection *****

class TestSelectTimestamps(unittest.TestCase):
  def times(self, output): return [timestamp for timestamp,_,_ in output]
  def test_opening_anchor_always_present(self):
    output = select_timestamps([(t*1000, 0.3) for t in range(15, 120, 15)], 120_000, FrameSettings(min_gap_s=10, max_gap_s=90))
    self.assertTrue(any(timestamp <= 1000 for timestamp in self.times(output)))
  def test_cap_is_scene_only(self):
    output = select_timestamps([(t*1000, 0.3) for t in range(15, 120, 15)], 120_000, FrameSettings(min_gap_s=10, max_gap_s=200, max_frames=1))
    self.assertEqual([trigger for _,trigger,_ in output], ["interval", "scene"])
  def test_auto_cap_is_duration_over_twenty_seconds(self):
    output = select_timestamps([(t*1000, 0.3) for t in range(10, 60, 10)], 60_000, FrameSettings(min_gap_s=0, max_gap_s=200))
    self.assertEqual(sum(1 for _,trigger,_ in output if trigger == "scene"), 3)
  def test_cap_keeps_highest_scores(self):
    output = select_timestamps([(15_000, 0.1), (30_000, 0.9), (45_000, 0.2)], 60_000, FrameSettings(min_gap_s=10, max_gap_s=200, max_frames=1))
    self.assertIn((30_000, "scene", 0.9), output)
  def test_min_gap_filters_rapid_cuts(self):
    output = select_timestamps([(11_000, 0.5), (12_000, 0.5), (13_000, 0.5)], 60_000, FrameSettings(min_gap_s=10, max_gap_s=200, max_frames=10))
    self.assertEqual(sum(1 for _,trigger,_ in output if trigger == "scene"), 1)
  def test_exact_multiple_gap_no_fill(self):
    self.assertEqual(len(select_timestamps([], 91_000, FrameSettings(max_gap_s=90, max_frames=100))), 1)
  def test_fill_spacing_bounded(self):
    output = select_timestamps([], 500_000, FrameSettings(max_gap_s=90, max_frames=100))
    timestamps = self.times(output) + [500_000]
    self.assertLessEqual(max(end-start for start,end in zip(timestamps, timestamps[1:])), 90_000)
  def test_output_sorted(self):
    output = select_timestamps([(50_000, 0.5), (20_000, 0.5)], 300_000, FrameSettings(min_gap_s=10, max_gap_s=90, max_frames=10))
    self.assertEqual(self.times(output), sorted(self.times(output)))
  def test_short_video_opening_at_midpoint(self): self.assertEqual(select_timestamps([], 1000, FrameSettings())[0][0], 500)

class TestSceneDetection(unittest.TestCase):
  @patch("video_consumer.frames.subprocess.run")
  def test_verbose_streams_stderr_but_captures_metadata(self, run:Mock):
    run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="frame:1 pts_time:4.5\nlavfi.scene_score=0.42\n")
    self.assertEqual(detect_scene_changes(Path("video.mp4"), 0.1, verbose=True), [(4500, 0.42)])
    self.assertEqual(run.call_args.kwargs["stdout"], subprocess.PIPE)
    self.assertIsNone(run.call_args.kwargs["stderr"])

class TestFrameId(unittest.TestCase):
  def test_format(self): self.assertEqual(_frame_id(7, 872_000), "007_1432")
  def test_zero(self): self.assertEqual(_frame_id(1, 0), "001_0000")

# ***** frames: dedupe *****

class TestDedupe(unittest.TestCase):
  def _make(self, directory:Path, name:str, color:tuple[int, int, int]) -> Frame:
    Image.new("RGB", (64, 36), color).save(path:=directory/f"{name}.jpg")
    return Frame(id=name, t_ms=0, path=path, trigger="scene")
  def test_identical_dropped_different_color_kept(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      directory = Path(temp_dir)
      frames = [self._make(directory, "a", (0, 0, 200)), self._make(directory, "b", (0, 0, 200)), self._make(directory, "c", (200, 0, 0))]
      self.assertEqual([item.id for item in dedupe_frames(frames, phash_distance=8)], ["a", "c"])
      self.assertFalse((directory/"b.jpg").exists())

# ***** align *****

class TestBuildSections(unittest.TestCase):
  def test_no_frames_one_section(self):
    output = build_sections([], [seg(0, 1000)], 10_000)
    self.assertEqual((len(output), output[0].frame, len(output[0].segments)), (1, None, 1))
  def test_first_section_starts_at_zero(self): self.assertEqual(build_sections([frame(1000)], [], 10_000)[0].start_ms, 0)
  def test_segments_assigned_by_start_time(self):
    output = build_sections([frame(1000, "a"), frame(5000, "b")], [seg(0, 900), seg(4500, 6000), seg(7000, 8000)], 10_000)
    self.assertEqual([len(section.segments) for section in output], [2, 1])

# ***** transcribe *****

class TestParseSegments(unittest.TestCase):
  def test_parse_and_skip_empty(self):
    data = {"transcription": [{"offsets": {"from": 0, "to": 1000}, "text": " hi "}, {"offsets": {"from": 1000, "to": 2000}, "text": "  "}]}
    with tempfile.TemporaryDirectory() as temp_dir:
      (path:=Path(temp_dir)/"whisper.json").write_text(json.dumps(data))
      self.assertEqual(parse_segments(path), [Segment(start_ms=0, end_ms=1000, text="hi")])

# ***** render *****

class TestFmtTs(unittest.TestCase):
  def test_minutes(self): self.assertEqual(fmt_ts(63_000), "01:03")
  def test_hours(self): self.assertEqual(fmt_ts(3_723_000), "1:02:03")
  def test_zero(self): self.assertEqual(fmt_ts(0), "00:00")

# ***** cli *****

class TestArgRange(unittest.TestCase):
  def test_lo_open_rejects_zero(self): self.assertRaises(argparse.ArgumentTypeError, _arg_range(float, 0, lo_open=True), "0")
  def test_hi_rejects(self): self.assertRaises(argparse.ArgumentTypeError, _arg_range(float, 0, 1, lo_open=True), "1.5")
  def test_accepts(self): self.assertEqual(_arg_range(int, 1)("5"), 5)
  def test_non_numeric(self): self.assertRaises(argparse.ArgumentTypeError, _arg_range(int, 1), "abc")

class TestStamps(unittest.TestCase):
  def test_fresh_needs_stamp_and_artifact(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      directory = Path(temp_dir)
      stamps, fingerprint, artifact = Stamps(directory/"stamps.json"), _fp("input", {"key": 1}), directory/"artifact"
      self.assertFalse(stamps.fresh("frames", fingerprint, artifact))
      stamps.update("frames", fingerprint)
      self.assertFalse(stamps.fresh("frames", fingerprint, artifact))
      artifact.touch()
      self.assertTrue(stamps.fresh("frames", fingerprint, artifact))
      self.assertFalse(stamps.fresh("frames", _fp("input", {"key": 2}), artifact))
      self.assertTrue(Stamps(directory/"stamps.json").fresh("frames", fingerprint, artifact))

if __name__ == "__main__": unittest.main()
