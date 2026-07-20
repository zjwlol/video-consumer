"""Pure-logic tests: no ffmpeg or whisper needed."""
from __future__ import annotations
import argparse, json, tempfile, unittest
from pathlib import Path
from PIL import Image
from video_consumer.align import build_sections
from video_consumer.cli import Stamps, _arg_range, _fp
from video_consumer.frames import Frame, FrameSettings, _frame_id, dedupe_frames, select_timestamps
from video_consumer.render import fmt_ts
from video_consumer.transcribe import Segment, parse_segments

def seg(a:int, b:int, text="x") -> Segment: return Segment(start_ms=a, end_ms=b, text=text)
def frame(t_ms:int, fid="f") -> Frame: return Frame(id=fid, t_ms=t_ms, path=Path(f"{fid}.jpg"), trigger="scene")

# ***** frames: timestamp selection *****

class TestSelectTimestamps(unittest.TestCase):
  def times(self, out): return [t for t,_,_ in out]

  def test_opening_anchor_always_present(self):
    out = select_timestamps([(t*1000, 0.3) for t in range(15, 120, 15)], 120_000, FrameSettings(min_gap_s=10, max_gap_s=90))
    self.assertTrue(any(t <= 1000 for t in self.times(out)))

  def test_cap_is_scene_only(self):
    # max_frames=1 -> opening + exactly 1 scene
    out = select_timestamps([(t*1000, 0.3) for t in range(15, 120, 15)], 120_000, FrameSettings(min_gap_s=10, max_gap_s=200, max_frames=1))
    self.assertEqual([trig for _,trig,_ in out], ["interval", "scene"])

  def test_cap_keeps_highest_scores(self):
    out = select_timestamps([(15_000, 0.1), (30_000, 0.9), (45_000, 0.2)], 60_000, FrameSettings(min_gap_s=10, max_gap_s=200, max_frames=1))
    self.assertIn((30_000, "scene", 0.9), out)

  def test_min_gap_filters_rapid_cuts(self):
    out = select_timestamps([(11_000, 0.5), (12_000, 0.5), (13_000, 0.5)], 60_000, FrameSettings(min_gap_s=10, max_gap_s=200, max_frames=10))
    self.assertEqual(sum(1 for _,trig,_ in out if trig == "scene"), 1)

  def test_exact_multiple_gap_no_fill(self):
    self.assertEqual(len(select_timestamps([], 91_000, FrameSettings(max_gap_s=90, max_frames=100))), 1)

  def test_fill_spacing_bounded(self):
    out = select_timestamps([], 500_000, FrameSettings(max_gap_s=90, max_frames=100))
    ts = self.times(out) + [500_000]
    self.assertLessEqual(max(b-a for a,b in zip(ts, ts[1:])), 90_000)

  def test_output_sorted(self):
    out = select_timestamps([(50_000, 0.5), (20_000, 0.5)], 300_000, FrameSettings(min_gap_s=10, max_gap_s=90, max_frames=10))
    self.assertEqual(self.times(out), sorted(self.times(out)))

  def test_short_video_opening_at_midpoint(self): self.assertEqual(select_timestamps([], 1000, FrameSettings())[0][0], 500)

class TestFrameId(unittest.TestCase):
  def test_format(self): self.assertEqual(_frame_id(7, 872_000), "007_1432")
  def test_zero(self): self.assertEqual(_frame_id(1, 0), "001_0000")

# ***** frames: dedupe *****

class TestDedupe(unittest.TestCase):
  def _mk(self, tmp:Path, name:str, color:tuple[int, int, int]) -> Frame:
    Image.new("RGB", (64, 36), color).save(p:=tmp/f"{name}.jpg")
    return Frame(id=name, t_ms=0, path=p, trigger="scene")

  def test_identical_dropped_different_color_kept(self):
    with tempfile.TemporaryDirectory() as td:
      tmp = Path(td)
      frames = [self._mk(tmp, "a", (0, 0, 200)), self._mk(tmp, "b", (0, 0, 200)), self._mk(tmp, "c", (200, 0, 0))]
      kept = dedupe_frames(frames, phash_distance=8)
      self.assertEqual([f.id for f in kept], ["a", "c"])
      self.assertFalse((tmp/"b.jpg").exists())

# ***** align *****

class TestBuildSections(unittest.TestCase):
  def test_no_frames_one_section(self):
    out = build_sections([], [seg(0, 1000)], 10_000)
    self.assertEqual((len(out), out[0].frame, len(out[0].segments)), (1, None, 1))

  def test_first_section_starts_at_zero(self): self.assertEqual(build_sections([frame(1000)], [], 10_000)[0].start_ms, 0)

  def test_segments_assigned_by_start_time(self):
    # 4500 starts before frame b at 5000 -> belongs to section a
    out = build_sections([frame(1000, "a"), frame(5000, "b")], [seg(0, 900), seg(4500, 6000), seg(7000, 8000)], 10_000)
    self.assertEqual([len(s.segments) for s in out], [2, 1])

# ***** transcribe *****

class TestParseSegments(unittest.TestCase):
  def test_parse_and_skip_empty(self):
    data = {"transcription": [{"offsets": {"from": 0, "to": 1000}, "text": " hi "}, {"offsets": {"from": 1000, "to": 2000}, "text": "  "}]}
    with tempfile.TemporaryDirectory() as td:
      (p:=Path(td)/"w.json").write_text(json.dumps(data))
      self.assertEqual(parse_segments(p), [Segment(start_ms=0, end_ms=1000, text="hi")])

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
    with tempfile.TemporaryDirectory() as td:
      tmp = Path(td)
      st, fp = Stamps(tmp/"stamps.json"), _fp("input", {"k": 1})
      self.assertFalse(st.fresh("frames", fp, tmp/"art"))
      st.update("frames", fp)
      self.assertFalse(st.fresh("frames", fp, tmp/"art"))  # artifact missing
      (tmp/"art").touch()
      self.assertTrue(st.fresh("frames", fp, tmp/"art"))
      self.assertFalse(st.fresh("frames", _fp("input", {"k": 2}), tmp/"art"))  # settings changed
      self.assertTrue(Stamps(tmp/"stamps.json").fresh("frames", fp, tmp/"art"))  # persists

if __name__ == "__main__": unittest.main()
