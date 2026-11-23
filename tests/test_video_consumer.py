"""Pure-logic tests: no ffmpeg or whisper needed."""
from __future__ import annotations
import argparse, contextlib, io, json, subprocess, tempfile, unittest
from pathlib import Path
from unittest.mock import Mock, patch
from PIL import Image
from video_consumer.align import Section, build_sections
from video_consumer.cli import Stamps, _arg_range, _clear_cache, _fp, _load_cached_frames, build_parser, main, run
from video_consumer.frames import Frame, FrameSettings, _frame_id, clear_frames, dedupe_frames, detect_scene_changes
from video_consumer.frames import extract_frames, select_timestamps
from video_consumer.probe import VideoInfo, probe
from video_consumer.render import fmt_ts, render_markdown
from video_consumer.transcribe import Segment, parse_segments, transcribe

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
  def test_strongest_change_wins_inside_the_min_gap(self):
    self.assertEqual(select_timestamps([(30_000, 0.05), (34_000, 0.9)], 94_000, FrameSettings(min_gap_s=10, max_gap_s=200)),
                     [(1000, "interval", 0.0), (34_000, "scene", 0.9)])
  def test_settle_frame_after_a_second_change_inside_the_min_gap(self):
    self.assertEqual(select_timestamps([(30_000, 1.0), (34_000, 0.63)], 94_000, FrameSettings(min_gap_s=10, max_gap_s=200)),
                     [(1000, "interval", 0.0), (30_000, "scene", 1.0), (40_000, "settle", 0.0)])
  def test_no_settle_frame_when_the_next_scene_is_close(self):
    self.assertEqual(select_timestamps([(30_000, 1.0), (34_000, 0.63), (45_000, 0.9)], 94_000, FrameSettings(min_gap_s=10, max_gap_s=200)),
                     [(1000, "interval", 0.0), (30_000, "scene", 1.0), (45_000, "scene", 0.9)])
  def test_a_change_right_after_the_opening_gets_a_settle_frame(self):
    self.assertEqual(select_timestamps([(5_000, 0.5)], 60_000, FrameSettings(min_gap_s=10, max_gap_s=200)),
                     [(1000, "interval", 0.0), (11_000, "settle", 0.0)])
  def test_a_settle_frame_before_the_end_needs_no_room_after_it(self):
    self.assertEqual(select_timestamps([(30_000, 1.0), (34_000, 0.63)], 45_000, FrameSettings(min_gap_s=10, max_gap_s=200)),
                     [(1000, "interval", 0.0), (30_000, "scene", 1.0), (40_000, "settle", 0.0)])
  def test_no_settle_frame_at_the_end_of_the_video(self):
    self.assertEqual(select_timestamps([(30_000, 1.0), (34_000, 0.63)], 40_000, FrameSettings(min_gap_s=10, max_gap_s=200)),
                     [(1000, "interval", 0.0), (30_000, "scene", 1.0)])

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

# ***** frames: extraction *****

class TestExtractFrames(unittest.TestCase):
  def extract(self, timestamps:list[tuple[int, str, float]], results:list[str|None]) -> tuple[list[Frame], Mock]:
    with tempfile.TemporaryDirectory() as temp_dir, patch("video_consumer.frames._extract_frame", side_effect=results) as extract_frame:
      with contextlib.redirect_stdout(io.StringIO()): frames = extract_frames(Path("video.mp4"), timestamps, Path(temp_dir), FrameSettings())
    return frames, extract_frame
  def test_an_interval_frame_past_the_video_is_skipped(self):
    frames, _ = self.extract([(1000, "interval", 0.0), (65_500, "interval", 0.0)], [None, "no frame"])
    self.assertEqual([item.t_ms for item in frames], [1000])
  def test_a_failed_scene_frame_is_skipped(self):
    frames, _ = self.extract([(1000, "interval", 0.0), (30_000, "scene", 0.5), (40_000, "scene", 0.4)], [None, "no frame", None])
    self.assertEqual([item.t_ms for item in frames], [1000, 40_000])
  def test_the_opening_frame_falls_back_to_the_first_frame(self):
    frames, extract_frame = self.extract([(500, "interval", 0.0)], ["no frame", None])
    self.assertEqual(([item.t_ms for item in frames], extract_frame.call_args_list[1].args[1]), ([500], 0))
  def test_an_opening_frame_that_fails_twice_raises(self):
    self.assertRaises(RuntimeError, self.extract, [(500, "interval", 0.0)], ["no frame", "no frame"])

class TestClearFrames(unittest.TestCase):
  def test_only_frame_jpegs_are_deleted(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      directory = Path(temp_dir)
      for name in ("001_0001.jpg", "012_10203.jpg", "001_0001.jpg.bak", "notes.txt", "source.mp4", "photo.jpg"): (directory/name).touch()
      (directory/"003_0001.jpg").mkdir()
      clear_frames(directory)
      left = sorted(path.name for path in directory.iterdir())
      self.assertEqual(left, ["001_0001.jpg.bak", "003_0001.jpg", "notes.txt", "photo.jpg", "source.mp4"])

# ***** frames: dedupe *****

class TestDedupe(unittest.TestCase):
  def _make(self, directory:Path, name:str, color:tuple[int, int, int], trigger="interval") -> Frame:
    Image.new("RGB", (64, 36), color).save(path:=directory/f"{name}.jpg")
    return Frame(id=name, t_ms=0, path=path, trigger=trigger)
  def test_identical_dropped_different_color_kept(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      directory = Path(temp_dir)
      frames = [self._make(directory, "a", (0, 0, 200)), self._make(directory, "b", (0, 0, 200)), self._make(directory, "c", (200, 0, 0))]
      self.assertEqual([item.id for item in dedupe_frames(frames, phash_distance=8)], ["a", "c"])
      self.assertFalse((directory/"b.jpg").exists())
  def test_identical_scene_and_settle_frames_are_kept(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      directory = Path(temp_dir)
      frames = [self._make(directory, "a", (0, 0, 200)), self._make(directory, "b", (0, 0, 200), trigger="scene"),
                self._make(directory, "c", (0, 0, 200), trigger="settle")]
      self.assertEqual([item.id for item in dedupe_frames(frames, phash_distance=8)], ["a", "b", "c"])

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
  def test_a_cut_file_raises(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      (path:=Path(temp_dir)/"whisper.json").write_text('{"transcription": [')
      self.assertRaises(RuntimeError, parse_segments, path)
  def test_a_missing_transcription_key_raises(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      (path:=Path(temp_dir)/"whisper.json").write_text('{"result": {}}')
      self.assertRaises(RuntimeError, parse_segments, path)
  def test_offset_moves_both_ends(self):
    data = {"transcription": [{"offsets": {"from": 0, "to": 1000}, "text": "hi"}]}
    with tempfile.TemporaryDirectory() as temp_dir:
      (path:=Path(temp_dir)/"whisper.json").write_text(json.dumps(data))
      self.assertEqual(parse_segments(path, 6000), [Segment(start_ms=6000, end_ms=7000, text="hi")])

class TestTranscribe(unittest.TestCase):
  @patch("video_consumer.transcribe.subprocess.run")
  def test_exit_0_without_json_does_not_reuse_the_old_file(self, run:Mock):
    run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="error: unknown language 'xx'\n")
    with tempfile.TemporaryDirectory() as temp_dir:
      directory = Path(temp_dir)
      (model:=directory/"model.bin").touch()
      (out_json:=directory/"whisper.json").write_text('{"transcription": []}')
      with self.assertRaisesRegex(RuntimeError, "unknown language"): transcribe(directory/"audio.wav", model, out_json, "xx")
      self.assertFalse(out_json.exists())

# ***** render *****

class TestFmtTs(unittest.TestCase):
  def test_minutes(self): self.assertEqual(fmt_ts(63_000), "01:03")
  def test_hours(self): self.assertEqual(fmt_ts(3_723_000), "1:02:03")
  def test_zero(self): self.assertEqual(fmt_ts(0), "00:00")

class TestRenderMarkdown(unittest.TestCase):
  def bodies(self, sections:list[Section]) -> list[str]:
    with tempfile.TemporaryDirectory() as temp_dir:
      out = render_markdown("title", "video.mp4", 30_000, "model.bin", sections, Path(temp_dir)/"transcript.md")
      return [line for line in out.read_text().splitlines() if line and line[0] not in "#-!"]
  def test_a_section_under_a_running_segment_is_not_called_silent(self):
    sections = [Section(frame=None, start_ms=0, end_ms=10_000, segments=[seg(0, 15_000, "hello")]),
                Section(frame=None, start_ms=10_000, end_ms=20_000), Section(frame=None, start_ms=20_000, end_ms=30_000)]
    self.assertEqual(self.bodies(sections), ["hello", "*[speech continues from the previous section]*", "*[no speech]*"])
  def test_a_long_segment_still_counts_after_a_shorter_one(self):
    sections = [Section(frame=None, start_ms=0, end_ms=10_000, segments=[seg(0, 25_000, "long")]),
                Section(frame=None, start_ms=10_000, end_ms=20_000, segments=[seg(10_500, 11_500, "short")]),
                Section(frame=None, start_ms=20_000, end_ms=30_000)]
    self.assertEqual(self.bodies(sections), ["long", "short", "*[speech continues from the previous section]*"])

# ***** probe *****

class TestProbe(unittest.TestCase):
  def audio_start_ms(self, container_start:str, first_audio_pts:str) -> int:
    streams = [{"codec_type": "video", "width": 640, "height": 360}, {"codec_type": "audio"}]
    replies = [{"format": {"duration": "12.0", "start_time": container_start}, "streams": streams}, {"packets": [{"pts_time": first_audio_pts}]}]
    results = [subprocess.CompletedProcess(args=[], returncode=0, stdout=json.dumps(reply)) for reply in replies]
    with tempfile.NamedTemporaryFile() as video, patch("video_consumer.probe.subprocess.run", side_effect=results):
      return probe(Path(video.name)).audio_start_ms
  def test_late_audio_start_is_measured_from_the_container_start(self): self.assertEqual(self.audio_start_ms("1.600000", "7.553011"), 5953)
  def test_audio_before_the_container_start_counts_as_zero(self): self.assertEqual(self.audio_start_ms("0.000000", "-0.046440"), 0)

# ***** cli *****

class TestArgRange(unittest.TestCase):
  def test_lo_open_rejects_zero(self): self.assertRaises(argparse.ArgumentTypeError, _arg_range(float, 0, lo_open=True), "0")
  def test_hi_rejects(self): self.assertRaises(argparse.ArgumentTypeError, _arg_range(float, 0, 1, lo_open=True), "1.5")
  def test_accepts(self): self.assertEqual(_arg_range(int, 1)("5"), 5)
  def test_non_numeric(self): self.assertRaises(argparse.ArgumentTypeError, _arg_range(int, 1), "abc")
  def test_nan_is_rejected(self): self.assertRaises(argparse.ArgumentTypeError, _arg_range(float, 0), "nan")
  def test_inf_is_rejected(self): self.assertRaises(argparse.ArgumentTypeError, _arg_range(float, 0), "inf")

class TestParser(unittest.TestCase):
  def rejects(self, *options:str):
    with contextlib.redirect_stderr(io.StringIO()): self.assertRaises(SystemExit, build_parser().parse_args, ["video.mp4", *options])
  def test_max_gap_under_a_second_is_rejected(self): self.rejects("--max-gap", "0.0004")
  def test_max_gap_too_big_for_milliseconds_is_rejected(self): self.rejects("--max-gap", "1e306")

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
  def test_a_dropped_stage_is_not_fresh_after_a_reload(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      directory = Path(temp_dir)
      stamps, artifact = Stamps(directory/"stamps.json"), directory/"audio.wav"
      artifact.touch()
      stamps.update("audio", "fingerprint")
      stamps.drop("audio")
      self.assertFalse(Stamps(directory/"stamps.json").fresh("audio", "fingerprint", artifact))
  def test_a_file_that_is_not_an_object_reads_as_empty(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      directory = Path(temp_dir)
      (directory/"stamps.json").write_text("[]")
      artifact = directory/"audio.wav"
      artifact.touch()
      self.assertFalse(Stamps(directory/"stamps.json").fresh("audio", "fingerprint", artifact))

class TestClearCache(unittest.TestCase):
  def test_a_forced_audio_stage_is_not_fresh_when_a_cut_wav_reappears(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      cache = Path(temp_dir)
      (wav:=cache/"audio.wav").touch()
      (frames_json:=cache/"frames.json").touch()
      stamps = Stamps(cache/"stamps.json")
      stamps.update("audio", "fingerprint")
      stamps.update("frames", "fingerprint")
      _clear_cache("audio", cache, cache/"frames", stamps)
      wav.touch()
      self.assertFalse(stamps.fresh("audio", "fingerprint", wav))
      self.assertTrue(stamps.fresh("frames", "fingerprint", frames_json))

class TestCachedFrames(unittest.TestCase):
  def test_a_cut_frames_json_is_a_miss(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      directory = Path(temp_dir)
      (frames_json:=directory/"frames.json").write_text("[{")
      self.assertIsNone(_load_cached_frames(frames_json, directory))
  def test_a_frames_json_without_frames_is_a_miss(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      directory = Path(temp_dir)
      (frames_json:=directory/"frames.json").write_text("{}")
      self.assertIsNone(_load_cached_frames(frames_json, directory))

class TestRunStamps(unittest.TestCase):
  def interrupt(self, has_audio:bool, stage_function:str) -> dict:
    with tempfile.TemporaryDirectory() as temp_dir:
      directory = Path(temp_dir)
      (video:=directory/"video.mp4").touch()
      (model:=directory/"model.bin").touch()
      (cache:=directory/"out"/".cache").mkdir(parents=True)
      (cache/"stamps.json").write_text('{"audio": "old", "transcribe": "old", "frames": "old"}')
      args = build_parser().parse_args([str(video), "-o", str(directory/"out"), "-m", str(model)])
      info = VideoInfo(path=video, duration_ms=60_000, width=640, height=360, has_audio=has_audio)
      with patch("video_consumer.cli.require_binaries"), patch("video_consumer.cli.probe", return_value=info):
        with patch(f"video_consumer.cli.{stage_function}", side_effect=KeyboardInterrupt), contextlib.redirect_stdout(io.StringIO()):
          self.assertRaises(KeyboardInterrupt, run, args)
      return json.loads((cache/"stamps.json").read_text())
  def test_interrupted_audio_extraction_leaves_no_audio_stamp(self): self.assertNotIn("audio", self.interrupt(True, "extract_audio"))
  def test_interrupted_audio_extraction_leaves_no_transcribe_stamp(self): self.assertNotIn("transcribe", self.interrupt(True, "extract_audio"))
  def test_interrupted_frame_build_leaves_no_frames_stamp(self): self.assertNotIn("frames", self.interrupt(False, "build_frames"))
  def test_a_new_version_rebuilds_the_frames(self):
    def build(video:Path, duration_ms:int, frames_dir:Path, settings:FrameSettings, verbose:bool) -> list[Frame]:
      frames_dir.mkdir(exist_ok=True)
      (path:=frames_dir/"001_0001.jpg").touch()
      return [Frame(id="001_0001", t_ms=1000, path=path, trigger="interval")]
    with tempfile.TemporaryDirectory() as temp_dir:
      (video:=Path(temp_dir)/"video.mp4").touch()
      args = build_parser().parse_args([str(video), "-o", str(Path(temp_dir)/"out")])
      info = VideoInfo(path=video, duration_ms=60_000, width=640, height=360, has_audio=False)
      with patch("video_consumer.cli.require_binaries"), patch("video_consumer.cli.probe", return_value=info):
        with patch("video_consumer.cli.build_frames", side_effect=build) as build_frames, contextlib.redirect_stdout(io.StringIO()):
          run(args)
          run(args)
          with patch("video_consumer.cli.__version__", "9.9.9"): run(args)
    self.assertEqual(build_frames.call_count, 2)

class TestMain(unittest.TestCase):
  @patch("sys.argv", ["video-consumer", "video.mp4"])
  @patch("video_consumer.cli.run", side_effect=NotADirectoryError("out is a file"))
  def test_a_filesystem_error_exits_with_1(self, _run:Mock):
    with contextlib.redirect_stderr(io.StringIO()):
      with self.assertRaises(SystemExit) as caught: main()
    self.assertEqual(caught.exception.code, 1)

if __name__ == "__main__": unittest.main()
