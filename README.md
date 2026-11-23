# video-consumer

Turn a video (meeting recording, knowledge-share, screencast) into a **digest
package**: a timestamped transcript interleaved with screenshots taken at
visually meaningful moments. Built for handing to an agent to summarize and
harvest into a context store.

Fully local. No network, no APIs — audio and frames never leave this machine.

## Requirements

System tools:

- `ffmpeg` / `ffprobe` (`brew install ffmpeg`), checked at startup
- `whisper-cli` (`brew install whisper-cpp`) and a whisper ggml model, default
  `~/whisper-models/ggml-small.bin`. Only needed when the video has an audio track.

## Install

```sh
python3 -m venv .venv
.venv/bin/pip install -e .
```

## Usage

```sh
./.venv/bin/video-consumer <video>
```

Output lands in `output/team-sync/`:

```
output/team-sync/
├── transcript.md      # timestamped transcript, one section per visual change,
│                      # each section headed by a screenshot
├── manifest.json      # machine-readable segments + frame alignment
├── frames/            # 640px JPEGs, named NNN_MMSS.jpg (sequence + timestamp)
└── .cache/            # wav, raw whisper json, frame list, stage stamps: makes re-runs instant
```

Then hand the folder to an agent: *"digest output/team-sync into brain"*.

## How frames are chosen

1. **Scene detection** — one ffmpeg pass emits timestamps where visual content
   changes (slide flips, window switches). Threshold `--scene-threshold 0.02`;
   text-only slide flips score ~0.03, camera cuts score ~0.3+.
2. **Min gap** — changes are taken strongest first, and one closer than
   `--min-gap 10` seconds to a frame already kept is dropped, so fast-cut
   content stays bounded.
3. **Cap** — scene frames are capped at ~1 per 20s of video (`--max-frames`
   to override), strongest changes first.
4. **Settle frame** — when the picture changed again within the min gap after
   a kept frame, one more frame is taken once the gap has passed, so the picture
   that stayed on screen is captured too. It is left out when the next scene
   frame follows within another min gap or the video ends first.
5. **Interval fallback** — any stretch longer than `--max-gap 90` seconds with
   no frame gets evenly spaced fills, so static talking-head video still gets
   sampled for changes scene detection missed.
6. **Dedupe** — a perceptual hash drops fill frames near-identical to the
   previous kept frame: a 10-minute static slide keeps one frame, not seven
   copies. Scene and settle frames are never dropped here.

The opening frame is always kept. Settle and fill frames are extra to the cap.
Timestamps where ffmpeg returns no frame (audio that runs longer than the
video, a cut-off file) are skipped.

The frame count depends on the content. Busy recordings and full-screen
webcam video reach the cap; pass `--scene-threshold 0.10` for fewer frames.

## Options

```
-o, --out DIR           output dir (default ./output/<input-stem>/)
-m, --model PATH        whisper ggml model (default ~/whisper-models/ggml-small.bin)
--language LANG         spoken language (default en)
--scene-threshold F     scene score to trigger a frame (default 0.02)
--min-gap SECONDS       minimum spacing between frames (default 10)
--max-gap SECONDS       fallback interval for static stretches (default 90, minimum 1)
--frame-width PX        frame JPEG width (default 640)
--max-frames N          cap on scene frames (default duration/20s)
--force [STAGE]         redo everything, or one of: audio, transcribe, frames.
                        Put it after the input: video-consumer talk.mp4 --force
-v, --verbose           surface ffmpeg/whisper output
```

## Caching

Each stage (audio, transcribe, frames) records a fingerprint of the inputs that
produced it: the source file's path, size and modified time, the model, the
settings, and the video-consumer version. A stage re-runs automatically when
its inputs change and is skipped otherwise, so tweaking frame settings never
re-runs transcription (the slow part: about 26 seconds per 10 minutes of audio
with `ggml-small.bin` on an M3 Pro). Rendering always runs. `--force [stage]`
overrides the cache manually; forcing `audio` also redoes the transcript.
A stage's fingerprint is removed before its files are rewritten, so an
interrupted run is redone, not reused. The source is not hashed: after swapping
in a different file with the same path, size and modified time, use `--force`.

## Notes

- No speaker diarization in v1 — whisper small doesn't do it. Speakers get
  inferred from context during digestion.
- A video with no audio track produces a frames-only digest.
- Any container ffmpeg can read works (mp4, mov, mkv, webm, ...) as long as it
  reports a duration. A live-captured WebM without one is rejected.
