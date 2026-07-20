# video-consumer

Turn a video (meeting recording, knowledge-share, screencast) into a **digest
package**: a timestamped transcript interleaved with screenshots taken at
visually meaningful moments. Built for handing to an agent to summarize and
harvest into a context store.

Fully local. No network, no APIs — audio and frames never leave this machine.

## Requirements

Already-installed system tools (checked at startup):

- `ffmpeg` / `ffprobe` (`brew install ffmpeg`)
- `whisper-cli` (`brew install whisper-cpp`)
- a whisper ggml model, default `~/whisper-models/ggml-small.bin`

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
└── .cache/            # wav + raw whisper json, makes re-runs instant
```

Then hand the folder to an agent: *"digest output/team-sync into brain"*.

## How frames are chosen

1. **Scene detection** — one ffmpeg pass emits timestamps where visual content
   changes (slide flips, window switches). Threshold `--scene-threshold 0.10`;
   slide flips in screenshares score low (~0.1), camera cuts score high (~0.3+).
2. **Min gap** — candidates closer than `--min-gap 10` seconds to the previous
   kept frame are dropped, so fast-cut content stays bounded.
3. **Interval fallback** — any stretch longer than `--max-gap 90` seconds with
   no frame gets evenly spaced fills, so static talking-head video still gets
   sampled for changes scene detection missed.
4. **Dedupe** — perceptual hash drops frames near-identical to the previous
   one. This applies to fills too, on purpose: a 10-minute static slide keeps
   one frame, not seven copies.
5. **Cap** — scene frames are capped at ~1 per 20s of video (`--max-frames`
   to override); the opening frame and gap fills are always kept.

Typical 45-minute meeting: 30–60 frames.

## Options

```
-o, --out DIR           output dir (default ./output/<input-stem>/)
-m, --model PATH        whisper ggml model (default ~/whisper-models/ggml-small.bin)
--language LANG         spoken language (default en)
--scene-threshold F     scene score to trigger a frame (default 0.10)
--min-gap SECONDS       minimum spacing between frames (default 10)
--max-gap SECONDS       fallback interval for static stretches (default 90)
--frame-width PX        frame JPEG width (default 640)
--max-frames N          cap on scene frames (default duration/20s)
--force [STAGE]         redo everything, or one of: audio, transcribe, frames, render
-v, --verbose           surface ffmpeg/whisper output
```

## Caching

Each stage records a fingerprint of the inputs that produced it (source file,
model, settings). A stage re-runs automatically when its inputs change and is
skipped otherwise, so tweaking frame settings never re-runs transcription (the
slow part, roughly a third of realtime on Apple Silicon). `--force [stage]`
overrides the cache manually.

## Notes

- No speaker diarization in v1 — whisper small doesn't do it. Speakers get
  inferred from context during digestion.
- A video with no audio track produces a frames-only digest.
- Any container ffmpeg can read works: mp4, mov, mkv, webm, ...
