# Ham to Text

Offline speech-to-text for ham radio and MARS (Military Auxiliary Radio System) audio. Processes pre-recorded files and live audio streams, optimized for narrowband HF SSB voice.

## Features

- Fully offline — no cloud dependencies
- Progressive JSON streaming output (JSONL)
- Optimized for narrowband ham/MARS radio audio (8kHz-16kHz SSB)
- WebRTC VAD with energy-based fallback for reliable speech detection
- Optional spectral gating denoiser (noisereduce)
- SoX preprocessing with configurable EQ, bandpass, and compression
- Conversational context carries between segments for better accuracy
- Cross-platform: macOS, Windows, Linux
- Configurable via TOML files or CLI flags

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- [SoX](http://sox.sourceforge.net/) installed and on PATH

### Install SoX

```bash
# macOS
brew install sox

# Ubuntu/Debian
sudo apt install sox

# Windows
choco install sox
```

## Quick Start

Clone the repo and run directly with `uv run` — no install step needed:

```bash
git clone https://github.com/shadowcodex/ham-to-text.git
cd ham-to-text

# Transcribe a file
uv run ham-to-text file audio.wav

# With noisereduce denoiser (recommended)
uv run --extra noisereduce ham-to-text file audio.wav --denoiser noisereduce

# Transcribe with JSON output
uv run ham-to-text file audio.wav --json

# Stream from default microphone (requires stream extra)
uv run --extra stream ham-to-text stream --json

# List audio devices
uv run --extra stream ham-to-text devices

# Use a different model
uv run ham-to-text file audio.wav --model small
```

### Optional Extras

```bash
# With noisereduce denoiser (recommended for ham/MARS audio)
uv run --extra noisereduce ham-to-text file audio.wav --denoiser noisereduce

# With live streaming support
uv run --extra stream ham-to-text stream

# With all extras
uv run --extra all ham-to-text file audio.wav
```

## Denoisers

| Denoiser | Install | Best For |
|----------|---------|----------|
| `none` | Built-in | Clean signals, no processing needed |
| `noisereduce` | `--extra noisereduce` | **Recommended.** Narrowband ham/MARS audio (8-16kHz). Spectral gating, lightweight |
| `deepfilter` | `--extra deepfilter` | Wideband (48kHz) speech. Not recommended for narrowband radio audio |

Set the denoiser via CLI flag or config:

```bash
uv run --extra noisereduce ham-to-text file audio.wav --denoiser noisereduce
```

## Whisper Models

This project uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2). The default model is `distil-large-v3`. Models are downloaded automatically on first use (~1-3 GB depending on size).

| Model | Size | Speed | Accuracy | Best For |
|---|---|---|---|---|
| `tiny` | ~75 MB | Fastest | Low | Quick testing |
| `base` | ~150 MB | Very fast | Fair | Low-resource machines |
| `small` | ~500 MB | Fast | Good | General use |
| `medium` | ~1.5 GB | Moderate | Very good | Better accuracy |
| `large-v3` | ~3 GB | Slow | Best | Maximum accuracy |
| `distil-large-v3` | ~1.5 GB | Fast | Very good | **Default** — best speed/accuracy tradeoff |

Set the model via CLI flag or config file:

```bash
uv run ham-to-text file audio.wav --model small
```

## Configuration

Create a `hamstt.toml` in your working directory or `~/.config/hamstt/config.toml` for global settings.

**Precedence** (highest wins): CLI flags > `--config` file > `./hamstt.toml` > `~/.config/hamstt/config.toml` > defaults

### Example `hamstt.toml`

```toml
[whisper]
model = "distil-large-v3"    # See model table above
language = "en"
beam_size = 5
best_of = 5
temperature = 0.0
compute_type = "int8"        # "int8", "float16", "float32"
device = "cpu"               # "cpu" or "cuda"
context_segments = 5         # Prior segments fed as context (0 to disable)

[denoiser]
name = "noisereduce"         # "none", "noisereduce", or "deepfilter"

[noisereduce]
stationary = false           # false = non-stationary mode (better for varying radio noise)
prop_decrease = 0.75         # Noise reduction strength (0.0-1.0)
n_fft = 512                  # FFT size
time_constant_s = 2.0        # Smoothing window

[sox]
highpass_hz = 200            # High-pass filter cutoff
lowpass_hz = 3400            # Low-pass filter cutoff
eq_center_hz = 1800          # Clarity EQ center frequency (0 boost to disable)
eq_boost_db = 6.0            # Clarity EQ boost in dB
norm_level_db = -3.0         # Normalization level

[vad]
filter = true                # Enable voice activity detection
aggressiveness = 0           # 0 = least aggressive (more speech), 3 = most aggressive
frame_ms = 30                # Frame size: 10, 20, or 30 ms
min_silence_ms = 300         # Min silence to split segments
speech_pad_ms = 300          # Padding around speech segments
energy_threshold = 0.02      # RMS threshold for energy-based gap recovery

[deepfilter]
attenuation_limit = 80.0     # Max noise suppression in dB
post_filter = true           # Extra suppression of noisy bins

[streaming]
chunk_duration_s = 0.5
buffer_duration_s = 30.0
silence_timeout_s = 1.5
sample_rate = 44100
# input_device = 0           # Audio device index (from `devices` command)
```

You can also point to a specific config file:

```bash
uv run ham-to-text file audio.wav --config my-config.toml
```

## Debugging Audio Stages

Use `--debug-audio` to save intermediate WAV files after each pipeline stage:

```bash
uv run --extra noisereduce ham-to-text file audio.wav --denoiser noisereduce --debug-audio /tmp/debug
```

This produces:

```
/tmp/debug/
├── 00_input.wav                  # Raw input audio
├── 01_sox_preprocess.wav         # After bandpass/EQ/compand/normalize
├── 02_noisereduce_seg000.wav     # After denoiser (per VAD segment)
├── 02_noisereduce_seg001.wav
└── ...
```

See [docs/audio-processing-guide.md](docs/audio-processing-guide.md) for detailed tuning guidance.

## JSON Output Format

Output is newline-delimited JSON (JSONL). Each line has a `"type"` field:

```jsonl
{"type":"transcription","text":"CQ CQ this is W1AW","is_valid":true,...}
{"type":"error","error":"Device not found","code":"STREAM_ERROR"}
```

## Development

```bash
# Run tests
uv run pytest                                # fast tests
uv run pytest -m slow                        # include model-loading tests
uv run pytest -m requires_sox                # include SoX integration tests
uv run pytest --audio-file recording.wav     # test with real audio files
```
