# Ham to Text

Offline speech-to-text for ham radio audio. Processes pre-recorded files and live audio streams, emitting progressive JSON output.

## Features

- Fully offline — no cloud dependencies
- Progressive JSON streaming output (JSONL)
- Pluggable denoiser pipeline (DeepFilterNet 3 optional)
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
# With live streaming support
uv run --extra stream ham-to-text stream

# With DeepFilterNet 3 denoiser
uv run --extra deepfilter ham-to-text file audio.wav --denoiser deepfilter

# With all extras
uv run --extra all ham-to-text file audio.wav
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
initial_prompt = "Ham radio communication. CQ de W1AW, 73, QSL..."

[sox]
highpass_hz = 200            # High-pass filter cutoff
lowpass_hz = 3400            # Low-pass filter cutoff
norm_level_db = -3.0         # Normalization level

[vad]
filter = true                # Enable voice activity detection
min_silence_duration_ms = 500
speech_pad_ms = 200

[denoiser]
name = "none"                # "none" or "deepfilter"

[deepfilter]
attenuation_limit = 100.0
post_filter = true

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
