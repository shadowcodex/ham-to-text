# Ham Radio STT

Offline speech-to-text for ham radio audio. Processes pre-recorded files and live audio streams, emitting progressive JSON output.

## Features

- Fully offline — no cloud dependencies
- Progressive JSON streaming output (JSONL)
- Pluggable denoiser pipeline (DeepFilterNet 3 optional)
- Cross-platform: macOS, Windows, Linux
- Configurable via TOML files or CLI flags

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
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

## Install

```bash
# Core (file transcription)
uv pip install ham-radio-stt

# With live streaming support
uv pip install "ham-radio-stt[stream]"

# With DeepFilterNet 3 denoiser
uv pip install "ham-radio-stt[deepfilter]"

# Everything
uv pip install "ham-radio-stt[all]"
```

## Usage

```bash
# Transcribe a file
ham-radio-stt file audio.wav

# Transcribe with JSON output
ham-radio-stt file audio.wav --json

# Stream from default microphone
ham-radio-stt stream --json

# List audio devices
ham-radio-stt devices

# Use a different model
ham-radio-stt file audio.wav --model small
```

## Configuration

Create a `hamstt.toml` in your working directory or `~/.config/hamstt/config.toml` for global settings. See `hamstt.toml.example` for all options.

Config precedence (highest wins):
CLI flags > --config file > ./hamstt.toml > ~/.config/hamstt/config.toml > defaults

## JSON Output Format

Output is newline-delimited JSON (JSONL). Each line has a `"type"` field:

```jsonl
{"type":"transcription","text":"CQ CQ this is W1AW","is_valid":true,...}
{"type":"error","error":"Device not found","code":"STREAM_ERROR"}
```

## Development

```bash
uv pip install -e ".[dev,all]"
pytest                                    # fast tests
pytest -m slow                            # include model-loading tests
pytest -m requires_sox                    # include SoX integration tests
pytest --audio-file recording.wav         # test with real audio files
```
