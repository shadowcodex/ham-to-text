# Ham Radio STT Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI-first speech-to-text tool for ham radio audio with pluggable pipeline, progressive JSON streaming output, and cross-platform support.

**Architecture:** A four-stage pipeline (SoX preprocessing -> pluggable denoiser -> Whisper transcription) with a `PipelineStage` protocol for composability. CLI emits newline-delimited JSON (JSONL) for machine consumption. Config layered from defaults -> global TOML -> local TOML -> CLI flags.

**Tech Stack:** Python 3.11+, faster-whisper (CTranslate2), SoX (subprocess), sounddevice (optional), DeepFilterNet 3 (optional), numpy, soundfile, pytest

**Spec:** `docs/superpowers/specs/2026-03-14-ham-radio-stt-design.md`

---

## File Map

| File | Responsibility | Created in Task |
|---|---|---|
| `pyproject.toml` | Package metadata, deps, extras, scripts entry point | 1 |
| `.gitignore` | Python/IDE/audio ignores | 1 |
| `README.md` | Project overview, install, usage | 1 |
| `hamstt.toml.example` | Example config file | 1 |
| `ham_radio_stt/__init__.py` | Public API exports, exception hierarchy | 2 |
| `ham_radio_stt/result.py` | `TranscriptionResult` dataclass | 2 |
| `ham_radio_stt/config.py` | `PipelineConfig` dataclass, TOML loading | 3 |
| `ham_radio_stt/stages/__init__.py` | `PipelineStage` protocol | 4 |
| `ham_radio_stt/stages/sox_preprocess.py` | SoX subprocess stage | 4 |
| `ham_radio_stt/stages/denoise.py` | Denoiser registry + `NoOpDenoiser` | 5 |
| `ham_radio_stt/stages/deepfilter.py` | DFN3 denoiser (optional) | 5 |
| `ham_radio_stt/transcribe.py` | Whisper transcription wrapper | 6 |
| `ham_radio_stt/pipeline.py` | `Pipeline` class — composes stages | 7 |
| `ham_radio_stt/cli.py` | argparse CLI, JSON output formatting | 8 |
| `ham_radio_stt/__main__.py` | `python -m ham_radio_stt` entry point | 8 |
| `ham_radio_stt/streaming.py` | `StreamingSession` — capture + flush | 9 |
| `tests/conftest.py` | Fixtures, `--audio-file` option, markers | 2 |
| `tests/test_result.py` | `TranscriptionResult` tests | 2 |
| `tests/test_config.py` | Config loading/layering tests | 3 |
| `tests/test_stages.py` | SoX + denoiser stage tests | 4, 5 |
| `tests/test_transcribe.py` | Whisper transcription tests | 6 |
| `tests/test_pipeline.py` | Pipeline integration tests | 7 |
| `tests/test_cli.py` | CLI output format + exit code tests | 8 |
| `tests/test_streaming.py` | Streaming session tests | 9 |

---

## Chunk 1: Project Scaffolding & Core Data Types

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`
- Create: `hamstt.toml.example`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "ham-radio-stt"
version = "0.1.0"
description = "CLI speech-to-text for ham radio audio — offline, pluggable, fast"
readme = "README.md"
requires-python = ">=3.11"
license = "MIT"
dependencies = [
    "faster-whisper>=1.0.0",
    "soundfile>=0.12.0",
    "numpy>=1.24.0",
]

[project.optional-dependencies]
deepfilter = [
    "deepfilternet>=0.5.0",
    "scipy>=1.10.0",
]
stream = [
    "sounddevice>=0.4.6",
]
all = [
    "ham-radio-stt[deepfilter,stream]",
]
dev = [
    "pytest>=7.0.0",
    "pytest-mock>=3.10.0",
]

[project.scripts]
ham-radio-stt = "ham_radio_stt.cli:main"
```

- [ ] **Step 2: Create `.gitignore`**

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
*.egg-info/
dist/
build/
*.egg
.eggs/

# Virtual environments
.venv/
venv/
env/

# IDE
.idea/
.vscode/
*.swp
*.swo
*~

# Testing
.pytest_cache/
.coverage
htmlcov/

# OS
.DS_Store
Thumbs.db

# Audio files (don't commit test recordings)
*.wav
*.mp3
*.flac
*.ogg

# Models (downloaded by faster-whisper)
models/

# Config (user-specific)
hamstt.toml
```

- [ ] **Step 3: Create `README.md`**

```markdown
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
pip install ham-radio-stt

# With live streaming support
pip install ham-radio-stt[stream]

# With DeepFilterNet 3 denoiser
pip install ham-radio-stt[deepfilter]

# Everything
pip install ham-radio-stt[all]
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
pip install -e ".[dev,all]"
pytest                                    # fast tests
pytest -m slow                            # include model-loading tests
pytest --audio-file recording.wav         # test with real audio
```
```

- [ ] **Step 4: Create `hamstt.toml.example`**

```toml
# Ham Radio STT Configuration
# Copy to ./hamstt.toml or ~/.config/hamstt/config.toml

[whisper]
model = "distil-large-v3"       # Options: distil-large-v3, large-v3-turbo, medium, small, base, tiny
language = "en"
beam_size = 5
best_of = 5
temperature = 0.0               # 0 = deterministic, reduces hallucination
compute_type = "int8"            # int8 is fastest on CPU
device = "cpu"
initial_prompt = "Ham radio communication. CQ de W1AW, 73, QSL, QRM, QRN, QSB, QTH, over, copy, roger."

[denoiser]
name = "none"                    # Options: none, deepfilter

[sox]
highpass_hz = 200                # Ham voice starts ~300Hz, be conservative
lowpass_hz = 3400                # SSB/AM upper cutoff
norm_level_db = -3.0

[vad]
filter = true
min_silence_duration_ms = 500
speech_pad_ms = 200

[streaming]
chunk_duration_s = 0.5
buffer_duration_s = 30.0
silence_timeout_s = 1.5
sample_rate = 44100
# input_device = 2              # Uncomment and set to specific device index

[deepfilter]
attenuation_limit = 100.0       # dB, max noise attenuation
post_filter = true              # Slight over-attenuation of very noisy sections
```

- [ ] **Step 5: Create package directories**

Run: `mkdir -p ham_radio_stt/stages tests`

- [ ] **Step 6: Commit scaffolding**

```bash
git add pyproject.toml .gitignore README.md hamstt.toml.example
git commit -m "feat: add project scaffolding with pyproject.toml, README, gitignore"
```

---

### Task 2: TranscriptionResult dataclass, exceptions, and test fixtures

**Files:**
- Create: `ham_radio_stt/__init__.py`
- Create: `ham_radio_stt/result.py`
- Create: `tests/conftest.py`
- Create: `tests/test_result.py`

- [ ] **Step 1: Write tests for `TranscriptionResult`**

Create `tests/test_result.py`:

```python
import pytest
from ham_radio_stt.result import TranscriptionResult


def _make_result(**overrides) -> TranscriptionResult:
    defaults = dict(
        text="CQ CQ this is W1AW",
        segments=[{"start": 0.0, "end": 3.0, "text": "CQ CQ this is W1AW"}],
        language="en",
        duration_s=3.0,
        processing_time_s=1.5,
        real_time_factor=0.5,
        audio_source="file:test.wav",
        avg_log_prob=-0.2,
        no_speech_prob=0.03,
    )
    defaults.update(overrides)
    return TranscriptionResult(**defaults)


class TestIsLikelyValid:
    def test_valid_speech(self):
        result = _make_result()
        assert result.is_likely_valid() is True

    def test_high_no_speech_prob(self):
        result = _make_result(no_speech_prob=0.8)
        assert result.is_likely_valid() is False

    def test_low_confidence(self):
        result = _make_result(avg_log_prob=-1.5)
        assert result.is_likely_valid() is False

    def test_empty_text(self):
        result = _make_result(text="")
        assert result.is_likely_valid() is False

    def test_whitespace_only_text(self):
        result = _make_result(text="   ")
        assert result.is_likely_valid() is False

    def test_boundary_no_speech_prob_exclusive(self):
        """0.6 is NOT < 0.6, so boundary value is invalid."""
        result = _make_result(no_speech_prob=0.6)
        assert result.is_likely_valid() is False

    def test_just_below_no_speech_threshold(self):
        result = _make_result(no_speech_prob=0.59)
        assert result.is_likely_valid() is True

    def test_boundary_avg_log_prob_exclusive(self):
        """-1.0 is NOT > -1.0, so boundary value is invalid."""
        result = _make_result(avg_log_prob=-1.0)
        assert result.is_likely_valid() is False

    def test_just_above_log_prob_threshold(self):
        result = _make_result(avg_log_prob=-0.99)
        assert result.is_likely_valid() is True


class TestToJsonDict:
    def test_file_mode_fields(self):
        result = _make_result(segment_index=0, offset_s=0.0, is_final=True)
        d = result.to_json_dict()
        assert d["type"] == "transcription"
        assert d["segment_index"] == 0
        assert d["offset_s"] == 0.0
        assert d["is_final"] is True
        assert "timestamp" not in d

    def test_stream_mode_fields(self):
        result = _make_result(timestamp="2026-03-14T18:30:01Z")
        d = result.to_json_dict()
        assert d["type"] == "transcription"
        assert d["timestamp"] == "2026-03-14T18:30:01Z"
        assert "segment_index" not in d

    def test_is_valid_included(self):
        result = _make_result()
        d = result.to_json_dict()
        assert d["is_valid"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_result.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ham_radio_stt'`

- [ ] **Step 3: Create `ham_radio_stt/__init__.py`**

```python
"""Ham Radio STT — Offline speech-to-text for ham radio audio."""

__version__ = "0.1.0"


class HamSTTError(Exception):
    """Base exception for ham-radio-stt."""


class AudioProcessingError(HamSTTError):
    """SoX failed, bad audio format, or corrupt file."""


class ModelLoadError(HamSTTError):
    """Model download failed, disk full, or missing dependency."""


class StreamError(HamSTTError):
    """Audio device not found or disconnected."""


class ConfigError(HamSTTError):
    """Bad TOML config or invalid parameter values."""
```

- [ ] **Step 4: Create `ham_radio_stt/result.py`**

```python
"""TranscriptionResult dataclass."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional


@dataclass
class TranscriptionResult:
    text: str
    segments: list[dict]
    language: str
    duration_s: float
    processing_time_s: float
    real_time_factor: float
    audio_source: str
    avg_log_prob: float
    no_speech_prob: float

    # File mode
    segment_index: Optional[int] = None
    offset_s: Optional[float] = None
    is_final: Optional[bool] = None

    # Stream mode
    timestamp: Optional[str] = None

    def is_likely_valid(self) -> bool:
        return (
            self.no_speech_prob < 0.6
            and self.avg_log_prob > -1.0
            and len(self.text.strip()) > 0
        )

    def to_json_dict(self) -> dict:
        d: dict = {"type": "transcription"}
        d["text"] = self.text
        d["is_valid"] = self.is_likely_valid()
        d["duration_s"] = self.duration_s
        d["processing_time_s"] = self.processing_time_s
        d["real_time_factor"] = self.real_time_factor
        d["avg_log_prob"] = self.avg_log_prob
        d["no_speech_prob"] = self.no_speech_prob
        d["segments"] = self.segments

        # Include mode-specific fields only if set
        if self.segment_index is not None:
            d["segment_index"] = self.segment_index
        if self.offset_s is not None:
            d["offset_s"] = self.offset_s
        if self.is_final is not None:
            d["is_final"] = self.is_final
        if self.timestamp is not None:
            d["timestamp"] = self.timestamp

        return d
```

- [ ] **Step 5: Create `tests/conftest.py`**

```python
"""Shared test fixtures and pytest configuration."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf


def pytest_addoption(parser):
    parser.addoption(
        "--audio-file",
        action="append",
        default=[],
        help="Path to real audio file(s) for integration testing",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests that load ML models")
    config.addinivalue_line("markers", "requires_sox: marks tests that need SoX on PATH")
    config.addinivalue_line("markers", "real_audio: marks tests that use --audio-file recordings")


def pytest_collection_modifyitems(config, items):
    audio_files = config.getoption("--audio-file")
    if not audio_files:
        skip_real = pytest.mark.skip(reason="needs --audio-file option")
        for item in items:
            if "real_audio" in item.keywords:
                item.add_marker(skip_real)


@pytest.fixture
def noisy_tone_wav(tmp_path):
    """Generate a 3-second WAV with 440Hz tone + white noise."""
    sr = 44100
    t = np.linspace(0, 3, sr * 3, dtype=np.float32)
    tone = 0.3 * np.sin(2 * np.pi * 440 * t)
    noise = 0.1 * np.random.default_rng(42).standard_normal(len(t)).astype(np.float32)
    audio = tone + noise
    path = tmp_path / "noisy_tone.wav"
    sf.write(str(path), audio, sr)
    return path


@pytest.fixture
def silence_wav(tmp_path):
    """Generate a 2-second silent WAV."""
    sr = 16000
    audio = np.zeros(sr * 2, dtype=np.float32)
    path = tmp_path / "silence.wav"
    sf.write(str(path), audio, sr)
    return path


@pytest.fixture
def multi_segment_wav(tmp_path):
    """Generate a WAV with tone-silence-tone pattern."""
    sr = 44100
    tone_len = sr * 2  # 2 seconds of tone
    silence_len = sr * 1  # 1 second of silence
    t_tone = np.linspace(0, 2, tone_len, dtype=np.float32)
    tone = 0.3 * np.sin(2 * np.pi * 440 * t_tone)
    silence = np.zeros(silence_len, dtype=np.float32)
    audio = np.concatenate([tone, silence, tone])
    path = tmp_path / "multi_segment.wav"
    sf.write(str(path), audio, sr)
    return path


@pytest.fixture
def real_audio_files(request):
    """Yield paths from --audio-file option. Skips if none provided."""
    from pathlib import Path
    files = request.config.getoption("--audio-file")
    if not files:
        pytest.skip("No --audio-file provided")
    return [Path(f) for f in files]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_result.py -v`
Expected: All 12 tests PASS

- [ ] **Step 7: Commit**

```bash
git add ham_radio_stt/__init__.py ham_radio_stt/result.py tests/conftest.py tests/test_result.py
git commit -m "feat: add TranscriptionResult dataclass, exceptions, and test fixtures"
```

---

### Task 3: PipelineConfig and TOML loading

**Files:**
- Create: `ham_radio_stt/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write config tests**

Create `tests/test_config.py`:

```python
import pytest
from pathlib import Path
from ham_radio_stt.config import PipelineConfig, load_config_from_toml


class TestPipelineConfigDefaults:
    def test_default_model(self):
        config = PipelineConfig()
        assert config.whisper_model == "distil-large-v3"

    def test_default_denoiser(self):
        config = PipelineConfig()
        assert config.denoiser == "none"

    def test_default_sample_rate(self):
        config = PipelineConfig()
        assert config.target_sample_rate == 16000

    def test_default_silence_timeout(self):
        config = PipelineConfig()
        assert config.stream_silence_timeout_s == 1.5


class TestLoadConfigFromToml:
    def test_load_whisper_section(self, tmp_path):
        toml_file = tmp_path / "test.toml"
        toml_file.write_text('[whisper]\nmodel = "small"\nbeam_size = 3\n')
        config, set_fields = load_config_from_toml(toml_file)
        assert config.whisper_model == "small"
        assert config.whisper_beam_size == 3
        assert set_fields == {"whisper_model", "whisper_beam_size"}
        # Other defaults preserved
        assert config.whisper_language == "en"

    def test_load_denoiser_section(self, tmp_path):
        toml_file = tmp_path / "test.toml"
        toml_file.write_text('[denoiser]\nname = "deepfilter"\n')
        config, set_fields = load_config_from_toml(toml_file)
        assert config.denoiser == "deepfilter"
        assert "denoiser" in set_fields

    def test_load_sox_section(self, tmp_path):
        toml_file = tmp_path / "test.toml"
        toml_file.write_text("[sox]\nhighpass_hz = 300\n")
        config, _ = load_config_from_toml(toml_file)
        assert config.sox_highpass_hz == 300
        assert config.sox_lowpass_hz == 3400  # default preserved

    def test_load_streaming_section(self, tmp_path):
        toml_file = tmp_path / "test.toml"
        toml_file.write_text("[streaming]\nsilence_timeout_s = 3.0\n")
        config, _ = load_config_from_toml(toml_file)
        assert config.stream_silence_timeout_s == 3.0

    def test_load_deepfilter_section(self, tmp_path):
        toml_file = tmp_path / "test.toml"
        toml_file.write_text("[deepfilter]\nattenuation_limit = 80.0\npost_filter = false\n")
        config, _ = load_config_from_toml(toml_file)
        assert config.dfn_attenuation_limit == 80.0
        assert config.dfn_post_filter is False

    def test_load_vad_section(self, tmp_path):
        toml_file = tmp_path / "test.toml"
        toml_file.write_text("[vad]\nmin_silence_duration_ms = 300\n")
        config, _ = load_config_from_toml(toml_file)
        assert config.vad_min_silence_duration_ms == 300

    def test_nonexistent_file_returns_defaults(self):
        config, set_fields = load_config_from_toml(Path("/nonexistent/path.toml"))
        assert config == PipelineConfig()
        assert set_fields == set()

    def test_invalid_toml_raises_config_error(self, tmp_path):
        from ham_radio_stt import ConfigError
        toml_file = tmp_path / "bad.toml"
        toml_file.write_text("this is not valid toml [[[")
        with pytest.raises(ConfigError):
            load_config_from_toml(toml_file)

    def test_unknown_key_raises_config_error(self, tmp_path):
        from ham_radio_stt import ConfigError
        toml_file = tmp_path / "test.toml"
        toml_file.write_text("[whisper]\nnonexistent_key = 42\n")
        with pytest.raises(ConfigError):
            load_config_from_toml(toml_file)

    def test_partial_toml_does_not_clobber_earlier_layers(self, tmp_path):
        """A partial TOML should only override the fields it specifies."""
        from ham_radio_stt.config import load_config
        from dataclasses import replace

        global_toml = tmp_path / "global.toml"
        global_toml.write_text('[whisper]\nmodel = "small"\n')
        local_toml = tmp_path / "local.toml"
        local_toml.write_text("[sox]\nhighpass_hz = 300\n")

        # Simulate layering: global sets model, local sets sox — model should survive
        global_config, _ = load_config_from_toml(global_toml)
        local_config, local_fields = load_config_from_toml(local_toml)
        changes = {f: getattr(local_config, f) for f in local_fields}
        merged = replace(global_config, **changes)

        assert merged.whisper_model == "small"  # from global, not clobbered
        assert merged.sox_highpass_hz == 300     # from local
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ham_radio_stt.config'`

- [ ] **Step 3: Implement `ham_radio_stt/config.py`**

```python
"""PipelineConfig dataclass and TOML config loading."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Optional

from ham_radio_stt import ConfigError

# Maps TOML [section] -> dataclass field prefix
_SECTION_PREFIX = {
    "whisper": "whisper_",
    "sox": "sox_",
    "streaming": "stream_",
    "deepfilter": "dfn_",
    "vad": "vad_",
}

# Special mappings where TOML key doesn't follow the prefix pattern
_SPECIAL_KEYS = {
    ("denoiser", "name"): "denoiser",
    ("whisper", "model"): "whisper_model",
    ("whisper", "device"): "whisper_device",
    ("sox", "norm_level_db"): "sox_norm_level_db",
}


@dataclass
class PipelineConfig:
    # Denoiser
    denoiser: str = "none"

    # Whisper
    whisper_model: str = "distil-large-v3"
    whisper_language: str = "en"
    whisper_beam_size: int = 5
    whisper_best_of: int = 5
    whisper_temperature: float = 0.0
    whisper_compute_type: str = "int8"
    whisper_device: str = "cpu"
    whisper_initial_prompt: str = (
        "Ham radio communication. "
        "CQ de W1AW, 73, QSL, QRM, QRN, QSB, QTH, over, copy, roger, "
        "phonetic alphabet: Alpha Bravo Charlie Delta Echo Foxtrot Golf "
        "Hotel India Juliet Kilo Lima Mike November Oscar Papa Quebec "
        "Romeo Sierra Tango Uniform Victor Whiskey X-ray Yankee Zulu."
    )

    # SoX
    sox_highpass_hz: int = 200
    sox_lowpass_hz: int = 3400
    sox_norm_level_db: float = -3.0
    target_sample_rate: int = 16000
    target_channels: int = 1

    # VAD
    vad_filter: bool = True
    vad_min_silence_duration_ms: int = 500
    vad_speech_pad_ms: int = 200

    # Streaming
    stream_chunk_duration_s: float = 0.5
    stream_buffer_duration_s: float = 30.0
    stream_silence_timeout_s: float = 1.5
    stream_sample_rate: int = 44100
    stream_input_device: Optional[int] = None

    # DeepFilterNet
    dfn_attenuation_limit: float = 100.0
    dfn_post_filter: bool = True


def _field_names() -> set[str]:
    return {f.name for f in fields(PipelineConfig)}


def _toml_key_to_field(section: str, key: str) -> str:
    special = _SPECIAL_KEYS.get((section, key))
    if special:
        return special
    prefix = _SECTION_PREFIX.get(section)
    if prefix:
        return f"{prefix}{key}"
    return key


def load_config_from_toml(path: Path) -> tuple[PipelineConfig, set[str]]:
    """Load config from TOML. Returns (config, set_of_field_names_that_were_set)."""
    if not path.exists():
        return PipelineConfig(), set()

    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"Invalid TOML in {path}: {e}") from e

    valid_fields = _field_names()
    overrides: dict = {}

    for section, values in data.items():
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            field_name = _toml_key_to_field(section, key)
            if field_name not in valid_fields:
                raise ConfigError(
                    f"Unknown config key [{section}] {key} "
                    f"(mapped to '{field_name}') in {path}"
                )
            overrides[field_name] = value

    return replace(PipelineConfig(), **overrides), set(overrides.keys())


def global_config_path() -> Path:
    return Path.home() / ".config" / "hamstt" / "config.toml"


def load_config(
    config_file: Optional[Path] = None,
    cli_overrides: Optional[dict] = None,
) -> PipelineConfig:
    config, _ = load_config_from_toml(global_config_path())

    local = Path("hamstt.toml")
    if local.exists():
        local_config, local_fields = load_config_from_toml(local)
        # Only override fields actually present in local TOML
        changes = {f: getattr(local_config, f) for f in local_fields}
        config = replace(config, **changes)

    if config_file:
        file_config, file_fields = load_config_from_toml(config_file)
        changes = {f: getattr(file_config, f) for f in file_fields}
        config = replace(config, **changes)

    if cli_overrides:
        valid_fields = _field_names()
        for key in cli_overrides:
            if key not in valid_fields:
                raise ConfigError(f"Unknown CLI config key: {key}")
        config = replace(config, **cli_overrides)

    return config
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_config.py -v`
Expected: All 13 tests PASS

- [ ] **Step 5: Commit**

```bash
git add ham_radio_stt/config.py tests/test_config.py
git commit -m "feat: add PipelineConfig with TOML loading and layered config"
```

---

## Chunk 2: Pipeline Stages (SoX, Denoisers)

### Task 4: PipelineStage protocol and SoX preprocessing

**Files:**
- Create: `ham_radio_stt/stages/__init__.py`
- Create: `ham_radio_stt/stages/sox_preprocess.py`
- Create: `tests/test_stages.py`

- [ ] **Step 1: Write SoX stage tests**

Create `tests/test_stages.py`:

```python
import shutil
import subprocess
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import soundfile as sf

from ham_radio_stt.config import PipelineConfig
from ham_radio_stt.stages.sox_preprocess import SoxPreprocess
from ham_radio_stt import AudioProcessingError


@pytest.fixture
def sox_stage():
    """Create SoxPreprocess — only usable when SoX is on PATH."""
    return SoxPreprocess(PipelineConfig())


class TestSoxPreprocess:
    @pytest.mark.requires_sox
    def test_output_is_16khz_mono(self, sox_stage, noisy_tone_wav):
        audio, sr = sf.read(str(noisy_tone_wav))
        processed, out_sr = sox_stage.process(audio, 44100)
        assert out_sr == 16000
        assert processed.ndim == 1  # mono

    @pytest.mark.requires_sox
    def test_duration_preserved(self, sox_stage, noisy_tone_wav):
        audio, sr = sf.read(str(noisy_tone_wav))
        input_duration = len(audio) / sr
        processed, out_sr = sox_stage.process(audio, 44100)
        output_duration = len(processed) / out_sr
        assert abs(input_duration - output_duration) / input_duration < 0.05

    @pytest.mark.requires_sox
    def test_output_dtype_float32(self, sox_stage, noisy_tone_wav):
        audio, sr = sf.read(str(noisy_tone_wav))
        processed, out_sr = sox_stage.process(audio, 44100)
        assert processed.dtype == np.float32

    def test_sox_not_found_raises(self):
        with patch("shutil.which", return_value=None):
            with pytest.raises(AudioProcessingError, match="[Ss]o[Xx]"):
                SoxPreprocess(PipelineConfig())

    @pytest.mark.requires_sox
    def test_name_attribute(self, sox_stage):
        assert sox_stage.name == "sox_preprocess"

    @pytest.mark.requires_sox
    def test_sox_failure_raises_audio_error(self, sox_stage):
        with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "sox", stderr=b"error")):
            audio = np.zeros(16000, dtype=np.float32)
            with pytest.raises(AudioProcessingError):
                sox_stage.process(audio, 16000)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_stages.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Create `ham_radio_stt/stages/__init__.py`**

```python
"""Pipeline stage protocol and stage exports."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class PipelineStage(Protocol):
    name: str

    def process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        """Process audio, return (processed_audio, sample_rate)."""
        ...
```

- [ ] **Step 4: Create `ham_radio_stt/stages/sox_preprocess.py`**

```python
"""Stage 1: SoX preprocessing — bandpass filter, compand, normalize, resample."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from ham_radio_stt import AudioProcessingError
from ham_radio_stt.config import PipelineConfig


class SoxPreprocess:
    name: str = "sox_preprocess"

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._sox_path = shutil.which("sox")
        if self._sox_path is None:
            import platform
            system = platform.system()
            install_hint = {
                "Darwin": "brew install sox",
                "Linux": "sudo apt install sox",
                "Windows": "choco install sox",
            }.get(system, "Install SoX and ensure it is on your PATH")
            raise AudioProcessingError(
                f"SoX not found on PATH. Install it: {install_hint}"
            )

    def process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        cfg = self._config
        input_tmp = None
        output_tmp = None
        try:
            input_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            output_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            input_tmp.close()
            output_tmp.close()

            sf.write(input_tmp.name, audio, sample_rate)

            cmd = [
                self._sox_path,
                input_tmp.name,
                "--rate", str(cfg.target_sample_rate),
                "--channels", str(cfg.target_channels),
                "--encoding", "signed-integer",
                "--bits", "16",
                output_tmp.name,
                "highpass", str(cfg.sox_highpass_hz),
                "lowpass", str(cfg.sox_lowpass_hz),
                "compand", "0.01,0.2",
                "-60,-60,-30,-10,0,-3",
                "-3", "-60", "0.1",
                "norm", str(cfg.sox_norm_level_db),
            ]

            try:
                subprocess.run(cmd, capture_output=True, check=True)
            except subprocess.CalledProcessError as e:
                stderr = e.stderr.decode(errors="replace") if e.stderr else "unknown error"
                raise AudioProcessingError(f"SoX failed: {stderr}") from e

            processed, out_sr = sf.read(output_tmp.name, dtype="float32")
            return processed, out_sr

        finally:
            for tmp in (input_tmp, output_tmp):
                if tmp is not None:
                    Path(tmp.name).unlink(missing_ok=True)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_stages.py -v`
Expected: PASS (requires_sox tests skip if SoX not installed, others pass)

- [ ] **Step 6: Commit**

```bash
git add ham_radio_stt/stages/__init__.py ham_radio_stt/stages/sox_preprocess.py tests/test_stages.py
git commit -m "feat: add PipelineStage protocol and SoX preprocessing stage"
```

---

### Task 5: Denoiser registry, NoOpDenoiser, and DeepFilterNet stage

**Files:**
- Create: `ham_radio_stt/stages/denoise.py`
- Create: `ham_radio_stt/stages/deepfilter.py`
- Modify: `tests/test_stages.py`

- [ ] **Step 1: Write denoiser tests**

Append to `tests/test_stages.py`:

```python
from ham_radio_stt.stages.denoise import (
    NoOpDenoiser,
    get_denoiser,
    register_denoiser,
    registered_denoisers,
)
from ham_radio_stt import ModelLoadError


class TestNoOpDenoiser:
    def test_passthrough(self):
        denoiser = NoOpDenoiser(PipelineConfig())
        audio = np.random.default_rng(42).standard_normal(16000).astype(np.float32)
        out, sr = denoiser.process(audio, 16000)
        np.testing.assert_array_equal(out, audio)
        assert sr == 16000

    def test_name(self):
        denoiser = NoOpDenoiser(PipelineConfig())
        assert denoiser.name == "none"


class TestDenoiserRegistry:
    def test_none_registered_by_default(self):
        assert "none" in registered_denoisers()

    def test_get_none_denoiser(self):
        denoiser = get_denoiser("none", PipelineConfig())
        assert isinstance(denoiser, NoOpDenoiser)

    def test_unknown_denoiser_raises(self):
        with pytest.raises(ModelLoadError, match="Unknown denoiser"):
            get_denoiser("nonexistent", PipelineConfig())

    def test_register_custom_denoiser(self, monkeypatch):
        class FakeDenoiser:
            name = "fake"
            def __init__(self, config): pass
            def process(self, audio, sr): return audio, sr

        # Use monkeypatch to avoid polluting global registry
        from ham_radio_stt.stages import denoise
        original = denoise._REGISTRY.copy()
        monkeypatch.setattr(denoise, "_REGISTRY", {**original})

        register_denoiser("fake", FakeDenoiser)
        denoiser = get_denoiser("fake", PipelineConfig())
        assert denoiser.name == "fake"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_stages.py::TestNoOpDenoiser -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Create `ham_radio_stt/stages/denoise.py`**

```python
"""Denoiser registry and NoOp denoiser."""

from __future__ import annotations

from typing import Any

import numpy as np

from ham_radio_stt import ModelLoadError
from ham_radio_stt.config import PipelineConfig


class NoOpDenoiser:
    name: str = "none"

    def __init__(self, config: PipelineConfig) -> None:
        pass

    def process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        return audio, sample_rate


_REGISTRY: dict[str, type] = {"none": NoOpDenoiser}


def register_denoiser(name: str, cls: type) -> None:
    _REGISTRY[name] = cls


def get_denoiser(name: str, config: PipelineConfig) -> Any:
    if name not in _REGISTRY:
        raise ModelLoadError(
            f"Unknown denoiser: {name}. Available: {list(_REGISTRY.keys())}. "
            f"For DeepFilterNet: pip install ham-radio-stt[deepfilter]"
        )
    return _REGISTRY[name](config)


def registered_denoisers() -> list[str]:
    return list(_REGISTRY.keys())
```

- [ ] **Step 4: Create `ham_radio_stt/stages/deepfilter.py`**

```python
"""DeepFilterNet 3 denoiser stage (optional — requires deepfilternet + PyTorch)."""

from __future__ import annotations

import logging

import numpy as np

from ham_radio_stt import ModelLoadError
from ham_radio_stt.config import PipelineConfig

logger = logging.getLogger(__name__)

try:
    import math

    import torch
    from df.enhance import enhance, init_df
    from scipy.signal import resample_poly

    class DeepFilterDenoiser:
        name: str = "deepfilter"

        def __init__(self, config: PipelineConfig) -> None:
            self._config = config
            try:
                self._model, self._df_state, _ = init_df()
            except Exception as e:
                raise ModelLoadError(f"Failed to load DeepFilterNet: {e}") from e
            logger.info("DeepFilterNet 3 model loaded")

        def process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
            # DFN3 requires 48kHz input
            if sample_rate != 48000:
                # Resample to 48kHz using integer ratios
                g = math.gcd(sample_rate, 48000)
                up = 48000 // g
                down = sample_rate // g
                audio = resample_poly(audio, up, down).astype(np.float32)

            # DFN3 expects shape (channels, samples)
            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
            enhanced = enhance(
                self._model,
                self._df_state,
                audio_tensor,
                atten_lim_db=self._config.dfn_attenuation_limit,
            )
            enhanced_np = enhanced.squeeze(0).numpy()

            # Apply post-filter if configured (additional noise suppression)
            # Note: post_filter handling depends on DFN3 version — enhance() may
            # accept a post_filter param. Check df.enhance API at implementation time.

            # Resample back to 16kHz
            out = resample_poly(enhanced_np, 1, 3).astype(np.float32)  # 48k -> 16k
            return out, 16000

    # Auto-register
    from ham_radio_stt.stages.denoise import register_denoiser
    register_denoiser("deepfilter", DeepFilterDenoiser)
    logger.debug("DeepFilterNet denoiser registered")

except ImportError:
    # deepfilternet not installed — that's fine, it's optional
    pass
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_stages.py -v`
Expected: All denoiser tests PASS

- [ ] **Step 6: Commit**

```bash
git add ham_radio_stt/stages/denoise.py ham_radio_stt/stages/deepfilter.py tests/test_stages.py
git commit -m "feat: add denoiser registry with NoOp and optional DeepFilterNet stage"
```

---

## Chunk 3: Transcription, Pipeline, and CLI

### Task 6: Whisper transcription wrapper

**Files:**
- Create: `ham_radio_stt/transcribe.py`
- Create: `tests/test_transcribe.py`

- [ ] **Step 1: Write transcription tests**

Create `tests/test_transcribe.py`:

```python
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ham_radio_stt.config import PipelineConfig
from ham_radio_stt.transcribe import WhisperTranscriber
from ham_radio_stt.result import TranscriptionResult
from ham_radio_stt import ModelLoadError


class TestWhisperTranscriber:
    def test_transcribe_returns_result(self):
        """Test with mocked WhisperModel."""
        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 3.0
        mock_segment.text = " CQ CQ this is W1AW"
        mock_segment.avg_logprob = -0.2
        mock_segment.no_speech_prob = 0.03

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.duration = 3.0

        with patch("ham_radio_stt.transcribe.WhisperModel") as MockModel:
            instance = MockModel.return_value
            instance.transcribe.return_value = (iter([mock_segment]), mock_info)

            transcriber = WhisperTranscriber(PipelineConfig())
            audio = np.zeros(16000 * 3, dtype=np.float32)
            result = transcriber.transcribe(audio, 16000, audio_source="file:test.wav")

        assert isinstance(result, TranscriptionResult)
        assert "CQ CQ this is W1AW" in result.text
        assert result.language == "en"
        assert result.duration_s == 3.0
        assert result.avg_log_prob == pytest.approx(-0.2)
        assert result.no_speech_prob == pytest.approx(0.03)
        assert result.processing_time_s >= 0
        assert result.audio_source == "file:test.wav"

    def test_empty_segments_returns_empty_text(self):
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.duration = 2.0

        with patch("ham_radio_stt.transcribe.WhisperModel") as MockModel:
            instance = MockModel.return_value
            instance.transcribe.return_value = (iter([]), mock_info)

            transcriber = WhisperTranscriber(PipelineConfig())
            audio = np.zeros(16000 * 2, dtype=np.float32)
            result = transcriber.transcribe(audio, 16000, audio_source="file:test.wav")

        assert result.text == ""
        assert result.segments == []

    def test_model_load_failure_raises(self):
        with patch("ham_radio_stt.transcribe.WhisperModel", side_effect=Exception("download failed")):
            with pytest.raises(ModelLoadError, match="download failed"):
                WhisperTranscriber(PipelineConfig())

    @pytest.mark.slow
    def test_real_model_loads(self):
        """Verify model can be loaded. Requires model download on first run."""
        config = PipelineConfig(whisper_model="tiny")
        transcriber = WhisperTranscriber(config)
        assert transcriber is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_transcribe.py -v -m "not slow"`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `ham_radio_stt/transcribe.py`**

```python
"""Whisper transcription wrapper using faster-whisper."""

from __future__ import annotations

import logging
import time

import numpy as np
from faster_whisper import WhisperModel

from ham_radio_stt import ModelLoadError
from ham_radio_stt.config import PipelineConfig
from ham_radio_stt.result import TranscriptionResult

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        try:
            logger.info("Loading Whisper model: %s", config.whisper_model)
            start = time.monotonic()
            self._model = WhisperModel(
                config.whisper_model,
                device=config.whisper_device,
                compute_type=config.whisper_compute_type,
            )
            elapsed = time.monotonic() - start
            logger.info("Whisper model loaded in %.1fs", elapsed)
        except Exception as e:
            raise ModelLoadError(f"Failed to load Whisper model: {e}") from e

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        audio_source: str,
    ) -> TranscriptionResult:
        cfg = self._config
        start = time.monotonic()

        segments_gen, info = self._model.transcribe(
            audio,
            language=cfg.whisper_language,
            beam_size=cfg.whisper_beam_size,
            best_of=cfg.whisper_best_of,
            temperature=cfg.whisper_temperature,
            initial_prompt=cfg.whisper_initial_prompt,
            vad_filter=cfg.vad_filter,
            vad_parameters=dict(
                min_silence_duration_ms=cfg.vad_min_silence_duration_ms,
                speech_pad_ms=cfg.vad_speech_pad_ms,
            ),
            condition_on_previous_text=False,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
        )

        # Materialize once — generator is lazy and stateful
        segments = list(segments_gen)

        processing_time = time.monotonic() - start
        duration = info.duration

        text = " ".join(s.text.strip() for s in segments).strip()
        segment_dicts = [
            {"start": s.start, "end": s.end, "text": s.text.strip()}
            for s in segments
        ]

        avg_log_prob = (
            sum(s.avg_logprob for s in segments) / len(segments)
            if segments
            else 0.0
        )
        no_speech_prob = (
            sum(s.no_speech_prob for s in segments) / len(segments)
            if segments
            else 1.0
        )

        return TranscriptionResult(
            text=text,
            segments=segment_dicts,
            language=info.language,
            duration_s=duration,
            processing_time_s=processing_time,
            real_time_factor=processing_time / duration if duration > 0 else 0.0,
            audio_source=audio_source,
            avg_log_prob=avg_log_prob,
            no_speech_prob=no_speech_prob,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_transcribe.py -v -m "not slow"`
Expected: 3 tests PASS, 1 skipped (slow)

- [ ] **Step 5: Commit**

```bash
git add ham_radio_stt/transcribe.py tests/test_transcribe.py
git commit -m "feat: add Whisper transcription wrapper with TranscriptionResult output"
```

---

### Task 7: Pipeline class — compose stages and run

**Files:**
- Create: `ham_radio_stt/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write pipeline tests**

Create `tests/test_pipeline.py`:

```python
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ham_radio_stt.config import PipelineConfig
from ham_radio_stt.pipeline import Pipeline
from ham_radio_stt.result import TranscriptionResult


@pytest.fixture
def mock_pipeline():
    """Pipeline with mocked transcriber and real NoOp denoiser."""
    with patch("ham_radio_stt.pipeline.WhisperTranscriber") as MockTranscriber:
        mock_result = TranscriptionResult(
            text="CQ CQ",
            segments=[{"start": 0.0, "end": 1.0, "text": "CQ CQ"}],
            language="en",
            duration_s=1.0,
            processing_time_s=0.5,
            real_time_factor=0.5,
            audio_source="file:test.wav",
            avg_log_prob=-0.2,
            no_speech_prob=0.03,
        )
        MockTranscriber.return_value.transcribe.return_value = mock_result

        with patch("ham_radio_stt.pipeline.SoxPreprocess") as MockSox:
            MockSox.return_value.process.return_value = (
                np.zeros(16000, dtype=np.float32),
                16000,
            )
            pipeline = Pipeline(PipelineConfig())
            yield pipeline


class TestPipeline:
    def test_transcribe_audio(self, mock_pipeline):
        audio = np.zeros(16000, dtype=np.float32)
        result = mock_pipeline.transcribe_audio(audio, 16000)
        assert isinstance(result, TranscriptionResult)
        assert result.text == "CQ CQ"

    def test_transcribe_file(self, mock_pipeline):
        with patch("soundfile.read", return_value=(np.zeros(16000, dtype=np.float32), 16000)):
            result = mock_pipeline.transcribe_file("test.wav")
        assert isinstance(result, TranscriptionResult)

    def test_transcribe_file_not_found(self, mock_pipeline):
        from ham_radio_stt import AudioProcessingError
        with pytest.raises(AudioProcessingError, match="not found"):
            mock_pipeline.transcribe_file("/nonexistent/file.wav")

    def test_stages_run_in_order(self):
        call_order = []

        class FakeStage1:
            name = "stage1"
            def __init__(self, config): pass
            def process(self, audio, sr):
                call_order.append("stage1")
                return audio, sr

        class FakeStage2:
            name = "stage2"
            def __init__(self, config): pass
            def process(self, audio, sr):
                call_order.append("stage2")
                return audio, sr

        with patch("ham_radio_stt.pipeline.WhisperTranscriber") as MockTranscriber:
            mock_result = TranscriptionResult(
                text="test", segments=[], language="en",
                duration_s=1.0, processing_time_s=0.5,
                real_time_factor=0.5, audio_source="test",
                avg_log_prob=-0.2, no_speech_prob=0.03,
            )
            MockTranscriber.return_value.transcribe.return_value = mock_result

            with patch("ham_radio_stt.pipeline.SoxPreprocess", FakeStage1):
                with patch("ham_radio_stt.pipeline.get_denoiser", return_value=FakeStage2(None)):
                    pipeline = Pipeline(PipelineConfig())
                    pipeline.transcribe_audio(np.zeros(16000, dtype=np.float32), 16000)

        assert call_order == ["stage1", "stage2"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `ham_radio_stt/pipeline.py`**

```python
"""Pipeline — composes stages and runs them in sequence."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from ham_radio_stt import AudioProcessingError
from ham_radio_stt.config import PipelineConfig
from ham_radio_stt.result import TranscriptionResult
from ham_radio_stt.stages.sox_preprocess import SoxPreprocess
from ham_radio_stt.stages.denoise import get_denoiser
from ham_radio_stt.transcribe import WhisperTranscriber

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        logger.info("Initializing pipeline...")

        self._sox_stage = SoxPreprocess(config)
        self._denoiser = get_denoiser(config.denoiser, config)
        self._stages = [self._sox_stage, self._denoiser]
        self._transcriber = WhisperTranscriber(config)

        stage_names = [s.name for s in self._stages]
        logger.info("Pipeline stages: %s -> transcribe", " -> ".join(stage_names))

    def transcribe_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        audio_source: str = "audio",
    ) -> TranscriptionResult:
        # Run through all stages
        for stage in self._stages:
            audio, sample_rate = stage.process(audio, sample_rate)

        return self._transcriber.transcribe(audio, sample_rate, audio_source)

    def transcribe_file(self, path: str | Path) -> TranscriptionResult:
        path = Path(path)
        if not path.exists():
            raise AudioProcessingError(f"Audio file not found: {path}")

        audio, sr = sf.read(str(path), dtype="float32")
        return self.transcribe_audio(audio, sr, audio_source=f"file:{path}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add ham_radio_stt/pipeline.py tests/test_pipeline.py
git commit -m "feat: add Pipeline class composing stages with transcription"
```

---

### Task 8: CLI with JSON streaming output

**Files:**
- Create: `ham_radio_stt/cli.py`
- Create: `ham_radio_stt/__main__.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write CLI tests**

Create `tests/test_cli.py`:

```python
import json
import subprocess
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ham_radio_stt.cli import main, format_json_line, format_error_json


class TestFormatJsonLine:
    def test_transcription_has_type(self):
        from ham_radio_stt.result import TranscriptionResult
        result = TranscriptionResult(
            text="CQ", segments=[], language="en",
            duration_s=1.0, processing_time_s=0.5,
            real_time_factor=0.5, audio_source="file:test.wav",
            avg_log_prob=-0.2, no_speech_prob=0.03,
        )
        line = format_json_line(result)
        parsed = json.loads(line)
        assert parsed["type"] == "transcription"
        assert parsed["text"] == "CQ"
        assert "is_valid" in parsed

    def test_file_mode_fields(self):
        from ham_radio_stt.result import TranscriptionResult
        result = TranscriptionResult(
            text="CQ", segments=[], language="en",
            duration_s=1.0, processing_time_s=0.5,
            real_time_factor=0.5, audio_source="file:test.wav",
            avg_log_prob=-0.2, no_speech_prob=0.03,
            segment_index=0, offset_s=0.0, is_final=True,
        )
        line = format_json_line(result)
        parsed = json.loads(line)
        assert parsed["segment_index"] == 0
        assert parsed["is_final"] is True


class TestFormatErrorJson:
    def test_error_has_type(self):
        line = format_error_json("Device not found", "STREAM_ERROR")
        parsed = json.loads(line)
        assert parsed["type"] == "error"
        assert parsed["error"] == "Device not found"
        assert parsed["code"] == "STREAM_ERROR"


class TestCliEntryPoint:
    def test_version_flag(self):
        result = subprocess.run(
            [sys.executable, "-m", "ham_radio_stt", "--version"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "0.1.0" in result.stdout

    def test_no_args_shows_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "ham_radio_stt"],
            capture_output=True, text=True,
        )
        # argparse shows help or error on no subcommand
        assert result.returncode != 0 or "usage" in result.stderr.lower() or "usage" in result.stdout.lower()

    def test_file_missing_path_exits_2(self):
        result = subprocess.run(
            [sys.executable, "-m", "ham_radio_stt", "file"],
            capture_output=True, text=True,
        )
        assert result.returncode == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `ham_radio_stt/cli.py`**

```python
"""CLI entry point for ham-radio-stt."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
from pathlib import Path

import ham_radio_stt
from ham_radio_stt.config import PipelineConfig, load_config
from ham_radio_stt.result import TranscriptionResult

logger = logging.getLogger("ham_radio_stt")


def format_json_line(result: TranscriptionResult) -> str:
    return json.dumps(result.to_json_dict(), ensure_ascii=False)


def format_error_json(error: str, code: str) -> str:
    return json.dumps({"type": "error", "error": error, "code": code})


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.WARNING),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _build_config(args: argparse.Namespace) -> PipelineConfig:
    cli_overrides: dict = {}
    if hasattr(args, "model") and args.model is not None:
        cli_overrides["whisper_model"] = args.model
    if hasattr(args, "denoiser") and args.denoiser is not None:
        cli_overrides["denoiser"] = args.denoiser
    if hasattr(args, "device") and args.device is not None:
        cli_overrides["stream_input_device"] = args.device

    config_file = getattr(args, "config", None)
    config_path = Path(config_file) if config_file else None

    return load_config(config_file=config_path, cli_overrides=cli_overrides or None)


def _cmd_file(args: argparse.Namespace) -> int:
    config = _build_config(args)
    use_json = getattr(args, "json", False)

    try:
        from ham_radio_stt.pipeline import Pipeline
        pipeline = Pipeline(config)
    except ham_radio_stt.HamSTTError as e:
        if use_json:
            print(format_error_json(str(e), "MODEL_LOAD_ERROR"), flush=True)
        else:
            print(f"Error: {e}", file=sys.stderr)
        return _exit_code_for(e)

    path = Path(args.path)
    if not path.exists():
        msg = f"File not found: {path}"
        if use_json:
            print(format_error_json(msg, "AUDIO_PROCESSING_ERROR"), flush=True)
        else:
            print(f"Error: {msg}", file=sys.stderr)
        return 1

    try:
        # For now, single-pass transcription. Progressive chunking in a later task.
        result = pipeline.transcribe_file(str(path))
        result.segment_index = 0
        result.offset_s = 0.0
        result.is_final = True

        if use_json:
            print(format_json_line(result), flush=True)
        else:
            print(result.text)

    except ham_radio_stt.HamSTTError as e:
        if use_json:
            print(format_error_json(str(e), type(e).__name__.upper()), flush=True)
        else:
            print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


def _cmd_stream(args: argparse.Namespace) -> int:
    config = _build_config(args)
    use_json = getattr(args, "json", False)

    try:
        from ham_radio_stt.pipeline import Pipeline
        from ham_radio_stt.streaming import StreamingSession
        pipeline = Pipeline(config)
    except ham_radio_stt.HamSTTError as e:
        if use_json:
            print(format_error_json(str(e), "MODEL_LOAD_ERROR"), flush=True)
        else:
            print(f"Error: {e}", file=sys.stderr)
        return _exit_code_for(e)

    def on_result(result: TranscriptionResult) -> None:
        if use_json:
            print(format_json_line(result), flush=True)
        else:
            if result.is_likely_valid():
                print(result.text, flush=True)

    try:
        import time
        with StreamingSession(pipeline, config, on_result) as session:
            signal.signal(signal.SIGINT, lambda *_: session.stop())
            while session.is_running:
                time.sleep(0.1)
    except ham_radio_stt.HamSTTError as e:
        if use_json:
            print(format_error_json(str(e), type(e).__name__.upper()), flush=True)
        else:
            print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


def _cmd_devices(args: argparse.Namespace) -> int:
    use_json = getattr(args, "json", False)
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        if use_json:
            device_list = []
            for i, d in enumerate(devices):
                device_list.append({
                    "index": i,
                    "name": d["name"],
                    "max_input_channels": d["max_input_channels"],
                    "max_output_channels": d["max_output_channels"],
                    "default_samplerate": d["default_samplerate"],
                })
            print(json.dumps(device_list, indent=2))
        else:
            for i, d in enumerate(devices):
                if d["max_input_channels"] > 0:
                    print(f"  {i}: {d['name']} (inputs: {d['max_input_channels']})")
    except ImportError:
        msg = "sounddevice not installed. Run: pip install ham-radio-stt[stream]"
        if use_json:
            print(format_error_json(msg, "MISSING_DEPENDENCY"), flush=True)
        else:
            print(f"Error: {msg}", file=sys.stderr)
        return 3

    return 0


def _exit_code_for(error: Exception) -> int:
    if isinstance(error, ham_radio_stt.ConfigError):
        return 2
    if isinstance(error, ham_radio_stt.ModelLoadError):
        return 3
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ham-radio-stt",
        description="Offline speech-to-text for ham radio audio",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {ham_radio_stt.__version__}")
    parser.add_argument("--log-level", default="WARNING", help="Logging level (default: WARNING)")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # file subcommand
    file_parser = subparsers.add_parser("file", help="Transcribe an audio file")
    file_parser.add_argument("path", help="Path to audio file")
    file_parser.add_argument("--json", action="store_true", help="Output as JSONL")
    file_parser.add_argument("--model", help="Whisper model name")
    file_parser.add_argument("--denoiser", help="Denoiser name")
    file_parser.add_argument("--config", help="Path to TOML config file")

    # stream subcommand
    stream_parser = subparsers.add_parser("stream", help="Stream from audio device")
    stream_parser.add_argument("--json", action="store_true", help="Output as JSONL")
    stream_parser.add_argument("--device", type=int, help="Audio device index")
    stream_parser.add_argument("--model", help="Whisper model name")
    stream_parser.add_argument("--denoiser", help="Denoiser name")
    stream_parser.add_argument("--config", help="Path to TOML config file")

    # devices subcommand
    devices_parser = subparsers.add_parser("devices", help="List audio devices")
    devices_parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args(argv)
    _setup_logging(args.log_level)

    # Handle SIGPIPE gracefully
    try:
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except (AttributeError, OSError):
        pass  # SIGPIPE not available on Windows

    handlers = {
        "file": _cmd_file,
        "stream": _cmd_stream,
        "devices": _cmd_devices,
    }

    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Create `ham_radio_stt/__main__.py`**

```python
"""Allow running as: python -m ham_radio_stt"""

import sys
from ham_radio_stt.cli import main

sys.exit(main())
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_cli.py -v`
Expected: All 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add ham_radio_stt/cli.py ham_radio_stt/__main__.py tests/test_cli.py
git commit -m "feat: add CLI with file/stream/devices commands and JSON output"
```

---

## Chunk 4: Streaming and Real Audio Tests

### Task 9: StreamingSession — capture thread and flush logic

**Files:**
- Create: `ham_radio_stt/streaming.py`
- Create: `tests/test_streaming.py`

- [ ] **Step 1: Write streaming tests**

Create `tests/test_streaming.py`:

```python
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ham_radio_stt.config import PipelineConfig
from ham_radio_stt.streaming import StreamingSession
from ham_radio_stt.result import TranscriptionResult
from ham_radio_stt import StreamError


@pytest.fixture
def mock_pipeline():
    pipeline = MagicMock()
    pipeline.transcribe_audio.return_value = TranscriptionResult(
        text="CQ CQ",
        segments=[{"start": 0.0, "end": 1.0, "text": "CQ CQ"}],
        language="en",
        duration_s=1.0,
        processing_time_s=0.5,
        real_time_factor=0.5,
        audio_source="stream:test",
        avg_log_prob=-0.2,
        no_speech_prob=0.03,
    )
    return pipeline


class TestStreamingSession:
    def test_starts_and_stops(self, mock_pipeline):
        config = PipelineConfig()
        results = []

        with patch("ham_radio_stt.streaming.sd") as mock_sd:
            # Mock InputStream context manager
            mock_stream = MagicMock()
            mock_sd.InputStream.return_value.__enter__ = MagicMock(return_value=mock_stream)
            mock_sd.InputStream.return_value.__exit__ = MagicMock(return_value=False)

            session = StreamingSession(mock_pipeline, config, results.append)
            with session:
                assert session.is_running
                time.sleep(0.2)
                session.stop()

            assert not session.is_running

    def test_pause_and_resume(self, mock_pipeline):
        config = PipelineConfig()

        with patch("ham_radio_stt.streaming.sd") as mock_sd:
            mock_stream = MagicMock()
            mock_sd.InputStream.return_value.__enter__ = MagicMock(return_value=mock_stream)
            mock_sd.InputStream.return_value.__exit__ = MagicMock(return_value=False)

            session = StreamingSession(mock_pipeline, config, lambda r: None)
            with session:
                session.pause()
                assert not session.is_running
                session.resume()
                assert session.is_running
                session.stop()

    def test_silence_calibration_logged(self, mock_pipeline):
        config = PipelineConfig()

        with patch("ham_radio_stt.streaming.sd") as mock_sd:
            mock_stream = MagicMock()
            mock_sd.InputStream.return_value.__enter__ = MagicMock(return_value=mock_stream)
            mock_sd.InputStream.return_value.__exit__ = MagicMock(return_value=False)

            # Simulate calibration audio (low-level noise)
            with patch("ham_radio_stt.streaming.StreamingSession._calibrate") as mock_cal:
                mock_cal.return_value = 0.01
                session = StreamingSession(mock_pipeline, config, lambda r: None)
                with session:
                    assert session._silence_threshold > 0
                    session.stop()

    def test_missing_sounddevice_raises(self, mock_pipeline):
        config = PipelineConfig()
        with patch("ham_radio_stt.streaming.sd", None):
            with pytest.raises(StreamError):
                StreamingSession(mock_pipeline, config, lambda r: None)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_streaming.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `ham_radio_stt/streaming.py`**

```python
"""Streaming session — capture audio from device, VAD-gated flush, transcribe."""

from __future__ import annotations

import logging
import queue
import threading
import time
from datetime import datetime, timezone

import numpy as np

from ham_radio_stt import StreamError
from ham_radio_stt.config import PipelineConfig
from ham_radio_stt.result import TranscriptionResult

logger = logging.getLogger(__name__)

try:
    import sounddevice as sd
except ImportError:
    sd = None


class StreamingSession:
    def __init__(
        self,
        pipeline,
        config: PipelineConfig,
        callback,
    ) -> None:
        if sd is None:
            raise StreamError(
                "sounddevice not installed. Run: pip install ham-radio-stt[stream]"
            )

        self._pipeline = pipeline
        self._config = config
        self._callback = callback
        self._running = False
        self._paused = False
        self._chunk_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()

        chunk_samples = int(config.stream_sample_rate * config.stream_chunk_duration_s)
        self._chunk_samples = chunk_samples
        self._silence_threshold = 0.01  # will be calibrated

        self._capture_thread: threading.Thread | None = None
        self._process_thread: threading.Thread | None = None

    def _calibrate(self) -> float:
        """Capture 1s of ambient audio and compute silence threshold."""
        logger.info("Calibrating silence threshold (1 second)...")
        samples = []
        duration = 0.0
        chunk_dur = self._config.stream_chunk_duration_s

        try:
            with sd.InputStream(
                samplerate=self._config.stream_sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self._chunk_samples,
                device=self._config.stream_input_device,
            ) as stream:
                while duration < 1.0:
                    data, _ = stream.read(self._chunk_samples)
                    samples.append(data.flatten())
                    duration += chunk_dur
        except Exception as e:
            raise StreamError(f"Failed to calibrate: {e}") from e

        ambient = np.concatenate(samples)
        rms = float(np.sqrt(np.mean(ambient**2)))
        threshold = 3.0 * rms
        logger.info("Silence threshold calibrated: %.6f (ambient RMS: %.6f)", threshold, rms)
        return threshold

    def __enter__(self) -> StreamingSession:
        self._silence_threshold = self._calibrate()
        self._running = True
        self._stop_event.clear()

        self._process_thread = threading.Thread(
            target=self._processing_loop,
            name="stt-processing",
            daemon=True,
        )
        self._process_thread.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()

    def stop(self) -> None:
        self._stop_event.set()
        self._running = False
        if self._process_thread and self._process_thread.is_alive():
            self._process_thread.join(timeout=10)

    def pause(self) -> None:
        self._paused = True
        self._running = False

    def resume(self) -> None:
        self._paused = False
        self._running = True

    @property
    def is_running(self) -> bool:
        return self._running

    def _processing_loop(self) -> None:
        cfg = self._config
        rolling_buffer: list[np.ndarray] = []
        buffer_duration = 0.0
        silence_duration = 0.0

        try:
            with sd.InputStream(
                samplerate=cfg.stream_sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self._chunk_samples,
                device=cfg.stream_input_device,
            ) as stream:
                while not self._stop_event.is_set():
                    if self._paused:
                        time.sleep(0.05)
                        continue

                    data, overflowed = stream.read(self._chunk_samples)
                    if overflowed:
                        logger.warning("Audio input overflow detected")

                    chunk = data.flatten()
                    rolling_buffer.append(chunk)
                    chunk_dur = len(chunk) / cfg.stream_sample_rate
                    buffer_duration += chunk_dur

                    # Silence detection
                    rms = float(np.sqrt(np.mean(chunk**2)))
                    if rms < self._silence_threshold:
                        silence_duration += chunk_dur
                    else:
                        silence_duration = 0.0

                    # Flush conditions
                    should_flush = False
                    if silence_duration >= cfg.stream_silence_timeout_s and buffer_duration > silence_duration:
                        should_flush = True
                    elif buffer_duration >= cfg.stream_buffer_duration_s:
                        should_flush = True

                    if should_flush:
                        audio = np.concatenate(rolling_buffer)
                        rolling_buffer.clear()
                        buffer_duration = 0.0
                        silence_duration = 0.0
                        self._flush(audio, cfg.stream_sample_rate)

                # Final flush on stop
                if rolling_buffer:
                    audio = np.concatenate(rolling_buffer)
                    if len(audio) > 0:
                        self._flush(audio, cfg.stream_sample_rate)

        except Exception as e:
            logger.error("Streaming error: %s", e)
            self._running = False

    def _flush(self, audio: np.ndarray, sample_rate: int) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        try:
            result = self._pipeline.transcribe_audio(
                audio, sample_rate,
                audio_source=f"stream:{timestamp}",
            )
            result.timestamp = timestamp
            self._callback(result)
        except Exception as e:
            logger.error("Transcription error during flush: %s", e)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_streaming.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add ham_radio_stt/streaming.py tests/test_streaming.py
git commit -m "feat: add StreamingSession with silence-calibrated VAD flush"
```

---

### Task 10: Real audio test support and final wiring

**Files:**
- Modify: `tests/conftest.py` (already has `--audio-file` support)
- Create: `tests/test_real_audio.py`
- Modify: `ham_radio_stt/__init__.py` (add public API exports)

- [ ] **Step 1: Write real audio tests**

Create `tests/test_real_audio.py`:

```python
import pytest
from pathlib import Path

from ham_radio_stt.result import TranscriptionResult


@pytest.mark.real_audio
@pytest.mark.slow
@pytest.mark.requires_sox
class TestRealAudio:
    def test_transcription_produces_result(self, real_audio_files):
        from ham_radio_stt.pipeline import Pipeline
        from ham_radio_stt.config import PipelineConfig

        config = PipelineConfig(whisper_model="tiny")
        pipeline = Pipeline(config)

        for audio_path in real_audio_files:
            result = pipeline.transcribe_file(str(audio_path))
            assert isinstance(result, TranscriptionResult)
            assert result.duration_s > 0
            assert result.processing_time_s > 0

            print(f"\n--- {audio_path.name} ---")
            print(f"Text: {result.text}")
            print(f"Duration: {result.duration_s:.1f}s")
            print(f"RTF: {result.real_time_factor:.2f}x")
            print(f"Confidence: {result.avg_log_prob:.2f}")
            print(f"No-speech prob: {result.no_speech_prob:.2f}")
            print(f"Valid: {result.is_likely_valid()}")
```

- [ ] **Step 2: Update `ham_radio_stt/__init__.py` with public exports**

Append to `ham_radio_stt/__init__.py`:

```python
from ham_radio_stt.config import PipelineConfig
from ham_radio_stt.result import TranscriptionResult

__all__ = [
    "__version__",
    "PipelineConfig",
    "TranscriptionResult",
    "HamSTTError",
    "AudioProcessingError",
    "ModelLoadError",
    "StreamError",
    "ConfigError",
]
```

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/ -v -m "not slow and not real_audio"`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_real_audio.py ham_radio_stt/__init__.py
git commit -m "feat: add real audio test support and public API exports"
```

---

### Task 11: Progressive file chunking

**Files:**
- Modify: `ham_radio_stt/pipeline.py`
- Modify: `ham_radio_stt/cli.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write progressive chunking test**

Add to `tests/test_pipeline.py`:

```python
class TestProgressiveTranscription:
    def test_transcribe_file_progressive_yields_results(self):
        """Verify that transcribe_file_progressive yields multiple results for multi-segment audio."""
        from ham_radio_stt.pipeline import Pipeline
        from ham_radio_stt.config import PipelineConfig

        with patch("ham_radio_stt.pipeline.WhisperTranscriber") as MockTranscriber:
            mock_result = TranscriptionResult(
                text="CQ", segments=[{"start": 0.0, "end": 1.0, "text": "CQ"}],
                language="en", duration_s=1.0, processing_time_s=0.5,
                real_time_factor=0.5, audio_source="file:test.wav",
                avg_log_prob=-0.2, no_speech_prob=0.03,
            )
            MockTranscriber.return_value.transcribe.return_value = mock_result

            with patch("ham_radio_stt.pipeline.SoxPreprocess") as MockSox:
                MockSox.return_value.process.return_value = (
                    np.zeros(16000 * 5, dtype=np.float32), 16000,
                )
                with patch("ham_radio_stt.pipeline.Pipeline._vad_segment") as mock_vad:
                    # Simulate 2 speech segments
                    mock_vad.return_value = [
                        (0, 16000 * 2),    # 0-2s
                        (16000 * 3, 16000 * 5),  # 3-5s
                    ]
                    with patch("soundfile.read", return_value=(np.zeros(44100 * 5, dtype=np.float32), 44100)):
                        pipeline = Pipeline(PipelineConfig())
                        results = list(pipeline.transcribe_file_progressive("test.wav"))

                assert len(results) == 2
                assert results[0].segment_index == 0
                assert results[0].is_final is False
                assert results[1].segment_index == 1
                assert results[1].is_final is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline.py::TestProgressiveTranscription -v`
Expected: FAIL — `AttributeError: 'Pipeline' object has no attribute 'transcribe_file_progressive'`

- [ ] **Step 3: Add `transcribe_file_progressive` and `_vad_segment` to Pipeline**

Add to `ham_radio_stt/pipeline.py`:

```python
from typing import Generator

# Add to Pipeline class:

    def _vad_segment(self, audio: np.ndarray, sample_rate: int) -> list[tuple[int, int]]:
        """Use faster-whisper's VAD to find speech segments. Returns list of (start_sample, end_sample)."""
        max_segment_samples = 30 * sample_rate  # 30s max per spec

        try:
            from faster_whisper.vad import VadOptions, get_speech_timestamps
        except ImportError:
            # Fallback: split into max_segment_samples chunks
            return self._split_long_segments([(0, len(audio))], max_segment_samples)

        vad_options = VadOptions(
            min_silence_duration_ms=self._config.vad_min_silence_duration_ms,
            speech_pad_ms=self._config.vad_speech_pad_ms,
        )
        timestamps = get_speech_timestamps(audio, vad_options)

        if not timestamps:
            return [(0, len(audio))]

        segments = [(int(ts["start"]), int(ts["end"])) for ts in timestamps]

        # Enforce 30s max segment boundary
        return self._split_long_segments(segments, max_segment_samples)

    @staticmethod
    def _split_long_segments(
        segments: list[tuple[int, int]], max_samples: int
    ) -> list[tuple[int, int]]:
        """Split any segment exceeding max_samples into chunks."""
        result = []
        for start, end in segments:
            while end - start > max_samples:
                result.append((start, start + max_samples))
                start += max_samples
            result.append((start, end))
        return result

    def transcribe_file_progressive(
        self,
        path: str | Path,
    ) -> Generator[TranscriptionResult, None, None]:
        path = Path(path)
        if not path.exists():
            raise AudioProcessingError(f"Audio file not found: {path}")

        audio, sr = sf.read(str(path), dtype="float32")

        # Pass 1: SoX preprocess entire file
        processed, processed_sr = self._sox_stage.process(audio, sr)

        # Pass 2: VAD segmentation
        segments = self._vad_segment(processed, processed_sr)

        # Pass 3: Transcribe each segment
        for i, (start, end) in enumerate(segments):
            chunk = processed[start:end]
            is_final = i == len(segments) - 1

            # Run remaining stages (denoiser etc.) on chunk
            stage_audio = chunk
            stage_sr = processed_sr
            for stage in self._stages[1:]:
                stage_audio, stage_sr = stage.process(stage_audio, stage_sr)

            offset_s = start / processed_sr
            result = self._transcriber.transcribe(
                stage_audio, stage_sr,
                audio_source=f"file:{path}",
            )
            result.segment_index = i
            result.offset_s = offset_s
            result.is_final = is_final
            yield result
```

- [ ] **Step 4: Update CLI to use progressive mode**

In `ham_radio_stt/cli.py`, update `_cmd_file`:

Replace the try block in `_cmd_file` with:

```python
    try:
        for result in pipeline.transcribe_file_progressive(str(path)):
            if use_json:
                print(format_json_line(result), flush=True)
            else:
                # Human mode: always print text. JSON consumers use is_valid to filter.
                if result.text.strip():
                    print(result.text, flush=True)
    except ham_radio_stt.HamSTTError as e:
        if use_json:
            print(format_error_json(str(e), type(e).__name__.upper()), flush=True)
        else:
            print(f"Error: {e}", file=sys.stderr)
        return 1
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add ham_radio_stt/pipeline.py ham_radio_stt/cli.py tests/test_pipeline.py
git commit -m "feat: add progressive file transcription with VAD segmentation"
```

---

### Task 12: Final integration — install and smoke test

- [ ] **Step 1: Install package in dev mode**

Run: `pip install -e ".[dev]"`
Expected: Installs successfully

- [ ] **Step 2: Verify CLI entry point**

Run: `ham-radio-stt --version`
Expected: `ham-radio-stt 0.1.0`

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/ -v -m "not slow and not real_audio and not requires_sox"`
Expected: All tests PASS

- [ ] **Step 4: Run SoX tests (if SoX installed)**

Run: `python -m pytest tests/ -v -m "requires_sox"`
Expected: SoX tests PASS

- [ ] **Step 5: Commit any final adjustments (if any files were modified)**

```bash
git status
# Stage only specific modified files if any adjustments were needed
# git add <specific files>
# git commit -m "chore: final integration adjustments"
```
