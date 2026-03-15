# Ham Radio STT — Design Spec

**Date:** 2026-03-14
**Status:** Approved
**Platform:** macOS, Windows, Linux — Python 3.11+, fully offline, no cloud dependencies

---

## 1. Overview

A CLI-first speech-to-text tool for ham radio audio. Processes pre-recorded files and live audio streams through a pluggable pipeline, emitting progressive JSON-line output suitable for human use or machine consumption (e.g., a Java application spawning the CLI as a subprocess).

```
Audio Input (file or audio device)
        |
        v
+------------------+
|  Stage 1: SoX    |  Bandpass filter, compand, normalize, resample to 16kHz mono
+--------+---------+
         |
         v
+------------------+
|  Stage 2: Denoiser|  Pluggable — NoOp (default), DeepFilterNet 3, or custom
+--------+---------+
         |
         v
+------------------+
|  Transcriber     |  faster-whisper with built-in Silero VAD
+--------+---------+
         |
         v
   JSON line to stdout
```

### Key Decisions (diverging from pre_spec.md)

| Topic | Pre-spec | This design |
|---|---|---|
| Platform | macOS Intel only | Cross-platform (Mac/Win/Linux) |
| Denoiser | DFN3 hardcoded as Stage 2 | Pluggable registry, DFN3 optional |
| VAD | Separate Silero VAD stage | Handled by faster-whisper internally |
| Primary interface | Library-first | CLI-first, JSON streaming for Java interop |
| File output | Single result at end | Progressive JSONL, chunked by pre-pass VAD |
| Config | Code-only dataclass | Layered: defaults -> global TOML -> local TOML -> CLI flags |
| Audio input | Mic + SDR | Standard audio devices only (SDR out of scope) |

---

## 2. Pipeline Architecture

### 2.1 PipelineStage Protocol

All audio processing stages implement a common interface:

```python
class PipelineStage(Protocol):
    name: str

    def process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        """Process audio, return (processed_audio, sample_rate)."""
        ...
```

Stages are composed into an ordered list at init time. The pipeline runs them sequentially. The transcriber sits outside the stage chain — it consumes the final audio and produces text.

**SoX and the PipelineStage protocol:** SoX is a subprocess that operates on files, but the `PipelineStage.process()` interface takes numpy arrays. The SoX stage internally writes a temp WAV, runs the subprocess, reads back the result, and cleans up in a `try/finally` block. This temp-file round-trip adds ~10-50ms overhead per call, which is well within the <2s processing budget. For streaming mode, this happens once per flush (every few seconds), not per chunk.

### 2.2 Stage Composition

```python
stages = [
    SoxPreprocess(config),        # always first — normalizes to 16kHz mono
    get_denoiser(config.denoiser, config),  # "none" -> NoOpDenoiser, "deepfilter" -> DFN3
]
transcriber = WhisperTranscriber(config)  # always last, produces text not audio
```

### 2.3 Denoiser Registry

Dict-based plugin system:

```python
_REGISTRY: dict[str, type[PipelineStage]] = {"none": NoOpDenoiser}

def register_denoiser(name: str, cls: type[PipelineStage]):
    _REGISTRY[name] = cls

def get_denoiser(name: str, config: PipelineConfig) -> PipelineStage:
    if name not in _REGISTRY:
        raise ModelLoadError(f"Unknown denoiser: {name}. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name](config)
```

DeepFilterNet 3 auto-registers on import if PyTorch is available:

```python
# deepfilter.py
try:
    from df.enhance import init_df, enhance
    class DeepFilterDenoiser:
        # ... implementation ...
    register_denoiser("deepfilter", DeepFilterDenoiser)
except ImportError:
    pass
```

If a user requests a denoiser that isn't installed, they get:
`"Denoiser 'deepfilter' requires: pip install ham-radio-stt[deepfilter]"`

---

## 3. Project Structure

```
ham_radio_stt/
├── __init__.py              # exports: public API, exceptions
├── config.py                # PipelineConfig dataclass, TOML loading, config layering
├── pipeline.py              # Pipeline class — composes stages, runs them
├── stages/
│   ├── __init__.py          # PipelineStage protocol
│   ├── sox_preprocess.py    # Stage 1: SoX subprocess
│   ├── denoise.py           # Denoiser registry + NoOpDenoiser
│   └── deepfilter.py        # DFN3 denoiser (optional import)
├── transcribe.py            # Whisper transcription (produces text, not audio)
├── result.py                # TranscriptionResult dataclass
├── streaming.py             # StreamingSession — capture thread + flush logic
├── cli.py                   # argparse CLI, JSON streaming output
└── __main__.py              # python -m ham_radio_stt entry point

tests/
├── test_stages.py           # unit tests for each stage in isolation
├── test_pipeline.py         # integration tests
├── test_cli.py              # CLI output format tests
├── test_streaming.py        # streaming session tests (mocked audio)
└── conftest.py              # shared fixtures, --audio-file option

pyproject.toml
README.md
.gitignore
hamstt.toml.example          # example config file
```

### Cross-Platform Notes

- SoX: `brew install sox` (mac), `apt install sox` (linux), `choco install sox` (windows). Checked via `shutil.which("sox")` at init with platform-specific error messages.
- sounddevice/PortAudio: pip package handles cross-platform.
- No OS-specific APIs in the core pipeline.
- Temp files via `tempfile` module (works everywhere).

### Packaging

Standard `pyproject.toml` with optional extras and a console script entry point:

```toml
[project.scripts]
ham-radio-stt = "ham_radio_stt.cli:main"
```

- `pip install ham-radio-stt` — core (SoX + faster-whisper)
- `pip install ham-radio-stt[deepfilter]` — adds DeepFilterNet 3 + PyTorch
- `pip install ham-radio-stt[stream]` — adds sounddevice for live capture
- `pip install ham-radio-stt[all]` — everything

---

## 4. Configuration

### 4.1 PipelineConfig Dataclass

```python
@dataclass
class PipelineConfig:
    # Denoiser
    denoiser: str = "none"              # "none", "deepfilter", or any registered name

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

    # VAD (passed to faster-whisper)
    vad_filter: bool = True
    vad_min_silence_duration_ms: int = 500
    vad_speech_pad_ms: int = 200

    # Streaming
    stream_chunk_duration_s: float = 0.5
    stream_buffer_duration_s: float = 30.0
    stream_silence_timeout_s: float = 1.5
    stream_sample_rate: int = 44100
    stream_input_device: Optional[int] = None

    # DeepFilterNet (only used when denoiser="deepfilter")
    dfn_attenuation_limit: float = 100.0      # dB, max noise attenuation
    dfn_post_filter: bool = True              # slight over-attenuation of very noisy sections
```

### 4.2 Config Layering

Precedence (highest wins):

```
CLI flags  >  --config file  >  ./hamstt.toml  >  ~/.config/hamstt/config.toml  >  defaults
```

Config files are TOML (stdlib since Python 3.11). TOML sections map to dataclass field prefixes:

```toml
[whisper]
model = "small"          # -> whisper_model
language = "en"          # -> whisper_language
beam_size = 3            # -> whisper_beam_size

[denoiser]
name = "deepfilter"      # -> denoiser

[sox]
highpass_hz = 300        # -> sox_highpass_hz
lowpass_hz = 3000        # -> sox_lowpass_hz

[streaming]
silence_timeout_s = 3.0  # -> stream_silence_timeout_s
chunk_duration_s = 0.5   # -> stream_chunk_duration_s

[deepfilter]
attenuation_limit = 80.0 # -> dfn_attenuation_limit
post_filter = false      # -> dfn_post_filter
```

**Mapping convention:** TOML key `[section] key` maps to dataclass field `{section_prefix}_{key}`. The section-to-prefix mapping is:

| TOML section | Field prefix |
|---|---|
| `[whisper]` | `whisper_` |
| `[denoiser]` | (special: `name` -> `denoiser`) |
| `[sox]` | `sox_` |
| `[streaming]` | `stream_` |
| `[deepfilter]` | `dfn_` |
| `[vad]` | `vad_` |

---

## 5. CLI Design

### 5.1 Commands

```bash
# Transcribe a file (progressive output)
ham-radio-stt file audio.wav
ham-radio-stt file audio.wav --json

# Stream from default audio device
ham-radio-stt stream
ham-radio-stt stream --json
ham-radio-stt stream --device 2 --json

# List audio devices
ham-radio-stt devices
ham-radio-stt devices --json

# Override config
ham-radio-stt file audio.wav --model small --denoiser deepfilter
ham-radio-stt file audio.wav --config custom.toml

# Version
ham-radio-stt --version
```

### 5.2 Output Discipline

- **stdout:** Only transcription text (human mode) or JSON (--json mode). Nothing else.
- **stderr:** All logging, progress, diagnostics. Controlled by `--log-level` (default: WARNING).

### 5.3 JSON Streaming Format

Both file and stream modes emit newline-delimited JSON (JSONL). Each line is a complete JSON object, flushed immediately. All JSONL objects include a `"type"` field as a discriminator:

- `"type": "transcription"` — a transcription result
- `"type": "error"` — an error (has `"error"` and `"code"` fields, no `"text"` field)

**File mode:**

```jsonl
{"type":"transcription","text":"CQ CQ this is W1AW","segment_index":0,"offset_s":0.0,"duration_s":3.2,"processing_time_s":1.1,"real_time_factor":0.34,"is_valid":true,"avg_log_prob":-0.18,"no_speech_prob":0.02,"segments":[{"start":0.0,"end":3.2,"text":"CQ CQ this is W1AW"}]}
{"type":"transcription","text":"K9ABC roger, five nine","segment_index":1,"offset_s":4.5,"duration_s":2.1,"processing_time_s":0.9,"real_time_factor":0.43,"is_valid":true,"avg_log_prob":-0.22,"no_speech_prob":0.04,"segments":[{"start":0.0,"end":2.1,"text":"K9ABC roger, five nine"}],"is_final":true}
```

File-mode specific fields: `segment_index`, `offset_s`, `is_final` (true on last segment).

**Stream mode:**

```jsonl
{"type":"transcription","text":"CQ CQ this is W1AW","is_valid":true,"duration_s":3.2,"processing_time_s":1.1,"real_time_factor":0.34,"avg_log_prob":-0.18,"no_speech_prob":0.02,"timestamp":"2026-03-14T18:30:01Z","segments":[{"start":0.0,"end":3.2,"text":"CQ CQ this is W1AW"}]}
```

Stream-mode specific field: `timestamp` (ISO 8601 wall-clock time).

**Error output (--json mode):**

```jsonl
{"type":"error","error":"Device not found: index 5","code":"STREAM_ERROR"}
```

Consumers can dispatch on the `"type"` field. Transcription objects always have `"text"`, error objects always have `"error"`.

### 5.4 File Mode Chunking

For progressive output in file mode, the pipeline performs a **three-pass approach**:

1. **Pass 1 — SoX preprocess** the entire file (fast, sub-second for most files)
2. **Pass 2 — VAD pre-segmentation** using Silero VAD (via `faster-whisper`'s `get_speech_timestamps`) to identify speech boundaries in the preprocessed audio. If no speech boundary within 30s, force a segment boundary.
3. **Pass 3 — Transcribe each segment independently** through the denoiser + Whisper, emitting a JSON line immediately after each segment completes.

This ensures progressive output without relying on Whisper's internal segmentation (which requires materializing all segments before any output). Each segment is an independent Whisper call, so results stream as they're ready.

### 5.5 Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Runtime error (pipeline failure, device lost) |
| 2 | Bad arguments / invalid config |
| 3 | Missing dependency (SoX not on PATH, deepfilter not installed) |

### 5.6 Signal Handling

- `SIGINT` / `SIGTERM`: Graceful shutdown — flush current buffer, emit final result, exit 0.
- `SIGPIPE`: Exit silently (consumer disconnected).

---

## 6. Streaming Mode

### 6.1 Thread Architecture

```
Audio Device -> capture_thread -> raw_chunk_queue -> processing_thread -> stdout
```

**Capture thread:** Reads audio from sounddevice in chunks of `stream_chunk_duration_s` (default 0.5s). Pushes raw float32 chunks onto a thread-safe queue. No processing.

**Processing thread:** A single dedicated thread (not a ThreadPoolExecutor). Runs the processing loop synchronously to ensure ordering:

1. Drain queue into rolling numpy buffer
2. Run energy-based silence detection on latest chunk
3. If silence for >= `stream_silence_timeout_s` (default 1.5s) -> flush
4. If buffer >= `stream_buffer_duration_s` (default 30s) -> force flush
5. On flush: copy buffer, reset, run full pipeline (SoX -> denoiser -> transcribe), emit JSON line

Since processing is single-threaded and sequential, no lock is needed for model access within streaming mode. A `threading.Lock` is only required if `transcribe_file()` or `transcribe_audio()` is called from a different thread while a stream is active.

### 6.2 Silence Calibration

On startup, capture 1s of ambient audio. Set silence threshold to `3 * rms_of_ambient`. Log calibrated threshold to stderr.

### 6.3 StreamingSession Lifecycle

```python
class StreamingSession:
    def __enter__(self) -> "StreamingSession": ...
    def __exit__(self, *args): ...
    def stop(self): ...       # graceful shutdown, flush remaining buffer
    def pause(self): ...      # stop capture without tearing down models
    def resume(self): ...
    @property
    def is_running(self) -> bool: ...
```

---

## 7. Stage Implementations

### 7.1 SoX Preprocessing (stages/sox_preprocess.py)

Runs SoX as a subprocess. Normalizes any input to 16kHz mono with bandpass filtering and dynamic compression.

SoX command (order matters):

```bash
sox {input_path} \
    --rate 16000 --channels 1 --encoding signed-integer --bits 16 \
    {output_path} \
    highpass {sox_highpass_hz} \
    lowpass {sox_lowpass_hz} \
    compand 0.01,0.2 -60,-60,-30,-10,0,-3 -3 -60 0.1 \
    norm {sox_norm_level_db}
```

- Uses `subprocess.run(..., capture_output=True, check=True)`
- Re-raises stderr as `AudioProcessingError`
- For numpy array input: write to temp WAV via soundfile, run SoX, read back, cleanup in `try/finally`

### 7.2 DeepFilterNet 3 (stages/deepfilter.py)

Optional denoiser. DFN3 operates at 48kHz internally. Reads its tuning parameters (`dfn_attenuation_limit`, `dfn_post_filter`) from `PipelineConfig`.

Processing flow: resample 16kHz -> 48kHz (scipy.signal.resample_poly, ratio 3:1) -> enhance -> resample 48kHz -> 16kHz.

Uses `scipy.signal.resample_poly` with integer ratios for quality. Not librosa.

### 7.3 Whisper Transcription (transcribe.py)

```python
segments, info = model.transcribe(
    audio,
    language=config.whisper_language,
    beam_size=config.whisper_beam_size,
    best_of=config.whisper_best_of,
    temperature=config.whisper_temperature,
    initial_prompt=config.whisper_initial_prompt,
    vad_filter=config.vad_filter,
    vad_parameters=dict(
        min_silence_duration_ms=config.vad_min_silence_duration_ms,
        speech_pad_ms=config.vad_speech_pad_ms,
    ),
    condition_on_previous_text=False,  # prevents hallucination loops
    log_prob_threshold=-1.0,
    no_speech_threshold=0.6,
)
segments = list(segments)  # materialize once — generator is lazy and stateful
```

`condition_on_previous_text=False` is critical for automation use cases.

---

## 8. Data Types

### 8.1 TranscriptionResult

```python
@dataclass
class TranscriptionResult:
    text: str
    segments: list[dict]
    language: str
    duration_s: float
    processing_time_s: float
    real_time_factor: float       # processing_time_s / duration_s
    audio_source: str             # "file:<path>" or "stream:<timestamp>"

    # Confidence
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
```

---

## 9. Error Handling

### 9.1 Exception Hierarchy

```python
class HamSTTError(Exception):            # base
class AudioProcessingError(HamSTTError):  # SoX failed, bad format
class ModelLoadError(HamSTTError):        # model download failed, missing dep
class StreamError(HamSTTError):           # device not found, disconnected
class ConfigError(HamSTTError):           # bad TOML, invalid parameters
```

All public methods only raise from this hierarchy, wrapping third-party exceptions.

### 9.2 Dependency Checking

At startup:
- SoX: `shutil.which("sox")` — if missing, platform-specific install instructions
- Optional deps: checked only when requested, clear pip install message

---

## 10. Testing Strategy

### 10.1 Synthetic Fixtures (conftest.py)

Generated programmatically with numpy + soundfile:
- Noisy tone: 440Hz sine + white noise
- Silence: zero-filled WAV
- Multi-segment: tone-silence-tone pattern

### 10.2 Test Layers

| Layer | What | Mocking |
|---|---|---|
| Unit: stages | Each stage in isolation | SoX subprocess mocked for CI without sox |
| Unit: config | TOML parsing, layered overrides, validation | Temp TOML files |
| Unit: result | `is_likely_valid()` edge cases | None |
| Integration: pipeline | Full pipeline on synthetic WAV | Whisper runs for real (marked slow) |
| CLI: output | JSON parses correctly, exit codes, `type` discriminator | Pipeline mocked |
| Streaming: lifecycle | Start, receive, flush, stop | sounddevice mocked |

### 10.3 Real Audio Testing

Custom pytest option `--audio-file` for testing with actual ham radio recordings:

```bash
pytest                                    # synthetic fixtures only
pytest --audio-file recording.wav         # include real audio test
pytest --audio-file a.wav --audio-file b.wav  # multiple recordings
```

Tests using real audio are marked `@pytest.mark.real_audio` and skip if no `--audio-file` provided.

### 10.4 Test Markers

- `@pytest.mark.slow` — loads Whisper model
- `@pytest.mark.requires_sox` — needs SoX on PATH
- `@pytest.mark.real_audio` — needs `--audio-file` flag

---

## 11. Performance Targets

| Metric | Target |
|---|---|
| Pipeline processing time (per chunk) | < 2s |
| Model load time (cached) | < 5s |
| File mode RTF | < 0.5x (10s audio in < 5s) |
| Stream flush latency | < 3.5s after speech ends (1.5s silence detection + <2s processing) |
| Memory footprint | < 3GB RSS with distil-large-v3 |

---

## 12. Known Gotchas

1. **DFN3 requires 48kHz** — resample before and after, not just after.
2. **faster-whisper generator** — `list(segments)` once is mandatory; the generator is lazy and stateful.
3. **`condition_on_previous_text=False`** — critical for automation; Whisper loops on noise otherwise.
4. **SoX `compand` argument order** — transfer function points are `in,out` pairs. Wrong order = silent output with no error.
5. **Temp file cleanup** — always `try/finally` or `contextlib.ExitStack` for temp WAVs.
6. **Thread safety** — WhisperModel and DeepFilterNet model are not thread-safe. Use `threading.Lock` only if calling `transcribe_file()`/`transcribe_audio()` concurrently with an active stream.
7. **sounddevice block size** — set to `int(sample_rate * chunk_duration_s)`. Mismatch causes dropouts.
8. **Float32 normalization** — sounddevice returns float32 [-1, 1]. Write via soundfile for correct conversion.
9. **SoX temp-file overhead** — the SoX stage writes/reads temp WAVs per call (~10-50ms). Acceptable within <2s budget but be aware for profiling.

---

## 13. Out of Scope

- Fine-tuning Whisper on ham radio audio
- SDR direct integration
- Speaker diarization / callsign extraction
- GUI

---

*Design approved 2026-03-14*
