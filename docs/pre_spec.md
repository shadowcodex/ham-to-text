# Ham to Text

This is a small application that can take pre-recorded or live audio from HAM Radio waves and convert to text fast with high accuracy.

Requirements is:

1. Completely offline use able
2. Can run on older intel based macbooks so CPU bound with no GPU/NPU systems (Maybe metal etc but tbd)
3. Accurate
4. Less than 2s response (Less than 100ms preferred)


Here is an output from claude.ai for research on how to build this...

# Ham Radio STT Pipeline — Implementation Spec
**Target:** Claude Code prototype  
**Platform:** Intel Mac (macOS), Python 3.11+, fully local, no cloud dependencies

---

## 1. Overview

A two-mode speech-to-text library for ham radio audio that produces clean, structured text
suitable for driving automation systems. The same processing pipeline is shared by both modes.

```
Audio Input (file or mic/SDR stream)
        │
        ▼
┌─────────────────┐
│  Stage 1: SoX   │  Bandpass filter, noise gate, normalize, resample to 16kHz mono
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│  Stage 2: DFN3       │  DeepFilterNet 3 — neural noise suppression
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  Stage 3: VAD        │  Silero VAD — drop non-speech frames before sending to Whisper
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  Stage 4: Whisper    │  faster-whisper distil-large-v3.5, int8 CPU
└────────┬─────────────┘
         │
         ▼
  TranscriptionResult (dataclass) → caller / automation system
```

---

## 2. Dependencies

```toml
# pyproject.toml / requirements.txt

# Core pipeline
deepfilternet          # DeepFilterNet 3 — pip install deepfilternet
faster-whisper         # CTranslate2-backed Whisper — pip install faster-whisper
soundfile              # WAV I/O
numpy

# Streaming mode only
sounddevice            # PortAudio bindings for mic input — pip install sounddevice

# System dependency (install separately)
# brew install sox
```

**Python:** 3.11+  
**SoX:** must be on PATH (`brew install sox`). The code must check and raise a clear error if missing.  
**Model download:** faster-whisper downloads models from HuggingFace on first use. The init
method should trigger this eagerly so the caller knows about it upfront, not mid-transcription.

---

## 3. Project Layout

```
ham_radio_stt/
├── __init__.py              # exports: HamRadioSTT, TranscriptionResult, PipelineConfig
├── config.py                # PipelineConfig dataclass (all tunable parameters)
├── pipeline.py              # HamRadioSTT — main public class
├── stages/
│   ├── __init__.py
│   ├── sox_preprocess.py    # Stage 1: SoX
│   ├── denoise.py           # Stage 2: DeepFilterNet 3
│   ├── vad.py               # Stage 3: VAD (Silero, via faster-whisper)
│   └── transcribe.py        # Stage 4: faster-whisper
├── streaming.py             # StreamingSession class
├── file_mode.py             # transcribe_file() helper
└── cli.py                   # thin CLI wrapper (click or argparse)

tests/
├── test_pipeline.py
├── test_sox.py
├── fixtures/
│   └── test_tone.wav        # short generated test fixture (440Hz + noise)
```

---

## 4. Public API

### 4.1 `PipelineConfig` (config.py)

```python
@dataclass
class PipelineConfig:

    # ── Whisper ───────────────────────────────────────────────────────────
    whisper_model: str = "distil-large-v3.5"
    # Other valid values: "large-v3-turbo", "medium", "small"
    # For fastest CPU option: "distil-large-v3"

    whisper_language: str = "en"
    whisper_beam_size: int = 5
    whisper_best_of: int = 5
    whisper_temperature: float = 0.0          # 0 = deterministic, reduces hallucination
    whisper_compute_type: str = "int8"        # int8 is fastest on Intel CPU
    whisper_device: str = "cpu"
    whisper_initial_prompt: str = (
        "Ham radio communication. "
        "CQ de W1AW, 73, QSL, QRM, QRN, QSB, QTH, over, copy, roger, "
        "phonetic alphabet: Alpha Bravo Charlie Delta Echo Foxtrot Golf "
        "Hotel India Juliet Kilo Lima Mike November Oscar Papa Quebec "
        "Romeo Sierra Tango Uniform Victor Whiskey X-ray Yankee Zulu."
    )
    # Extend with your own callsigns and common phrases for better accuracy.

    # ── DeepFilterNet ─────────────────────────────────────────────────────
    dfn_attenuation_limit: float = 100.0      # dB, max noise attenuation
    dfn_post_filter: bool = True              # slight over-attenuation of very noisy sections

    # ── SoX ───────────────────────────────────────────────────────────────
    sox_highpass_hz: int = 200                # ham voice starts ~300Hz, be conservative
    sox_lowpass_hz: int = 3400               # SSB/AM upper cutoff
    sox_noise_gate_db: float = -50.0         # silence below this threshold (dBFS)
    sox_noise_gate_attack_ms: int = 10
    sox_noise_gate_decay_ms: int = 200       # slow decay to avoid clipping tails
    sox_norm_level_db: float = -3.0          # normalize to this level
    target_sample_rate: int = 16000          # Whisper expects 16kHz
    target_channels: int = 1                 # mono

    # ── VAD ───────────────────────────────────────────────────────────────
    vad_filter: bool = True
    vad_min_silence_duration_ms: int = 500   # merge segments with gaps shorter than this
    vad_speech_pad_ms: int = 200             # pad each speech segment with silence

    # ── Streaming ─────────────────────────────────────────────────────────
    stream_chunk_duration_s: float = 0.5     # size of each audio chunk read from mic
    stream_buffer_duration_s: float = 30.0   # max rolling buffer before forced flush
    stream_silence_timeout_s: float = 2.0    # flush after this much silence (VAD-gated)
    stream_sample_rate: int = 44100          # capture rate (downsampled to 16kHz by SoX)
    stream_input_device: Optional[int] = None  # None = system default mic
```

### 4.2 `TranscriptionResult` (pipeline.py)

```python
@dataclass
class TranscriptionResult:
    text: str                        # cleaned, stripped transcript
    segments: list[dict]             # raw faster-whisper segment objects
    language: str                    # detected language code
    duration_s: float                # audio duration in seconds
    processing_time_s: float         # wall-clock time for full pipeline
    real_time_factor: float          # processing_time_s / duration_s (< 1.0 = faster than RT)
    audio_source: str                # "file:<path>" or "stream:<timestamp>"

    # Confidence proxy — faster-whisper gives per-segment log probs
    avg_log_prob: float              # mean of segment.avg_logprob; higher = more confident
    no_speech_prob: float            # mean of segment.no_speech_prob; > 0.6 = likely silence

    def is_likely_valid(self) -> bool:
        """Heuristic: probably real speech, not hallucination or silence."""
        return (
            self.no_speech_prob < 0.6
            and self.avg_log_prob > -1.0
            and len(self.text.strip()) > 0
        )
```

### 4.3 `HamRadioSTT` (pipeline.py) — Main Class

```python
class HamRadioSTT:

    def __init__(self, config: PipelineConfig = PipelineConfig()):
        """
        Initialise all models eagerly.
        Downloads faster-whisper model if not cached (~1.5GB for distil-large-v3.5).
        Raises EnvironmentError if SoX is not on PATH.
        Logs model load times.
        """

    def transcribe_file(self, path: str | Path) -> TranscriptionResult:
        """
        Transcribe a WAV/MP3/FLAC file through the full pipeline.
        Accepts any sample rate / channel count — SoX normalises to 16kHz mono.
        Returns TranscriptionResult.
        Raises FileNotFoundError, AudioProcessingError.
        """

    def transcribe_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> TranscriptionResult:
        """
        Transcribe a numpy float32 array (already at 16kHz mono preferred).
        Useful when audio is captured by other means (SDR, etc.).
        Runs SoX via temp file, then DFN3, then Whisper.
        """

    def start_stream(
        self,
        callback: Callable[[TranscriptionResult], None],
        device: Optional[int] = None,
    ) -> "StreamingSession":
        """
        Opens mic input. Returns a StreamingSession context manager.
        callback is called from the processing thread each time a speech
        segment is flushed and transcribed.

        Usage:
            def on_text(result):
                if result.is_likely_valid():
                    print(result.text)

            with stt.start_stream(on_text):
                time.sleep(60)  # listen for 60 seconds
        """

    def list_audio_devices(self) -> list[dict]:
        """Return sounddevice device list for device selection."""
```

### 4.4 `StreamingSession` (streaming.py)

```python
class StreamingSession:
    """
    Context manager returned by HamRadioSTT.start_stream().
    Manages the capture thread, rolling audio buffer, VAD-gated flush logic,
    and the processing thread pool.
    """

    def __enter__(self) -> "StreamingSession": ...
    def __exit__(self, *args): ...

    def stop(self): ...          # graceful shutdown, flushes remaining buffer
    def pause(self): ...         # stop capture without tearing down models
    def resume(self): ...

    @property
    def is_running(self) -> bool: ...
```

#### Streaming flush logic (implement exactly this):

```
capture thread  →  raw_chunk_queue  →  processing thread

Processing thread loop:
  1. Drain raw_chunk_queue into rolling_buffer (numpy concat)
  2. If rolling_buffer >= stream_buffer_duration_s  →  force flush
  3. Run lightweight energy VAD on last stream_chunk_duration_s of buffer:
       rms = np.sqrt(np.mean(chunk[-n_samples:]**2))
       is_silence = rms < SILENCE_THRESHOLD  (tune: ~0.01 for 16kHz float32)
  4. Increment silence_counter if is_silence, reset if speech
  5. If silence_counter * stream_chunk_duration_s >= stream_silence_timeout_s:
       flush_buffer()
       rolling_buffer = np.array([])

flush_buffer():
  1. Copy current rolling_buffer
  2. Reset rolling_buffer
  3. Submit copy to ThreadPoolExecutor (max_workers=1 to preserve order)
  4. Executor calls: result = pipeline.transcribe_audio(buffer, 16000)
  5. Call user callback(result)
```

**Silence threshold:** Do not hardcode. Compute it during a 1-second calibration period at
`start_stream()` time: capture 1s of ambient audio, set threshold to `3 * rms_of_ambient`.
Log the calibrated threshold so the user can see it.

---

## 5. Stage Implementations

### 5.1 SoX Stage (stages/sox_preprocess.py)

```python
def sox_preprocess(
    input_path: str,
    output_path: str,
    config: PipelineConfig,
) -> None:
    """
    Run SoX as a subprocess. Raise AudioProcessingError on non-zero exit.
    """
```

**SoX command to build (in this order, order matters):**

```bash
sox {input_path} \
    --rate {target_sample_rate} \
    --channels {target_channels} \
    --encoding signed-integer \
    --bits 16 \
    {output_path} \
    highpass {sox_highpass_hz} \
    lowpass {sox_lowpass_hz} \
    compand 0.01,0.2 -60,-60,-30,-10,0,-3 -3 -60 0.1 \   # compress dynamics
    norm {sox_norm_level_db}
```

Notes:
- The `compand` effect serves as both noise gate and soft limiter — it approximates AGC
  reversal common in radio receivers. Tune the transfer curve for your specific radio.
- Always write to a temp `.wav` file, never in-place.
- Use `subprocess.run(..., capture_output=True, check=True)` and re-raise stderr as
  `AudioProcessingError` with the SoX error message included.
- For streaming input (numpy array), write to a `tempfile.NamedTemporaryFile` first,
  run SoX, read back result, delete temp file. Use a `try/finally` block.

### 5.2 DeepFilterNet 3 Stage (stages/denoise.py)

```python
def load_dfn_model() -> tuple:
    """Load and cache DeepFilterNet 3 model. Call once at init."""
    from df.enhance import init_df, enhance
    model, df_state, _ = init_df()
    return model, df_state

def dfn_enhance(
    audio: np.ndarray,          # float32, any sample rate
    sample_rate: int,
    model,
    df_state,
    config: PipelineConfig,
) -> np.ndarray:
    """
    Run DeepFilterNet 3 enhancement.
    DFN3 requires 48kHz input. Resample if needed (use scipy.signal.resample_poly).
    After enhancement, resample back to target_sample_rate (16kHz).
    Returns float32 numpy array at 16kHz.
    """
    from df.enhance import enhance
    import torch
    # DFN3 expects float32 tensor, shape: (channels, samples)
    # Input MUST be 48kHz — resample to 48k before calling enhance()
    # Output is also 48kHz — resample back to 16k after
```

**Important DFN3 gotcha:** DeepFilterNet 3 operates at 48kHz internally. The pipeline
receives 16kHz audio from SoX. You must: resample 16kHz→48kHz → enhance → resample 48kHz→16kHz.
Use `scipy.signal.resample_poly` with integer ratios (e.g., 3:1 for 48k→16k) to avoid
quality loss. Do not use librosa for this — scipy is faster and has no hidden resampling
quality issues at these rates.

### 5.3 Transcription Stage (stages/transcribe.py)

```python
def load_whisper_model(config: PipelineConfig) -> WhisperModel:
    from faster_whisper import WhisperModel
    return WhisperModel(
        config.whisper_model,
        device=config.whisper_device,
        compute_type=config.whisper_compute_type,
        # Model cache at ~/.cache/huggingface/hub by default
    )

def transcribe(
    audio: np.ndarray,           # float32, 16kHz mono
    model: WhisperModel,
    config: PipelineConfig,
) -> dict:
    """
    Run faster-whisper transcription.
    Returns dict with: text, segments, language, duration_s, avg_log_prob, no_speech_prob
    """
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
        condition_on_previous_text=False,  # IMPORTANT: prevents hallucination loops
        log_prob_threshold=-1.0,           # reject segments below this confidence
        no_speech_threshold=0.6,           # suppress pure-silence segments
    )
    # Materialise the generator ONCE — iterating it twice gives empty results
    segments = list(segments)
    ...
```

**Critical:** `condition_on_previous_text=False` is essential for automation use cases.
Without it, Whisper can enter repetition loops on static/noise, generating a stream of
hallucinated repeated phrases that would poison an automation system.

---

## 6. Error Handling

Define a small exception hierarchy in `__init__.py`:

```python
class HamSTTError(Exception):          # base
class AudioProcessingError(HamSTTError):  # SoX failed, bad audio format
class ModelLoadError(HamSTTError):     # model download failed, disk full
class StreamError(HamSTTError):        # sounddevice error, device not found
```

All public methods should only raise from this hierarchy (wrap third-party exceptions).

---

## 7. CLI (cli.py)

Implement with `argparse` (no extra deps):

```
# Transcribe a file
python -m ham_radio_stt file path/to/audio.wav

# Stream from default mic
python -m ham_radio_stt stream

# Stream from specific device
python -m ham_radio_stt stream --device 2

# List audio devices
python -m ham_radio_stt devices

# Use a different model (faster but less accurate)
python -m ham_radio_stt file audio.wav --model large-v3-turbo

# Output as JSON (for piping to automation)
python -m ham_radio_stt file audio.wav --json
```

JSON output format (for `--json` flag):

```json
{
  "text": "W1AW this is K9ABC, copy you five by nine, over.",
  "is_valid": true,
  "duration_s": 4.2,
  "processing_time_s": 1.8,
  "real_time_factor": 0.43,
  "avg_log_prob": -0.21,
  "no_speech_prob": 0.03,
  "source": "file:audio.wav",
  "segments": [
    { "start": 0.0, "end": 4.2, "text": "W1AW this is K9ABC, copy you five by nine, over." }
  ]
}
```

---

## 8. Tests (tests/test_pipeline.py)

Write tests using `pytest`. Do not require a real microphone or internet connection.
Generate synthetic test fixtures programmatically.

```python
# Fixture: generate a WAV with a 440Hz tone + white noise
# Use numpy + soundfile — no external audio files needed
@pytest.fixture
def noisy_tone_wav(tmp_path):
    sr = 44100
    t = np.linspace(0, 3, sr * 3)
    tone = 0.3 * np.sin(2 * np.pi * 440 * t)
    noise = 0.1 * np.random.randn(len(t))
    audio = (tone + noise).astype(np.float32)
    path = tmp_path / "test_tone.wav"
    sf.write(str(path), audio, sr)
    return path
```

Required test cases:
1. `test_sox_preprocess_output_shape` — verify output is 16kHz mono
2. `test_sox_preprocess_duration_preserved` — output duration ≈ input duration ±5%
3. `test_dfn3_reduces_rms_noise` — RMS of enhanced audio < RMS of noisy input
4. `test_pipeline_config_defaults` — PipelineConfig() has all expected fields
5. `test_transcription_result_is_likely_valid` — test the heuristic with edge cases
6. `test_file_mode_end_to_end` — full pipeline on fixture, result is TranscriptionResult
7. `test_streaming_session_starts_and_stops` — mock sounddevice, verify no exceptions
8. `test_json_output_format` — CLI --json output parses correctly

---

## 9. Performance Targets (Intel Mac, 2019+)

These are aspirational targets to test against with your actual hardware:

| Metric | Target |
|---|---|
| Model load time (cold) | < 20s (network), < 5s (cached) |
| File mode RTF | < 0.5× (process 10s audio in < 5s) |
| Stream flush latency | < 3s after speech ends |
| Memory footprint | < 3GB RSS with distil-large-v3.5 |
| CPU during transcription | < 400% (4 cores) |

---

## 10. Known Gotchas to Document in Code Comments

1. **DFN3 requires 48kHz** — resample before and after, not just after.
2. **faster-whisper generator** — calling `list(segments)` once is mandatory; the generator
   is lazy and stateful.
3. **`condition_on_previous_text=False`** — critical for automation; Whisper will loop on noise otherwise.
4. **SoX `compand` argument order** — the transfer function points are `in,out` pairs, not gain curves.
   Getting this wrong produces distorted or silent output with no error.
5. **Temp file cleanup** — always use `try/finally` or `contextlib.ExitStack` when writing
   temp WAVs for SoX processing; a crash mid-pipeline will otherwise litter `/tmp`.
6. **Thread safety** — `WhisperModel` and DeepFilterNet model are not thread-safe. Use a
   single `threading.Lock` in `HamRadioSTT` if you ever call `transcribe_audio` concurrently.
7. **sounddevice block size** — set to `int(config.stream_sample_rate * config.stream_chunk_duration_s)`;
   mismatched block size causes audio dropouts.
8. **Float32 normalisation** — sounddevice returns float32 in [-1, 1]. SoX expects int16 by default.
   Write to WAV with `soundfile` (handles conversion) rather than raw bytes.

---

## 11. Out of Scope for Prototype

- Fine-tuning Whisper on custom ham radio audio (separate project, needs GPU)
- SDR (Software Defined Radio) direct integration — stub `transcribe_audio(np.ndarray)`
  is sufficient; SDR integration is a separate input adapter
- Speaker diarisation / callsign extraction — post-processing on `result.text`
- Windows / Linux support — macOS Intel only for now

---

*Spec version 1.0 — March 2026*