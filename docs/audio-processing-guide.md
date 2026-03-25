# Audio Processing & Filter Tuning Guide

This guide covers the audio processing pipeline in ham-to-text: how each stage works, how to tune it for different conditions, and how to troubleshoot common issues.

## Pipeline Overview

Audio flows through these stages in order:

```
Input Audio -> SoX Preprocess -> VAD Segmentation -> Denoiser (per segment) -> Whisper Transcription
```

1. **SoX Preprocess** -- bandpass filtering, clarity EQ, dynamic range compression, normalization
2. **VAD** -- WebRTC voice activity detection with energy-based gap recovery
3. **Denoiser** -- spectral gating noise suppression (noisereduce) per segment
4. **Whisper** -- speech-to-text with conversational context from prior segments

Use `--debug-audio DIR` to save intermediate WAV files after each stage for inspection:

```bash
ham-to-text file audio.wav --denoiser noisereduce --debug-audio /tmp/debug
```

This produces:

```
/tmp/debug/
  00_input.wav                  # Raw input
  01_sox_preprocess.wav         # After SoX
  02_noisereduce_seg000.wav     # After denoiser (per VAD segment)
  02_noisereduce_seg001.wav
  ...
```

---

## Stage 1: SoX Preprocessing

SoX applies five effects in sequence: highpass, lowpass, EQ boost, compand, and norm. The order matters -- each effect feeds into the next.

### Highpass Filter

Removes frequencies below the cutoff. Uses a 2nd-order Butterworth filter (12 dB/octave rolloff).

| Config | Default | TOML |
|--------|---------|------|
| `sox_highpass_hz` | `200` | `[sox] highpass_hz = 200` |

**What it removes:** Mains hum (50/60 Hz), power supply buzz, wind rumble, mechanical vibrations.

**Tuning guidance:**

| Value | Use Case |
|-------|----------|
| 100 Hz | Preserve deep male voices, more permissive |
| **200 Hz** | **Default.** Good general-purpose for ham radio |
| 300 Hz | Matches traditional SSB bandwidth (300-2700 Hz). More aggressive noise removal but thins male voices |

Human voice fundamentals range from ~85 Hz (deep male) to ~255 Hz (female). The 200 Hz default trims the lowest fundamentals while keeping most voice energy intact.

### Lowpass Filter

Removes frequencies above the cutoff. Also a 2nd-order Butterworth.

| Config | Default | TOML |
|--------|---------|------|
| `sox_lowpass_hz` | `3400` | `[sox] lowpass_hz = 3400` |

**What it removes:** High-frequency hiss, static, digital noise, adjacent-channel interference.

**Tuning guidance:**

| Value | Use Case |
|-------|----------|
| 2700 Hz | Strict SSB bandwidth. Maximum noise removal but can sound muffled |
| **3400 Hz** | **Default.** ITU telephony bandwidth. Good clarity/noise balance |
| 4000 Hz | FM signals or cleaner recordings. Preserves more consonant detail |

Consonants (s, t, f) carry energy up to 4-8 kHz. Keeping the cutoff at 3400 Hz preserves most intelligibility while cutting noise above the voice band.

**Together, highpass + lowpass form a bandpass filter** that isolates the voice-frequency range.

### Clarity EQ Boost

Narrowband ham radio audio (8kHz SSB) concentrates most energy below 800 Hz, making voices sound muffled. The EQ boost lifts the 1-3 kHz clarity range where consonants and speech intelligibility live.

| Config | Default | TOML |
|--------|---------|------|
| `sox_eq_center_hz` | `1800` | `[sox] eq_center_hz = 1800` |
| `sox_eq_boost_db` | `6.0` | `[sox] eq_boost_db = 6.0` |

The EQ uses a parametric equalizer with Q=1.5, centered at the configured frequency. Set `eq_boost_db = 0` to disable.

**Tuning guidance:**

| Boost | Use Case |
|-------|----------|
| 0 dB | Disabled. Use for already-clear audio or wideband FM |
| 3-4 dB | Light boost. Slightly improves clarity without changing character |
| **6 dB** | **Default.** Noticeable clarity improvement for narrowband SSB |
| 8-10 dB | Aggressive. Can sound harsh but maximizes intelligibility on very muffled signals |

**Center frequency:** 1800 Hz is a good default for SSB voice. Lower (1200-1500 Hz) emphasizes warmth, higher (2000-2500 Hz) emphasizes crispness.

### Compand (Dynamic Range Compression)

The compand effect is a dynamic range processor -- it adjusts volume based on signal level. This is the most complex effect in the chain.

The current (hardcoded) parameters are:

```
compand 0.01,0.2  -60,-60,-30,-10,0,-3  -3  -60  0.1
```

Here is what each part does:

#### Attack and Decay: `0.01,0.2`

- **Attack = 0.01s (10 ms):** How fast the compressor reacts to audio getting louder. Fast attack catches pops and bursts quickly.
- **Decay = 0.2s (200 ms):** How fast it reacts to audio getting quieter. Slower decay prevents choppy "pumping" artifacts, while still tracking speech cadence.

#### Transfer Function: `-60,-60,-30,-10,0,-3`

Read as input/output dB pairs defining how volume is remapped:

| Input Level | Output Level | Effect |
|-------------|-------------|--------|
| -60 dB | -60 dB | Noise floor -- no change (gate threshold) |
| -30 dB | -10 dB | **+20 dB boost** -- pulls up weak signals |
| 0 dB | -3 dB | **-3 dB reduction** -- soft-limits loud signals |

Between these points, SoX interpolates linearly in dB space. The net effect:

- **Below -60 dB:** Treated as silence/noise, passed through unchanged
- **-60 to -30 dB:** Weak signals get dramatically boosted (this is where faint radio signals live)
- **-30 to 0 dB:** Heavy compression -- the 30 dB input range maps to only 7 dB of output. Loud and medium signals come out nearly the same volume

This approximates an AGC (Automatic Gain Control) that normalizes voice levels across the dynamic range.

#### Gain: `-3`

Fixed gain applied after the transfer function. Shifts everything down 3 dB to prevent clipping after the boost from compression.

#### Initial Volume: `-60`

Assumed signal level at the start of processing, before the attack/decay envelope has analyzed the actual audio. Set to -60 dB (silence) so the compressor starts quiet and ramps up, preventing a pop at the beginning.

#### Delay (Look-ahead): `0.1`

Look-ahead of 100 ms. The compressor peeks 100 ms into the future to pre-attenuate bursts before they arrive. This produces much smoother output. Adds 100 ms latency (irrelevant for file processing, minor for streaming).

### Norm (Normalization)

Scales the entire audio so the peak sample hits a target level. This is a linear, whole-file operation -- it does not change dynamic range.

| Config | Default | TOML |
|--------|---------|------|
| `sox_norm_level_db` | `-3.0` | `[sox] norm_level_db = -3.0` |

**Tuning guidance:**

| Value | Use Case |
|-------|----------|
| -1 dB | Very hot signal, minimal headroom. Risks clipping in some codecs |
| **-3 dB** | **Default.** Standard for speech processing |
| -6 dB | Conservative. Use if downstream processing adds gain |

Norm must come last because it is a two-pass effect (scans for peak, then applies gain). Earlier effects would invalidate a prior normalization.

### Why This Order Matters

```
highpass -> lowpass -> EQ -> compand -> norm
```

1. **Filters first:** Highpass and lowpass remove out-of-band noise *before* the compressor sees it. If compand came first, it would react to 60 Hz hum or high-frequency hiss and incorrectly adjust gain based on noise rather than voice.
2. **EQ second:** Boosts the clarity range while the signal is still band-limited but before compression. This ensures the compressor sees the EQ'd frequency balance.
3. **Compand third:** Now operating on band-limited, EQ'd audio, level detection accurately reflects voice energy only.
4. **Norm last:** Guarantees the final output peaks at exactly the target level regardless of what earlier effects did.

**Bad orderings cause real problems:** Putting compand before the filters causes pumping artifacts. Putting norm before compand wastes the normalization.

### Adjustments for Common Scenarios

**Very noisy signals (heavy QRM/QRN):**
```toml
[sox]
highpass_hz = 300    # Tighter bandpass
lowpass_hz = 2700    # Strict SSB bandwidth
```

**Muffled/unclear voices:**
Increase the EQ boost or shift the center frequency higher:
```toml
[sox]
eq_center_hz = 2000
eq_boost_db = 8.0
```

**Weak/distant stations:**
The default compand curve already boosts weak signals by up to 20 dB. If signals are extremely weak, the denoiser stage is more effective than tightening SoX filters.

**FM signals (wider bandwidth):**
```toml
[sox]
lowpass_hz = 4000    # FM has wider bandwidth than SSB
eq_boost_db = 0      # FM audio is usually already clear
```

**Clipping/overdriven signals:**
Lower the norm target to give more headroom:
```toml
[sox]
norm_level_db = -6.0
```

---

## Stage 2: VAD (Voice Activity Detection)

After SoX preprocessing, the pipeline uses [WebRTC VAD](https://github.com/wiseman/py-webrtcvad) to segment audio into speech chunks before denoising and transcription. WebRTC VAD was chosen over neural VADs (Silero, pyannote) because it was designed for telephony-grade narrowband audio -- the same domain as ham radio SSB.

### How It Works

WebRTC VAD uses a GMM (Gaussian Mixture Model) approach to classify 10/20/30 ms audio frames as speech or non-speech. Unlike neural VADs trained on clean wideband speech, it doesn't have strong expectations about what speech "should" sound like at higher frequencies, making it much more reliable for degraded radio audio.

After WebRTC VAD segmentation, an **energy-based fallback** recovers speech in gaps the VAD missed. Any gap with RMS energy above the threshold is included as a segment. This catches speech that even WebRTC VAD misses on very noisy signals.

### Configuration

| Config | Default | TOML | Description |
|--------|---------|------|-------------|
| `vad_filter` | `true` | `[vad] filter = true` | Enable/disable VAD |
| `vad_aggressiveness` | `0` | `[vad] aggressiveness = 0` | 0=least aggressive (catches more speech), 3=most aggressive |
| `vad_frame_ms` | `30` | `[vad] frame_ms = 30` | Frame size: 10, 20, or 30 ms |
| `vad_min_silence_ms` | `300` | `[vad] min_silence_ms = 300` | Min silence gap to split segments |
| `vad_speech_pad_ms` | `300` | `[vad] speech_pad_ms = 300` | Padding around speech segments |
| `vad_energy_threshold` | `0.02` | `[vad] energy_threshold = 0.02` | RMS threshold for gap recovery |

### Tuning Guidance

**Missing speech (words/phrases cut off):**
- Lower aggressiveness: `aggressiveness = 0`
- Increase padding: `speech_pad_ms = 500`
- Lower energy threshold: `energy_threshold = 0.01`

**Too many false positives (noise transcribed as speech):**
- Raise aggressiveness: `aggressiveness = 2` or `3`
- Raise energy threshold: `energy_threshold = 0.05`

**Speech split into too many small segments:**
- Increase min silence: `min_silence_ms = 500` or `700`

**Words clipped at segment boundaries:**
- Increase padding: `speech_pad_ms = 400` or `500`

---

## Stage 3: Noisereduce Denoiser (Recommended)

[Noisereduce](https://github.com/timsainb/noisereduce) uses spectral gating to suppress noise. It estimates a noise profile per frequency band and attenuates bins below the threshold. Unlike DeepFilterNet, it works natively with any sample rate including 8kHz and 16kHz, making it ideal for narrowband ham radio audio.

### Installation

```bash
uv run --extra noisereduce ham-to-text file audio.wav --denoiser noisereduce
```

### Configuration

| Config | Default | TOML | Description |
|--------|---------|------|-------------|
| `denoiser` | `"none"` | `[denoiser] name = "noisereduce"` | Enable noisereduce |
| `nr_stationary` | `false` | `[noisereduce] stationary = false` | Noise mode (see below) |
| `nr_prop_decrease` | `0.75` | `[noisereduce] prop_decrease = 0.75` | Noise reduction strength (0.0-1.0) |
| `nr_n_fft` | `512` | `[noisereduce] n_fft = 512` | FFT size |
| `nr_time_constant_s` | `2.0` | `[noisereduce] time_constant_s = 2.0` | Smoothing window for noise estimation |

### Stationary vs Non-Stationary Mode

- **`stationary = false` (default, recommended):** Dynamically updates the noise estimate over time. Better for ham radio where noise varies (fading, QRM, band conditions changing).
- **`stationary = true`:** Uses a fixed noise estimate from the beginning of the audio. Better when the noise is constant (steady hiss, constant hum) and you want to avoid the algorithm adapting to speech.

### Noise Reduction Strength (`prop_decrease`)

Controls how much of the estimated noise to remove. Range 0.0 (no reduction) to 1.0 (full removal).

| Value | Behavior |
|-------|----------|
| 0.3-0.5 | Light. Preserves more of the original signal, some noise remains |
| **0.75** | **Default.** Good balance for most radio audio |
| 0.9-1.0 | Aggressive. Maximum noise removal but risks "musical noise" artifacts |

**For ham radio:** Start at 0.75. If speech sounds distorted or you hear chirping artifacts, lower to 0.5. If too much noise remains, raise to 0.9.

### Troubleshooting

**"Musical noise" / chirping artifacts:**
Lower the reduction strength:
```toml
[noisereduce]
prop_decrease = 0.5
```

**Speech sounds distorted:**
The reduction is too aggressive:
```toml
[noisereduce]
prop_decrease = 0.5
time_constant_s = 3.0    # Slower adaptation
```

**Not enough noise removed:**
```toml
[noisereduce]
prop_decrease = 0.9
```

---

## DeepFilterNet 3 Denoiser (Not Recommended for Narrowband)

DeepFilterNet 3 is a neural network denoiser that operates at 48 kHz internally. It is **not recommended for narrowband ham radio audio** (8-16 kHz) because:

- It was trained on wideband (48 kHz) clean speech + noise
- Narrowband audio upsampled to 48 kHz looks like heavily filtered/degraded audio to the model
- The model's recurrent layers progressively classify the narrowband signal as noise, causing audio to fade out within segments

DeepFilterNet may still work well for **wideband audio sources** (FM, internet streams, VoIP recordings). If you want to use it:

```bash
uv run --extra deepfilter ham-to-text file audio.wav --denoiser deepfilter
```

### Configuration

| Config | Default | TOML | Description |
|--------|---------|------|-------------|
| `dfn_attenuation_limit` | `100.0` | `[deepfilter] attenuation_limit = 100.0` | Max noise suppression in dB |
| `dfn_post_filter` | `true` | `[deepfilter] post_filter = true` | Extra suppression of noisy bins |

**Version constraints:** The `deepfilter` extra pins `torch` and `torchaudio` to 2.2.x due to API compatibility requirements with `deepfilternet 0.5.x`.

---

## Conversational Context

The pipeline carries transcription text forward between segments. When processing segment N, Whisper receives the text from the previous segments as part of its prompt. This helps maintain consistency for:

- Callsigns and station identifiers heard earlier in the conversation
- Names, ranks, and terminology established in prior segments
- Conversational flow and sentence structure

| Config | Default | TOML | Description |
|--------|---------|------|-------------|
| `whisper_context_segments` | `5` | `[whisper] context_segments = 5` | Number of prior segments to include as context |

Set to `0` to disable context carry-forward. Higher values provide more context but may cause Whisper to hallucinate or repeat phrases from earlier segments.

---

## Tuning Presets

### Clean, Strong Signal (local repeater, FM)

The signal is already good. Minimal processing needed:

```toml
[sox]
highpass_hz = 100
lowpass_hz = 4000
eq_boost_db = 0        # FM audio is usually clear

[denoiser]
name = "none"
```

### Typical HF SSB (moderate noise)

The default settings work well here. Add noisereduce for better results:

```toml
[denoiser]
name = "noisereduce"

[noisereduce]
prop_decrease = 0.75
```

### Weak/Distant Station (low SNR)

Tighten the bandpass to reduce noise, use aggressive denoising:

```toml
[sox]
highpass_hz = 300
lowpass_hz = 2700
eq_boost_db = 8.0       # Boost clarity on weak signals

[denoiser]
name = "noisereduce"

[noisereduce]
prop_decrease = 0.9

[vad]
aggressiveness = 0       # Catch every bit of speech
energy_threshold = 0.01  # Low threshold for weak signals
```

### Heavy QRM (Adjacent-channel Interference)

Narrow the bandwidth aggressively:

```toml
[sox]
highpass_hz = 300
lowpass_hz = 2400

[denoiser]
name = "noisereduce"

[noisereduce]
prop_decrease = 0.8
```

### MARS / Military Auxiliary Radio

MARS traffic uses SSB with relatively consistent signal quality. Good baseline:

```toml
[sox]
highpass_hz = 200
lowpass_hz = 3400
eq_center_hz = 1800
eq_boost_db = 6.0

[denoiser]
name = "noisereduce"

[noisereduce]
prop_decrease = 0.75

[vad]
aggressiveness = 0
speech_pad_ms = 300
```

---

## Debugging Workflow

When transcription quality is poor, use `--debug-audio` to isolate the problem:

1. **Listen to `00_input.wav`** -- Is the raw audio intelligible to a human? If not, no amount of processing will help.

2. **Listen to `01_sox_preprocess.wav`** -- Is the voice clearer after filtering?
   - If it sounds muffled: increase `eq_boost_db` or raise `eq_center_hz`
   - If there is still hum/rumble: tighten the highpass (`highpass_hz` higher)
   - If the volume is uneven: the compand is not aggressive enough for this signal

3. **Listen to `02_noisereduce_seg*.wav`** -- Did denoising help or hurt?
   - Chirping/musical noise: lower `prop_decrease`
   - Still noisy: raise `prop_decrease`
   - Speech distorted: lower `prop_decrease` and increase `time_constant_s`
   - Sounds worse than SoX output: try `--denoiser none` to skip it

4. **Check segment coverage** -- Are segments missing speech?
   - Compare segment count and durations against what you hear in `01_sox_preprocess.wav`
   - If speech is being missed: lower `vad_aggressiveness`, lower `vad_energy_threshold`
   - If segments are clipped: increase `vad_speech_pad_ms`

5. **Compare transcription** with and without the denoiser to see which produces better text.

### Quick A/B Test

```bash
# Without denoiser
ham-to-text file audio.wav --denoiser none

# With denoiser
ham-to-text file audio.wav --denoiser noisereduce

# With debug audio for listening
ham-to-text file audio.wav --denoiser noisereduce --debug-audio /tmp/debug
```

---

## Configuration Reference

All settings can be specified via TOML config files or CLI flags. Precedence (highest wins):

```
CLI flags > --config file > ./hamstt.toml > ~/.config/hamstt/config.toml > defaults
```

### Full Example Config

```toml
[whisper]
model = "distil-large-v3"
language = "en"
beam_size = 5
best_of = 5
temperature = 0.0
compute_type = "int8"
device = "cpu"
context_segments = 5

[sox]
highpass_hz = 200
lowpass_hz = 3400
eq_center_hz = 1800
eq_boost_db = 6.0
norm_level_db = -3.0

[denoiser]
name = "noisereduce"

[noisereduce]
stationary = false
prop_decrease = 0.75
n_fft = 512
time_constant_s = 2.0

[vad]
filter = true
aggressiveness = 0
frame_ms = 30
min_silence_ms = 300
speech_pad_ms = 300
energy_threshold = 0.02

[deepfilter]
attenuation_limit = 80.0
post_filter = true

[streaming]
chunk_duration_s = 0.5
buffer_duration_s = 30.0
silence_timeout_s = 1.5
sample_rate = 44100
```
