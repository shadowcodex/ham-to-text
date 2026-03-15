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
    tone_len = sr * 2
    silence_len = sr * 1
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
