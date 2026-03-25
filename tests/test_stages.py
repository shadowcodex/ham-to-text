import shutil
import subprocess
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from ham_to_text import AudioProcessingError, ModelLoadError
from ham_to_text.config import PipelineConfig
from ham_to_text.stages.sox_preprocess import SoxPreprocess
from ham_to_text.stages.denoise import (
    NoOpDenoiser,
    get_denoiser,
    register_denoiser,
    registered_denoisers,
)


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
        assert processed.ndim == 1

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

        from ham_to_text.stages import denoise
        original = denoise._REGISTRY.copy()
        monkeypatch.setattr(denoise, "_REGISTRY", {**original})

        register_denoiser("fake", FakeDenoiser)
        denoiser = get_denoiser("fake", PipelineConfig())
        assert denoiser.name == "fake"
