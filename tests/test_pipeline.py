from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ham_radio_stt.config import PipelineConfig
from ham_radio_stt.pipeline import Pipeline
from ham_radio_stt.result import TranscriptionResult


@pytest.fixture
def mock_pipeline():
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
            MockSox.return_value.name = "sox_preprocess"
            with patch("ham_radio_stt.pipeline.get_denoiser") as MockDenoiser:
                MockDenoiser.return_value.process.return_value = (
                    np.zeros(16000, dtype=np.float32),
                    16000,
                )
                MockDenoiser.return_value.name = "none"
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
            with patch("pathlib.Path.exists", return_value=True):
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
